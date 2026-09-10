import os
from os import path
import pandas as pd
import numpy as np
import tqdm
import pickle
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data.dataset import Dataset

from misc.utils import divide_chunks
from dataset.vocab_unsupervised import UnsupervisedVocabulary

logger = logging.getLogger(__name__)
log = logger


class ContabilDataset(Dataset):
    def __init__(self,
                 mlm,
                 group_by=None,
                 group_by_ids=None,
                 seq_len=10,
                 num_bins=10,
                 cached=True,
                 root="./data/contabil/",
                 fname="contabil_trans",
                 vocab_dir="checkpoints",
                 fextension="",
                 nrows=None,
                 flatten=False,
                 stride=5,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_group_col=False):

        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.fextension = f'_{fextension}' if fextension else ''
        self.cached = cached
        self.group_by_ids = group_by_ids
        self.return_labels = return_labels
        self.skip_group_col = skip_group_col

        self.group_by: str | None = group_by

        self.mlm = mlm
        self.trans_stride = stride

        self.flatten = flatten

        self.vocab = UnsupervisedVocabulary(adap_thres)
        self.seq_len = seq_len
        self.encoder_fit = {}

        self.trans_table = None
        self.data = []
        self.labels = []
        self.window_label = []

        self.ncols = None
        self.num_bins = num_bins
        self.encode_data()
        self.init_vocab()
        self.prepare_samples()
        self.save_vocab(vocab_dir)

    def __getitem__(self, index):
        if self.flatten:
            return_data = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_data = torch.tensor(self.data[index], dtype=torch.long).reshape(self.seq_len, -1)

        if self.return_labels:
            return_data = (return_data, torch.tensor(self.labels[index], dtype=torch.long))

        return return_data

    def __len__(self):
        return len(self.data)

    def save_vocab(self, vocab_dir):
        file_name = path.join(vocab_dir, f'vocab{self.fextension}.nb')
        log.info(f"saving vocab at {file_name}")
        self.vocab.save_vocab(file_name)

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    @staticmethod
    def timeEncoder(X):
        # X espera um DataFrame contendo 'entry_date' e opcionalmente 'creation_time'
        dt = pd.to_datetime(X['entry_date'], errors='coerce')
        
        # Se houver hora, podemos extrair a hora; senão default para 0
        hour = 0
        if 'creation_time' in X.columns:
            # Extrai apenas a hora do formato hh:mm:ss
            hour = pd.to_datetime(X['creation_time'], format='%H:%M:%S', errors='coerce').dt.hour.fillna(0)

        d = pd.DataFrame({
            'year': dt.dt.year.fillna(2026),
            'month': dt.dt.month.fillna(1),
            'day': dt.dt.day.fillna(1),
            'hour': hour
        })
        
        # Converte para um inteiro representativo unificado para o vocabulário
        d_int = (d['year'] * 1000000 + d['month'] * 10000 + d['day'] * 100 + d['hour']).astype(int)
        return pd.DataFrame(d_int)

    @staticmethod
    def amountEncoder(X_series):
        # 1. Converte para float e substitui nulos/NaNs por 0.0
        s = X_series.astype(float).fillna(0.0)

        # 2. Aplica o Signed Log: sign(x) * ln(1 + |x|)
        # Usamos np.log1p(x), que calcula ln(1 + x) com alta precisão numérica para valores próximos de zero.
        amt = np.sign(s) * np.log1p(np.abs(s))

        return pd.DataFrame(amt)


    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def nanZero(X):
        return X.where(pd.notnull(X), 0)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1  # Clip edges
        return quant_inputs

    def user_level_data(self):
        group_suffix = f"_{self.group_by}" if self.group_by else "_global"
        fname = path.join(self.root, f"preprocessed/{self.fname}{group_suffix}{self.fextension}.pkl")
        trans_data, trans_labels = [], []

        if self.cached and path.isfile(fname):
            log.info(f"Carregando dados do cache: {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            columns_names = cached_data["columns"]

        else:
            # Como não há rótulo, TODAS as colunas do trans_table são features!
            all_columns = list(self.trans_table.columns)
            feature_cols = all_columns.copy()

            if self.group_by and self.group_by in feature_cols and self.skip_group_col:
                feature_cols.remove(self.group_by)

            columns_names = feature_cols.copy()

            # CASO 1: Agrupado por uma coluna específica (ex: Conta, Lote, Usuario)
            if self.group_by and self.group_by in self.trans_table.columns:
                unique_groups = self.trans_table[self.group_by].unique()
                log.info(f"Agrupando transações pela coluna '{self.group_by}' ({len(unique_groups)} grupos encontrados).")

                for group_val in tqdm.tqdm(unique_groups):
                    group_df = self.trans_table.loc[self.trans_table[self.group_by] == group_val]
                    
                    
                    # Alternativa ultra-rápida ao iterrows:
                    group_trans = group_df[feature_cols].to_numpy().flatten().tolist()
                    group_labels = [0] * len(group_df)

                    trans_data.append(group_trans)
                    trans_labels.append(group_labels)

            # CASO 2: Linha do tempo global contínua
            else:
                log.info("Nenhum 'group_by' especificado. Tratando toda a base como uma linha do tempo global.")
                
                
                # Alternativa ultra-rápida ao iterrows:
                group_trans = self.trans_table[feature_cols].to_numpy().flatten().tolist()
                group_labels = [0] * len(self.trans_table)

                trans_data.append(group_trans)
                trans_labels.append(group_labels)

            with open(fname, 'wb') as cache_file:
                pickle.dump({"trans": trans_data, "labels": trans_labels, "columns": columns_names}, cache_file)

        return trans_data, trans_labels, columns_names
    
    def format_trans(self, trans_lst, column_names):
        # O tamanho de cada bloco de transação agora usa exatamente o número de colunas de features reais
        chunk_size = len(column_names)
        trans_lst = list(divide_chunks(trans_lst, chunk_size))
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            if self.mlm:  
                vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def prepare_samples(self):
        log.info("preparing user level data...")
        trans_data, trans_labels, columns_names = self.user_level_data()

        log.info("creating transaction samples with vocab")
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names)

            user_labels = trans_labels[user_idx]

            bos_token = self.vocab.get_id(self.vocab.bos_token, special_token=True)  # will be used for GPT2
            eos_token = self.vocab.get_id(self.vocab.eos_token, special_token=True)  # will be used for GPT2
            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]  # flattening
                if not self.mlm and self.flatten:  # for GPT2, need to add [BOS] and [EOS] tokens
                    ids = [bos_token] + ids + [eos_token]
                self.data.append(ids)

            for jdx in range(0, len(user_labels) - self.seq_len + 1, self.trans_stride):
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)

                fraud = 0
                if len(np.nonzero(ids)[0]) > 0:
                    fraud = 1
                self.window_label.append(fraud)

        assert len(self.data) == len(self.labels)

        '''
            ncols = total fields - 1 (special tokens) - 1 (label)
            if bert:
                ncols += 1 (for sep)
        '''
        self.ncols = len(self.vocab.field_keys) - 1 + (1 if self.mlm else 0)
        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")

    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows)
        
        if self.group_by_ids:
            if (not self.group_by):
                log.error("group_by_ids foi fornecido, mas group_by não foi definido. Não é possível filtrar sem a coluna de referência.")
                raise ValueError("group_by_ids foi fornecido, mas group_by não foi definido.")

            filter_col = self.group_by 

            if not filter_col in data.columns:
                log.error(f"A coluna '{filter_col}' não existe no DataFrame. Não é possível filtrar pelos IDs fornecidos.")
                raise KeyError(f"A coluna '{filter_col}' não existe no DataFrame.")

            
            log.info(f'Filtrando dados pela lista de contas em "{filter_col}": {self.group_by_ids}...')
            
            # Converte os IDs passados para o mesmo tipo de dado da coluna do DataFrame (evita int vs str bug)
            col_type = data[filter_col].dtype
            account_ids = [col_type.type(x) for x in self.group_by_ids]
            
            data = data[data[filter_col].isin(account_ids)]
            

        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def init_vocab(self):
        column_names = list(self.trans_table.columns)
        if self.skip_group_col:
            if not self.group_by:
                log.error("skip_group_col foi definido como True, mas group_by não foi fornecido. Não é possível remover a coluna de agrupamento.")
                raise ValueError("skip_group_col foi definido como True, mas group_by não foi fornecido.")
            column_names.remove(self.group_by)

        self.vocab.set_field_keys(column_names)

        for column in column_names:
            unique_values = self.trans_table[column].value_counts(sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        log.info(f"total columns: {list(column_names)}")
        log.info(f"total vocabulary size: {len(self.vocab.id2token)}")

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])
            log.info(f"column : {column}, vocab size : {vocab_size}")

            if vocab_size > self.vocab.adap_thres:
                log.info(f"\tsetting {column} for adaptive softmax")
                self.vocab.adap_sm_cols.add(column)

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.fextension}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")

        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.trans_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return

        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")

        log.info("Engenharia de Recursos Temporais e Auditoria Contábil...")
        # ==========================================
        # 1. ENGENHARIA DE DELTAS TEMPORAIS
        # ==========================================
        dt_entry = pd.to_datetime(data['entry_date'], errors='coerce')
        
        dt = pd.to_datetime(data['entry_date'], errors='coerce')

        # Quantas datas falharam na conversão?
        nulos_gerados = dt.isna().sum()
        total_linhas = len(data)
        log.info(f"Total de linhas: {total_linhas}")
        log.info(f"Datas que falharam e viraram NaT: {nulos_gerados} ({nulos_gerados / total_linhas * 100:.2f}%)")
        
        # 1.1 Atraso de Lançamento: creation_date vs entry_date (em dias)
        delta_cols = []
        if 'creation_date' in data.columns:
            log.info(f"Calculando atraso de lançamento (creation_date vs entry_date)...")
            dt_creation = pd.to_datetime(data['creation_date'], errors='coerce')
            # Diferença em dias (valores positivos indicam digitação após a data contábil)
            data['entry_delay_days'] = (dt_creation - dt_entry).dt.total_seconds() / (24 * 3600)
            data['entry_delay_days'] = data['entry_delay_days'].fillna(0.0)
            delta_cols.append('entry_delay_days')

        # 1.2 Intervalo para o Lançamento Anterior do mesmo Grupo (Velocidade)
        # Ordenamos temporalmente para o cálculo correto da série
        if self.group_by and self.group_by in data.columns:
            log.info(f"Calculando delta temporal relativo ao grupo '{self.group_by}'...")
            data = data.sort_values(by=[self.group_by, 'entry_date']).reset_index(drop=True)
            dt_entry_sorted = pd.to_datetime(data['entry_date'], errors='coerce')
            data['group_delta_days'] = data.groupby(self.group_by, group_keys=False).apply(
                lambda g: (pd.to_datetime(g['entry_date'], errors='coerce').diff().dt.total_seconds() / (24 * 3600))
            ).fillna(0.0)
        else:
            log.info("Calculando delta temporal da linha do tempo global...")
            data = data.sort_values(by='entry_date').reset_index(drop=True)
            dt_entry_sorted = pd.to_datetime(data['entry_date'], errors='coerce')
            data['group_delta_days'] = (dt_entry_sorted.diff().dt.total_seconds() / (24 * 3600)).fillna(0.0)

        delta_cols.append('group_delta_days')

        # ==========================================
        # 2. TRATAMENTO NUMÉRICO (VALORES E RAZÕES)
        # ==========================================
        log.info("Calculando razões de movimento contábil...")
        data['journal_entry_movement_ratio'] = (
                data['movement'] / data['journal_entry_movement']).replace([np.inf, -np.inf], 1)
        data['global_movement_ratio'] = (data['movement'] / data['global_movement']).replace(
                [np.inf, -np.inf], 1)

        
        log.info("Codificação monetária e normalização de razões...")
        val_cols = ['movement', 'journal_entry_movement', 'global_movement']
        for col in val_cols:
            if col in data.columns:
                data[col] = self.amountEncoder(data[col])

        ratio_cols = ['journal_entry_movement_ratio', 'global_movement_ratio']
        for col in ratio_cols:
            if col in data.columns:
                data[col] = data[col].astype(float).fillna(0.0)

        # ==========================================
        # 3. TRATAMENTO CATEGÓRICO E SEQUENCIAL
        # ==========================================
        categoricas = [
            'company', 'branch', 'exchange_id', 'company_exchange_id',
            'erp_book_account', 'erp_book_account_name', 'cosif_book_account',
            'cosif_book_account_name', 'supplier_number', 'asset_number',
            'journal_entry_origin', 'movement_topic', 'movement_description',
            'erp_doc_type', 'status_origin', 'partner', 'partner_name',
            'cost_center', 'cost_center_name', 'profit_center', 'profit_center_name',
            'erp_book_account_type', 'movement_type', 'entry_credit_or_debit',
            'erp_transaction_code', 'metadata_file_path', 'global_currency', 
        ]
        
        colunas_sequenciais = ['id', 'auxiliary_id', 'journal_entry_id', 'creator_user_id', 'sap_journal_entry_id']
        
        all_to_fit = categoricas + colunas_sequenciais
        for col in all_to_fit:
            if col in data.columns:
                data[col] = data[col].astype(str).where(pd.notnull(data[col]), 'None')

        log.info("Aplicando LabelEncoder em todas as colunas textuais/categóricas...")
        for col_name in tqdm.tqdm(all_to_fit):
            if col_name in data.columns:
                col_data = data[col_name]
                col_fit, col_data = self.label_fit_transform(col_data)
                self.encoder_fit[col_name] = col_fit
                data[col_name] = col_data

        # ==========================================
        # 4. PROCESSAMENTO DO TIMESTAMP BASE
        # ==========================================
        log.info("Processando Timestamp principal...")
        time_data = data[['entry_date']]
        if 'creation_time' in data.columns:
            time_data = data[['entry_date']].copy()
            time_data['creation_time'] = data['creation_time']

        timestamp = self.timeEncoder(time_data)
        timestamp_fit, timestamp = self.label_fit_transform(timestamp, enc_type="time")
        self.encoder_fit['Timestamp'] = timestamp_fit
        data['Timestamp'] = timestamp

        # ==========================================
        # 5. QUANTIZAÇÃO (BINNING) DE TODAS AS CONTÍNUAS
        # ==========================================
        log.info("Quantização (Binning) de Timestamps, Deltas, Valores e Razões...")
        
        # Lista unificada de todas as colunas numéricas contínuas a quantizar
        continuous_to_quant = ['Timestamp'] + \
                              [c for c in delta_cols if c in data.columns] + \
                              [c for c in val_cols if c in data.columns] + \
                              [c for c in ratio_cols if c in data.columns]

        for col in continuous_to_quant:
            coldata = np.array(data[col])
            bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
            data[col] = self._quantize(coldata, bin_edges)
            self.encoder_fit[f"{col}-Quant"] = [bin_edges, bin_centers, bin_widths]

        # ==========================================
        # 6. SELEÇÃO FINAL DAS COLUNAS (FEATURES)
        # ==========================================
        columns_to_select = (
            ['Timestamp'] + 
            [c for c in delta_cols if c in data.columns] +
            [c for c in val_cols if c in data.columns] + 
            [c for c in ratio_cols if c in data.columns] +
            [c for c in categoricas if c in data.columns] + 
            [c for c in colunas_sequenciais if c in data.columns]
        )

        self.trans_table = data[columns_to_select]

        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))

        encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))