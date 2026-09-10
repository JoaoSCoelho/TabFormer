from collections import OrderedDict
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class UnsupervisedVocabulary:
    def __init__(self, adap_thres=10000):
        # Tokens especiais para modelos autoregressivos (GPT) e de mascaramento (BERT)
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.adap_thres = adap_thres
        self.adap_sm_cols = set()

        self.special_field_tag = "SPECIAL"

        self.special_tokens = [
            self.unk_token, self.sep_token, self.pad_token,
            self.cls_token, self.mask_token, self.bos_token, self.eos_token
        ]

        self.token2id = OrderedDict()  # {field: {token: [global_id, local_id]}, ...}
        self.id2token = OrderedDict()  # {global_id: [token, field, local_id]}
        self.field_keys = OrderedDict()
        self.token2id[self.special_field_tag] = OrderedDict()

        self.filename = '' # this field is set in the `save_vocab` method

        # Inicializa os tokens especiais dentro do namespace "SPECIAL"
        for token in self.special_tokens:
            global_id = len(self.id2token)
            local_id = len(self.token2id[self.special_field_tag])

            self.token2id[self.special_field_tag][token] = [global_id, local_id]
            self.id2token[global_id] = [token, self.special_field_tag, local_id]

    def set_id(self, token, field_name, return_local=False):
        """Registra um token para um campo específico e retorna seu ID."""
        if token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])

            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            global_id, local_id = self.token2id[field_name][token]

        return local_id if return_local else global_id

    def get_id(self, token, field_name="", special_token=False, return_local=False):
        """Recupera o ID global ou local de um token."""
        if special_token:
            field_name = self.special_field_tag

        if token in self.token2id[field_name]:
            global_id, local_id = self.token2id[field_name][token]
        else:
            raise KeyError(f"Token '{token}' não encontrado no campo: '{field_name}'")

        return local_id if return_local else global_id

    def set_field_keys(self, keys):
        """Define os nomes das colunas de entrada e anexa o namespace SPECIAL."""
        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys[key] = None

        # Registra SPECIAL como uma coluna no vocabulário
        self.field_keys[self.special_field_tag] = None

    def get_field_ids(self, field_name, return_local=False):
        """Retorna todos os IDs associados a uma coluna contábil específica."""
        if field_name not in self.token2id:
            raise KeyError(f"Campo '{field_name}' é inválido.")

        selected_idx = 1 if return_local else 0
        return [ids[selected_idx] for ids in self.token2id[field_name].values()]

    def get_from_global_ids(self, global_ids, what_to_get='local_ids'):
        """Converte tensores de IDs globais de volta para IDs locais ou strings legíveis."""
        device = global_ids.device

        def map_global_ids_to_local_ids(gid):
            return self.id2token[gid][2] if gid != -100 else -100

        def map_global_ids_to_tokens(gid):
            return f'{self.id2token[gid][1]}_{self.id2token[gid][0]}' if gid != -100 else '-'

        if what_to_get == 'local_ids':
            return global_ids.cpu().apply_(map_global_ids_to_local_ids).to(device)
        elif what_to_get == 'tokens':
            vectorized_token_map = np.vectorize(map_global_ids_to_tokens)
            new_array_for_tokens = global_ids.detach().clone().cpu().numpy()
            return vectorized_token_map(new_array_for_tokens)
        else:
            raise ValueError("Parâmetro 'what_to_get' deve ser 'local_ids' ou 'tokens'.")

    def save_vocab(self, fname):
        """Salva o dicionário de tokens legíveis em arquivo de texto."""
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                fout.write(f"{field}_{token}\n")

    def get_field_keys(self, ignore_special=False, remove_target=False):
        """Retorna as colunas registradas, com opção de ignorar o namespace SPECIAL."""
        keys = list(self.field_keys.keys())
        if ignore_special and self.special_field_tag in keys:
            keys.remove(self.special_field_tag)
        return keys

    def get_special_tokens(self):
        """Retorna os tokens especiais mapeados."""
        special_tokens_map = {
            f"{name}_token": f"{self.special_field_tag}_{token}"
            for name, token in zip(
                ["unk", "sep", "pad", "cls", "mask", "bos", "eos"],
                self.special_tokens
            )
        }
        return AttrDict(special_tokens_map)

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        return f"UnsupervisedVocabulary: [{len(self)} tokens] [field_keys={list(self.field_keys.keys())}]"