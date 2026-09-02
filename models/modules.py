from misc.utils import ddict

from transformers.modeling_utils import PreTrainedModel
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel
)

from models.tabformer_tokenizer import TabFormerTokenizer
from models.hierarchical import TabFormerEmbeddings
from models.tabformer_bert import TabFormerBertForMaskedLM, TabFormerBertConfig
from models.tabformer_gpt2 import TabFormerGPT2LMHeadModel


class TabFormerBaseModel(PreTrainedModel):
    # Mudança 1: config obrigatoriamente como o primeiro argumento
    def __init__(self, config, hf_model, tab_embeddings):
        config._attn_implementation = "eager"

        super().__init__(config)
        

        self.model = hf_model
        self.tab_embeddings = tab_embeddings

    # Mudança 2: input_ids agora é opcional (nomeado) para receber do Trainer via **kwargs
    def forward(self, input_ids=None, **input_args):
        if input_ids is not None:
            inputs_embeds = self.tab_embeddings(input_ids)
            return self.model(inputs_embeds=inputs_embeds, **input_args)
        return self.model(**input_args)


class TabFormerHierarchicalLM(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        config._attn_implementation = "eager"

        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    # Mesma adaptação do input_ids opcional aqui
    def forward(self, input_ids=None, **input_args):
        if input_ids is not None:
            inputs_embeds = self.tab_embeddings(input_ids)
            return self.tb_model(inputs_embeds=inputs_embeds, **input_args)
        return self.tb_model(**input_args)


class TabFormerBertLM:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, field_hidden_size=768):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        hidden_size = field_hidden_size if flatten else (field_hidden_size * self.ncols)

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=self.ncols)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):

        if flatten and not field_ce:
            # flattened vanilla BERT
            model = BertForMaskedLM(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            model = TabFormerBertForMaskedLM(self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            model = TabFormerHierarchicalLM(self.config, self.vocab)

        return model


class TabFormerGPT2:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, field_hidden_size=768):

        self.vocab = vocab
        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.config = GPT2Config(vocab_size=len(self.vocab))

        self.tokenizer = TabFormerTokenizer(
            unk_token=special_tokens.unk_token,
            bos_token=special_tokens.bos_token,
            eos_token=special_tokens.eos_token
        )

        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):
        if field_ce:
            model = TabFormerGPT2LMHeadModel(self.config, self.vocab)
        else:
            model = GPT2LMHeadModel(self.config)
        if not flatten:
            tab_emb_config = ddict(
                vocab_size=len(self.vocab),
                hidden_size=self.config.hidden_size,
                field_hidden_size=self.field_hidden_size,
                ncols=self.ncols,
                pad_token_id=self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else 0
            )
            # Mudança 3: Passamos a configuração na ordem correta para o Base Model
            model = TabFormerBaseModel(self.config, model, TabFormerEmbeddings(tab_emb_config))

        return model