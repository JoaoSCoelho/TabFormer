import torch.nn as nn


class TabFormerConcatEmbeddings(nn.Module):
    """TabFormerConcatEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabulary that is common across columns
               - `sparse=True` in `nn.Embedding` speeds up gradient computation for large vocabs

        Args:
            config.ncols
            config.vocab_size
            config.hidden_size

        Inputs:
            - **input_ids** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.field_hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0),
            sparse=False
        )
        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

        self.hidden_size = config.hidden_size
        self.field_hidden_size = config.field_hidden_size

    def forward(self, input_ids):
        input_shape = input_ids.size()

        embeds_sz = list(input_shape[:-1]) + [input_shape[-1] * self.field_hidden_size]
        # Adicionado .contiguous() para garantir compatibilidade com PyTorch 2.x
        inputs_embeds = self.lin_proj(self.word_embeddings(input_ids).contiguous().view(embeds_sz))

        return inputs_embeds


class TabFormerEmbeddings(nn.Module):
    """TabFormerEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabulary that is common across columns

        Args:
            config.ncols
            config.num_layers (int): Number of transformer layers
            config.vocab_size
            config.hidden_size
            config.field_hidden_size

        Inputs:
            - **input_ids** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()

        if not hasattr(config, 'num_layers'):
            config.num_layers = 1
        if not hasattr(config, 'nhead'):
            config.nhead = 8

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.field_hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0),
            sparse=False
        )

        # batch_first=True evita permutas extras de tensor no PyTorch 2.x
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.field_hidden_size,
            nhead=config.nhead,
            dim_feedforward=config.field_hidden_size,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

    def forward(self, input_ids):
        # input_ids: [bsz, seq_len, ncols]
        inputs_embeds = self.word_embeddings(input_ids)
        embeds_shape = list(inputs_embeds.size())  # [bsz, seq_len, ncols, field_hidden_size]

        # Achata batch e seq_len para processar cada transação com o encoder de colunas
        inputs_embeds = inputs_embeds.contiguous().view([-1] + embeds_shape[-2:])  # [bsz * seq_len, ncols, field_hidden_size]
        
        # Com batch_first=True, dispensamos os permute(1, 0, 2)
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        
        # Reconstrói a dimensão [bsz, seq_len, ncols * field_hidden_size]
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0], embeds_shape[1], -1)

        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds