from typing import Dict
from transformers.tokenization_utils import PreTrainedTokenizer

class TabFormerTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        **kwargs
    ):
        super().__init__(
            bos_token=bos_token, 
            eos_token=eos_token, 
            unk_token=unk_token,
            **kwargs
        )

    # =====================================================================
    # MUDANÇA: Métodos obrigatórios (dummy) para compatibilidade com o HF v4
    # Evita o TypeError de "Abstract Class" e impede que o Trainer quebre no save
    # =====================================================================
    
    @property
    def vocab_size(self) -> int:
        return 0

    def get_vocab(self) -> Dict[str, int]:
        return {}

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        # Retorna uma tupla vazia sinalizando que não há arquivo de vocabulário para salvar
        return ()

    def _tokenize(self, text, **kwargs):
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        return 0

    def _convert_id_to_token(self, index: int) -> str:
        return self.unk_token