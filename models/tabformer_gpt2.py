from torch.nn import CrossEntropyLoss

# MUDANÇA 1: Importação atualizada para a v4.x
from transformers import GPT2LMHeadModel


class TabFormerGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, vocab):
        super().__init__(config)
        self.vocab = vocab

    def forward(
            self,
            input_ids=None,
            past_key_values=None,  # MUDANÇA 2: Renomeado de 'past' para 'past_key_values'
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
            **kwargs # Adicionado para absorver argumentos modernos do Trainer (ex: return_dict)
    ):
        
        # MUDANÇA 3: Passamos return_dict=False para manter a compatibilidade com a indexação [0]
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=False, 
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        # lm_logits : [bsz x seq_len x vsz]
        # labels    : [bsz x seq_len]
        # When flatten is set to True:
        # seq_len = num_transactions * (num_columns + 2)  --> plus 2 because each transaction has BOS and EOS padding

        outputs = (lm_logits,) + transformer_outputs[1:]
        
        if labels is not None:
            total_lm_loss = 0
            field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=True)

            # Verifica se está no modo Flatten (2D) ou Hierárquico (3D)
            if labels.dim() == 2:
                # ==========================================
                # MODO FLATTEN (Sequência linear de tokens)
                # ==========================================
                shift_labels = labels[:, 1:-1].contiguous()  # Remove [BOS] e [EOS]
                shift_logits = lm_logits[:, :-2, :].contiguous()

                seq_len = shift_logits.size(1)
                
                for field_idx, field_name in enumerate(field_names):
                    col_ids = list(range(field_idx, seq_len, len(field_names)))
                    global_ids_field = self.vocab.get_field_ids(field_name)
                    
                    lm_logits_field = shift_logits[:, col_ids, :][:, :, global_ids_field]
                    lm_labels_field = shift_labels[:, col_ids]
                    
                    lm_labels_local_field = self.vocab.get_from_global_ids(global_ids=lm_labels_field,
                                                                           what_to_get='local_ids')

                    loss_fct = CrossEntropyLoss()
                    lm_loss_field = loss_fct(lm_logits_field.view(-1, len(global_ids_field)),
                                             lm_labels_local_field.view(-1))
                    total_lm_loss += lm_loss_field

            else:
                # ==========================================
                # MODO HIERÁRQUICO (Transações x Colunas)
                # ==========================================
                # Para prever o futuro (GPT-2), deslocamos o tempo (dimensão 1)
                # labels não precisam prever a transação 0, então começamos da 1
                shift_labels = labels[:, 1:, :].contiguous()
                # logits usam o tempo de 0 até o penúltimo para prever o próximo
                shift_logits = lm_logits[:, :-1, :].contiguous()
                
                for field_idx, field_name in enumerate(field_names):
                    global_ids_field = self.vocab.get_field_ids(field_name)
                    
                    # Extrai as previsões correspondentes ao vocabulário dessa coluna
                    lm_logits_field = shift_logits[:, :, global_ids_field]
                    
                    # A resposta certa é simplesmente a coluna (field_idx) na dimensão 2
                    lm_labels_field = shift_labels[:, :, field_idx]
                    
                    lm_labels_local_field = self.vocab.get_from_global_ids(global_ids=lm_labels_field,
                                                                           what_to_get='local_ids')

                    loss_fct = CrossEntropyLoss()
                    # Agora sim: (Batch x Seq) para as previsões e para os alvos!
                    lm_loss_field = loss_fct(lm_logits_field.view(-1, len(global_ids_field)),
                                             lm_labels_local_field.view(-1))
                    total_lm_loss += lm_loss_field

            outputs = (total_lm_loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)