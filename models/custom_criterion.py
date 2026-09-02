from torch import Tensor
from torch.nn import AdaptiveLogSoftmaxWithLoss
from torch.nn.functional import log_softmax


class CustomAdaptiveLogSoftmax(AdaptiveLogSoftmaxWithLoss):
    def __init__(
            self, ignore_index=-100,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor):
        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        # Remove targets que devem ser ignorados (ex: padding ou tokens não previstos)
        consider_indices = (target != self.ignore_index)
        input = input[consider_indices, :]
        target = target[consider_indices]

        batch_size = target.size(0)
        
        # EARLY EXIT: Se todos os elementos do batch foram ignorados, retorna perda 0
        # Isso evita que o PyTorch estoure NaN ao calcular a média (.mean()) no final
        if batch_size == 0:
            return input.new_tensor(0.0, requires_grad=True)

        used_rows = 0
        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size, dtype=torch.long) # Forçamos tipo long (int64) para índices

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            
            # CORREÇÃO: as_tuple=True[0] garante que sempre será um tensor 1D, 
            # mesmo que haja 0 ou 1 correspondência, evitando o crash do .squeeze() antigo
            row_indices = target_mask.nonzero(as_tuple=True)[0]

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                cluster_logprob = log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     target.min().item(),
                                                     target.max().item()))

        head_output = self.head(input)
        head_logprob = log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze(1) # Mudado de squeeze() para squeeze(1) por segurança
        
        loss = (-output).mean()

        return loss