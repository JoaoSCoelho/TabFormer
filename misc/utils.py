import torch

class ddict(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def random_split_dataset(dataset, lengths, random_seed=20200706):
    """
    Divide o dataset de forma determinística usando um gerador isolado.
    Evita alterar a semente (seed) global do sistema, prática recomendada no PyTorch 2.x.
    """
    generator = torch.Generator().manual_seed(random_seed)
    
    # O PyTorch moderno aceita o gerador diretamente na função random_split
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        lengths, 
        generator=generator
    )

    return train_dataset, eval_dataset, test_dataset


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]