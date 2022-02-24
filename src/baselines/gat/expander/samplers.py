import torch
from torch import Tensor
import numpy as np
import itertools


def sampler(indim: int, outdim: int, density: float, sample_method: str="prabhu") -> Tensor:
    row_idx = []
    col_idx = []
    
    if sample_method.lower() == "prabhu":
        if outdim < indim:
            for i in range(outdim):
                num_connections = max(1, int(indim*density))
                x = np.random.permutation(indim)
                row_idx.extend([i]*num_connections)
                col_idx.extend(list(x[:num_connections]))
        else:
            for i in range(indim):
                num_connections = max(1, int(outdim*density))
                x = np.random.permutation(outdim)
                row_idx.extend(list(x[:num_connections]))
                col_idx.extend([i]*num_connections)

    elif sample_method.lower() == "identity":
        assert indim==outdim, "activation-only needs same input and output dimension"
        row_idx = [i for i in range(indim)]
        col_idx = [i for i in range(outdim)]
    
    elif sample_method.lower() == "random":
        num_connections = max(1, int(indim*outdim*density))
        t_ = np.random.permutation(list(itertools.product(range(outdim), range(indim))))
        row_idx = [x[0] for x in t_[:num_connections]]
        col_idx = [x[1] for x in t_[:num_connections]]
    
    elif sample_method.lower() == "regular-rotate":
        num_connections = max(1, int(indim*density))
        for i in range(outdim):
            row_idx.extend([i]*num_connections)
            col_idx.extend(range((i%indim, i+num_connections%indim)))
        
    row = torch.LongTensor(row_idx)
    col = torch.LongTensor(col_idx)
    edge_index = torch.stack([row, col], dim=0).long()
    return edge_index