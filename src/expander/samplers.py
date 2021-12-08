import torch
from torch import Tensor
import numpy as np


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
    
    elif sample_method.lower() == "random":
        num_connections = int(indim*outdim*density)
        while True:
            rind = torch.randint(outdim, size=(int(num_connections*1.5),)).reshape(-1, 1)
            cind = torch.randint(indim, size=(int(num_connections*1.5),)).reshape(-1, 1)
            t_ = torch.cat([rind, cind], dim=1)
            t_ = set([(int(x[0]), int(x[1])) for x in t_])
            if len(t_) >= num_connections:
                row_idx = [x[0] for x in t_]
                col_idx = [x[1] for x in t_]
                row_idx = row_idx[:num_connections]
                col_idx = col_idx[:num_connections]
                break
        
    row = torch.LongTensor(row_idx)
    col = torch.LongTensor(col_idx)
    edge_index = torch.stack([row, col], dim=0).long()
    return edge_index