import torch
import numpy as np


def regular_sampler(outdim, indim, density):
    row_idx = []
    col_idx = []
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
    row = torch.LongTensor(row_idx)
    col = torch.LongTensor(col_idx)
    return row, col