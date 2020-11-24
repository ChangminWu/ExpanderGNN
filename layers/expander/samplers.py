import torch
import numpy as np


def sampler(outdim, indim, density, method):
    mask = torch.zeros(outdim, indim)
    if method == "regular":
        if outdim < indim:
            for i in range(outdim):
                x = torch.randperm(indim)
                for j in range(int(indim*density)):
                    mask[i][x[j]] = 1
        else:
            for i in range(indim):
                x = torch.randperm(outdim)
                for j in range(int(outdim*density)):
                    mask[x[j]][i] = 1

        n_params = int(density * max(outdim, indim)) * min(outdim, indim)

    elif method == "random":
        n_params = int(density*outdim*indim)
        edges = [(j, i) for j in range(outdim) for i in range(indim)]
        inds = torch.randperm(len(edges))
        for i, ind in enumerate(inds):
            m, n = edges[ind]
            mask[m][n] = 1
        # for i in range(indim):
        #     for j in range(outdim):
        #         mask[j][i] = np.random.choice([0, 1], 1, p=[1-density, density])[0]
        # n_params = int(mask.sum().item())

    elif method == "rotate":
        k = int(density * outdim)
        for i in range(indim):
            mask[np.arange(i, i+k) % outdim, i] = 1
        n_params = int(mask.sum().item())

    return mask, n_params