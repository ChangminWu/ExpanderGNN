import torch


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



    return mask, n_params
