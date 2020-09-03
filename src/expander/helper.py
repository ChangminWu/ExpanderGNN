import torch


def sampler(in_features: int, out_features: int, density: float, sample_method: str="prabhu") -> Tensor:
    mask = torch.zeros(out_features, in_features)
    if sample_method.lower() == "prabhu":
        if out_features < in_features:
            for i in range(out_features):
                x = torch.randperm(in_features)
                for j in range(int(in_features*density)):
                    mask[i][x[j]] = 1
        else:
            for i in range(in_features):
                x = torch.randperm(out_features)
                for j in range(int(out_features*density)):
                    mask[x[j]][i] = 1

    elif sample_method.lower() == "prabhu-full":
        if out_features < in_features:
            n_connections = int(in_features * density)
            n_repeat = in_features // out_features
            x = torch.randperm(out_features)
            for i in range(in_features):
                mask[x[i % out_features]][i] = 1
                if (i+1) % out_features == 0:
                    x = torch.randperm(out_features)
            for i in range(out_features):
                x = torch.randperm(in_features)
                if i < in_features % out_features:
                    for j in range(n_connections-n_repeat-1):
                        mask[i][x[j]] = 1
                else:
                    for j in range(n_connections-n_repeat):
                        mask[i][x[j]] = 1
        else:
            n_connections = int(out_features * density)
            n_repeat = out_features // in_features
            x = torch.randperm(in_features)
            for i in range(out_features):
                mask[i][x[i % in_features]] = 1
                if (i+1) % in_features == 0:
                    x = torch.randperm(in_features)
            for i in range(in_features):
                x = torch.randperm(out_features)
                if i < out_features % in_features:
                    for j in range(n_connections-n_repeat-1):
                        mask[x[j]][i] = 1
                else:
                    for j in range(n_connections-n_repeat):
                        mask[x[j]][i] = 1
        assert torch.nonzero(mask).size(0) == int(density * max(out_features, in_features)) * min(out_features,
                                                                                                  in_features)

    elif sample_method.lower() == "random":
        n_connections = int(density * out_features * in_features)
        new_locs = []
        i = 0
        while len(new_locs) < n_connections:
            i += 1
            print("random sample tried {}th time".format(i))
            iind = torch.randint(in_features, size=(int(n_connections*1.5),)).reshape(-1, 1)
            oind = torch.randint(out_features, size=(int(n_connections*1.5),)).reshape(-1, 1)
            t_ = torch.cat([iind, oind], dim=1)
            t_ = set([(int(x[0]), int(x[1])) for x in t_])
            if len(t_) < n_connections:
                continue
            else:
                for tup in t_:
                    mask[tup[0]][tup[1]] = 1
        #         new_locs = [torch.tensor([tup[0], tup[1]]).int().reshape(1, -1) for tup in t_]
        #         new_locs = tuple(torch.cat(new_locs, dim=0).t())
        # mask[new_locs] = 1
        assert torch.nonzero(mask).size(0) == n_connections
    return mask
