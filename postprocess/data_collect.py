import pandas as pd 
import numpy as np
import glob

MODELS = ["GCN", "GIN", "GAT", "GraphSage", "GatedGCN", "PNA", "MLP"]
ACTIVS = ["ReLU", "PReLU", "RReLU", "SoftPlus", "Linear"]
DENSITY = [0.1, 0.5, 0.9]
RECORD = ["ACC", "Time per Epoch(s)", "Converge(#Epochs)"]

def collect_results(folder, dataset, name):
    ### regular
    types = ["Regular"] * len(MODELS)
    models = MODELS
    params = [""] * len(MODELS)
    
    ### expander
    types.extend(["Expander"]*len(MODELS)*len(DENSITY))
    models.extend([MODELS[i//len(DENSITY)] for i in range(len(DENSITY)*len(MODELS))])
    params.extend(["{:.0%}".format(i) for i in DENSITY]*len(MODELS))
    # list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in lst))
    
    ### activation
    types.extend(["Activation"]*len(MODELS)*len(ACTIVS))
    models.extend([MODELS[i//len(ACTIVS)] for i in range(len(ACTIVS)*len(MODELS))])
    params.extend(ACTIVS*len(MODELS))

    ### simple
    types.extend(["Simple"]*len(MODELS))
    models.extend(MODELS)
    params.extend([""]*len(MODELS))

    row_index = [types, models, params]
    if type(dataset) is not list:
        dataset = list([dataset])
    
    datasets = dataset * len(RECORD)
    records =  RECORD * len(dataset)

    column_index = [datasets, records]

    df = pd.DataFrame(index=row_index, columns=column_index, dtype='float')
    print(df)


if __name__ == "__main__":
    collect_results("./", ["IMDB_BINARY", "IMDB"], "none")
    # for file in glob.glob(folder+".txt"):

