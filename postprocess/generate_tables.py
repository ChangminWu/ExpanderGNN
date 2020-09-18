import pandas as pd 
import numpy as np
import glob
import argparse
import itertools

MODEL = ["GCN", "GIN", "GAT", "GraphSage", "GatedGCN", "PNA", "MLP"]
ACTIV = ["ReLU", "PReLU", "RReLU", "SoftPlus", "Linear", "Sigmoid", "Tanh"]
DENSITY = [0.1, 0.5, 0.9]
RECORD = ["ACC", "Time per Epoch(s)", "Converge(#Epochs)"]


def collect_results(folder, dataset, output_file):
    ### regular
    types = ["Regular"] * len(MODEL)
    models = MODEL.copy()
    params = ["---"] * len(MODEL)

    ### expander
    types.extend(["Expander"]*len(MODEL)*len(DENSITY))
    models.extend([MODEL[i//len(DENSITY)] for i in range(len(DENSITY)*len(MODEL))])
    params.extend(["{:.0%}".format(i) for i in DENSITY]*len(MODEL))
    # 

    ### activation
    types.extend(["Activations"]*len(MODEL)*len(ACTIV))
    models.extend([MODEL[i//len(ACTIV)] for i in range(len(ACTIV)*len(MODEL))])
    params.extend(ACTIV*len(MODEL))

    ### simple
    types.extend(["Simple"]*len(MODEL))
    models.extend(MODEL)
    params.extend(["---"]*len(MODEL))

    row_index = [types, models, params]
    if type(dataset) is not list:
        dataset = list([dataset])

    datasets = list(itertools.chain.from_iterable(itertools.repeat(x, len(RECORD)) for x in dataset))
    records = RECORD * len(dataset)
    column_index = [datasets, records]

    df = pd.DataFrame(index=row_index, columns=column_index, dtype='float')
    df = df.sort_index()
    for file in glob.glob(folder+"*.txt"):
        components = file.replace(folder, "").split("_")
        row_ind = tuple()
        col_ind = tuple()
        for i in dataset:
            if components[3].lower() == i.lower():
                col_ind = col_ind + (i,)

        row_ind = row_ind + (components[2].replace("Simple", "").replace("Activation", ""), )

        type_param = components[1].split("-")
        
        for i in ["Regular", "Expander", "Activations", "Simple"]:
            if i.lower() == type_param[0].lower():
                row_ind = (i, ) + row_ind
        
        if len(type_param) == 1:
            row_ind = row_ind + ("---", )
        elif len(type_param) == 2:
            for i in ACTIV:
                if i.lower() == type_param[-1].lower():
                    row_ind = row_ind + (i, )
            if len(row_ind) == 2:
                continue
        else:
            row_ind = row_ind + ("{:.0%}".format(float(type_param[-1])), )
        
        with open(file) as f:
            for line in f.readlines():
                if line.split(":")[0] == "TEST ACCURACY averaged":
                    acc = line.split(":")[1].split("with s.d.")[0]
                    col_acc = col_ind + ("ACC", )
                    df.loc[row_ind][col_acc] = float(acc)
                if line.split(":")[0] == "TEST MAE averaged":
                    acc = line.split(":")[1].split("with s.d.")[0]
                    col_acc = col_ind + ("MAE", )
                    df.loc[row_ind][col_acc] = float(acc)
                if line.split(":")[0] == "Average Convergence Time (Epochs)":
                    epochs = line.split(":")[1].split(" ")[1]
                    col_conv = col_ind + ("Converge(#Epochs)", )
                    df.loc[row_ind][col_conv] = int(float(epochs))
                if line.split(":")[0] == "Average Time Per Epoch":
                    time = line.split(":")[1].split(" ")[1]
                    col_time = col_ind + ("Time per Epoch(s)", )
                    df.loc[row_ind][col_time] = float(time)
            # print(row_ind)
        f.close()
    df.to_latex(buf=output_file, longtable=True, multicolumn=True, multirow=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        help="give path of folder that stores result files")
    parser.add_argument('--dataset', action="store", nargs="+", help="decide which datasets need to be presented")
    parser.add_argument('--output', help="give output file")
    args = parser.parse_args()

    collect_results(args.folder, args.dataset, args.output)