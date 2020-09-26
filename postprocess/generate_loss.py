import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd 
import numpy as np
import glob
import argparse
import itertools

VALUE = "MAE"
MODEL = ["GCN", "GIN", "GraphSage", "PNA", "MLP"]
ACTIV = ["ReLU", "PReLU", "Tanh"]
DENSITY = [0.1, 0.5, 0.9]
RECORD = [VALUE, "Time per Epoch(s)", "#Parameters"]
BARWIDTH = 0.3

def loss_plot(folder, output_file=None):
    df = None
    for file in glob.glob(folder+"*.csv"):
        file_name = file.replace(folder, "").split("_")

        # add experiment type
        name = None
        indicator = file_name[0].split("-")
        for i in ["Regular", "Expander", "Activations", "Simple"]:
            if indicator[1].lower() == i.lower():
                if i != "Regular":
                    name = i
                else:
                    name = "Vanilla"

        if len(indicator) == 3:
            for j in ACTIV:
                if indicator[-1].lower() == j.lower():
                    name = name+"-"+j
            if name is None:
                continue

        elif len(indicator) == 4:
            for j in DENSITY:
                if float(indicator[-1]) == j:
                    name = name + "-{:.0%}".format(j)
                if float(indicator[-1]) == 0.1:
                    break
            if name.split("-")[-1] in ["Regular", "Expander", "Activations", "Simple"]:
                continue

        gnn = file_name[1].replace("Simple", "").replace("Activation", "")
        for i in MODEL:
            if gnn.lower() == i.lower():
                name = name + " {}".format(i)
        if name is None or name.split(" ")[-1] not in MODEL:
            continue

        load_data = pd.read_csv(file)
        load_data["Model"] = name
        if df is None:
            df = load_data
        else:
            df = df.append(load_data)

    hue_order = ["Simple"]
    for a in ACTIV:
        hue_order.append("Activations-{}".format(a))
    for e in DENSITY:
        hue_order.append("Expander-{:.0%}".format(e))
    hue_order.append("Vanilla")

    hue_order = [x+" {}".format(MODEL[0]) for x in hue_order]
    
    # fig=plt.figure(figsize=(40,20))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("mako_r", 8)
    g = sns.lineplot(x="Step", y="Value", hue="Model", data=df, palette=palette, hue_order=hue_order)
    g.set_xlabel("Epoch")
    g.set_ylabel("Train Loss")
    g.figure.tight_layout()
    g.figure.savefig(output_file, dpi=1600)

    # df = pd.DataFrame(data=data, columns=["Dataset", "Model", "Type"]+RECORD)  #, "Value", "ValueType"])

    # sns.set_theme(style="whitegrid")
    # g = sns.catplot(x="Model", y=VALUE, hue="Type", col="Dataset", data=df, kind="bar", height=4, aspect=1.0, sharex=True, sharey=True, hue_order=hue_order, col_order=dataset)
    # #g = sns.lineplot(x="Type", y="Time per Epoch(s)", hue="Model", data=df)
    # g.set_xticklabels(rotation=45, horizontalalignment='right', fontweight='light')
    # g.savefig(output_file, dpi=1600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        help="give path of folder that stores result files")
    parser.add_argument('--output', help="give output file")
    args = parser.parse_args()

    loss_plot(args.folder, args.output)