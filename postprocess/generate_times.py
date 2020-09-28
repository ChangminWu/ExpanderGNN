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

VALUE = "Accuracy"
MODEL = ["GCN"] #, "GIN", "GraphSage", "PNA", "MLP"
ACTIV = ["ReLU", "PReLU", "Tanh"]
DENSITY = [0.1, 0.5, 0.9]
RECORD = [VALUE, "Time per Epoch(s)", "#Parameters"]
BARWIDTH = 0.3

def time_plot(folder, dataset, output_file):
    data = list()
    for file in glob.glob(folder+"*.txt"):
        components = file.replace(folder, "").split("_")
        single_row = list()

        # add dataset
        for i in dataset:
            if components[3].lower() == i.lower():
                single_row.append(i)
        if len(single_row) == 0:
            continue

        # add model name
        gnn = components[2].replace("Simple", "").replace("Activation", "")
        for i in MODEL:
            if gnn.lower() == i.lower():
                single_row.append(i)
        if len(single_row) == 1:
            continue

        # add experiment type
        types = components[1].split("-")
        for i in ["Regular", "Expander", "Activations", "Simple"]:
            if types[0].lower() == i.lower():
                if i != "Regular":
                    exp = i
                else:
                    exp = "Vanilla"
        if len(types) == 1:
            single_row.append(exp)
        elif len(types) == 2:
            for j in ACTIV:
                if j.lower() == types[1].lower():
                    single_row.append(exp+"-"+j)
            if len(single_row) == 2:
                continue
        elif len(types) == 3:
            for j in DENSITY:
                if float(types[-1]) == j:
                    single_row.append(exp+"-{:.0%}".format(j))
            if len(single_row) == 2:
                continue

        # add results, time, convergence
        rows = []
        with open(file) as f:
            for line in f.readlines():
                if line.split(":")[0] == "TEST ACCURACY averaged":
                    res = line.split(":")[1].split("with s.d.")[0]
                    # rows.append(single_row+[float(res), "Accuracy"]) 
                if line.split(":")[0] == "TEST MAE averaged":
                    res = line.split(":")[1].split("with s.d.")[0]
                    # rows.append(single_row+[float(res), "MAE"])              
                # if line.split(":")[0] == "Average Convergence Time (Epochs)":
                #     epochs = line.split(":")[1].split(" ")[1]
                #     rows.append(single_row+[int(float(epochs)), "Converge(#Epochs)"])
                if line.split(":")[0] == "Total Parameters":
                    n_params = line.split(":")[1].split(" ")[1]
                #     rows.append(single_row+[int(float(n_params)), "#Parameters)"])
                if line.split(":")[0] == "Average Time Per Epoch":
                    time = line.split(":")[1].split(" ")[1]
                    rows.append(single_row+[float(time), "Time per Epoch(s)"])
            single_row.extend([float(res), float(time), int(float(n_params))])
        f.close()
        data.append(single_row)
        # data.extend(rows)
    ticklabel = ["Simple"]
    for a in ACTIV:
        ticklabel.append("Activations-{}".format(a))
    for e in DENSITY:
        ticklabel.append("Expander-{:.0%}".format(e))
    ticklabel.append("Vanilla")

    df = pd.DataFrame(data=data, columns=["Dataset", "Model", "Type"]+RECORD)
    times = []
    nparam = []
    for tick in ticklabel:
        for idx, types in enumerate(df["Type"]):
            if tick == types:
                times.append(df["Time per Epoch(s)"].loc[idx])
                nparam.append(df["#Parameters"].loc[idx])

    df2 = pd.DataFrame({'Time per Epoch(s)': times,
                        '#Parameters': nparam})
    fig, ax1 = plt.subplots(figsize=(20, 20))
    ax1 = df2['#Parameters'].plot(kind='bar', color='r', alpha=0.2)
    ax2 = df2['Time per Epoch(s)'].plot(kind='line', marker='x', secondary_y=True, linewidth=5, markersize=20, color="g")
    ax1.set_xticklabels([x+" GCN" for x in ticklabel], fontsize=50, rotation=45, horizontalalignment='right', fontweight='light')
    plt.setp(ax1.get_yticklabels(), fontsize=50, color='r', alpha=0.4)
    plt.setp(ax2.get_yticklabels(), fontsize=50, color='g')
    ax1.set_ylabel("#Parameters", fontsize=70, color='r', alpha=0.4)
    ax2.set_ylabel("Time per Epoch(s)", fontsize=70, color='g')
    fig.tight_layout()
    plt.savefig(output_file, dpi=800)

    #   #, "Value", "ValueType"])
    # print(df)

    # sns.set_theme(style="whitegrid")
    # #g = sns.catplot(x="Type", y="Time per Epoch(s)", hue="Model", col="Dataset", data=df, kind="bar", height=4, aspect=1.0, sharex=True, sharey=True, hue_order=hue_order, col_order=dataset)
    # # palette = sns.color_palette("mako_r", 1)
    # g = sns.lineplot(x="Type", y="Time per Epoch(s)", data=df) #, palette=palette) #  hue="Model", 
    # # g.set_xticklabels(rotation=45, horizontalalignment='right', fontweight='light')
    # # g.set_xticklabels(fontsize=10, fontweight='light')
    # g.figure.tight_layout()
    # # plt.setp(g._legend.get_texts(), fontsize=10)
    # # plt.setp(g._legend.get_title(), fontsize=10)
    # # plt.setp(g._legend.get_texts(), fontsize=5)
    # g.figure.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        help="give path of folder that stores result files")
    parser.add_argument('--dataset', action="store", nargs="+", help="decide which datasets need to be presented")
    parser.add_argument('--output', help="give output file")
    args = parser.parse_args()

    time_plot(args.folder, args.dataset, args.output)