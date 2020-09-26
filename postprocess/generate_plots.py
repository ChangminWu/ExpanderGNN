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
MODEL = ["GCN", "GIN", "GraphSage", "PNA", "MLP"]
ACTIV = ["ReLU", "PReLU", "Tanh"]
DENSITY = [0.1, 0.5, 0.9]
RECORD = [VALUE, "Time per Epoch(s)", "#Parameters"]
BARWIDTH = 0.3

def bar_plot(folder, dataset, output_file):
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
    hue_order = ["Simple"]
    for a in ACTIV:
        hue_order.append("Activations-{}".format(a))
    for e in DENSITY:
        hue_order.append("Expander-{:.0%}".format(e))
    hue_order.append("Vanilla")


    df = pd.DataFrame(data=data, columns=["Dataset", "Model", "Type"]+RECORD)  #, "Value", "ValueType"])

    plt.figure(figsize=(20,60))
    sns.set_theme(style="whitegrid")
    g = sns.catplot(x="Model", y=VALUE, hue="Type", col="Dataset", data=df, kind="bar", height=4, aspect=1.0, sharex=True, sharey=True, hue_order=hue_order, col_order=dataset)
    #g = sns.lineplot(x="Type", y="Time per Epoch(s)", hue="Model", data=df)
    # g.set_xticklabels(rotation=45, horizontalalignment='right', fontweight='light')
    g.set_xticklabels(fontsize=10, fontweight='light')
    g.tight_layout()
    plt.setp(g._legend.get_texts(), fontsize=10)
    plt.setp(g._legend.get_title(), fontsize=10)
    # plt.setp(g._legend.get_texts(), fontsize=5)
    g.savefig(output_file, dpi=1600)

                    
# def draw_errorplot(dataset, exp_name="varing-sparsity-sparse-readout", path="d:\\results\\results\\", add_head=True, add_tail=True):
#     fig, ax = plt.subplots()

#     res = {}
#     for method in METHODS:
#         res[method] = {}
#         res[method]["acc"] = []
#         res[method]["std"] = []
#         sparsity_label = []
#         files = list(Path(path).rglob("result_{}_{}_{}_sparsity_*.txt".format(exp_name, method, dataset)))
#         sorted_files = sorted(files, key=lambda x: float(str(x).split("sparsity_")[-1].strip(".txt")))
#         for filename in sorted_files:
#             sparsity = float(str(filename).split("sparsity_")[-1].strip(".txt"))
#             with open(filename) as f:
#                 for line in f.readlines():
#                     if line.split(":")[0] == "TEST ACCURACY averaged":
#                         acc, std = line.split(":")[1].split("with s.d.")
#                 acc = float(acc)
#                 std = float(std)
#             f.close()
#             res[method]["acc"].append(acc)
#             res[method]["std"].append(std)
#             sparsity_label.append("{:.2%}".format(sparsity))
            
#         if add_head:
#             files = list(Path(path).rglob("result_zero-sparsity-sgcn_SimpleGCN_ENZYMES_sparsity_0.0.txt"))
#             for filename in files:
#                 sparsity = 0.0
#                 with open(filename) as f:
#                     for line in f.readlines():
#                         if line.split(":")[0] == "TEST ACCURACY averaged":
#                             acc, std = line.split(":")[1].split("with s.d.")
#                     acc = float(acc)
#                     std = float(std)
#                 f.close()
#                 res[method]["acc"].insert(0, acc)
#                 res[method]["std"].insert(0, std)
#                 sparsity_label.insert(0, "{:.2%}".format(sparsity))
                
#         if add_tail:
#             md = method.strip("Expander")
#             files = list(Path(path).rglob("result_varing-sparsity_{}_{}_sparsity_0.1.txt".format(md, dataset)))
#             for filename in files:
#                 sparsity = 1.0
#                 with open(filename) as f:
#                     for line in f.readlines():
#                         if line.split(":")[0] == "TEST ACCURACY averaged":
#                             acc, std = line.split(":")[1].split("with s.d.")
#                     acc = float(acc)
#                     std = float(std)
#                 f.close()
#                 res[method]["acc"].append(acc)
#                 res[method]["std"].append(std)
#                 sparsity_label.append("{:.2%}".format(sparsity))
        
#         ax.errorbar(np.arange(len(res[method]["acc"])), res[method]["acc"], res[method]["std"], uplims=True, lolims=True, label=method)
        
#     ax.set_ylabel("Classification Accuracy")
#     ax.set_xticks(np.arange(len(res[method]["acc"])))
#     ax.set_xticklabels(sparsity_label, rotation='vertical')
#     ax.set_ylim(0,80)
#     ax.set_title("Accuracy v.s. Sparsity for {}".format(dataset))
#     ax.yaxis.grid(True)
#     ax.legend()

#     plt.tight_layout()
#     plt.savefig(path+"{}_{}_acc.png".format(dataset, METHODS[0]))

# def draw_timeplot(dataset, exp_name="varing-sparsity-sparse-readout", path="d:\\results\\results\\", add_head=True, add_tail=True):
#     fig, ax = plt.subplots()
#     res = {}
#     for method in METHODS:
#         res[method] = {}
#         res[method]["time"] = []
#         res[method]["param"] = []
#         sparsity_label = []
#         files = list(Path(path).rglob("result_{}_{}_{}_sparsity_*.txt".format(exp_name, method, dataset)))
#         sorted_files = sorted(files, key=lambda x: float(str(x).split("sparsity_")[-1].strip(".txt")))
#         for filename in sorted_files:
#             sparsity = float(str(filename).split("sparsity_")[-1].strip(".txt"))
#             with open(filename) as f:
#                 for line in f.readlines():
#                     if line.split(":")[0] == "Average Time Per Epoch":
#                         time = line.split(":")[1].split(" ")[1]
#                     if line.split(":")[0] == "Total Parameters":
#                         num_params = line.split(":")[1]
#                 time = float(time)
#                 num_params = int(num_params)
#             f.close()
#             res[method]["time"].append(time)
#             res[method]["param"].append(num_params)
#             sparsity_label.append("{:.2%}".format(sparsity))
#         if add_head:
#             files = list(Path(path).rglob("result_zero-sparsity-sgcn_SimpleGCN_ENZYMES_sparsity_0.0.txt"))
#             for filename in files:
#                 sparsity = 0.0
#                 with open(filename) as f:
#                     for line in f.readlines():
#                         if line.split(":")[0] == "Average Time Per Epoch":
#                             time = line.split(":")[1].split(" ")[1]
#                         if line.split(":")[0] == "Total Parameters":
#                             num_params = line.split(":")[1]
#                     time = float(time)
#                     num_params = int(num_params)
#                 f.close()
#                 res[method]["time"].insert(0, time)
#                 res[method]["param"].insert(0, num_params)
#                 sparsity_label.insert(0, "{:.2%}".format(sparsity))
                
#         if add_tail:
#             md = method.strip("Expander")
#             files = list(Path(path).rglob("result_varing-sparsity_{}_{}_sparsity_0.1.txt".format(md, dataset)))
#             for filename in files:
#                 sparsity = 1.0
#                 with open(filename) as f:
#                     for line in f.readlines():
#                         if line.split(":")[0] == "Average Time Per Epoch":
#                             time = line.split(":")[1].split(" ")[1]
#                         if line.split(":")[0] == "Total Parameters":
#                             num_params = line.split(":")[1]
#                     time = float(time)
#                     num_params = int(num_params)
#                 f.close()
#                 res[method]["time"].append(time)
#                 res[method]["param"].append(num_params)
#                 sparsity_label.append("{:.2%}".format(sparsity))
    
#     fig, ax1 = plt.subplots()
    
#     color = 'tab:red'
#     ax1.set_ylabel("running time per epoch")
#     ax1.set_xticks(np.arange(len(res[method]["time"])))
#     ax1.set_xticklabels(sparsity_label, rotation='vertical')
#     ax1.set_xlabel("sparsification rate")
#     ax1.set_ylim(0,2)
#     ax1.plot(np.arange(len(res[method]["time"])), res[method]["time"], color=color)
#     ax1.set_title("Time and Flops for {} on {}".format(method, dataset))
#     ax1.yaxis.grid(True)
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()
#     color = 'tab:blue'
#     ax2.set_ylabel('flops (total number of parameters)', color=color)
#     ax2.plot(np.arange(len(res[method]["param"])), res[method]["param"], color=color)
#     ax2.tick_params(axis='y', labelcolor=color)

#     fig.tight_layout()
#     plt.savefig(path+"{}_{}_time.png".format(dataset, METHODS[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        help="give path of folder that stores result files")
    parser.add_argument('--dataset', action="store", nargs="+", help="decide which datasets need to be presented")
    parser.add_argument('--output', help="give output file")
    args = parser.parse_args()

    bar_plot(args.folder, args.dataset, args.output)