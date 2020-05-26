import json
import numpy as np
import sys
from operator import itemgetter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

METHODS = ["ExpanderGraphSage"] #"ExpanderGIN", "ExpanderMLP"
sparsities = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
BARWIDTH = 0.3

def draw_errorplot(dataset, exp_name="varing-sparsity-sparse-readout", path="d:\\results\\results\\", add_head=True, add_tail=True):
    fig, ax = plt.subplots()

    
    res = {}
    for method in METHODS:
        res[method] = {}
        res[method]["acc"] = []
        res[method]["std"] = []
        sparsity_label = []
        files = list(Path(path).rglob("result_{}_{}_{}_sparsity_*.txt".format(exp_name, method, dataset)))
        sorted_files = sorted(files, key=lambda x: float(str(x).split("sparsity_")[-1].strip(".txt")))
        for filename in sorted_files:
            sparsity = float(str(filename).split("sparsity_")[-1].strip(".txt"))
            with open(filename) as f:
                for line in f.readlines():
                    if line.split(":")[0] == "TEST ACCURACY averaged":
                        acc, std = line.split(":")[1].split("with s.d.")
                acc = float(acc)
                std = float(std)
            f.close()
            res[method]["acc"].append(acc)
            res[method]["std"].append(std)
            sparsity_label.append("{:.2%}".format(sparsity))
            
        if add_head:
            files = list(Path(path).rglob("result_zero-sparsity-sgcn_SimpleGCN_ENZYMES_sparsity_0.0.txt"))
            for filename in files:
                sparsity = 0.0
                with open(filename) as f:
                    for line in f.readlines():
                        if line.split(":")[0] == "TEST ACCURACY averaged":
                            acc, std = line.split(":")[1].split("with s.d.")
                    acc = float(acc)
                    std = float(std)
                f.close()
                res[method]["acc"].insert(0, acc)
                res[method]["std"].insert(0, std)
                sparsity_label.insert(0, "{:.2%}".format(sparsity))
                
        if add_tail:
            md = method.strip("Expander")
            files = list(Path(path).rglob("result_varing-sparsity_{}_{}_sparsity_0.1.txt".format(md, dataset)))
            for filename in files:
                sparsity = 1.0
                with open(filename) as f:
                    for line in f.readlines():
                        if line.split(":")[0] == "TEST ACCURACY averaged":
                            acc, std = line.split(":")[1].split("with s.d.")
                    acc = float(acc)
                    std = float(std)
                f.close()
                res[method]["acc"].append(acc)
                res[method]["std"].append(std)
                sparsity_label.append("{:.2%}".format(sparsity))
        
        ax.errorbar(np.arange(len(res[method]["acc"])), res[method]["acc"], res[method]["std"], uplims=True, lolims=True, label=method)
        
    ax.set_ylabel("Classification Accuracy")
    ax.set_xticks(np.arange(len(res[method]["acc"])))
    ax.set_xticklabels(sparsity_label, rotation='vertical')
    ax.set_ylim(0,80)
    ax.set_title("Accuracy v.s. Sparsity for {}".format(dataset))
    ax.yaxis.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path+"{}_{}_acc.png".format(dataset, METHODS[0]))

def draw_timeplot(dataset, exp_name="varing-sparsity-sparse-readout", path="d:\\results\\results\\", add_head=True, add_tail=True):
    fig, ax = plt.subplots()
    res = {}
    for method in METHODS:
        res[method] = {}
        res[method]["time"] = []
        res[method]["param"] = []
        sparsity_label = []
        files = list(Path(path).rglob("result_{}_{}_{}_sparsity_*.txt".format(exp_name, method, dataset)))
        sorted_files = sorted(files, key=lambda x: float(str(x).split("sparsity_")[-1].strip(".txt")))
        for filename in sorted_files:
            sparsity = float(str(filename).split("sparsity_")[-1].strip(".txt"))
            with open(filename) as f:
                for line in f.readlines():
                    if line.split(":")[0] == "Average Time Per Epoch":
                        time = line.split(":")[1].split(" ")[1]
                    if line.split(":")[0] == "Total Parameters":
                        num_params = line.split(":")[1]
                time = float(time)
                num_params = int(num_params)
            f.close()
            res[method]["time"].append(time)
            res[method]["param"].append(num_params)
            sparsity_label.append("{:.2%}".format(sparsity))
        if add_head:
            files = list(Path(path).rglob("result_zero-sparsity-sgcn_SimpleGCN_ENZYMES_sparsity_0.0.txt"))
            for filename in files:
                sparsity = 0.0
                with open(filename) as f:
                    for line in f.readlines():
                        if line.split(":")[0] == "Average Time Per Epoch":
                            time = line.split(":")[1].split(" ")[1]
                        if line.split(":")[0] == "Total Parameters":
                            num_params = line.split(":")[1]
                    time = float(time)
                    num_params = int(num_params)
                f.close()
                res[method]["time"].insert(0, time)
                res[method]["param"].insert(0, num_params)
                sparsity_label.insert(0, "{:.2%}".format(sparsity))
                
        if add_tail:
            md = method.strip("Expander")
            files = list(Path(path).rglob("result_varing-sparsity_{}_{}_sparsity_0.1.txt".format(md, dataset)))
            for filename in files:
                sparsity = 1.0
                with open(filename) as f:
                    for line in f.readlines():
                        if line.split(":")[0] == "Average Time Per Epoch":
                            time = line.split(":")[1].split(" ")[1]
                        if line.split(":")[0] == "Total Parameters":
                            num_params = line.split(":")[1]
                    time = float(time)
                    num_params = int(num_params)
                f.close()
                res[method]["time"].append(time)
                res[method]["param"].append(num_params)
                sparsity_label.append("{:.2%}".format(sparsity))
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_ylabel("running time per epoch")
    ax1.set_xticks(np.arange(len(res[method]["time"])))
    ax1.set_xticklabels(sparsity_label, rotation='vertical')
    ax1.set_xlabel("sparsification rate")
    ax1.set_ylim(0,2)
    ax1.plot(np.arange(len(res[method]["time"])), res[method]["time"], color=color)
    ax1.set_title("Time and Flops for {} on {}".format(method, dataset))
    ax1.yaxis.grid(True)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('flops (total number of parameters)', color=color)
    ax2.plot(np.arange(len(res[method]["param"])), res[method]["param"], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(path+"{}_{}_time.png".format(dataset, METHODS[0]))

if __name__ == "__main__":
    draw_errorplot("ENZYMES")
    draw_timeplot("ENZYMES")