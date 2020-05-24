import json
import numpy as np
import sys
from operator import itemgetter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

METHODS = ["ExpanderGCN", "ExpanderMLP", "ExpanderGIN"] #"ExpanderGIN", "ExpanderMLP"
SIZES = [0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]
BARWIDTH = 0.3

def draw_errorplot(dataset, path="d:\\projects\\dascim\\ExpanderGNN\\postprocess\\new_results\\"):
    fig, ax = plt.subplots()
    res = {}
    for method in METHODS:
        res[method] = {}
        res[method]["acc"] = []
        res[method]["std"] = []
        sizes_label = []
        files = list(Path(path).rglob("result_{}_{}_GPU1_*_expander_size_*.txt".format(method, dataset)))
        sorted_files = sorted(files, key=lambda x: float(str(x).split("expander_size_")[-1].strip(".txt")))
        for filename in sorted_files:
            size = float(str(filename).split("expander_size_")[-1].strip(".txt"))
            with open(filename) as f:
                for line in f.readlines():
                    if line.split(":")[0] == "TEST ACCURACY averaged":
                        acc, std = line.split(":")[1].split("with s.d.")
                acc = float(acc)
                std = float(std)
            f.close()
            res[method]["acc"].append(acc)
            res[method]["std"].append(std)
            sizes_label.append("{:.2%}".format(float(size/16)))
        
        
        
        ax.errorbar(np.arange(len(res[method]["acc"])), res[method]["acc"], res[method]["std"], uplims=True, lolims=True, label=method)
        
    ax.set_ylabel("Classification Accuracy")
    ax.set_xticks(np.arange(len(res[method]["acc"])))
    print(sizes_label)
    ax.set_xticklabels(sizes_label, rotation='vertical')
    ax.set_ylim(0,80)
    ax.set_title("Accuracy v.s. Sparsity for {}".format(dataset))
    ax.yaxis.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path+"{}_{}_acc.png".format(dataset, METHODS[0]))


if __name__ == "__main__":
    draw_errorplot("ENZYMES")