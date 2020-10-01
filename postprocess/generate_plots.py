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
from matplotlib.offsetbox import AnchoredText


VALUE = "Accuracy"
# MODEL = [["GCN", "GIN", "MLP"], ["GCN", "GIN", "MLP"], ["GCN", "GIN", "GraphSage", "PNA", "MLP"]]
MODEL = ["GCN", "GIN", "MLP"] #"GraphSage", "PNA"
ACTIV = ["ReLU", "PReLU", "Tanh"]
DENSITY = [0.1, 0.5, 0.9]
RECORD = [VALUE, "Time per Epoch(s)", "#Parameters"]
BARWIDTH = 0.3
GRAPHID = ["a", "b", "c", "e", "f", "g", "d", "h"]
DUPLICATES = 1000

def bar_plots(folder, dataset, output_file):
    dataframes = list()
    for k, ds in enumerate(dataset):
        data = list()
        for file in glob.glob(folder+"*{}*.txt".format(ds)):
            components = file.replace(folder, "").split("_")
            single_row = list()

            gnn = components[2].replace("Simple", "").replace("Activation", "")
            for i in MODEL:
                if gnn.lower() == i.lower():
                    single_row.append(i)
            if len(single_row) == 0:
                continue

            types = components[1].split("-")
            for i in ["Regular", "Expander", "Activations", "Simple"]:
                if types[0].lower() == i.lower():
                    if i != "Regular":
                        exp = i
                    else:
                        exp = "Vanilla"
                    if types[0].lower() == "activations" or types[0].lower() == "simple":
                        if gnn.lower() == "mlp":
                            continue
                        
            if len(types) == 1:
                single_row.append(exp)
            elif len(types) == 2:
                for j in ACTIV:
                    if j.lower() == types[1].lower():
                        single_row.append(exp+"-"+j)
                if len(single_row) == 1:
                    continue
            elif len(types) == 3:
                for j in DENSITY:
                    if float(types[-1]) == j:
                        single_row.append(exp+"-{:.0%}".format(j))
                if len(single_row) == 1:
                    continue

            rows = []
            with open(file) as f:
                for line in f.readlines():
                    if line.split(":")[0] == "TEST ACCURACY averaged":
                        res = line.split(":")[1].split("with s.d.")[0]
                        if len(line.split(":")[1].split("with s.d."))>1:
                            std = line.split(":")[1].split("with s.d.")[1].replace(" ", "")
                        else:
                            std = 0
                    if line.split(":")[0] == "TEST MAE averaged":
                        res = line.split(":")[1].split("with s.d.")[0]
                    # if line.split(":")[0] == "Average Convergence Time (Epochs)":
                    #     epochs = line.split(":")[1].split(" ")[1]
                    #     rows.append(single_row+[int(float(epochs)), "Converge(#Epochs)"])
                    if line.split(":")[0] == "Total Parameters":
                        n_params = line.split(":")[1].split(" ")[1]
                    if line.split(":")[0] == "Average Time Per Epoch":
                        time = line.split(":")[1].split(" ")[1]
                        rows.append(single_row+[float(time), "Time per Epoch(s)"])
                single_row.extend([float(res), float(time), int(float(n_params)), float(std)])
            f.close()
            data.append(single_row)

        df = pd.DataFrame(data=data, columns=["Model", "Type"]+RECORD+["std"])  #, "Value", "ValueType"])
        dataframes.append(df)

    hue_order = ["Simple"]
    for a in ACTIV:
        hue_order.append("Activations-{}".format(a))
    for e in DENSITY:
        hue_order.append("Expander-{:.0%}".format(e))
    hue_order.append("Vanilla")

    
    # sns.set(style="whitegrid")
    # fig, axlist = plt.subplots(1, len(dataset), figsize=(20*len(dataset), 20))
    # for i in range(len(dataset)):
    #     ax = plt.subplot(1, len(dataset), i+1)
    #     g = sns.barplot(x="Model", y=VALUE, hue="Type", data=dataframes[i], hue_order=hue_order, order=MODEL, palette="Set2")
    #     max_value = dataframes[i][VALUE].max()
    #     min_value = dataframes[i][VALUE].min()
    #     ax.set_title("{}".format(dataset[i]), fontsize=70)
    #     ax.set_xlabel("")
    #     # ax.set_grid_on(True)
    #     ax.set_frame_on(False)
    #     # ax.spines['top'].set_visible(False)
    #     # ax.spines['right'].set_visible(False)
    #     # ax.spines['left'].set_visible(False)
    #     # ax.grid(axis='y')
    #     #plt.box(False)
    #     ax.tick_params(axis='both', which='major', labelsize=48)
    #     # ax.set_yticklabels(fontsize=40, fontweight='light')
    #     # ax.set_xticklabels(fontsize=50, fontweight='light')
    #     y_min, y_max = ax.get_ylim() 
    #     ax.set_ylim(y_min+0.5*min_value, y_max+0.06*y_max)
        
    #     if i == 0:
    #         ax.set_ylabel("MAE", fontsize=60)
    #     elif i == 1:
    #         ax.set_ylabel("Accuracy", fontsize=60)
    #     else:
    #         ax.set_ylabel("")

    #     if i == len(dataset)-1:
    #         handles, labels = ax.get_legend_handles_labels()
    #     ax.get_legend().remove()
    #     anc = AnchoredText("({})".format(GRAPHID[i]), loc="upper left", frameon=False, prop=dict(fontweight="bold", fontsize=50))
    #     ax.add_artist(anc)
    #     # tl = ((ax.get_xlim()[1] - ax.get_xlim()[0])*0.010 + ax.get_xlim()[0], (ax.get_ylim()[1] - ax.get_ylim()[0])*0.98 + ax.get_ylim()[0])
    #     # ax.text(tl[0], tl[1], r"({})".format("g"), fontsize=30)
    #     # if i==2:
    #     #     ax.set_ylabel("MAE", fontsize=50)
    #     # if i == 1:
    #     #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4, fontsize=30)
    
    # fig.subplots_adjust(top=0.9, left=0.125, right=0.9, bottom=0.01)
    # # lgd = axlist.flatten()[-2].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4, bbox_transform=fig.transFigure) 
    # lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), fancybox=False, shadow=False, ncol=4, fontsize=50, frameon=False)

    # # 
    # plt.tight_layout(pad=20, w_pad=2.0)
    # plt.savefig(output_file, dpi=200, bbox_inches='tight') #,bbox_extra_artists=(lgd,),  

    sns.set(style="whitegrid")
    fig, axlist = plt.subplots(2, 4, figsize=(80, 40), sharex=True)
    for i in range(len(dataset)):
        if i<=2:
            ax = plt.subplot(2, 4, i+1)
        else:
            ax = plt.subplot(2, 4, i+2)
        
        if dataset[i].lower() != "MNIST".lower() and dataset[i].lower() != "CIFAR10".lower():
            dfCopy = dataframes[i].loc[dataframes[i].index.repeat(DUPLICATES)].copy()
            dfCopy[VALUE] = np.random.normal(dfCopy[VALUE].values, dfCopy['std'].values)
            g = sns.barplot(x="Model", y=VALUE, hue="Type", data=dfCopy, hue_order=hue_order, order=MODEL, palette="Set2", ci="sd", capsize=0.03)
        else:
            g = sns.barplot(x="Model", y=VALUE, hue="Type", data=dataframes[i], hue_order=hue_order, order=MODEL, palette="Set2")
        
        max_value = dataframes[i][VALUE].max()
        min_value = dataframes[i][VALUE].min()
        ax.set_title("{}".format(dataset[i]), fontsize=70)
        ax.set_xlabel("")
        # ax.set_grid_on(True)
        ax.set_frame_on(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.grid(axis='y')
        #plt.box(False)
        if i<=2:
            ax.set_xticks([])
        
        ax.tick_params(axis='both', which='major', labelsize=48)
            
        y_min, y_max = ax.get_ylim() 
        ax.set_ylim(y_min+0.2*min_value, y_max+0.02*y_max)
        
        if i == 0 or i == 3:
            ax.set_ylabel("Accuracy", fontsize=60)
        else:
            ax.set_ylabel("")

        if i == len(dataset)-1:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        anc = AnchoredText("({})".format(GRAPHID[i]), loc="upper left", frameon=False, prop=dict(fontweight="bold", fontsize=50))
        ax.add_artist(anc)
        # tl = ((ax.get_xlim()[1] - ax.get_xlim()[0])*0.010 + ax.get_xlim()[0], (ax.get_ylim()[1] - ax.get_ylim()[0])*0.98 + ax.get_ylim()[0])
        # ax.text(tl[0], tl[1], r"({})".format("g"), fontsize=30)
        # if i==2:
        #     ax.set_ylabel("MAE", fontsize=50)
        # if i == 1:
        #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4, fontsize=30)
    
    
    
    df = dataframes[2] #Proteins
    df_new = df[df["Model"]=="GCN"].copy()
    times = []
    nparam = []
    ticks = []
    for tick in hue_order:
        for idx, types in enumerate(df_new["Type"]):
            if tick == types:
                times.append(df_new["Time per Epoch(s)"].iloc[idx])
                nparam.append(df_new["#Parameters"].iloc[idx])
        ticks.append(tick)

    df2 = pd.DataFrame({'Time per Epoch(s)': times,
                        '#Parameters': nparam,
                        'Type': ticks})

    ax = plt.subplot(2, 4, 4)
    ax.plot(df2["Type"], df2["Time per Epoch(s)"], lw=5, ls="--")
    ax.scatter(df2["Type"], df2["Time per Epoch(s)"], c=sns.color_palette("Set2"), s=200, marker="o", linewidths=20)

    # max_value = df2["Time per Epoch(s)"].max()
    # min_value = df2["Time per Epoch(s)"].min()
    ax.set_title("{}".format("PROTEINS (GCN)"), fontsize=70)
    ax.set_xlabel("")
    ax.set_frame_on(False)
    # ax.grid(axis='y')
    ax.set_xticks([])
    ax.tick_params(axis='both', which='major', labelsize=48)
            
    # y_min, y_max = ax.get_ylim() 
    # ax.set_ylim(y_min+0.2*min_value, y_max+0.02*y_max)

    ax.set_ylabel("Time per Epoch(s)", fontsize=60)
    anc = AnchoredText("({})".format(GRAPHID[-2]), loc="upper left", frameon=False, prop=dict(fontweight="bold", fontsize=50))
    ax.add_artist(anc)

    sns.set(style="whitegrid")
    ax = plt.subplot(2, 4, 8)
    g = sns.barplot(x="Type", y="#Parameters", data=df2, palette="Set2")

    max_value = df2["#Parameters"].max()
    min_value = df2["#Parameters"].min()
    ax.set_title("{}".format("PROTEINS (GCN)"), fontsize=70)
    ax.set_xlabel("")
    ax.set_frame_on(False)
    # ax.grid(axis='y')
    ax.set_xticks([])
    yticks = [int(x/1000) for x in ax.get_yticks()]
    ax.set_yticklabels(yticks)
    ax.tick_params(axis='both', which='major', labelsize=48)
            
    y_min, y_max = ax.get_ylim() 
    ax.set_ylim(y_min+0.2*min_value, y_max+0.01*y_max)

    ax.set_ylabel(r"#Parameters($10^3$)", fontsize=60)
    anc = AnchoredText("({})".format(GRAPHID[-1]), loc="upper left", frameon=False, prop=dict(fontweight="bold", fontsize=50))
    ax.add_artist(anc)
    
    
    fig.subplots_adjust(top=0.9, left=0.125, right=0.9, bottom=0.01)
    # lgd = axlist.flatten()[-2].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4, bbox_transform=fig.transFigure) 
    lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), fancybox=False, shadow=False, ncol=4, fontsize=50, frameon=False)

    # 
    plt.tight_layout(pad=20, w_pad=2.0)
    plt.savefig(output_file, dpi=200, bbox_inches='tight') #,bbox_extra_artists=(lgd,),  




        
    # # sns.set_theme(style="whitegrid")
    # g = sns.catplot(x="Model", y=VALUE, hue="Type", col="Dataset", data=df, kind="bar", height=4, aspect=1.0, sharex=True, sharey=False, hue_order=hue_order, col_order=dataset)
    # #g = sns.lineplot(x="Type", y="Time per Epoch(s)", hue="Model", data=df)
    # # g.set_xticklabels(rotation=45, horizontalalignment='right', fontweight='light')
    # g.set_xticklabels(fontsize=10, fontweight='light')
    # g.tight_layout()
    # plt.setp(g._legend.get_texts(), fontsize=10)
    # plt.setp(g._legend.get_title(), fontsize=10)
    # # plt.setp(g._legend.get_texts(), fontsize=5)
    # g.savefig(output_file, dpi=200)


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

    plt.figure(figsize=(20,20))
    # sns.set_theme(style="whitegrid")
    g = sns.catplot(x="Model", y=VALUE, hue="Type", col="Dataset", data=df, kind="bar", height=4, aspect=1.0, sharex=True, sharey=False, hue_order=hue_order, col_order=dataset)
    #g = sns.lineplot(x="Type", y="Time per Epoch(s)", hue="Model", data=df)
    # g.set_xticklabels(rotation=45, horizontalalignment='right', fontweight='light')
    g.set_xticklabels(fontsize=10, fontweight='light')
    g.tight_layout()
    plt.setp(g._legend.get_texts(), fontsize=10)
    plt.setp(g._legend.get_title(), fontsize=10)
    # plt.setp(g._legend.get_texts(), fontsize=5)
    g.savefig(output_file, dpi=200)

                    
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

    bar_plots(args.folder, args.dataset, args.output)