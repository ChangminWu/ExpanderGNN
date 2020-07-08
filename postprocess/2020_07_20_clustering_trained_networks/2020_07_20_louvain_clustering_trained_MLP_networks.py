#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:55:39 2020

@author: johannes
"""


import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.stats import norm
import scipy.stats as stats




architectures = os.listdir("weighted_MLP_only")
architectures.sort()


architecture_networks = {}
nodes_per_layer = {}
num_layers = {}

#
# IMPORT the architectures
#


for architecture in architectures:
    GNN = architecture.split('_')[1]
    sparsity = float(architecture.split("sparsity_")[-1].split("Expander")[0])
    ExpanderLayers = nx.DiGraph()
    
    
    print('loading ' + GNN + ' with density: ' + str(sparsity))    
    

    LayerPaths = []
    for dirpath, dirnames, files in os.walk('weighted/'+architecture):
        for file_name in files:
            if file_name.endswith('.npy'):
                LayerPaths.append(os.path.join(dirpath, file_name))
    
    LayerPaths.sort(key = lambda x: x.split('.npy')[0][-1]) #sort paths according to the number preceeding '.gpickle' in the file name
        
                
    #ONLY use split0 for now
    LayerPaths = [Path for Path in LayerPaths if 'split0' in Path]
    
    
    #load the layers and build the architecture
    no_in_nodes = [18]
    for Layer in LayerPaths:
            weights = np.transpose(np.load(Layer, allow_pickle=True))
            no_outnodes,no_innodes = weights.shape
            
            upper_rows = np.concatenate((np.zeros((no_outnodes,no_outnodes)),weights), axis=1)
            lower_rows = np.zeros((no_innodes,no_outnodes+no_innodes))
            all_rows = np.concatenate((upper_rows, lower_rows), axis=0)                
            ExpanderLinearLayer = nx.from_numpy_matrix(all_rows, create_using=nx.DiGraph)

            
            mapping = {x: x+sum(no_in_nodes[:LayerPaths.index(Layer)]) for x in ExpanderLinearLayer.nodes}
            ExpanderLinearLayer = nx.relabel_nodes(ExpanderLinearLayer, mapping)
            ExpanderLayers = nx.compose(ExpanderLayers, ExpanderLinearLayer)
            
            no_in_nodes.append(len(ExpanderLinearLayer.nodes())-no_in_nodes[-1])
    
    if GNN == 'ExpanderMLP':
#        print(LayerPaths)
#        LayerPaths_saved = LayerPaths
        nodes_per_layer[GNN+'_'+str(sparsity)] = no_in_nodes
        num_layers[GNN+'_'+str(sparsity)] = len(LayerPaths) +1
    
    architecture_networks[GNN+'_'+str(sparsity)] = ExpanderLayers
    
        
    
        



#
#   Please insert the Louvain clustering here
#

    
    
    
    
    
#calculate node locations in the architecture plot 
    
node_positions = {}


for GNN in architecture_networks:
    ExpanderLayers = architecture_networks[GNN]
    
    plot_pos = dict(ExpanderLayers.nodes)
    layer_positions = np.array([], dtype=np.int64).reshape(0,2)
    for i in np.arange(num_layers[GNN]):
        new_layer_positions = np.concatenate((np.zeros((nodes_per_layer[GNN][i],1)) +i, -1/2 + np.arange(nodes_per_layer[GNN][i]).reshape((nodes_per_layer[GNN][i],1))/nodes_per_layer[GNN][i]), axis=1)
        layer_positions = np.vstack([layer_positions, new_layer_positions])
    i=0
    for d in plot_pos:
       plot_pos[d] = layer_positions[i,:]
       i+=1

    node_positions[GNN] = plot_pos



    
#
#    Then create a plot of the network with a code like the following. Here the main task is to assign a number of colors to the variable number of clusters
#

A_kmeans = KMeans(n_clusters=3).fit(A_evecs[GNN][:,-3:])    
A_spectral_clustering_result = [['b','r','g'][2*(n==1)+(n==2)] for n in A_kmeans.labels_] #['r' if y == 1 else 'b' for y in A_kmeans .labels_]
    
plt.figure(4,figsize=(25,25))
nx.draw_networkx(ExpanderLayers, pos=plot_pos[GNN], node_color = A_spectral_clustering_result )
#plt.axis('equal')
    
fig = plt.gcf()
fig.savefig('figures/plot_full_architecture_A_spectral_clustering3.eps', format='eps', dpi=1200) 


    
    
    
    
 
# sum(nx.adjacency_matrix(ExpanderLayers_undirected).toarray())   
# np.nonzero(nx.adjacency_matrix(ExpanderLayers_undirected).toarray())
# nx.adjacency_matrix(ExpanderLayers_undirected).toarray()[0,23]   
 
    
 
# plt.figure(20)

# #too large 
# #plt.hist(nx.adjacency_matrix(ExpanderLayers_undirected).toarray().reshape((1,-1)))
 
# #hence I filter the zeros
# plt.hist(nx.adjacency_matrix(ExpanderLayers_undirected).toarray()[np.nonzero(nx.adjacency_matrix(ExpanderLayers_undirected).toarray())], bins = 1000)    

# fig = plt.gcf()
# fig.suptitle("MLP 0.9", fontsize=14)
# fig.savefig('figures/weight_hist_0_9.eps', format='eps', dpi=1200) 
    
  
    
  
# plt.figure(21)

# #too large 
# #plt.hist(nx.adjacency_matrix(ExpanderLayers_undirected).toarray().reshape((1,-1)))
 
# #hence I filter the zeros
# plt.hist(nx.adjacency_matrix(architecture_networks['ExpanderMLP_0.9']).toarray()[np.nonzero(nx.adjacency_matrix(architecture_networks['ExpanderMLP_0.9']).toarray())], bins = 1000)    

# fig = plt.gcf()
# fig.suptitle("MLP 0.9", fontsize=14)
# fig.savefig('figures/weight_hist_0_9_pos_and_neg.eps', format='eps', dpi=1200)         





# plt.figure(22)

# #too large 
# #plt.hist(nx.adjacency_matrix(ExpanderLayers_undirected).toarray().reshape((1,-1)))
 
# #hence I filter the zeros
# weights = nx.adjacency_matrix(architecture_networks['ExpanderMLP_0.9']).toarray()[np.nonzero(nx.adjacency_matrix(architecture_networks['ExpanderMLP_0.9']).toarray())]
# plt.hist(weights, density = True, bins = 1000, color='b')    

# mu, std = norm.fit(weights)


# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)

# plt.show()

# fig = plt.gcf()
# fig.savefig('figures/weight_hist_0_9_pos_and_neg_dens.eps', format='eps', dpi=1200)         


# # fig = plt.gcf()
# # fig.suptitle("MLP 0.9", fontsize=14)
# # fig.savefig('figures/weight_hist_0_9_pos_and_neg.eps', format='eps', dpi=1200)         



# plt.figure(23)

# stats.probplot((weights-mu)/std, dist="norm", plot=plt)
# plt.title("Normal Q-Q plot")
# plt.show()
# fig = plt.gcf()
# fig.savefig('figures/weight_0_9_pos_and_neg_qq.eps', format='eps', dpi=1200)         










