#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:00:38 2020

@author: johannes
"""


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#
#   LOAD the network
#

path = "d:/projects/dascim/ExpanderGNN/out/TUs_graph_classification/expanders/ExpanderMLP_ENZYMES_GPU1_16h17m00s_on_Apr_21_2020_expander_size_2ExpanderMLPNet/Sequential/"
ExpanderLinearLayer_0 = np.load(path+"ExpanderLinearLayer_{}.gpickle".format(0), allow_pickle=True)
ExpanderLinearLayer_1 = np.load(path+"ExpanderLinearLayer_{}.gpickle".format(1), allow_pickle=True)
ExpanderLinearLayer_2 = np.load(path+"ExpanderLinearLayer_{}.gpickle".format(2), allow_pickle=True)
ExpanderLinearLayer_3 = np.load(path+"ExpanderLinearLayer_{}.gpickle".format(3), allow_pickle=True)

print(sum(sum(nx.adjacency_matrix(ExpanderLinearLayer_0).toarray())))
print(sum(sum(nx.adjacency_matrix(ExpanderLinearLayer_1).toarray())))
print(sum(sum(nx.adjacency_matrix(ExpanderLinearLayer_2).toarray())))
print(sum(sum(nx.adjacency_matrix(ExpanderLinearLayer_3).toarray())))

mapping = {x: x+18 for x in ExpanderLinearLayer_1.nodes}
ExpanderLinearLayer_1_relabeled = nx.relabel_nodes(ExpanderLinearLayer_1, mapping)
ExpanderLayers = nx.compose(ExpanderLinearLayer_0, ExpanderLinearLayer_1_relabeled)

mapping = {x: x+18+128 for x in ExpanderLinearLayer_2.nodes}
ExpanderLinearLayer_2_relabeled = nx.relabel_nodes(ExpanderLinearLayer_2, mapping)
ExpanderLayers = nx.compose(ExpanderLayers, ExpanderLinearLayer_2_relabeled)

mapping = {x: x+18+128+128 for x in ExpanderLinearLayer_3.nodes}
ExpanderLinearLayer_3_relabeled = nx.relabel_nodes(ExpanderLinearLayer_3, mapping)
ExpanderLayers = nx.compose(ExpanderLayers, ExpanderLinearLayer_3_relabeled)

print(sum(sum(nx.adjacency_matrix(ExpanderLayers).toarray())))

num_layers=5
#nodes_per_layer = [int(len(ExpanderLinearLayer_0)/2), int(len(ExpanderLinearLayer_0)/2)]
nodes_per_layer = [18, 128, 128, 128, 128]





#
#   DRAW graphs in nice prescribed layout
#

#plt.figure(1)
#nx.draw_planar(ExpanderLinearLayer_0)
#list(SBM_sample_k_core2.nodes)


plot_pos = dict(ExpanderLayers.nodes)
layer_positions = np.array([], dtype=np.int64).reshape(0,2)
for i in np.arange(num_layers):
    new_layer_positions = np.concatenate((np.zeros((nodes_per_layer[i],1)) +i, -1/2 + np.arange(nodes_per_layer[i]).reshape((nodes_per_layer[i],1))/nodes_per_layer[i]), axis=1)
    layer_positions = np.vstack([layer_positions, new_layer_positions])
i=0
for d in plot_pos:
   plot_pos[d] = layer_positions[i,:]
   i+=1

plt.figure(1)
nx.draw_networkx(ExpanderLayers, pos=plot_pos)
plt.axis('equal')



#
#    CALCULATE the eigenvalues
#

ExpanderLayers_undirected = ExpanderLayers.to_undirected()

print('\nEigenvalues of the undirected architecture:')

A_evals = np.sort(np.linalg.eigh(nx.adjacency_matrix(ExpanderLayers_undirected).toarray())[0])
L_evals = np.sort(np.linalg.eigh(nx.laplacian_matrix(ExpanderLayers_undirected).toarray())[0])
Lsym_evals = np.sort(np.linalg.eigh(nx.normalized_laplacian_matrix(ExpanderLayers_undirected).toarray())[0])

# ISSUE we got complex eigenvalues of a symmetric matrix
#sum(np.iscomplex(A_evals))
#sum(np.iscomplex(L_evals))
#sum(np.iscomplex(Lsym_evals))
#A = nx.adjacency_matrix(ExpanderLayers_undirected).toarray()
#print(sum(sum((np.abs(A-A.T) > 10**-8))))
# FIX use eigh instead of eig. This utilises the fact that the input matrix is symmetric



print('Lsym\t', Lsym_evals)
print('L\t', L_evals)
print('A\t', A_evals)


plt.figure(2)
plt.subplot(1,3,1)
plt.plot(np.arange(len(A_evals)), A_evals, 'b*')
plt.subplot(1,3,2)
plt.plot(np.arange(len(L_evals)), L_evals, 'bd')
plt.subplot(1,3,3)
plt.plot(np.arange(len(Lsym_evals)), Lsym_evals, 'bo')


#A_evals = np.sort(np.linalg.eig(nx.adjacency_matrix(ExpanderLayers).toarray())[0])
#Ldir_evals = np.sort(np.linalg.eig(nx.directed_laplacian_matrix(ExpanderLayers))[0])







