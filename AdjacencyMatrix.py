#!/usr/bin/env python
# coding: utf-8

# In[3]:


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import collections
import seaborn as sns       # package for better viewing of networks
import operator 
from networkx.algorithms import community
from networkx.algorithms import hits
import networkx.algorithms.community
import re
from matplotlib import pyplot, patches
import scipy
from scipy import io
from scipy import sparse
import community     #from package python-louvain


# In[4]:


#import modules for building edge weighted networks
from EdgeWeightedNetworkBuilding import build_network_from_excel, build_network_from_df  


# In[2]:


#build edge weighted network
df_getNetwork = build_network_from_excel(file_path = "/Users/iris/Documents/QMUL-2018/Individual_Project/coding/datasets/mcf7_ntera2_hl60_ksea.xlsm",key="MCF7", sheet_name = "zScorenodes.edges", threshold = 0.2)
cell_line = "MCF7"


# In[5]:


#show adjacency metrix and save to file
mat_ad = nx.adjacency_matrix(df_getNetwork).todense()
mat_ad
type(mat_ad)


# In[12]:


scipy_mat_ad = sparse.csr_matrix(mat_ad)
G = nx.from_scipy_sparse_matrix(scipy_mat_ad)
draw_adjacency_matrix(G)


# In[13]:


def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)


# In[14]:


# Run louvain community finding algorithm
louvain_community_dict = community.best_partition(G, resolution=1.2)
louvain_community_dict


# In[15]:


from collections import defaultdict


# In[17]:


# Run louvain community finding algorithm
louvain_community_dict = community.best_partition(G, resolution=1.3)


# Convert community assignmet dict into list of communities
louvain_comms = defaultdict(list)
for node_index, comm_id in louvain_community_dict.iteritems():
    louvain_comms[comm_id].append(node_index)
louvain_comms = louvain_comms.values()

nodes_louvain_ordered = [node for comm in louvain_comms for node in comm]
draw_adjacency_matrix(G, nodes_louvain_ordered, [louvain_comms], ["blue"])

print nodes_louvain_ordered
print louvain_comms


# In[20]:


louvain_community_dict = community.best_partition(df_getNetwork, resolution=1.3)

louvain_comms=collections.defaultdict(list)

for node_index, comm_id in louvain_community_dict.items():
    louvain_comms[comm_id].append(node_index)
louvain_comms = louvain_comms.values()

nodes_louvain_ordered = [node for comm in louvain_comms for node in comm]

adjacency_matrix = nx.convert_matrix.to_pandas_adjacency(df_getNetwork,nodelist=nodes_louvain_ordered)

#Plot adjacency matrix in toned-down black and white
plt.figure(figsize=(25, 25))
sns.heatmap(adjacency_matrix,cmap="Blues")

partitions=[louvain_comms]
colors=["Black"]

assert len(partitions) == len(colors)
ax = plt.gca()
for partition, color in zip(partitions, colors):
    current_idx = 0
    for module in partition:
        ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                    len(module), # Width
                                    len(module), # Height
                                    facecolor="none",
                                    edgecolor=color,
                                    linestyle="--",
                                    linewidth="2"))
        current_idx += len(module)

plt.savefig("heatmap.svg",format="svg")
plt.savefig("heatmap.png",format="png")
plt.show()

