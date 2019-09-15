#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import collections
import seaborn as sns       # package for better viewing of networks
import operator 
from networkx.algorithms import *     
from networkx.algorithms.community import * 
from networkx import community
from networkx.algorithms import community  #package for girvan_newman

import re
import community     #from package python-louvain


# In[2]:


#import modules for building edge weighted networks
from EdgeWeightedNetworkBuilding import build_network_from_excel, build_network_from_df  


# In[3]:


df_getNetwork = build_network_from_excel(file_path = "/Users/iris/Documents/QMUL-2018/Individual_Project/coding/datasets/mcf7_ntera2_hl60_ksea.xlsm",key="MCF7", sheet_name = "zScorenodes.edges", threshold = 0.2)
cell_line = "MCF7"


# In[4]:


'''Next Cells Calculate Network Communities Using Different Algorithms'''
#Louvain
partition = community.best_partition(df_getNetwork, resolution=1.2)
print "Louvain Modularity: ", community.modularity(partition, df_getNetwork)
print "Louvain Partition: ", partition


# In[5]:


#drawing Louvain
size = float(len(set(partition.values())))
pos = nx.spring_layout(df_getNetwork)
count = 0.
for com in set(partition.values()) :
    count += 1.
    list_nodes = [nodes for nodes in partition.keys()
        if partition[nodes] == com]
    nx.draw_networkx_nodes(df_getNetwork, pos, list_nodes, node_size = 20,
        node_color = str(count / size))
nx.draw_networkx_edges(df_getNetwork, pos, alpha=0.5)
plt.show()


# In[6]:


#girvan_newman
from networkx.algorithms import community
communities_generator = community.girvan_newman(df_getNetwork)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
sorted(map(sorted, next_level_communities))


# In[7]:


# Bipartitions
Kernighan_Lin = kernighan_lin_bisection(df_getNetwork, partition=None, max_iter=100, seed=None)
Kernighan_Lin


# In[8]:


#Modularity-based communities
Modularity_based = greedy_modularity_communities(df_getNetwork)
Modularity_based


# In[9]:


df_getNetwork
k_clique = [k_clique_communities(df_getNetwork,10)]
k_clique = list(k_clique)
print k_clique

#c = list(nx.k_clique_communities(df_getNetwork, 4))
#print list(k_clique[1])


# In[10]:


def k_clique_communities(G, k, cliques=None):
    K5 = nx.convert_node_labels_to_integers(G,first_label=2)
    df_getNetwork.add_edges_from(K5.edges())
    c = list(k_clique_communities(G, 4))
    sorted(list(c[0]))
    list(k_clique_communities(G, 10))
    if k < 2:
        raise nx.NetworkXError("k=%d, k must be greater than 1." % k)
    if cliques is None:
        cliques = nx.find_cliques(G)
    cliques = [frozenset(c) for c in cliques if len(c) >= k]

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    # For each clique, see which adjacent cliques percolate
    perc_graph = nx.Graph()
    perc_graph.add_nodes_from(cliques)
    for clique in cliques:
        for adj_clique in _get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= (k - 1):
                perc_graph.add_edge(clique, adj_clique)

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(perc_graph):
        yield(frozenset.union(*component))


def _get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques



# In[13]:


#test k_clique
g=nx.gnm_random_graph(5,5)
k_clique = [k_clique_communities(g,10)]
k_clique = list(k_clique)

