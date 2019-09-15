#!/usr/bin/env python
# coding: utf-8

# In[5]:


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import collections
import seaborn as sns       # package for better viewing of networks
import operator 


# In[6]:


#Function to do basic analysis with/without Threshold for weights
def network_with_basic_analysis(df,key,threshold=None,k1="k1",k2="k2",save_path="/Users/iris/Documents/QMUL-2018/Individual_Project/coding/"):
    if threshold is not None:
        df=df[df[key]>threshold]
    output_net=nx.from_pandas_edgelist(df,k1,k2,[key])
    #nx.draw_networkx(output_net, with_labels=True)
    net_density = nx.density(output_net)
    net_ave_degree = output_net.number_of_edges()/float(len(output_net))
    #net_ave_degree = nx.average_degree_connectivity(output_net)
    net_clust_coefficients = nx.clustering(output_net)
    # Average clustering coefficient
    net_avg_clust = sum(net_clust_coefficients.values()) / len(net_clust_coefficients)
    
    return {"keys":key,
            "density":net_density,
            "average_degree":net_ave_degree,
            "clustering_coefficients":net_clust_coefficients,
            "average_coefficients":net_avg_clust }


# In[7]:


# Read in Dataset using absolute path
df_getNetwork = pd.read_excel('/Users/iris/Documents/QMUL-2018/Individual_Project/coding/datasets/test1.xlsx',sheet_name="zscore.nodes.edges")
# Use the Header of Dataset as a Name List for the Series of Networks(could be cell lines or compound names)
netNameList = df_getNetwork.columns.tolist()
del netNameList[0:2]

outputlist = []
for i in netNameList:
    outputlist.append(network_with_basic_analysis(df_getNetwork,i,0))
    
df_basic_analysis = pd.DataFrame(outputlist)


# In[9]:


df_basic_analysis = df_basic_analysis[['keys', 'density', 'average_degree', 'clustering_coefficients', 'average_coefficients']]
df_basic_analysis
df_basic_analysis = df_basic_analysis.to_csv (r'/Users/iris/Documents/QMUL-2018/Individual_Project/coding/analysis_results/cell_line_basic_analysis.csv', index = None, header=True) 


# In[ ]:




