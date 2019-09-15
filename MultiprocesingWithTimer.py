#!/usr/bin/env python
# coding: utf-8

# In[21]:


import multiprocessing as mp    #multiprocessing
from tqdm import tqdm           #timer
import time                     #timer


# In[22]:


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import re


# In[23]:


# Read in Dataset using absolute path
df_getNetwork = pd.read_excel('/Users/iris/Documents/QMUL-2018/Individual_Project/coding/datasets/test1.xlsx',sheet_name="AMLbiopsies.zscore.nodes.edges")
# Use the Header of Dataset as a Name List for the Series of Networks(could be cell lines or compound names)
netNameList = df_getNetwork.columns.tolist()
del netNameList[0:2]
netNameList[:10]


# In[24]:


#Function to multi-process the program, with a timer to calculate the running time
def partialed_func(key):
    return build_network_from_excel(file_path = "/Users/iris/Documents/QMUL-2018/Individual_Project/coding/datasets/AML comparison Activities with kinase expression.xlsm",key=key, sheet_name = "zScorenodes.edges", threshold = 0.1)

p=mp.Pool()
r = list(tqdm(p.imap(partialed_func, netNameList[0:4]), total=len(netNameList[0:4])))   #4 is the set intervals within whole process

