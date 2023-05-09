# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:25:34 2022

@author: sverremh
"""

# ### Imports
import pandas as pd
import numpy as np
#import random
import igraph as ig
#import math
import matplotlib.pyplot as plt
import sys
from itertools import product
from itertools import compress
# ### Methods

def apply_matching(d_id, s_id):
  # add to match_map:
  match_map.loc[d_id, 'Supply_id'] = s_id
  # remove already used:
  try:
    match_matrix.drop(d_id, inplace=True)
    match_matrix.drop(s_id, axis=1, inplace=True)
  except KeyError:
    pass
def apply_matching2(d_id, s_id):
    # add to match_map: 
    match_map.loc[d_id, 'Supply_id'] = s_id

def matrix_to_graph(demand, supply, match_matrix):
    vertices = [0]*len(demand.index) + [1] * len(supply.index)
    edges = []
    weights = []

    is_na = match_matrix.isna()  
    row_inds = np.arange(match_matrix.shape[0]).tolist()
    col_inds = np.arange(len(demand.index), len(demand.index)+ match_matrix.shape[1]).tolist()
    for i in row_inds:
        combs = list(product([i], col_inds) )
        mask =  ~is_na.iloc[i]
        edges.extend( (list(compress(combs, mask) ) ) )
        weights.extend(list(compress(match_matrix.iloc[i], mask)))
    
    weights = 1 / np.array(weights) # invert to minimize objective
    
    graph = ig.Graph.Bipartite(vertices,  edges)

    graph.es["label"] = weights
    
    graph.vs["label"] = list(demand.index)+list(supply.index) # vertice names
    return graph

def display_graph(graph):
    fig, ax = plt.subplots(figsize=(20, 10))
    ig.plot(
        graph,
        target=ax,
        layout=graph.layout_bipartite(),
        vertex_size=0.4,
        vertex_label=graph.vs["label"],
        palette=ig.RainbowPalette(),
        vertex_color=[v*80+50 for v in graph.vs["type"]],
        edge_width=graph.es["label"],
        edge_label=[round(1/w,2) for w in graph.es["label"]]  # invert weight, to see real LCA
    )
    plt.show()

def best_matching(match_graph):
    # Match by finding one best coandidate 
    result = 0
    matching = ig.Graph.maximum_bipartite_matching(match_graph, weights=match_graph.es["label"])
    for match_edge in matching.edges():        
        apply_matching2(match_edge.source_vertex["label"], match_edge.target_vertex["label"])
    return sum(matching.edges()["label"]) #LCA


# ### Input data

f_GWP_new = 28.9
f_GWP_old = f_GWP_new * 0.0778

# read input arguments
path = sys.argv[0]
demand_json = sys.argv[1]
supply_json = sys.argv[2]
result_file = sys.argv[3]

#demand_json = "C:\\Users\\sverremh\\AppData\\Roaming\\Grasshopper\\Libraries\\FirstPython\\Debug\\net48\\Files\\demand_input.json"
#supply_json = "C:\\Users\\sverremh\\AppData\\Roaming\\Grasshopper\\Libraries\\FirstPython\\Debug\\net48\Files\\supply_input.json"

#read and clean demand df
demand = pd.read_json(demand_json)
demand_header = demand.iloc[0]
demand.columns = demand_header
demand.drop(axis = 1, index= 0, inplace=True)
demand.reset_index(drop = True, inplace = True)
#read and clean supply df
supply = pd.read_json(supply_json)
supply_header = supply.iloc[0]
supply.columns = supply_header
supply.drop(axis = 1, index= 0, inplace=True)
supply['Is_new'] = False
supply.reset_index(drop = True, inplace = True)

#demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height'])
#supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Is_new'])
# TODO add 'Distance'
# TODO add 'Price'
# TODO add 'Material'
# TODO add 'Density'
# TODO add 'Imperfections'
# TODO add 'Is_column'
# TODO add 'Utilisation'
# TODO add 'Group'
# TODO add 'Quality'
# TODO add 'Max_height' ?

# Add a perfect matching pair
#demand.loc['D1'] = {'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
#supply.loc['R1'] = {'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}

# Add non-matchable demand
#demand.loc['D2'] = {'Length': 13.00, 'Area': 0.001, 'Inertia_moment':0.00001, 'Height': 0.05}

# Add non-matchable supply
#supply.loc['R2'] = {'Length': 0.1, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}

# Add one with two good matches, where second slighlty better
#demand.loc['D3'] = {'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
#supply.loc['R3'] = {'Length': 5.20, 'Area': 0.042, 'Inertia_moment':0.00015, 'Height': 0.22, 'Is_new':False}
#supply.loc['R4'] = {'Length': 5.10, 'Area': 0.041, 'Inertia_moment':0.00014, 'Height': 0.21, 'Is_new':False}


# Add new products for each demand element:
demand_duplicate = demand.copy(deep = True) # duplicate demand database
demand_duplicate['Is_new'] = True # set them as new elements

supply = pd.concat((supply, demand_duplicate), ignore_index=True) # add perfectly matching new elements to supply

#for index, row in demand.iterrows():
#  supply.loc['N'+index[1:]] = {'Length': row['Length'], 'Area': row['Area'], 'Inertia_moment': row['Inertia_moment'], 'Height': row['Height'], 'Is_new': True}
  

# --- Matching ---

# Create `match_map` vector of matched ID's without values.
match_map = pd.DataFrame(None, index=demand.index.values.tolist() , columns=['Supply_id'] )

# Create incidence matrix called `match_matrix`

match_matrix = pd.DataFrame(np.nan, index=demand.index.values.tolist() , columns=supply.index.values.tolist() )


# BOTTLENECK # Takes more than a minute when iterating through both of them. Why? 
# assign LCA but only when pass all the criteria: 

"""     
for i, D_row in demand.iterrows():
    for j, S_row in supply.iterrows():
        # find all matches that satisfy criteria
        if demand.loc[i]['Length'] <= supply.loc[j]['Length'] and demand.loc[i]['Area'] <= supply.loc[j]['Area'] and demand.loc[i]['Inertia_moment'] <= supply.loc[j]['Inertia_moment'] and demand.loc[i]['Height'] <= supply.loc[j]['Height']:
          # fill with LCA
          if supply.loc[j]['Is_new']:
            match_matrix.loc[i, j] = round(demand.loc[i]['Length']*supply.loc[j]['Area']*f_GWP_new, 2)
          else:
            match_matrix.loc[i, j] = round(demand.loc[i]['Length']*supply.loc[j]['Area']*f_GWP_old, 2)
"""

match_new = lambda sup_row : row[1] <= sup_row['Length'] and row[2] <= sup_row['Area'] and row[3] <= sup_row['Inertia_moment'] and row[4] <= sup_row['Height'] and sup_row['Is_new'] == True
match_old = lambda sup_row : row[1] <= sup_row['Length'] and row[2] <= sup_row['Area'] and row[3] <= sup_row['Inertia_moment'] and row[4] <= sup_row['Height'] and sup_row['Is_new'] == False
for row in demand.itertuples():
    bool_match_new = supply.apply(match_new, axis = 1).tolist()
    bool_match_old = supply.apply(match_old, axis = 1).tolist()
    
    match_matrix.loc[row[0], bool_match_new] = row[1] * supply.loc[bool_match_new, 'Area'] * f_GWP_new
    match_matrix.loc[row[0], bool_match_old] = row[1] * supply.loc[bool_match_old, 'Area'] * f_GWP_old
match_matrix = match_matrix.round(2)



match_graph = matrix_to_graph(demand, supply, match_matrix)
best_matching(match_graph)
print(match_map)
print(match_matrix.head())
match_map.to_csv(result_file)

#(match_graph)