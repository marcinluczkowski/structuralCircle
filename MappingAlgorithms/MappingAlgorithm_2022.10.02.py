# ### Imports
import pandas as pd
import numpy as np
import random
import igraph as ig
import math
import matplotlib.pyplot as plt


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

def matrix_to_graph(demand, supply, match_matrix):
    vertices = [0] * len(demand.index) + [1] * len(supply.index) #['D1', 'D2', 'R1', 'R2']
    edges=[]
    weights=[]  # weights is the maximizing objective, in our case 1/LCA
    i = 0
    for index, row in match_matrix.iterrows():
      j = len(demand.index)
      for v in row:
        if v>0:
          edges.append((i,j))
          weights.append(round(1/v,2))
        j+=1
      i+=1
    graph = ig.Graph.Bipartite(
        vertices,
        edges
    )
    graph.es["label"] = weights
    graph.vs["label"] = list(demand.index)+list(supply.index) # vertice names
    return graph

def display_graph(graph):
    # TODO add display of matching
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
        apply_matching(match_edge.source_vertex["label"], match_edge.target_vertex["label"])
    return sum(matching.edges()["label"]) #LCA


# ### Input data

f_GWP_new = 28.9
f_GWP_old = f_GWP_new * 0.0778

demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height'])
supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Is_new'])
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
demand.loc['D1'] = {'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R1'] = {'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}

# Add non-matchable demand
demand.loc['D2'] = {'Length': 13.00, 'Area': 0.001, 'Inertia_moment':0.00001, 'Height': 0.05}

# Add non-matchable supply
supply.loc['R2'] = {'Length': 0.1, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}

# Add one with two good matches, where second slighlty better
demand.loc['D3'] = {'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R3'] = {'Length': 5.20, 'Area': 0.042, 'Inertia_moment':0.00015, 'Height': 0.22, 'Is_new':False}
supply.loc['R4'] = {'Length': 5.10, 'Area': 0.041, 'Inertia_moment':0.00014, 'Height': 0.21, 'Is_new':False}


# Add new products for each demand element:
for index, row in demand.iterrows():
  supply.loc['N'+index[1:]] = {'Length': row['Length'], 'Area': row['Area'], 'Inertia_moment': row['Inertia_moment'], 'Height': row['Height'], 'Is_new': True}
  

# ### Matching

# Create `match_map` vector of matched ID's without values.
match_map = pd.DataFrame(None, index=demand.index.values.tolist() , columns=['Supply_id'] )

# Create incidence matrix called `match_matrix`
match_matrix = pd.DataFrame(np.nan, index=demand.index.values.tolist() , columns=supply.index.values.tolist() )

# assign LCA but only when pass all the criteria: 
for i, D_row in demand.iterrows():
    for j, S_row in supply.iterrows():
        # find all matches that satisfy criteria
        if demand.loc[i]['Length'] <= supply.loc[j]['Length'] and demand.loc[i]['Area'] <= supply.loc[j]['Area'] and demand.loc[i]['Inertia_moment'] <= supply.loc[j]['Inertia_moment'] and demand.loc[i]['Height'] <= supply.loc[j]['Height']:
          # fill with LCA
          if supply.loc[j]['Is_new']:
            match_matrix.loc[i, j] = round(demand.loc[i]['Length']*supply.loc[j]['Area']*f_GWP_new, 2)
          else:
            match_matrix.loc[i, j] = round(demand.loc[i]['Length']*supply.loc[j]['Area']*f_GWP_old, 2)

match_graph = matrix_to_graph(demand, supply, match_matrix)
best_matching(match_graph)

match_map
# display_graph(match_graph)