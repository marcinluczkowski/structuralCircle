# -*- coding: utf-8 -*-

import sys
from itertools import product, compress
import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
# import random
# import math


class Matching():
    """Class describing the matching problem, with its constituent parts"""
    def __init__(self, demand, supply, add_new=False, multi=False):
        self.demand = demand
        if add_new:
            # add perfectly matching new elements to supply
            demand_copy = demand.copy(deep = True)
            demand_copy['Is_new'] = True # set them as new elements
            self.supply = pd.concat((supply, demand_copy), ignore_index=True)
        else:
            self.supply = supply
        self.multi = multi
        self.graph = None
        self.result = None
        self.pairs = pd.DataFrame(None, index=self.demand.index.values.tolist(), columns=['Supply_id'])
        self.incidence = pd.DataFrame(np.nan, index=self.demand.index.values.tolist(), columns=self.supply.index.values.tolist())

    def evaluate(self):
        """Populates incidence matrix with weights based on the criteria"""
        can_match = lambda sup_row : row[1] <= sup_row['Length'] and row[2] <= sup_row['Area'] and row[3] <= sup_row['Inertia_moment'] and row[4] <= sup_row['Height']
        for row in self.demand.itertuples():
            bool_match = self.supply.apply(can_match, axis = 1).tolist()
            self.incidence.loc[row[0], bool_match] =  calculate_lca(row[1], self.supply.loc[bool_match, 'Area'], is_new=True)

    def add_pair(self, demand_id, supply_id):
        """Execute matrix matching"""
        # add to match_map:
        self.pairs.loc[demand_id, 'Supply_id'] = supply_id
        # remove already used:
        try:
            self.incidence.drop(demand_id, inplace=True)
            self.incidence.drop(supply_id, axis=1, inplace=True)
        except KeyError:
            pass

    def add_graph(self):
        """Add a graph notation based on incidence matrix"""
        vertices = [0]*len(self.demand.index) + [1]*len(self.supply.index)
        edges = []
        weights = []

        is_na = self.incidence.isna()
        row_inds = np.arange(self.incidence.shape[0]).tolist()
        col_inds = np.arange(len(self.demand.index), len(self.demand.index)+ self.incidence.shape[1]).tolist()
        for i in row_inds:
            combs = list(product([i], col_inds) )
            mask =  ~is_na.iloc[i]
            edges.extend( (list(compress(combs, mask) ) ) )
            weights.extend(list(compress(self.incidence.iloc[i], mask)))
        weights = 1 / np.array(weights)
        graph = ig.Graph.Bipartite(vertices,  edges)
        graph.es["label"] = weights
        graph.vs["label"] = list(self.demand.index)+list(self.supply.index) #vertice names
        self.graph = graph

    def display_graph(self):
        """Plot the graph and matching result"""
        if self.graph:
            # TODO add display of matching
            fig, ax = plt.subplots(figsize=(20, 10))
            ig.plot(
                self.graph,
                target=ax,
                layout=self.graph.layout_bipartite(),
                vertex_size=0.4,
                vertex_label=self.graph.vs["label"],
                palette=ig.RainbowPalette(),
                vertex_color=[v*80+50 for v in self.graph.vs["type"]],
                edge_width=self.graph.es["label"],
                edge_label=[round(1/w,2) for w in self.graph.es["label"]]  # invert weight, to see real LCA
            )
            plt.show()

    def match_bipartite_graph(self):
        """Match using Maximum Bipartite Graphs to find best indyvidual mapping candidates"""
        # TODO multiple assignment won't work.
        if not self.graph:
            self.add_graph()
        bipartite_matching = ig.Graph.maximum_bipartite_matching(self.graph, weights=self.graph.es["label"])
        for match_edge in bipartite_matching.edges():
            self.add_pair(match_edge.source_vertex["label"], match_edge.target_vertex["label"])
        self.result = sum(matching.edges()["label"])
        return self.result

    def match_bin_packing(self):
        #TODO
        pass

# class Elements(pd.DataFrame):
#     def read_json(self):
#         super().read_json()
#         self.columns = self.iloc[0]
#         self.drop(axis = 1, index= 0, inplace=True)
#         self.reset_index(drop = True, inplace = True)


def calculate_lca(length, area, gwp=28.9, is_new=True):
    """ Calculate Life Cycle Assessment """
    # TODO add distance, processing and other impact categories than GWP
    if not is_new:
        gwp = gwp * 0.0778
    lca = length * area * gwp
    return lca


# if __name__ == "__main__":

# read input arguments
PATH = sys.argv[0]
DEMAND_JSON = r"C:\Code\structuralCircle\MatchingAlgorithms\sample_demand_input.json" #sys.argv[1]
SUPPLY_JSON = r"C:\Code\structuralCircle\MatchingAlgorithms\sample_supply_input.json" #sys.argv[2]
RESULT_FILE = r"C:\Code\structuralCircle\MatchingAlgorithms\result.csv" #sys.argv[3]
#DEMAND_JSON = "C:\\Users\\sverremh\\AppData\\Roaming\\Grasshopper\\Libraries\\FirstPython\\Debug\\net48\\Files\\sample_demand_input.json"
#SUPPLY_JSON = "C:\\Users\\sverremh\\AppData\\Roaming\\Grasshopper\\Libraries\\FirstPython\\Debug\\net48\Files\\sample_supply_input.json"

#read and clean demand df
demand = pd.read_json(DEMAND_JSON)
demand_header = demand.iloc[0]
demand.columns = demand_header
demand.drop(axis = 1, index= 0, inplace=True)
demand.reset_index(drop = True, inplace = True)
#read and clean supply df
supply = pd.read_json(SUPPLY_JSON)
supply_header = supply.iloc[0]
supply.columns = supply_header
supply.drop(axis = 1, index= 0, inplace=True)
supply['Is_new'] = False
supply.reset_index(drop = True, inplace = True)

matching = Matching(demand, supply, add_new=True, multi=False)
matching.evaluate()
matching.match_bipartite_graph()

matching.pairs.to_csv(RESULT_FILE)
# print(matching.result)
# matching.display_graph()