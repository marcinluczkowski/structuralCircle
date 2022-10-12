# -*- coding: utf-8 -*-

import sys
from itertools import product, compress
import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
import numexpr as ne
# import random
# import math
import logging
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s_%(asctime)s_%(message)s',
    datefmt='%H:%M:%S',
    # filename='log.log',
    # filemode='w'
    )

class Matching():
    """Class describing the matching problem, with its constituent parts"""
    def __init__(self, demand, supply, add_new=False, multi=False, constraints = {}):
        self.demand = demand
        if add_new:
            # add perfectly matching new elements to supply
            demand_copy = demand.copy(deep = True)
            demand_copy['Is_new'] = True # set them as new elements
            try:
                demand_copy.rename(index=dict(zip(demand.index.values.tolist(), [sub.replace('D', 'N') for sub in demand.index.values.tolist()] )), inplace=True)
            except AttributeError:
                pass
            self.supply = pd.concat((supply, demand_copy), ignore_index=False)
        else:
            self.supply = supply
        self.multi = multi
        self.graph = None
        self.result = None  #saves latest result of the matching
        self.pairs = pd.DataFrame(None, index=self.demand.index.values.tolist(), columns=['Supply_id']) #saves latest array of pairs
        self.incidence = pd.DataFrame(np.nan, index=self.demand.index.values.tolist(), columns=self.supply.index.values.tolist())
        self.constraints = constraints

        logging.info("Matching object created with %s demand, and %s supply elements", len(demand), len(supply))
    
    def evaluate(self):
        """Populates incidence matrix with true values where the element fit constraint criteria"""    
        
        #TODO Where to add weights to this incidence matrix
        start = time.time()
        bool_array = np.full((self.demand.shape[0], self.supply.shape[0]), True) # initiate empty array
        for param, compare in self.constraints.items():
            cond_list = []
            for var in self.supply[param]:
                demand_array = self.demand[param]
                bool_col = ne.evaluate(f"{var} {compare} demand_array") # numpy array of boolean
                cond_list.append(bool_col)
            cond_array = np.column_stack(cond_list) #create new 2D-array of conditionals
            bool_array = ne.evaluate("cond_array & bool_array") # is this working
            #bool_array = np.logical_and(bool_array, cond_array)
        bool_df = pd.DataFrame(bool_array, columns= self.incidence.columns, index= self.incidence.index)

        end = time.time()
        #self.incidence = bool_array
        logging.info("Weight evaluation NEW METHODexecution time: %s sec", round(end - start,3))
        return bool_df



    def evaluate2(self):
        """Populates incidence matrix with weights based on the criteria"""
        # TODO optimize the evaluation.
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
        
        
        start = time.time()
        match_new = lambda sup_row : row[1] <= sup_row['Length'] and row[2] <= sup_row['Area'] and row[3] <= sup_row['Inertia_moment'] and row[4] <= sup_row['Height'] and sup_row['Is_new'] == True
        match_old = lambda sup_row : row[1] <= sup_row['Length'] and row[2] <= sup_row['Area'] and row[3] <= sup_row['Inertia_moment'] and row[4] <= sup_row['Height'] and sup_row['Is_new'] == False
        for row in self.demand.itertuples():
            bool_match_new = self.supply.apply(match_new, axis = 1).tolist()
            bool_match_old = self.supply.apply(match_old, axis = 1).tolist()
            
            self.incidence.loc[row[0], bool_match_new] = calculate_lca(row[1], self.supply.loc[bool_match_new, 'Area'], is_new=True)
            self.incidence.loc[row[0], bool_match_old] = calculate_lca(row[1], self.supply.loc[bool_match_old, 'Area'], is_new=False)
        end = time.time()
        logging.info("Weight evaluation execution time: %s sec", round(end - start,3))

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
        if not self.graph:
            self.add_graph()
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

    def _matching_decorator(func):
        """Set of repetitive tasks for all matching methods"""
        def wrapper(self, *args, **kwargs):
            # Before:
            start = time.time()
            # empty result of previous matching:
            self.result = 0  
            self.pairs = pd.DataFrame(None, index=self.demand.index.values.tolist(), columns=['Supply_id'])
            # The actual method:
            func(self, *args, **kwargs)
            # After:
            end = time.time()
            logging.info("Matched: %s to %s (%s %%) of %s elements using %s, resulting in LCA (GWP): %s kgCO2eq, in: %s sec.",
                len(self.pairs['Supply_id'].unique()),
                self.pairs['Supply_id'].count(),
                100*self.pairs['Supply_id'].count()/len(self.demand),
                self.supply.shape[0],
                func.__name__,
                round(self.result, 2),
                round(end - start,3)
            )                
            return [self.result, self.pairs]
        return wrapper

    @_matching_decorator
    def match_nested_loop(self, plural_assign=False):
        """Simplest brute force matching that iterates all elemnts."""
        demand_sorted = self.demand.sort_values(by=['Length', 'Area'], axis=0, ascending=False)
        supply_sorted = self.supply.sort_values(by=['Is_new', 'Length', 'Area'], axis=0, ascending=True)
        for demand_index, demand_row in demand_sorted.iterrows():
            match=False
            logging.debug("-- Attempt to find a match for %s", demand_index)                
            for supply_index, supply_row in supply_sorted.iterrows():
                # TODO replace constraints with evalute string
                if demand_row.Length <= supply_row.Length and demand_row.Area <= supply_row.Area and demand_row.Inertia_moment <= supply_row.Inertia_moment and demand_row.Height <= supply_row.Height:
                    match=True
                    self.add_pair(demand_index, supply_index)
                if match:
                    if plural_assign:
                        # shorten the supply element:
                        supply_row.Length = supply_row.Length - demand_row.Length
                        # sort the supply list
                        supply_sorted = supply_sorted.sort_values(by=['Is_new', 'Length', 'Area'], axis=0, ascending=True)  # TODO move this element instead of sorting whole list
                        self.result += calculate_lca(demand_row.Length, supply_row.Area, is_new=supply_row.Is_new)
                        logging.debug("---- %s is a match, that results in %s m cut.", supply_index, supply_row.Length)
                    else:
                        self.result += calculate_lca(supply_row.Length, supply_row.Area, is_new=supply_row.Is_new)
                        logging.debug("---- %s is a match and will be utilized fully.", supply_index)
                        supply_sorted.drop(supply_index)
                    break
                else:
                    logging.debug("---- %s is not matching.", supply_index)

    @_matching_decorator
    def match_bipartite_graph(self):
        """Match using Maximum Bipartite Graphs"""
        # TODO multiple assignment won't work OOTB.
        if not self.graph:
            self.add_graph()
        bipartite_matching = ig.Graph.maximum_bipartite_matching(self.graph, weights=self.graph.es["label"])
        for match_edge in bipartite_matching.edges():
            self.add_pair(match_edge.source_vertex["label"], match_edge.target_vertex["label"])
        self.result = sum(bipartite_matching.edges()["label"])

    @_matching_decorator
    def match_bin_packing(self):
        """Match using Bin Packing"""
        #TODO
        pass

    @_matching_decorator
    def match_knapsacks(self):
        """Match using Knapsacks"""
        #TODO

        def constraint_inds():
            """Construct the constraint array"""
            rows = self.demand.shape[0]
            cols = self.supply.shape[0]
            bool_array = np.full((rows, cols), False)

            # iterate through constraints
            for key, val in self.constraints.items():
                cond_list = []
                for var in self.supply[key]:
                    array = self.demand[key]
                    col = ne.evaluate(f'array {val} var')
                    cond_list.append(col) # add results to cond_list
                conds = np.column_stack(cond_list) # create 2d array of tests
                bool_array = np.logical_or(bool_array, conds)

            constraint_inds = np.transpose(np.where(bool_array)) # convert to nested list if [i,j] indices
            return constraint_inds

        data = {}
        data ['lengths'] = self.demand.Length.astype(float)
        data['values'] = self.demand.Area.astype(float)
        
        assert len(data['lengths']) == len(data['values']) # The same check is done indirectly in the dataframe
        data['num_items'] = len(data['values'])
        data['all_items'] = range(data['num_items'])
        data['areas'] = self.demand.Area

        data['bin_capacities'] = self.supply.Length # these would be the bins
        data['bin_areas'] = self.supply.Area.to_numpy(dtype = int)
        data['num_bins'] = len(data['bin_capacities'])
        data['all_bins'] = range(data['num_bins'])

        #get constraint ids
        c_inds = constraint_inds()

        # create solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if solver is None:
            print('SCIP Solver is unavailable')
            return

        # --- Variables ---
        # x[i,j] = 1 if item i is backed in bin j. 0 else
        x = {}
        for i in data['all_items']:
            for j in data['all_bins']:
                x[i,j] = solver.BoolVar(f'x_{i}_{j}') 
  
        print(f'Number of variables = {solver.NumVariables()}') 

        # --- Constraints ---
        # each item can only be assigned to one bin
        for i in data['all_items']:
            solver.Add(sum(x[i,j] for j in data['all_bins']) <= 1)

        # the amount packed in each bin cannot exceed its capacity.
        for j in data['all_bins']:
            solver.Add(
                sum(x[i,j] * data['lengths'][i] for i in data['all_items'])
                    <= data['bin_capacities'][j])

        # fix the variables where the area of the element is too small to fit
        # in this case it is the supply element 0 which has an area of 10
        for inds in c_inds:
            i = int(inds[0])
            j = int(inds[1])
            solver.Add(x[i,j] == 0)

        print(f'Number of contraints = {solver.NumConstraints()}')
        # --- Objective ---
        # maximise total value of packed items
        objective = solver.Objective()
        for i in data['all_items']:
            for j in data['all_bins']:
              objective.SetCoefficient(x[i,j], float(data['areas'][i]))      
        objective.SetMaximization()
        # Starting solver
        print('Starting solver')
        status = solver.Solve()
        print('Computation done')
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:

            results = {}
            print('Solution found! \n ------RESULTS-------\n')
            total_length = 0
            for j in data['all_bins']:
                results[j] = []
                print(f'Bin {j}')
                bin_length = 0
                bin_value = 0
                for i in data['all_items']:
                    if x[i, j].solution_value() > 0:
                        results[j].append(i)
                        print(
                            f"Item {i} Length: {data['lengths'][i]} area: {data['areas'][i]}"
                        )
                        bin_length += data['lengths'][i]
                        bin_value += data['areas'][i]
                print(f'Packed bin lengths: {bin_length}')
                print(f'Packed bin value: {bin_value}')
                total_length += bin_length
                print(f'Total packed Lenghtst: {total_length}\n')

        # return the results as a DataFrame like the bin packing problem
        # Or a dictionary. One key per bin/supply, and a list of ID's for the
        # elements which should go within that bin. 

        return [self.result, self.pairs]

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


if __name__ == "__main__":
    PATH = sys.argv[0]
    #DEMAND_JSON = sys.argv[1]
    #SUPPLY_JSON = sys.argv[2]
    #RESULT_FILE = sys.argv[3]

    DEMAND_JSON = r"MatchingAlgorithms\sample_demand_input.json"
    SUPPLY_JSON = r"MatchingAlgorithms\sample_supply_input.json"
    RESULT_FILE = r"MatchingAlgorithms\result.csv"

