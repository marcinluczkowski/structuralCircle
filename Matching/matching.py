# -*- coding: utf-8 -*-

import logging
import random
import sys
import time
from itertools import compress, product
from copy import copy, deepcopy
import igraph as ig
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import pandas as pd
import pygad
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from scipy.optimize import milp, LinearConstraint, NonlinearConstraint, Bounds
import helper_methods as hm
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s_%(asctime)s_%(message)s',
    datefmt='%H:%M:%S',
    # filename='log.log',
    # filemode='w'
    )

TIMBER_GWP = 28.9           # based on NEPD-3442-2053-EN, previously: 10.0
REUSE_GWP_RATIO = 2.25      # 0.0778*28.9 = 2.25 based on Eberhardt, previously: 0.1

class Matching():
    """Class describing the matching problem, with its constituent parts."""
    def __init__(self, demand, supply, include_transportation, add_new=False, multi=False, constraints = {}, solution_limit = 60):
        self.demand = demand.infer_objects()
        # TODO calculate new elements first, and not add them to supply bank.
        if add_new:
            # add perfectly matching new elements to supply
            demand_copy = demand.copy(deep = True)
            demand_copy['Is_new'] = True # set them as new elements
            try:
                # This works only when the indices are already named "D"
                demand_copy.rename(index=dict(zip(demand.index.values.tolist(), [sub.replace('D', 'N') for sub in demand.index.values.tolist()] )), inplace=True)
            except AttributeError:
                pass
            self.supply = pd.concat((supply, demand_copy), ignore_index=False).infer_objects()
            
        else:
            self.supply = supply.infer_objects()
        self.include_transportation = include_transportation
        self.multi = multi
        self.graph = None
        self.result = None  #saves latest result of the matching
        self.pairs = pd.DataFrame(None, index=self.demand.index.values.tolist(), columns=['Supply_id']) #saves latest array of pairs
        self.incidence = pd.DataFrame(np.nan, index=self.demand.index.values.tolist(), columns=self.supply.index.values.tolist())
        self.weights = None
        self.constraints = constraints
        self.solution_time = None
        self.solution_limit = solution_limit           
        # create incidence and weight for the method
        self.evaluate()
        self.weight_incidence()

        #calculate LCA of original elements
        self.demand.eval(f"LCA = Length * Area * {TIMBER_GWP}", inplace = True) #TODO Discuss with Artur where and how to make this more general
        self.supply.eval(f"LCA = Length * Area * {REUSE_GWP_RATIO}", inplace = True)
        logging.info("Matching object created with %s demand, and %s supply elements", len(demand), len(supply))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result    

    def evaluate(self):
        """Populates incidence matrix with true values where the element fit constraint criteria"""    
        # TODO optimize the evaluation.
        # TODO add 'Distance' 'Price' 'Material' 'Density' 'Imperfections' 'Is_column' 'Utilisation' 'Group' 'Quality' 'Max_height' ?
        #TODO Create standalone method for evaluating one column Rj of the incidence matrix. Need this for cutoffs in greedy algorithm as well. 
        start = time.time()
        bool_array = np.full((self.demand.shape[0], self.supply.shape[0]), True) # initiate empty array
        for param, compare in self.constraints.items():
            cond_list = []
            for var in self.supply[param]:
                demand_array = self.demand[param].to_list()
                bool_col = ne.evaluate(f'{var} {compare} demand_array') # numpy array of boolean
                cond_list.append(bool_col)
            cond_array = np.column_stack(cond_list) #create new 2D-array of conditionals
            bool_array = ne.evaluate("cond_array & bool_array") # 
            #bool_array = np.logical_and(bool_array, cond_array)
        
        self.incidence = pd.DataFrame(bool_array, columns= self.incidence.columns, index= self.incidence.index)
        end = time.time()
        logging.info("Create incidence matrix from constraints: %s sec", round(end - start,3))

    def evaluate_column(self, supply_val, parameter, compare, current_bool):
        """Evaluates a column in the incidence matrix according to the constraints
            Returns a np array that can substitute the input column."""
        demand_array = self.demand[parameter].to_numpy(dtype = float) # array of demand parameters to evaluate. 
        compare_array = ne.evaluate(f"{supply_val} {compare} demand_array")        
        return ne.evaluate("current_bool & compare_array")
        

    
    def weight_incidence(self):
        """Assign weights to elements in the incidence matrix. At the moment only LCA is taken into\
        account. This method should replace the last step in the original evaluate method."""
        start = time.time()
        
        # TODO implement constraints
        el_locs0 = np.where(self.incidence) # tuple of rows and columns as list
        el_locs = np.transpose(el_locs0) # array of row-column pairs where incidence matrix is true. 
        areas = self.supply.Area.iloc[el_locs[:, 1]].to_numpy(dtype=float).astype(float) # array of areas 
        lengths = self.demand.Length.iloc[el_locs[:, 0]].to_numpy(dtype=float).astype(float) # array of element lenghts
        el_new = self.supply.Is_new.iloc[el_locs[:,1]].to_numpy(dtype=float).astype(bool) # array of booleans for element condition.  
        gwp = TIMBER_GWP
        gwp_array = np.where(el_new, gwp, REUSE_GWP_RATIO)
        #get_gwp = ne.evaluate("gwp if el_new else gwp*REUSE_GWP_RATIO")
        #gwp = [7.28 if tr else ]
        # TODO function as an input 
        lca_array = ne.evaluate("areas * lengths * gwp_array")
       
        
        lca_mat = np.empty(shape = (self.incidence.shape[0], self.incidence.shape[1]))
        lca_mat[:] = np.nan
        lca_mat[el_locs0[0], el_locs0[1]] = lca_array
        self.weights = pd.DataFrame(lca_mat, index = self.incidence.index, columns = self.incidence.columns)
        """
        self.incidence = self.incidence.apply(lambda el: np.where(el, \
            calculate_lca(self.demand[el.index, "Length"], \
                self.supply[el.name, "Area"], \
                is_new = self.supply.loc[el.name, "Is_new"]), \
                np.nan))
        """
        end = time.time()  
        logging.info("Weight evaluation of incidence matrix: %s sec", round(end - start, 3))

    def add_pair(self, demand_id, supply_id):
        """Execute matrix matching"""
        # add to match_map:
        self.pairs.loc[demand_id, 'Supply_id'] = supply_id
        
    def add_graph(self):
        """Add a graph notation based on incidence matrix"""
        
        vertices = [0]*len(self.demand.index) + [1]*len(self.supply.index)
        num_rows = len(self.demand.index)
        edges = []
        weights = []    
        #is_n = self.incidence    
        #is_n = ~self.weights.isna() # get and invert the booleans
        row_inds = np.arange(self.incidence.shape[0]).tolist()
        col_inds = np.arange(len(self.demand.index), len(self.demand.index)+ self.incidence.shape[1]).tolist()
        for i in row_inds:
            combs = list(product([i], col_inds) )
            mask =  self.incidence.iloc[i] 
            edges.extend( (list(compress(combs, mask) ) ) )
            weights.extend(list(compress(self.weights.iloc[i], mask)))
        # initial LCA minus the replacement LCA:
        weights = np.array([self.demand.LCA[edge[0]] for edge in edges ]) - np.array(weights)
        #weights = max(weights) - np.array(weights)
        graph = ig.Graph.Bipartite(vertices, edges)
        graph.es["label"] = weights
        graph.vs["label"] = list(self.demand.index)+list(self.supply.index) #vertice names
        self.graph = graph

    def display_graph(self, graph_type='rows', show_weights=True, show_result=True):
        """Plot the graph and matching result"""
        if not self.graph:
            self.add_graph()
        weight = None
        if show_weights:
            # weight = list(np.absolute(np.array(self.graph.es["label"]) - 8).round(decimals=2)) 
            weight = list(np.array(self.graph.es["label"]).round(decimals=2)) 
        edge_color = None
        edge_width = self.graph.es["label"]
        if show_result and not self.pairs.empty:
            edge_color = ["gray"] * len(self.graph.es)
            edge_width = [0.7] * len(self.graph.es)
            # TODO could be reformatted like this https://igraph.readthedocs.io/en/stable/tutorials/bipartite_matching.html#tutorials-bipartite-matching
            not_found = 0
            for index, pair in self.pairs.iterrows():
                source = self.graph.vs.find(label=index) 
                try:
                    target = self.graph.vs.find(label=pair['Supply_id'])
                    edge = self.graph.es.select(_between = ([source.index], [target.index]))
                    edge_color[edge.indices[0]] = "black" #"red"
                    edge_width[edge.indices[0]] = 2.5
                except ValueError:
                    not_found+=1
                
            if not_found > 0:
                logging.error("%s elements not found - probably no new elements supplied.", not_found)
        vertex_color = []
        for v in self.graph.vs:
            if 'D' in v['label']:
                vertex_color.append("lightgray")
            elif 'S' in v['label']:
                vertex_color.append("slategray")
            else:
                vertex_color.append("pink")
        layout = self.graph.layout_bipartite()
        if graph_type == 'rows':
            layout = self.graph.layout_bipartite()
        elif graph_type == 'circle':
            layout = self.graph.layout_circle()

        if self.graph:
            fig, ax = plt.subplots(figsize=(15, 10))
            ig.plot(
                self.graph,
                target=ax,
                layout=layout,
                vertex_size=0.4,
                vertex_label=self.graph.vs["label"],
                palette=ig.RainbowPalette(),
                vertex_color=vertex_color,
                edge_width=edge_width,
                edge_label=weight,
                edge_color=edge_color,
                edge_curved=0.15
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
            #Calculate the result of the matching
            self.calculate_result()
            # After:
            end = time.time()
            self.solution_time = round(end - start, 3)
            all_string_series = self.pairs.fillna('nan') # have all entries as string before search
            
            num_old = len(all_string_series.loc[all_string_series.Supply_id.str.contains('S')].Supply_id.unique())
            num_new = len(all_string_series.loc[all_string_series.Supply_id.str.contains('N')].Supply_id.unique())
            num_matched = len(self.pairs.dropna())
            logging.info("Matched %s old and %s new elements to %s demand elements (%s %%) using %s. Resulting in LCA (GWP) %s kgCO2eq, in %s seconds.", 
                num_old, num_new, num_matched, round(100 * num_matched / len(self.pairs), 2), func.__name__, round(self.result, 2), round(end - start, 3))
            return [self.result, self.pairs]
        return wrapper

    def calculate_result(self):
        """Evaluates the result based on the final matching of elements"""
        # if rows without pairing, remove those    
        local_pairs = self.pairs.dropna()

        #get the index of columns in weight df which are paired
        #TODO Make the supply and demand id_s numerical 
        col_inds = local_pairs.Supply_id.apply(lambda label: self.weights.columns.get_loc(label))
        row_inds = list( map(lambda name: self.weights.index.get_loc(name), local_pairs.index) )
        #row_inds = np.arange(0, local_pairs.shape[0], 1) # the row inds are the same here and in the weights
        self.result = (self.weights.to_numpy()[row_inds, col_inds]).sum()

        # add the LCA of original elements that are not substituted
        mask = self.pairs.Supply_id.isna().to_numpy()
        original_LCA = self.demand.LCA[mask].sum()
        self.result += original_LCA
        #TODO: add LCA from transportation of the used timber
        if self.include_transportation: #Include GWP from transportation of used elements
            demand_indices = local_pairs.index.values.tolist()
            demand_copy = self.demand.loc[demand_indices]
            demand_copy = demand_copy[["Latitude", "Longitude"]]

            supply_copy = self.supply.loc[local_pairs["Supply_id"].tolist()]
            supply_copy.rename(columns = {"Latitude": "s_latitude", "Longitude": "s_longitude"}, inplace = True)

            supply_copy["d_latitude"] = demand_copy["Latitude"].tolist()
            supply_copy["d_longitude"] = demand_copy["Longitude"].tolist()
            
            supply_copy["distance"] = supply_copy.apply(lambda x: self.calculate_driving_distance(x.d_latitude, x.d_longitude, x.s_latitude, x.s_longitude), axis = 1) #Dataframe with calculated distances
            supply_copy["volume"] = supply_copy["Area"] * supply_copy["Length"]

            supply_copy["transportation_LCA"] = supply_copy.apply(lambda x: self.calculate_transportation_LCA(x.volume, 400, x.distance, factor = 96), axis = 1)

            sum_transportation_LCA = supply_copy["transportation_LCA"].sum()
            print(f"Transportation LCA:", sum_transportation_LCA)
            self.result += sum_transportation_LCA

        

    def calculate_driving_distance(self, demand_lat, demand_lon, supply_lat, supply_lon):
        """Calculates the driving distance between two coordinates and returns the result in meters
        - Coordinates as a String
        """
        try:
            url = f"http://router.project-osrm.org/route/v1/car/{demand_lon},{demand_lat};{supply_lon},{supply_lat}?overview=false"
            req = requests.get(url)
            driving_distance_meter = req.json()["routes"][0]["distance"]
            distance = driving_distance_meter / 1000 #driving distance in km
        except:
            logging.error("Was not able to get the driving distance from OSRM-API")
            distance = 0
        return  distance

    def calculate_transportation_LCA(self, volume, density, distance, factor = 96.0):
        """Calculates the CO2 equivalents of driving one element a specific distance
        - volume in float
        - density in float
        - distance in float
        - factor in float
        
        """
        density = density / 1000 #convert kg/m^3 to tonne/m^3
        factor = factor / 1000 #convert gram to kg
        return volume * density * distance * factor #C02 equivalents in kg

        #CALCULATE GWP FROM TRANSPORTATION

    @_matching_decorator
    def match_greedy_algorithm(self, plural_assign=False):
        """Algorithm that takes one best element at each iteration, based on sorted lists, not considering any alternatives."""
        # TODO consider opposite sorting (as we did in Gh), small chance but better result my occur
        demand_sorted = self.demand.copy(deep =True)
        supply_sorted = self.supply.copy(deep =True)
        #Change indices to integers for both demand and supply
        demand_sorted.index = np.array(range(len(demand_sorted.index)))
        supply_sorted.index = np.array(range(len(supply_sorted.index)))

        #sort the supply and demand
        #demand_sorted.sort_values(by=['Length', 'Area'], axis=0, ascending=False, inplace = True)
        demand_sorted.sort_values(by=['LCA'], axis=0, ascending=False, inplace = True)
        #supply_sorted.sort_values(by=['Is_new', 'Length', 'Area'], axis=0, ascending=True, inplace = True)
        supply_sorted.sort_values(by=['Is_new', 'LCA'], axis=0, ascending=True, inplace = True) # FIXME Need to make this work "optimally"
        incidence_np = self.incidence.copy(deep=True).values      

        columns = self.supply.index.to_list()
        rows = self.demand.index.to_list()
        min_length = self.demand.Length.min() # the minimum lenght of a demand element
        
        for demand_tuple in demand_sorted.itertuples():            
            match=False
            logging.debug("-- Attempt to find a match for %s", demand_tuple.Index)                
            for supply_tuple in supply_sorted.itertuples():                 
                if incidence_np[demand_tuple.Index,supply_tuple.Index]:           
                    match=True
                    self.add_pair(rows[demand_tuple.Index], columns[supply_tuple.Index])
                if match:
                    new_length = supply_tuple.Length - demand_tuple.Length
                    if plural_assign and new_length >= min_length:                    
                        # shorten the supply element:
                        supply_sorted.loc[supply_tuple.Index, "Length"] = new_length
                        
                        temp_row = supply_sorted.loc[supply_tuple.Index].copy(deep=True)
                        temp_row['LCA'] = temp_row.Length * temp_row.Area * REUSE_GWP_RATIO * TIMBER_GWP
                        supply_sorted.drop(supply_tuple.Index, axis = 0, inplace = True)
                        
                        #new_ind = supply_sorted['LCA'].searchsorted([False ,temp_row['LCA']], side = 'left') #get index to insert new row into #TODO Can this be sorted also by 'Area' and any other constraint?
                        new_ind = supply_sorted[supply_sorted['Is_new'] == False]['LCA'].searchsorted(temp_row['LCA'], side = 'left')
                        part1 = supply_sorted[:new_ind].copy(deep=True)
                        part2 = supply_sorted[new_ind:].copy(deep=True)
                        supply_sorted = pd.concat([part1, pd.DataFrame(temp_row).transpose().infer_objects(), part2]) #TODO Can we make it simpler
                        
                        new_incidence_col = self.evaluate_column(new_length, "Length", self.constraints['Length'], incidence_np[:, supply_tuple.Index])
                        #new_incidence_col = self.evaluate_column(supply_tuple.Index, new_length, "Length", self.constraints["Length"], incidence_np[:, supply_tuple.Index])
                        #incidence__np[:, columns.index(supply_tuple.Index)] = new_incidence_col

                        #incidence_copy.loc[:, columns[supply_tuple.Index]] = new_incidence_col #TODO If i get the indicies to work. Try using this as an np array instead of df.
                        incidence_np[:,supply_tuple.Index] = new_incidence_col
                        
                        logging.debug("---- %s is a match, that results in %s m cut.", supply_tuple.Index, supply_tuple.Length)
                    else:
                        #self.result += calculate_lca(supply_row.Length, supply_row.Area, is_new=supply_row.Is_new)
                        logging.debug("---- %s is a match and will be utilized fully.", supply_tuple.Index)
                        supply_sorted.drop(supply_tuple.Index, inplace = True)
                    break
                        
            else:
                logging.debug("---- %s is not matching.", supply_tuple.Index)

    @_matching_decorator
    def match_bipartite_graph(self):
        """Match using Maximum Bipartite Graphs. A maximum matching is a set of edges such that each vertex is
        incident on at most one matched edge and the weight of such edges in the set is as large as possible."""
        # TODO multiple assignment won't work OOTB.
        if not self.graph:
            self.add_graph()
        if self.graph.is_connected():
            # TODO separate disjoint graphs for efficiency
            logging.info("graph contains unconnected subgraphs that could be separated")
        bipartite_matching = ig.Graph.maximum_bipartite_matching(self.graph, weights=self.graph.es["label"])
        for match_edge in bipartite_matching.edges():
            self.add_pair(match_edge.source_vertex["label"], match_edge.target_vertex["label"])  
        
    @_matching_decorator
    def match_genetic_algorithm(self):
        """Match using Evolutionary/Genetic Algorithm"""

        # supply capacity - length:
        capacity = self.supply['Length'].to_numpy()
        lengths = self.demand['Length'].to_numpy()

        # demand_mapping (column - demand):
        initial_population = np.zeros((len(self.supply), len(self.demand)))
        # for each column add one random 0/1.
        for col in range(len(self.demand)):
            row = random.randint(0, len(self.supply)-1)
            initial_population[row, col] = random.randint(0, 1)

        def fitness_func(solution, solution_idx):
            # output = np.sum(solution*function_inputs) #LCA!
            total_length = np.sum(solution*lengths)
            if np.sum(total_length > capacity) != len(capacity):
                output = 10e4  # penalty
            elif np.argwhere(np.sum(solution, axis=0) > 1):
                output = 10e4  # penalty
            else:
                # LCA:
                output = np.sum(solution*self.demand['Length'])
            fitness = 1.0 / output
            return fitness
        
        ga_instance = pygad.GA(
            num_generations=20,
            num_parents_mating=2,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=initial_population.size, #len(initial_population),
            # binary representation of the problem with help from: https://blog.paperspace.com/working-with-different-genetic-algorithm-representations-python/
            # (also possible with: gene_space=[0, 1])
            init_range_low=0,
            random_mutation_min_val=0,
            init_range_high=2,   # upper bound exclusive, so only 0 and 1
            random_mutation_max_val=2,   # upper bound exclusive, so only 0 and 1
            mutation_by_replacement=True,
            gene_type=int,

            parent_selection_type="sss",    # steady_state_selection() https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#steady-state-selection
            keep_parents=1,
            crossover_type="single_point",  # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#steady-state-selection

            mutation_type="random",  # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#steady-state-selection
            mutation_num_genes=1,
            # mutation_percent_genes=10,

            initial_population=initial_population
            )
        
        ga_instance.run()

        logging.debug(ga_instance.initial_population)
        logging.debug(ga_instance.population)

        solution, solution_fitness, solution_idx = ga_instance.best_solution() 
        logging.debug("Parameters of the best solution: %s", solution)
        logging.debug("Fitness value of the best solution = %s", solution_fitness)

        # prediction = np.sum(np.array(function_inputs)*solution)
        # logging.debug("Predicted output based on the best solution: %s", prediction)

        self.result += 1234 #calculate_lca(supply_row.Length, supply_row.Area, is_new=supply_row.Is_new)

    @_matching_decorator
    def match_mixed_integer_programming_OLD(self):
        """Match using SCIP - Solving Constraint Integer Programs, branch-and-cut algorithm, type of mixed integer programming (MIP)"""

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

        # --- Create the data needed for the solver ---        
        data = {} # initiate empty dictionary
        data ['lengths'] = self.demand.Length.astype(float)
        data['areas'] = self.demand.Area.astype(float)
        
        assert len(data['lengths']) == len(data['areas']) # The same check is done indirectly in the dataframe
        data['num_items'] = len(data['areas'])
        data['all_items'] = range(data['num_items'])
        data['all_items'] = range(data['num_items'])


        data['bin_capacities'] = self.supply.Length # these would be the bins
        data['num_bins'] = len(data['bin_capacities'])
        data['all_bins'] = range(data['num_bins'])

        #get constraint ids
        #get constraint ids
        c_inds = constraint_inds()
        # create solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if solver is None:
            logging.debug('SCIP Solver is unavailable')
            return

        # --- Variables ---
        # x[i,j] = 1 if item i is backed in bin j. 0 else
        var_array = np.full((self.incidence.shape), 0)
        x = {}
        for i in data['all_items']:
            for j in data['all_bins']:
                x[i,j] = solver.BoolVar(f'x_{i}_{j}') 
  
        logging.debug('Number of variables = %s', solver.NumVariables()) 

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
        for inds in c_inds:
            i = int(inds[0])
            j = int(inds[1])
            solver.Add(x[i,j] == 0)

        logging.debug('Number of contraints = %s', solver.NumConstraints())
        # --- Objective ---
        # maximise total value of packed items
        # coefficients
        coeff_array = self.weights.replace(np.nan, self.weights.max().max() * 1000).to_numpy() # play with different values here. 

        objective = solver.Objective()
        for i in data['all_items']:
            for j in data['all_bins']:
                objective.SetCoefficient(x[i,j], 1 / coeff_array[i,j]) # maximise the sum of 1/sum(weights)
                #objective.SetCoefficient(x[i,j], float(data['areas'][i]))      
        objective.SetMaximization()
        #objective.SetMinimization()

        # Starting solver
        # Starting solver
        status = solver.Solve()
        print('Computation done')
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            gwp_sum = 0
            for i in data['all_items']:
                for j in data['all_bins']:
                    if x[i,j].solution_value() > 0: 
                        self.pairs.iloc[i] = j # add the matched pair. 
                        gwp_sum += coeff_array[i, j]
                        continue # only one x[0, j] can be 1. the rest are 0. Continue
            self.result = gwp_sum           
            
            results = {}
            logging.debug('Solution found! \n ------RESULTS-------\n')
            total_length = 0
            for j in data['all_bins']:
                results[j] = []
                logging.debug('Bin %s', j)
                bin_length = 0
                bin_value = 0
                for i in data['all_items']:
                    if x[i, j].solution_value() > 0:
                        results[j].append(i)
                        logging.debug("Item %s Length: %s area: %s", i, data['lengths'][i], data['areas'][i])
                        bin_length += data['lengths'][i]
                        bin_value += data['areas'][i]
                logging.debug('Packed bin lengths: %s', bin_length)
                logging.debug('Packed bin value: %s', bin_value)
                total_length += bin_length
                logging.debug('Total packed Lenghtst: %s\n', total_length)

        # return the results as a DataFrame like the bin packing problem
        # Or a dictionary. One key per bin/supply, and a list of ID's for the
        # elements which should go within that bin. 
        # elements which should go within that bin. 

        # TODO temp result:
        return [self.result, self.pairs]

    @_matching_decorator
    def match_mixed_integer_programming(self):
        """This method is the same as the previous one, but uses a CP model instead of a MIP model in order to stop at a given number of 
        feasible solutions. """
        #TODO Evaluate if the cost function is the best we can have. 
        # the CP Solver works only on integers. Consequently, all values are multiplied by 1000 before solving the
        m_fac = 10000
        max_time = self.solution_limit
        # --- Create the data needed for the solver ---        
        data = {} # initiate empty dictionary
        data ['lengths'] = (self.demand.Length * m_fac).astype(int)
        data['values'] = (self.demand.Area * m_fac).astype(int)
        
        assert len(data['lengths']) == len(data['values']) # The same check is done indirectly in the dataframe
        data['num_items'] = len(data['values']) # don't need this. TODO Delete it. 
        data['all_items'] = range(data['num_items'])
        #data['areas'] = self.demand.Area

        data['bin_capacities'] = (self.supply.Length * m_fac).astype(int)  # these would be the bins
        #data['bin_areas'] = self.supply.Area.to_numpy(dtype = int)
        data['num_bins'] = len(data['bin_capacities'])
        data['all_bins'] = range(data['num_bins'])

        #get constraint ids
        #c_inds = constraint_inds()
        c_inds = np.transpose(np.where(~self.incidence)) # get the position of the substitutions that cannot be used
        
        # create model
        model = cp_model.CpModel()

        # --- Variables ---
        # x[i,j] = 1 if item i is backed in bin j. 0 else
        var_array = np.full((self.incidence.shape), 0) #TODO Implement this for faster extraction of results later. Try to avoid nested loops
        x = {}
        for i in data['all_items']:
            for j in data['all_bins']:
                x[i,j] = model.NewBoolVar(f'x_{i}_{j}')   
        #print(f'Number of variables = {solver.NumVariables()}') 

        # --- Constraints ---
        # each item can only be assigned to one bin
        for i in data['all_items']:
            model.AddAtMostOne(x[i, j] for j in data['all_bins'])
            
        # the amount packed in each bin cannot exceed its capacity.
        for j in data['all_bins']:
            model.Add(sum(x[i, j] * data['lengths'][i]
            for i in data['all_items']) <= data['bin_capacities'][j])

        # from the already calculated incidence matrix we add constraints to the elements i we know
        # cannot fit into bin j.
        for inds in c_inds:
            i = int(inds[0])
            j = int(inds[1])
            model.Add(x[i,j] == 0)
            #model.AddHint(x[i,j], 0)    
        
        # --- Objective ---
        # maximise total inverse of total gwp
        # coefficients
        coeff_array = self.weights.values * m_fac
        np.nan_to_num(coeff_array, False, nan = 0.0)
        #coeff_array = coeff_array.replace(np.nan, coeff_array.max().max() * 1000).to_numpy() # play with different values here. 
        #coeff_array = coeff_array.astype(int)
        
        objective = []
        for i in data['all_items']:
            for j in data['all_bins']:
                objective.append(
                    #cp_model.LinearExpr.Term(x[i,j], coeff_array[i,j])
                    cp_model.LinearExpr.Term(x[i,j], (self.demand.LCA[i]*m_fac - coeff_array[i,j]))
                    )
                                
        #model.Maximize(cp_model.LinearExpr.Sum(objective))
        model.Maximize(cp_model.LinearExpr.Sum(objective))
        # --- Solve ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time
        status = solver.Solve(model)
        test = solver.ObjectiveValue()
        index_series = self.supply.index
        # --- RESULTS ---
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            gwp_sum = 0
            for i in data['all_items']:
                for j in data['all_bins']:
                    if solver.Value(x[i,j]) > 0: 
                        self.pairs.iloc[i] = index_series[j] # add the matched pair.                         
                        break # only one x[0, j] can be 1. the rest are 0. Continue or break?
            
    @_matching_decorator
    def match_scipy_milp(self):
        max_time = self.solution_limit
        #costs = np.nan_to_num(self.weights.to_numpy(), nan = 0.0).reshape(-1,)*100 # set as 1d array like the variables below
        #initial_gwp = pd.eval('self.demand.Length * self.demand.Area * TIMBER_GWP').sum()
        #costs = self.weights.to_numpy(dtype = float)
        weights = np.nan_to_num(self.weights.to_numpy().astype(float), nan = 0) 
        lca = self.demand.Length.to_numpy(dtype = float).reshape((-1,1)) 
        costs = np.subtract(lca, weights).reshape((-1,))
        
        #costs = costs 
        #np.nan_to_num(costs, copy = False, nan = -110)
        # What should be the costs of assigning an element?
        # parameters x
        x_mat = np.zeros(self.weights.shape, dtype= int) # need to flatten this to use scipy
        x_arr = np.reshape(x_mat, (-1, ))
        
        # parameter bounds
        lb = np.full_like(x_arr, 0)
        ub = np.where(self.weights.isna().to_numpy().reshape((-1,)), 0 ,1)
        bounds = Bounds(lb = lb, ub = ub)#, ub = np.full_like(x_arr, 1)) # can modify these later to restrict solutions that we already know are infeasible.
        
        
        # constraints

        #Try creating a constraints list
        rows, cols = x_mat.shape
        A1 = np.zeros((rows, rows*cols))
        # fill a with ones: 
        for i in range(rows):
            A1[i, i*cols : (i+1)*cols] = 1
        cons = [] # Constraints dictionary
        
        max_constr = lambda vec: np.sum(vec)
        constraints1 = LinearConstraint(A = A1 , lb = 0, ub = 1)

        A2 = np.zeros((cols, rows * cols))
        demand_lengths = self.demand.Length.to_numpy()
        #constraints2 = LinearConstraint(A = A2, lb = 0, ub = self.supply.Length)
        for j in range(cols):
            A2[j, j::cols] = demand_lengths
            #A2[j, j*rows : (j+1)*rows] = demand_lengths

        constraints2 = LinearConstraint(A = A2, lb = 0., ub = self.supply.Length.to_numpy())
                       
        integrality = np.full_like(x_arr, True) # force decision variables to be 0 or 1
        constraints = [constraints1, constraints2]       
       
        # Run optimisation:
        time_limit = max_time
        options = {'disp':False, 'time_limit': time_limit, 'presolve' : True}
        
        res = milp(c=  costs* (-1), constraints = constraints, bounds = bounds, integrality = integrality, options = options)
        #res = milp(c= -np.ones_like(x_arr), constraints = constraints, bounds = bounds, integrality = integrality, options = options)
        # ======= POST PROCESS ===========
        try:
            assigment_array = res.x.reshape(rows, cols) 
        except AttributeError:# If no solution res.x is None. No substitutions exists. 
            assigment_array = np.zeros_like(x_mat)
        demand_ind, supply_ind = np.where(assigment_array == 1)

        demand_id = self.demand.index[demand_ind]
        supply_id = self.supply.index[supply_ind]

        self.pairs.loc[demand_id] = supply_id.to_numpy().reshape((-1,1))

        # Create dataframe to see if constraints are kept. 
        #capacity_df = pd.concat([self.pairs, self.demand.Length], axis = 1).groupby('Supply_id').sum()
        #compare_df = capacity_df.join(self.supply.Length, how = 'inner', lsuffix = ' Assigned', rsuffix = ' Capacity')
        #compare_df['OK'] = np.where(compare_df['Length Assigned'] <= compare_df['Length Capacity'], True, False)
        
      
      
      
def run_matching( demand, supply, include_transportation = False, constraints = None, add_new = True, bipartite = True, greedy_single = True, greedy_plural = True, genetic = False, milp = False, sci_milp = False):
    """Run selected matching algorithms and returns results for comparison.
    By default, bipartite, and both greedy algorithms are run. Activate and deactivate as wished."""
    #TODO Can **kwargs be used instead of all these arguments
    # create matching object 
    matching = Matching(demand=demand, supply= supply, include_transportation = include_transportation, constraints=constraints, add_new= add_new, multi = True)

    matches =[] # results to return
    headers = []
    if bipartite:
        matching.match_bipartite_graph()
        matches.append({'Name': 'Bipartite', 'Match object': copy(matching)})
    
    if greedy_single:
        matching.match_greedy_algorithm(plural_assign=False)
        matches.append({'Name': 'Greedy_single','Match object': copy(matching)})

    if greedy_plural:
        matching.match_greedy_algorithm(plural_assign=True)
        matches.append({'Name': 'Greedy_plural', 'Match object': copy(matching)})
    
    if milp:
        matching.match_mixed_integer_programming()
        matches.append({'Name': 'MILP','Match object': copy(matching)})

    if sci_milp:
        matching.match_scipy_milp()
        matches.append({'Name': 'Scipy MILP','Match object': copy(matching)})
    if genetic:
        matching.match_genetic_algorithm()
        matches.append({'Name': 'Genetic','Match object': copy(matching)})
    
    #convert list of dfs to single df

    return matches


# class Elements(pd.DataFrame):
#     def read_json(self):
#         super().read_json()
#         self.columns = self.iloc[0]
#         self.drop(axis = 1, index= 0, inplace=True)
#         self.reset_index(drop = True, inplace = True)

def calculate_lca(length, area, is_new=True, gwp=TIMBER_GWP):
    """ Calculate Life Cycle Assessment """
    # TODO add distance, processing and other impact categories than GWP
    if not is_new:
        gwp = gwp * REUSE_GWP_RATIO
    #gwp_array = np.where(is_new, gwp, gwp * REUSE_GWP_RATIO)
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
