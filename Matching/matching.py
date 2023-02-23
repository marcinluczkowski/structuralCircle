# -*- coding: utf-8 -*-

import logging
import random
import sys
import time
from itertools import compress, product
from copy import copy, deepcopy
import igraph as ig
import numexpr as ne
import numpy as np
import pandas as pd
import pygad
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from scipy.optimize import milp, LinearConstraint, NonlinearConstraint, Bounds
import helper_methods as hm
import LCA as lca


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s_%(asctime)s_%(message)s',
    datefmt='%H:%M:%S',
    # filename='log.log',
    # filemode='w'
    )


class Matching():
    """Class describing the matching problem, with its constituent parts."""
    def __init__(self, demand, supply, score_function_string, add_new=False, multi=False, constraints={}, solution_limit=60):
        """_summary_

        :param demand: _description_
        :type demand: _type_
        :param supply: _description_
        :type supply: _type_
        :param score_function_string: _description_
        :type score_function_string: _type_
        :param add_new: _description_, defaults to False
        :type add_new: bool, optional
        :param multi: _description_, defaults to False
        :type multi: bool, optional
        :param constraints: _description_, defaults to {}
        :type constraints: dict, optional
        :param solution_limit: _description_, defaults to 60
        :type solution_limit: int, optional
        """
        self.demand = demand.infer_objects()
        if add_new: # just copy designed to supply set, so that they act as new products
            demand_copy = self.demand.copy(deep = True)
            try:
                # Rename Dx to Nx. This works only when the indices are already named "D"
                demand_copy.rename(index=dict(zip(demand.index.values.tolist(), [sub.replace('D', 'N') for sub in demand.index.values.tolist()] )), inplace=True)
            except AttributeError:
                pass
            self.supply = pd.concat((supply, demand_copy), ignore_index=False).infer_objects()
        else:
            self.supply = supply.infer_objects()
        self.multi = multi
        self.graph = None
        self.result = None  #saves latest result of the matching
        self.pairs = pd.DataFrame(None, index=self.demand.index.values.tolist(), columns=['Supply_id']) #saves latest array of pairs
        self.incidence = pd.DataFrame(np.nan, index=self.demand.index.values.tolist(), columns=self.supply.index.values.tolist())
        # self.weights = None
        self.constraints = constraints
        self.score_function_string = score_function_string
        self.solution_time = None
        self.solution_limit = solution_limit           
       
        self.demand['Score'] = self.demand.eval(score_function_string)
        self.supply['Score'] = self.supply.eval(score_function_string)

        # create incidence and weight for the method
        self.incidence = self.evaluate_incidence()
        self.weights = self.evaluate_weights()

        logging.info("Matching object created with %s demand, and %s supply elements", len(demand), len(supply))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result    

    def evaluate_incidence(self):
        """Returns incidence matrix with true values where the element fit constraint criteria"""    
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
        # for simplicity I restrict the incidence of new elements to only be True for the "new" equivalent
        inds = self.supply.index[self.supply.index.map(lambda s: 'N' in s)] # Get the indices for new elements
        if len(inds) > 0:
            diag_mat = np.full((len(inds), len(inds)), False)
            np.fill_diagonal(diag_mat, True) # create a diagonal with True on diag, False else. 
            bool_array = np.hstack((bool_array[:, :-len(inds)], diag_mat))

        end = time.time()
        logging.info("Create incidence matrix from constraints: %s sec", round(end - start,3))
        return pd.DataFrame(bool_array, columns= self.incidence.columns, index= self.incidence.index)

    def evaluate_column(self, supply_val, parameter, compare, current_bool):
        """Evaluates a column in the incidence matrix according to the constraints
            Returns a np array that can substitute the input column."""
        demand_array = self.demand[parameter].to_numpy(dtype = float) # array of demand parameters to evaluate. 
        compare_array = ne.evaluate(f"{supply_val} {compare} demand_array")        
        return ne.evaluate("current_bool & compare_array")
            
    def evaluate_weights(self):
        """Return matrix of weights for elements in the incidence matrix. The lower the weight the better."""
        start = time.time()
        weights = np.full(self.incidence.shape, np.nan)
        el_locs0 = np.where(self.incidence) # tuple of rows and columns positions, as a list
        el_locs = np.transpose(el_locs0) # array of row-column pairs where incidence matrix is true. 
        # create a new dataframe with values from supply, except for the Length, which is from demand set (cut supply)
        eval_df = self.supply.iloc[el_locs0[1]].reset_index(drop=True)
        eval_df['Length'] = self.demand.iloc[el_locs0[0]]['Length'].reset_index(drop=True)
        eval_score = eval_df.eval(self.score_function_string)
        weights[el_locs0[0], el_locs0[1]] = eval_score.to_numpy()     
        end = time.time()  
        logging.info("Weight evaluation of incidence matrix: %s sec", round(end - start, 3))
        return pd.DataFrame(weights, index = self.incidence.index, columns = self.incidence.columns)

    def add_pair(self, demand_id, supply_id):
        """Execute matrix matching"""
        # add to match_map:
        self.pairs.loc[demand_id, 'Supply_id'] = supply_id
        
    def add_graph(self):
        """Add a graph notation based on incidence matrix"""
        vertices = [0]*len(self.demand.index) + [1]*len(self.supply.index)
        num_rows = len(self.demand.index)
        edges = np.transpose(np.where(self.incidence))
        edges = [[edge[0], edge[1]+num_rows] for edge in edges]
        edge_weights = self.weights.to_numpy().reshape(-1,)
        edge_weights = edge_weights[~np.isnan(edge_weights)]
        # We need to reverse the weights, so that the higher the better. Because of this edge weights are initial score minus the replacement score:
        edge_weights = (np.array([self.demand.Score[edge[0]] for edge in edges ])+0.0000001) - edge_weights 
        # assemble graph
        graph = ig.Graph.Bipartite(vertices, edges)
        graph.es["label"] = edge_weights
        graph.vs["label"] = list(self.demand.index)+list(self.supply.index) #vertice names
        self.graph = graph

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
            logging.info("Matched %s old and %s new elements to %s demand elements (%s %%) using %s. Resulting in score %s, in %s seconds.", 
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
        # add the score of original elements that are not substituted
        mask = self.pairs.Supply_id.isna().to_numpy()
        original_score = self.demand.Score[mask].sum()
        self.result += original_score

    ### MATCHING ALGORITHMS

    @_matching_decorator
    def match_brute(self, plural_assign=False):
        """..."""
        # TODO implement it
        pass

    @_matching_decorator
    def match_greedy(self, plural_assign=False):
        """Algorithm that takes one best element at each iteration, based on sorted lists, not considering any alternatives."""

        sorted_weights = self.weights.join(self.demand.Score)
        sorted_weights = sorted_weights.sort_values(by='Score', axis=0, ascending=False)
        sorted_weights = sorted_weights.drop(columns=['Score'])
        #sorted_weights.replace(np.nan, np.inf, inplace=True)  

        score = self.supply.Score.copy()

        for i in range(sorted_weights.shape[0]):
            row_id = sorted_weights.iloc[[i]].index[0]
            vals = np.array(sorted_weights.iloc[[i]])[0]
#            if np.any(vals):    # checks if not empty row (no matches)
            if sum(~np.isnan(vals)) > 0: # check if there it at least one element not np.nan
                lowest = np.nanmin(vals)
                col_id = sorted_weights.columns[np.where(vals == lowest)][0]
                self.add_pair(row_id, col_id)
                if plural_assign:
                    # check if this column makes sense if remaining rows (score < initial_score-score_used), remove if not
                    # sorted_weights[col_id] = sorted_weights[col_id].apply(hm.remove_alternatives, args=(self.supply.loc[col_id].Score))
                    score.loc[col_id] = score.loc[col_id] - lowest
                    sorted_weights[col_id] = sorted_weights[col_id].apply((lambda x: hm.remove_alternatives(x, score.loc[col_id])))
                else:
                    # empty the column that was used
                    sorted_weights[col_id] = np.nan

    @_matching_decorator
    def match_greedy_DEPRECIATED(self, plural_assign=False):
        """Algorithm that takes one best element at each iteration, based on sorted lists, not considering any alternatives."""
        # TODO consider opposite sorting (as we did in Gh), small chance but better result my occur
        demand_sorted = self.demand.copy(deep =True)
        supply_sorted = self.supply.copy(deep =True)
        #Change indices to integers for both demand and supply
        demand_sorted.index = np.array(range(len(demand_sorted.index)))
        supply_sorted.index = np.array(range(len(supply_sorted.index)))

        #sort the supply and demand
        #demand_sorted.sort_values(by=['Length', 'Area'], axis=0, ascending=False, inplace = True)
        demand_sorted.sort_values(by=['Score'], axis=0, ascending=False, inplace = True)
        #supply_sorted.sort_values(by=['Is_new', 'Length', 'Area'], axis=0, ascending=True, inplace = True)
        supply_sorted.sort_values(by=['Is_new', 'Score'], axis=0, ascending=True, inplace = True) # FIXME Need to make this work "optimally"
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
                        temp_row['LCA'] = temp_row.Length * temp_row.Area * lca.TIMBER_REUSE_GWP
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
        

    # TODO (SIGURD) WORK IN PROGRESS: MAKING A NEW GENETIC ALGORITHM
    @_matching_decorator
    def match_genetic_algorithm_SIGURD(self):
        #ASSUMING THAT WE WANT TO OPTIMIZE ON MINIMIZING LCA
        solutions_per_population = len(self.supply)
        number_of_buckets = len(self.demand)

        #NOTE DELETE AFTER IF NOT WORKING: Testing with not including new elements in the genetic algorithm search
        #
        #
        supply_names = self.supply.index.tolist()
        index_first_new = self.supply.index.tolist().index("N0")
        supply_names_only_reuse = supply_names[:index_first_new]
        #supply_without_new = self.supply[[col_names_only_reuse]].copy()

        #Initializing a random population
        initial_population = np.array(([[random.randint(0,1) for x in range(len(supply_names_only_reuse)*len(self.demand))] for y in range(len(supply_names_only_reuse)*10)]))

        #test = self.weights["N0"]
        #weight_cols = self.weights.columns.values.tolist()
        #test2 = np.array_split(initial_population[0], number_of_buckets)
        #toNum = self.weights.to_numpy().flatten()
        test3 = 3
        #Fitness function to calculate fitness value of chromosomes
        #Genetic algorithm expects a maximization fitness function => when we are minimizing lca we must divide by 1/LCA
        
        def fitness_func(solution, solution_idx):
            fitness = 0
            supply_names = self.supply.index.tolist()
            index_first_new = self.supply.index.tolist().index("N0")
            supply_names_only_reuse = supply_names[:index_first_new]
            weight_only_reuse = self.weights[supply_names_only_reuse].copy()
            weights_1d_array = weight_only_reuse.to_numpy().flatten() 
            
            """
            for i in range(len(solution)):
                #if solution[i] == 1 and not np.isnan(weights_1d_array[i]):
                #    fitness -= weights_1d_array[i]
                if solution[i] == 1:
                    if np.isnan(weights_1d_array[i]): #Element cannot be matched => penalty
                        fitness += 10e4
                    else:
                        fitness += weights_1d_array[i]
            if fitness == 0: #To ensure that not matching elements is a reward for the algorithm
                fitness = 10e10
            #    else:
            #        if np.isnan(weights_1d_array[i]): #Element is not matched and should not be matched => reward
            #            fitness += 0.1
            #        else: #Element is not matched, but could have been matched
            #            fitness += 10e5

            return 1.0/fitness
            """

            
            #Trying something new where each bucket must have one solution
            solutions = np.array_split(solution, number_of_buckets)
            weights = np.array_split(weights_1d_array, number_of_buckets)
            max_weight = np.max(weights_1d_array[~np.isnan(weights_1d_array)])
            penalty = 3*max_weight
            #penalty = 10e5
            indexes_of_matches = []
            for i in range(len(solutions)):
                num_matches_in_bracket = 0
                for j in range(len(solutions[i])):
                    if solutions[i][j] == 1:
                        num_matches_in_bracket += 1
                        indexes_of_matches.append(j)
                        if np.isnan(weights[i][j]): #Element cannot be matched => penalty
                            fitness += penalty #Penalty
                        else:
                            fitness += weights[i][j]
                            
                if num_matches_in_bracket > 1 or num_matches_in_bracket < 1:
                    fitness += penalty #Penalty
            
            index_duplicates = {x for x in indexes_of_matches if indexes_of_matches.count(x) > 1}
            if len(index_duplicates) > 0: #Means some demand elements are assigned the same supply element
                fitness += penalty
                  

            return 1.0/fitness
            
            
        ga_instance = pygad.GA(
            num_generations=50,
            num_parents_mating=int(np.ceil(solutions_per_population/2)*2),
            fitness_func=fitness_func, #len(initial_population),
            # binary representation of the problem with help from: https://blog.paperspace.com/working-with-different-genetic-algorithm-representations-python/
            # (also possible with: gene_space=[0, 1])
            #random_mutation_min_val=0,
            #random_mutation_max_val=2,   # upper bound exclusive, so only 0 and 1
            #mutation_by_replacement=True,
            gene_type=int,
            parent_selection_type="sss",    # steady_state_selection() https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#steady-state-selection
            keep_parents=-1, #keep all parents
            crossover_type="single_point",  # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#steady-state-selection
            #mutation_type="swap",  # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#steady-state-selection
            #mutation_num_genes=int(solutions_per_population/5), Not needed if mutation_probability is set
            #mutation_probability = 0.1,
            #mutation_percent_genes=0.1,
            initial_population=initial_population
            )
        ga_instance.run()
        logging.debug(ga_instance.initial_population)
        logging.debug(ga_instance.population)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        
        see_result_from_genetic = hm.extract_genetic_solution(self.weights, solution, number_of_buckets)
        test4 = 4

    @_matching_decorator
    def match_genetic_algorithm(self):
        """Match using Evolutionary/Genetic Algorithm"""
        # TODO implement the method
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
            # output = np.sum(solution*function_inputs) #score!
            total_length = np.sum(solution*lengths)
            if np.sum(total_length > capacity) != len(capacity):
                output = 10e4  # penalty
            elif np.argwhere(np.sum(solution, axis=0) > 1):
                output = 10e4  # penalty
            else:
                # score:
                output = np.sum(solution*self.demand['Length'])
            fitness = 1.0 / output
            return fitness
        ga_instance = pygad.GA(
            num_generations=30,
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
        # TODO Don't use the method as it is :)
        self.result += 1234 #calculate_score(supply_row.Length, supply_row.Area, is_new=supply_row.Is_new)

    @_matching_decorator
    def match_mixed_integer_programming_DEPRECIATED(self):
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
        data['lengths'] = self.demand.Length.astype(float)
        data['areas'] = self.demand.Area.astype(float)
        assert len(data['lengths']) == len(data['areas']) # The same check is done indirectly in the dataframe
        data['num_items'] = len(data['areas'])
        data['all_items'] = range(data['num_items'])
        data['all_items'] = range(data['num_items'])
        data['bin_capacities'] = self.supply.Length # these would be the bins
        data['num_bins'] = len(data['bin_capacities'])
        data['all_bins'] = range(data['num_bins'])
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
        status = solver.Solve()
        logging.debug('Computation done')
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            score = 0
            for i in data['all_items']:
                for j in data['all_bins']:
                    if x[i,j].solution_value() > 0: 
                        self.pairs.iloc[i] = j # add the matched pair. 
                        score += coeff_array[i, j]
                        continue # only one x[0, j] can be 1. the rest are 0. Continue
            self.result = score           
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
        data['lengths'] = (self.demand.Length * m_fac).astype(int)
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
        #logging.debug(f'Number of variables = {solver.NumVariables()}') 
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
        # maximise total inverse of total score
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
                    cp_model.LinearExpr.Term(x[i,j], (self.demand.Score[i]*m_fac - coeff_array[i,j]))
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
            score = 0
            for i in data['all_items']:
                for j in data['all_bins']:
                    if solver.Value(x[i,j]) > 0: 
                        self.pairs.iloc[i] = index_series[j] # add the matched pair.                         
                        break # only one x[0, j] can be 1. the rest are 0. Continue or break?
            
    @_matching_decorator
    def match_scipy_milp(self):
        max_time = self.solution_limit

        weights = np.nan_to_num(self.weights.to_numpy().astype(float), nan = 0) 
        score = self.demand.Score.to_numpy(dtype = float).reshape((-1,1)) 
        costs = np.subtract(score, weights).reshape((-1,))
        
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
        #TODO Make sure once more that the costs here are the same as what we describe in the text.
        res = milp(c=  (costs+0.0000001)* (-1), constraints = constraints, bounds = bounds, integrality = integrality, options = options)
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
        
  
def run_matching(demand, supply, score_function_string, constraints = None, add_new = True, solution_limit = 120,
                bipartite = True, greedy_single = True, greedy_plural = True, genetic = False, milp = False, sci_milp = False):
    """Run selected matching algorithms and returns results for comparison.
    By default, bipartite, and both greedy algorithms are run. Activate and deactivate as wished."""
    #TODO Can **kwargs be used instead of all these arguments
    # create matching object 
    matching = Matching(demand=demand, supply=supply, score_function_string=score_function_string, constraints=constraints, add_new=add_new, multi = True, solution_limit=solution_limit)
    matches =[] # results to return
    headers = []
    if greedy_single:
        matching.match_greedy(plural_assign=False)
        matches.append({'Name': 'Greedy_single','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if greedy_plural:
        matching.match_greedy(plural_assign=True)
        matches.append({'Name': 'Greedy_plural', 'Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if bipartite:
        matching.match_bipartite_graph()
        matches.append({'Name': 'Bipartite', 'Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if milp:
        matching.match_mixed_integer_programming()
        matches.append({'Name': 'MILP','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if sci_milp:
        matching.match_scipy_milp()
        matches.append({'Name': 'Scipy_MILP','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if genetic:
        matching.match_genetic_algorithm()
        matches.append({'Name': 'Genetic','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})

        #NOTE DELETE AFTER (SIGURD)
        matching.match_genetic_algorithm_SIGURD()
    # TODO convert list of dfs to single df
    return matches


if __name__ == "__main__":
    #DEMAND_JSON = sys.argv[1]
    #SUPPLY_JSON = sys.argv[2]
    #RESULT_FILE = sys.argv[3]
    DEMAND_JSON = r"MatchingAlgorithms\sample_demand_input.json"
    SUPPLY_JSON = r"MatchingAlgorithms\sample_supply_input.json"
    RESULT_FILE = r"MatchingAlgorithms\result.csv"
    
    constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='} # dictionary of constraints to add to the method
    demand, supply = hm.create_random_data(demand_count=10, supply_count=5)
    score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"
    result = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=False, bipartite=False, genetic=True)
    simple_pairs = hm.extract_pairs_df(result)
    simple_results = hm.extract_results_df(result)
    print("Simple pairs:")
    print(simple_pairs)
    print()
    print("Simple results")
    print(simple_results)