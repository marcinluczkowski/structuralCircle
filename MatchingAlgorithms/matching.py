# -*- coding: utf-8 -*-

import sys
from itertools import product, compress
import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import pygad
import numexpr as ne
import logging
import time
import random


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s_%(asctime)s_%(message)s',
    datefmt='%H:%M:%S',
    # filename='log.log',
    # filemode='w'
    )


class Matching():
    """Class describing the matching problem, with its constituent parts."""
    def __init__(self, demand, supply, add_new=False, multi=False, constraints = {}):
        self.demand = demand
        if add_new:
            # add perfectly matching new elements to supply
            demand_copy = demand.copy(deep = True)
            demand_copy['Is_new'] = True # set them as new elements
            try:
                # This works only when the indices are already named "D"
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
        self.weights = None
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
                demand_array = self.demand[param].to_list()
                bool_col = ne.evaluate(f'{var} {compare} demand_array') # numpy array of boolean
                cond_list.append(bool_col)
            cond_array = np.column_stack(cond_list) #create new 2D-array of conditionals
            bool_array = ne.evaluate("cond_array & bool_array") # 
            #bool_array = np.logical_and(bool_array, cond_array)
        self.incidence = pd.DataFrame(bool_array, columns= self.incidence.columns, index= self.incidence.index)
        end = time.time()
        logging.info("Create incidence matrix from constraints: %s sec", round(end - start,3))

    # def evaluate2(self):
    #     """OBSOLETE"""
    #     # TODO optimize the evaluation.
    #     # TODO add 'Distance'
    #     # TODO add 'Price'
    #     # TODO add 'Material'
    #     # TODO add 'Density'
    #     # TODO add 'Imperfections'
    #     # TODO add 'Is_column'
    #     # TODO add 'Utilisation'
    #     # TODO add 'Group'
    #     # TODO add 'Quality'
    #     # TODO add 'Max_height' ?
    #     start = time.time()
    #     match_new = lambda sup_row : row[1] <= sup_row['Length'] and row[2] <= sup_row['Area'] and row[3] <= sup_row['Inertia_moment'] and row[4] <= sup_row['Height'] and sup_row['Is_new'] == True
    #     match_old = lambda sup_row : row[1] <= sup_row['Length'] and row[2] <= sup_row['Area'] and row[3] <= sup_row['Inertia_moment'] and row[4] <= sup_row['Height'] and sup_row['Is_new'] == False
    #     for row in self.demand.itertuples():
    #         bool_match_new = self.supply.apply(match_new, axis = 1).tolist()
    #         bool_match_old = self.supply.apply(match_old, axis = 1).tolist()
            
    #         self.incidence.loc[row[0], bool_match_new] = calculate_lca(row[1], self.supply.loc[bool_match_new, 'Area'], is_new=True)
    #         self.incidence.loc[row[0], bool_match_old] = calculate_lca(row[1], self.supply.loc[bool_match_old, 'Area'], is_new=False)
    #     end = time.time()
    #     logging.info("Weight evaluation execution time: %s sec", round(end - start,3))

    def weight_incidence(self):
        """Assign wegihts to elements in the incidence matrix. At the moment only LCA is taken into\
        account. This method should replace the last step in the original evaluate method."""
        start = time.time()
        el_locs0 = np.where(self.incidence) # tuple of rows and columns as list
        el_locs = np.transpose(el_locs0) # array of row-column pairs where incidence matrix is true. 
        areas = self.supply.Area.iloc[el_locs[:, 1]].to_list() # array of areas 
        lenghts = self.demand.Length.iloc[el_locs[:, 0]].to_list() # array of element lenghts
        el_new = self.supply.Is_new.iloc[el_locs[:,1]].to_list() # array of booleans for element condition.  
        gwp = 28.9
        gwp_array = np.where(el_new, gwp, gwp * 0.0778)
        #get_gwp = ne.evaluate("gwp if el_new else gwp*0.0778")
        #gwp = [7.28 if tr else ]
        lca_array = ne.evaluate("areas * lenghts * gwp_array")
       
        
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
        # remove already used:
        # TODO not change the initial incidence...
        # try:
        #     self.incidence.drop(demand_id, inplace=True)
        #     self.incidence.drop(supply_id, axis=1, inplace=True)
        # except KeyError:
        #     pass

    def add_graph(self):
        """Add a graph notation based on incidence matrix"""
        #FIXME The method assigns new elements to demand items although there should be available old elemements.  
        vertices = [0]*len(self.demand.index) + [1]*len(self.supply.index)
        edges = []
        weights = []
        #is_na = self.incidence.isna() #TODO delete this line if the below line works
        is_na = self.weights.isna()
        row_inds = np.arange(self.incidence.shape[0]).tolist()
        col_inds = np.arange(len(self.demand.index), len(self.demand.index)+ self.incidence.shape[1]).tolist()
        for i in row_inds:
            combs = list(product([i], col_inds) )
            mask =  ~is_na.iloc[i] # invert the booleans
            edges.extend( (list(compress(combs, mask) ) ) )
            weights.extend(list(compress(self.weights.iloc[i], mask)))
        weights = 1 / np.array(weights)
        graph = ig.Graph.Bipartite(vertices,  edges)
        graph.es["label"] = weights
        graph.vs["label"] = list(self.demand.index)+list(self.supply.index) #vertice names
        self.graph = graph

    def display_graph(self, show_weight=True):
        """Plot the graph and matching result"""
        if not self.graph:
            self.add_graph()
        weight = None
        if show_weight:
            weight = [round(1/w,2) for w in self.graph.es["label"]]  # invert weight, to see real LCA
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
                edge_label=weight
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
            # logging.info("Matched: %s to %s (%s %%) of %s elements using %s, resulting in LCA (GWP): %s kgCO2eq, in: %s sec.",
            #     len(self.pairs['Supply_id'].unique()),
            #     self.pairs['Supply_id'].count(),
            #     100*self.pairs['Supply_id'].count()/len(self.demand),
            #     self.supply.shape[0],
            #     func.__name__,
            #     round(self.result, 2),
            #     round(end - start,3)
            all_string_series = self.pairs.fillna('nan') # have all entries as string before search
            num_old = len(all_string_series.loc[all_string_series.Supply_id.str.contains('R')].Supply_id.unique())
            num_new = len(all_string_series.loc[all_string_series.Supply_id.str.contains('N')].Supply_id.unique())
            num_matched = len(self.pairs.dropna())
            logging.info(f"""Matched {num_old} old and {num_new} new elements to {num_matched} demand elements ({100 * num_matched / len(self.pairs)}%) 
            using {func.__name__}. Resulting in LCA (GWP) {round(self.result, 2)} kgCO2eq, in {round(end - start,3)} seconds.""")

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

    @_matching_decorator
    def match_greedy_algorithm(self, plural_assign=False):
        """Algorithm that takes one best element at each iteration, based on sorted lists, not considering any alternatives."""
        # TODO consider opposite sorting (as we did in Gh), small chance but better result my occur
        demand_sorted = self.demand.sort_values(by=['Length', 'Area'], axis=0, ascending=False)
        supply_sorted = self.supply.sort_values(by=['Is_new', 'Length', 'Area'], axis=0, ascending=True)

        min_length = demand_sorted.iloc[-1].Length # the minimum lenght of a demand element
        for demand_index, demand_row in demand_sorted.iterrows():
            match=False
            logging.debug("-- Attempt to find a match for %s", demand_index)                
            for supply_index, supply_row in supply_sorted.iterrows():
                # TODO replace constraints with evalute string
                if demand_row.Length <= supply_row.Length and demand_row.Area <= supply_row.Area and demand_row.Inertia_moment <= supply_row.Inertia_moment and demand_row.Height <= supply_row.Height:
                    match=True
                    self.add_pair(demand_index, supply_index)
                if match:
                    new_length = supply_row.Length - demand_row.Length
                    if plural_assign and new_length >= min_length:
                        # shorten the supply element:
                        #supply_row.Length = supply_row.Length - demand_row.Length # this does not do anything to the dataframe?
                        
                        supply_sorted.loc[supply_index, "Length"] = new_length
                        # if the total length becomes zero, then we should remove it.
                        # sort the supply list
                        supply_sorted = supply_sorted.sort_values(by=['Is_new', 'Length', 'Area'], axis=0, ascending=True)  # TODO move this element instead of sorting whole list
                        #self.result += calculate_lca(demand_row.Length, supply_row.Area, is_new=supply_row.Is_new)
                        logging.debug("---- %s is a match, that results in %s m cut.", supply_index, supply_row.Length)
                    else:
                        #self.result += calculate_lca(supply_row.Length, supply_row.Area, is_new=supply_row.Is_new)
                        logging.debug("---- %s is a match and will be utilized fully.", supply_index)
                        supply_sorted.drop(supply_index, inplace = True)
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
        #self.result = sum(bipartite_matching.edges()["label"]) #TODO Remove this if new method works.


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
    def match_mixed_integer_programming(self):
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
    def match_cp_solver(self):
        """This method is the same as the previous one, but uses a CP model instead of a MIP model in order to stop at a given number of 
        feasible solutions. """

        # the CP Solver works only on integers. Consequently, all values are multiplied by 1000 before solving the
        m_fac = 10000
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
        c_inds = np.transpose(np.where(~self.incidence)) # get the position of the element which cannot be used
        
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
        coeff_array = self.weights * m_fac
        coeff_array = coeff_array.replace(np.nan, coeff_array.max().max() * 1000).to_numpy() # play with different values here. 
        coeff_array = coeff_array.astype(int)
        objective = []
        for i in data['all_items']:
            for j in data['all_bins']:
                objective.append(
                    cp_model.LinearExpr.Term(x[i,j], 1 / coeff_array[i,j]))
                                
        model.Maximize(cp_model.LinearExpr.Sum(objective))
        
        # --- Solve ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 100
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
        #TODO Try using scipy for computational speed
        pass
      

# class Elements(pd.DataFrame):
#     def read_json(self):
#         super().read_json()
#         self.columns = self.iloc[0]
#         self.drop(axis = 1, index= 0, inplace=True)
#         self.reset_index(drop = True, inplace = True)

def calculate_lca(length, area, is_new=True, gwp=28.9, ):
    """ Calculate Life Cycle Assessment """
    # TODO add distance, processing and other impact categories than GWP
    if not is_new:
        gwp = gwp * 0.0778
    #gwp_array = np.where(is_new, gwp, gwp * 0.0778)
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

