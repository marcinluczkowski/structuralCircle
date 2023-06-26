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
import itertools
from itertools import combinations


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
        self.supply = supply.infer_objects()
        self.score_function_string = score_function_string.replace(" ", "")
        self.evaluate_transportation()
        
        pd.set_option('display.max_columns', 10)

        #Calculate the score and transportation score for the demand and supply elements
        self.demand['Score'], self.demand["Transportation"] = self.demand.eval(self.score_function_string)
        self.supply['Score'], self.supply["Transportation"] = self.supply.eval(self.score_function_string)


        if add_new: # just copy designed to supply set, so that they act as new products
            demand_copy = self.demand.copy(deep = True)
            try:
                # Rename Dx to Nx. This works only when the indices are already named "D"
                demand_copy.rename(index=dict(zip(self.demand.index.values.tolist(), [sub.replace('D', 'N') for sub in self.demand.index.values.tolist()] )), inplace=True)
            except AttributeError:
                pass
            self.supply = pd.concat((self.supply, demand_copy), ignore_index=False).infer_objects()
        else:
            self.supply = self.supply.infer_objects()
        self.multi = multi
        self.graph = None
        self.result = None  #saves latest result of the matching
        self.result_transport = None
        self.pairs = pd.DataFrame(None, index=self.demand.index.values.tolist(), columns=['Supply_id']) #saves latest array of pairs
        self.incidence = pd.DataFrame(np.nan, index=self.demand.index.values.tolist(), columns=self.supply.index.values.tolist())
        self.constraints = constraints
        self.solution_time = None
        self.solution_limit = solution_limit           
        #Create incidence and weight for the method
        self.demand['Score'] = self.demand.eval(score_function_string)
        self.supply['Score'] = self.supply.eval(score_function_string)
        self.incidence = self.evaluate_incidence()
        self.weights, self.weights_transport = self.evaluate_weights()
        logging.info("Matching object created with %s demand, and %s supply elements", len(demand), len(supply))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result    
    
    def evaluate_transportation(self):
        """Evaluates the driving distance for supply and demand elements"""
        if "include_transportation=True" in self.score_function_string:
            #Evaluating driving distance of supply elements

            self.supply["Distance"] = 0
            self.demand["Distance"] = 0
            coord_dict_supply = {}
            coord_dict_demand = {}

            #Unique coordinates of supply elements
            for idx, row in self.supply.iterrows():
                # create coordinate tuple
                coord = (row['Latitude'], row['Longitude'])
                # check if coordinate tuple already exists in dictionary
                if coord in coord_dict_supply:
                    # if it does, append index to list
                    coord_dict_supply[coord].append(idx)
                else:
                    # if it doesn't, create new list with index
                    coord_dict_supply[coord] = [idx]

            #Unique coordinates of demand elements
            for idx, row in self.demand.iterrows():
                # create coordinate tuple
                coord = (row['Latitude'], row['Longitude'])
                # check if coordinate tuple already exists in dictionary
                if coord in coord_dict_demand:
                    # if it does, append index to list
                    coord_dict_demand[coord].append(idx)
                else:
                    # if it doesn't, create new list with index
                    coord_dict_demand[coord] = [idx]
            
            site_lat = self.supply.iloc[0]["Site_lat"]
            site_lon = self.supply.iloc[0]["Site_lon"]
            distances_supply = {key: 0 for key in coord_dict_supply.keys()}
            distances_demand = {key: 0 for key in coord_dict_demand.keys()}

            #API call for supply elements
            for key in distances_supply:
                distances_supply[key] = lca.calculate_driving_distance(key[0], key[1], site_lat, site_lon)
            #API call for demand elements
            for key in distances_demand:
                distances_demand[key] = lca.calculate_driving_distance(key[0], key[1], site_lat, site_lon)
            
            #Add distances to supply column
            for key, value in coord_dict_supply.items():
                for idx in value:
                    self.supply.loc[idx, ["Distance"]] = distances_supply[key]
            
            #Add distances to demand column
            for key, value in coord_dict_demand.items():
                for idx in value:
                    self.demand.loc[idx, ["Distance"]] = distances_demand[key]
            #Assumes that all demand elements comes from the same location!!!
            #first_demand = self.demand.iloc[:1]
            #demand_distance = lca.calculate_driving_distance(first_demand["Supply_lat"], first_demand["Supply_lon"], first_demand["Demand_lat"], first_demand["Demand_lon"])
            #self.demand["Distance"] = demand_distance
        else:
            self.supply["Distance"] = np.NaN
            self.demand["Distance"] = np.NaN
        
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
                if isinstance(demand_array[0], str): #Assumes that target is to compare two text strings, "Timber" == "Timber"
                    bool_col = np.array(eval(f"['{var}' {compare} x for x in demand_array]"))
                else:
                    bool_col = ne.evaluate(f'{var} {compare} demand_array') # numpy array of boolean NOTE: Does not work when evaluating material given as a "String"
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
        weights_transport = np.full(self.incidence.shape, np.nan)
        el_locs0 = np.where(self.incidence) # tuple of rows and columns positions, as a list
        el_locs = np.transpose(el_locs0) # array of row-column pairs where incidence matrix is true. 
        # create a new dataframe with values from supply, except for the Length, which is from demand set (cut supply)
        eval_df = self.supply.iloc[el_locs0[1]].reset_index(drop=True)
        eval_df['Length'] = self.demand.iloc[el_locs0[0]]['Length'].reset_index(drop=True)
        eval_scores = eval_df.eval(self.score_function_string)
        eval_score = eval_scores[0]
        eval_score_transport = eval_scores[1]
        weights[el_locs0[0], el_locs0[1]] = eval_score
        weights_transport[el_locs0[0], el_locs0[1]] = eval_score_transport
        end = time.time()  
        logging.info("Weight evaluation of incidence matrix: %s sec", round(end - start, 3))
        return pd.DataFrame(weights, index = self.incidence.index, columns = self.incidence.columns), pd.DataFrame(weights_transport, index = self.incidence.index, columns = self.incidence.columns)

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
        self.result_transport = (self.weights_transport.to_numpy()[row_inds, col_inds]).sum()
        # add the score of original elements that are not substituted
        mask = self.pairs.Supply_id.isna().to_numpy()
        original_score = self.demand.Score[mask].sum()
        original_transport = self.demand.Transportation[mask].sum()
        self.result += original_score
        self.result_transport += original_transport

    ###########################
    ### MATCHING ALGORITHMS ###
    ###########################

    @_matching_decorator
    def match_brute(self, plural_assign=False):
        """Brute Force Approach that investigates all possible solutions"""
        weights = self.weights
        bestmatch=[]
        lowest_lca=10e10
        possible_solutions=hm.extract_brute_possibilities(self.incidence)
        arrayweights=weights.to_numpy()
        for subset in itertools.product(*possible_solutions):
            column_sum=np.sum(list(subset),axis=0)[:-1]
            invalid_solution=len([*filter(lambda x:x>1,column_sum)])>0
            if not invalid_solution:
                multiplum=np.multiply(arrayweights,subset)
                LCAsum=np.nansum(multiplum)
                if LCAsum<lowest_lca:
                    lowest_lca=LCAsum
                    bestmatch=subset
        bestmatch=pd.DataFrame(data=list(bestmatch),index=weights.index,columns=weights.columns)
        coordinates_of_pairs = [(f"D{x}", bestmatch.columns[y]) for x, y in zip(*np.where(bestmatch.values == 1))]
        for pair in coordinates_of_pairs:
            self.add_pair(pair[0],pair[1])
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

    # @_matching_decorator
    # def match_greedy_DEPRECIATED(self, plural_assign=False):
    #     """Algorithm that takes one best element at each iteration, based on sorted lists, not considering any alternatives."""
    #     # TODO consider opposite sorting (as we did in Gh), small chance but better result my occur
    #     demand_sorted = self.demand.copy(deep =True)
    #     supply_sorted = self.supply.copy(deep =True)
    #     #Change indices to integers for both demand and supply
    #     demand_sorted.index = np.array(range(len(demand_sorted.index)))
    #     supply_sorted.index = np.array(range(len(supply_sorted.index)))
    #     #sort the supply and demand
    #     #demand_sorted.sort_values(by=['Length', 'Area'], axis=0, ascending=False, inplace = True)
    #     demand_sorted.sort_values(by=['Score'], axis=0, ascending=False, inplace = True)
    #     #supply_sorted.sort_values(by=['Is_new', 'Length', 'Area'], axis=0, ascending=True, inplace = True)
    #     supply_sorted.sort_values(by=['Is_new', 'Score'], axis=0, ascending=True, inplace = True) # FIXME Need to make this work "optimally"
    #     incidence_np = self.incidence.copy(deep=True).values      
    #     columns = self.supply.index.to_list()
    #     rows = self.demand.index.to_list()
    #     min_length = self.demand.Length.min() # the minimum lenght of a demand element
    #     for demand_tuple in demand_sorted.itertuples():            
    #         match=False
    #         logging.debug("-- Attempt to find a match for %s", demand_tuple.Index)                
    #         for supply_tuple in supply_sorted.itertuples():                 
    #             if incidence_np[demand_tuple.Index,supply_tuple.Index]:           
    #                 match=True
    #                 self.add_pair(rows[demand_tuple.Index], columns[supply_tuple.Index])
    #             if match:
    #                 new_length = supply_tuple.Length - demand_tuple.Length
    #                 if plural_assign and new_length >= min_length:                    
    #                     # shorten the supply element:
    #                     supply_sorted.loc[supply_tuple.Index, "Length"] = new_length
    #                     temp_row = supply_sorted.loc[supply_tuple.Index].copy(deep=True)
    #                     temp_row['LCA'] = temp_row.Length * temp_row.Area * lca.TIMBER_REUSE_GWP
    #                     supply_sorted.drop(supply_tuple.Index, axis = 0, inplace = True)
    #                     #new_ind = supply_sorted['LCA'].searchsorted([False ,temp_row['LCA']], side = 'left') #get index to insert new row into #TODO Can this be sorted also by 'Area' and any other constraint?
    #                     new_ind = supply_sorted[supply_sorted['Is_new'] == False]['LCA'].searchsorted(temp_row['LCA'], side = 'left')
    #                     part1 = supply_sorted[:new_ind].copy(deep=True)
    #                     part2 = supply_sorted[new_ind:].copy(deep=True)
    #                     supply_sorted = pd.concat([part1, pd.DataFrame(temp_row).transpose().infer_objects(), part2]) #TODO Can we make it simpler
    #                     new_incidence_col = self.evaluate_column(new_length, "Length", self.constraints['Length'], incidence_np[:, supply_tuple.Index])
    #                     #new_incidence_col = self.evaluate_column(supply_tuple.Index, new_length, "Length", self.constraints["Length"], incidence_np[:, supply_tuple.Index])
    #                     #incidence__np[:, columns.index(supply_tuple.Index)] = new_incidence_col
    #                     #incidence_copy.loc[:, columns[supply_tuple.Index]] = new_incidence_col #TODO If i get the indicies to work. Try using this as an np array instead of df.
    #                     incidence_np[:,supply_tuple.Index] = new_incidence_col
    #                     logging.debug("---- %s is a match, that results in %s m cut.", supply_tuple.Index, supply_tuple.Length)
    #                 else:
    #                     #self.result += calculate_lca(supply_row.Length, supply_row.Area, is_new=supply_row.Is_new)
    #                     logging.debug("---- %s is a match and will be utilized fully.", supply_tuple.Index)
    #                     supply_sorted.drop(supply_tuple.Index, inplace = True)
    #                 break     
    #         else:
    #             logging.debug("---- %s is not matching.", supply_tuple.Index)

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
    def match_bipartite_plural(self):
        """Match using Maximum Bipartite Graphs. A maximum matching is a set of edges such that each vertex is
        incident on at most one matched edge and the weight of such edges in the set is as large as possible.
        
        Runs Maximum Bipartite Matching once, cuts the matches if possible and runs Maximum Biparite Matching once more
        """
        self.add_graph()
        if self.graph.is_connected():
            # TODO separate disjoint graphs for efficiency
            logging.info("graph contains unconnected subgraphs that could be separated")
        bipartite_matching = ig.Graph.maximum_bipartite_matching(self.graph, weights=self.graph.es["label"])
        
        #Store the original data
        original_supply = self.supply.copy()
        original_demand = self.demand.copy()
        original_weights = self.weights.copy()
        original_incidence = self.incidence.copy()

        weights_np = list(self.weights.to_numpy())
        weights_np_trans = list(np.transpose(weights_np))
        incidence_np_trans = list(np.transpose(self.incidence.to_numpy()))
        supply_np = list(self.supply.to_numpy())
        weights_columns = list(self.weights.columns)
        weights_rows = list(self.weights.index)
        score_index = list(self.supply.columns).index("Score")
        any_cutoff_found = False

        #Iterate through matches
        for match_edge in bipartite_matching.edges():
            demand_name = match_edge.source_vertex["label"]
            supply_name = match_edge.target_vertex["label"]
            if "N" in supply_name: #Skip if a New element is found
                continue
            demand_index = weights_rows.index(demand_name)
            supply_index = weights_columns.index(supply_name)
            
            new_score = supply_np[supply_index][score_index] - weights_np[demand_index][supply_index]
            if new_score > 0:
                any_cutoff_found = True
                row_copy = supply_np[supply_index].copy()
                row_copy[score_index] = new_score
                supply_np[supply_index][score_index] = weights_np[demand_index][supply_index]
                #Calculate weights and incidence for the cutoff-elements
                cutoff_weights = list(map(lambda x: hm.remove_alternatives(x, row_copy[score_index]), weights_np_trans[supply_index]))
                cutoff_incidence = list(map(lambda x: not(np.isnan(x)), cutoff_weights))
                #Update the supply weights and incidence after being cut
                updated_weights = list(map(lambda x: hm.remove_alternatives(x, supply_np[supply_index][score_index]), weights_np_trans[supply_index]))
                updated_incidence = list(map(lambda x: not(np.isnan(x)), updated_weights))
                weights_np_trans[supply_index] = updated_weights
                incidence_np_trans[supply_index] = updated_incidence
                
                incidence_np_trans.append(np.array(cutoff_incidence))
                weights_np_trans.append(np.array(cutoff_weights))
                supply_np.append(np.array(row_copy))
                weights_columns.append(supply_name + "C")

        if any_cutoff_found: #Add cutoffs and run the algorithm one more time. If not, the algorithm is NOT rerunned
            #Creating dataframes
            weights_np = np.transpose(weights_np_trans)
            incidence_np = np.transpose(incidence_np_trans)

            #Editing the self-dataframes
            self.weights = pd.DataFrame(weights_np, index = weights_rows, columns = weights_columns)
            self.incidence = pd.DataFrame(incidence_np, index = weights_rows, columns = weights_columns)
            self.supply = pd.DataFrame(supply_np, index = weights_columns, columns = list(self.supply.columns))
            #Evaluate new possible matches and run Maximum Bipartite Matching once more
            self.add_graph()
            bipartite_matching = ig.Graph.maximum_bipartite_matching(self.graph, weights=self.graph.es["label"])

            #Reset the dataframes to the originals without any cutoff
            self.supply = original_supply
            self.demand = original_demand
            self.weights = original_weights
            self.incidence = original_incidence

        #Extract the matches
        for match_edge in bipartite_matching.edges():
            demand_index = match_edge.source_vertex["label"]
            supply_index = match_edge.target_vertex["label"]
            if "C" in supply_index:
                c_indx = supply_index.index("C")
                supply_index = supply_index[:c_indx] #remove the "C" from the cut-off-elements
            self.add_pair(demand_index, supply_index)

    @_matching_decorator
    def match_bipartite_plural_multiple(self):
        """Match using Maximum Bipartite Graphs. A maximum matching is a set of edges such that each vertex is
        incident on at most one matched edge and the weight of such edges in the set is as large as possible.
        
        Runs Maximum Bipartite Matching once, cuts the matches if possible and runs Maximum Biparite Matching once more
        """
        self.add_graph()
        if self.graph.is_connected():
            # TODO separate disjoint graphs for efficiency
            logging.info("graph contains unconnected subgraphs that could be separated")
        bipartite_matching = ig.Graph.maximum_bipartite_matching(self.graph, weights=self.graph.es["label"])
        
        #Store the original data
        original_supply = self.supply.copy()
        original_demand = self.demand.copy()
        original_weights = self.weights.copy()
        original_incidence = self.incidence.copy()

        weights_np = list(self.weights.to_numpy())
        weights_np_trans = list(np.transpose(weights_np))
        incidence_np = list(self.incidence.to_numpy())
        incidence_np_trans = list(np.transpose(incidence_np))
        supply_np = list(self.supply.to_numpy())
        weights_columns = list(self.weights.columns)
        weights_rows = list(self.weights.index)
        score_index = list(self.supply.columns).index("Score")

        any_cutoff_found = True
        iteration = 0
        while any_cutoff_found:
            any_cutoff_found = False
            
            #Iterate through matches
            for match_edge in bipartite_matching.edges():
                demand_name = match_edge.source_vertex["label"]
                supply_name = match_edge.target_vertex["label"]
                if "N" in supply_name: #Skip if a New element is found
                    continue
                demand_index = weights_rows.index(demand_name)
                supply_index = weights_columns.index(supply_name)

                new_score = supply_np[supply_index][score_index] - weights_np[demand_index][supply_index]
                if new_score > 0:
                    any_cutoff_found = True
                    row_copy = supply_np[supply_index].copy()
                    row_copy[score_index] = new_score
                    supply_np[supply_index][score_index] = weights_np[demand_index][supply_index]
                    #Calculate weights and incidence for the cutoff-elements
                    cutoff_weights = list(map(lambda x: hm.remove_alternatives(x, row_copy[score_index]), weights_np_trans[supply_index]))
                    cutoff_incidence = list(map(lambda x: not(np.isnan(x)), cutoff_weights))
                    #Update the supply weights and incidence after being cut
                    updated_weights = list(map(lambda x: hm.remove_alternatives(x, supply_np[supply_index][score_index]), weights_np_trans[supply_index]))
                    updated_incidence = list(map(lambda x: not(np.isnan(x)), updated_weights))
                    weights_np_trans[supply_index] = updated_weights
                    incidence_np_trans[supply_index] = updated_incidence

                    incidence_np_trans.append(np.array(cutoff_incidence))
                    weights_np_trans.append(np.array(cutoff_weights))
                    supply_np.append(np.array(row_copy))
                    if "C" not in supply_name:
                        weights_columns.append(supply_name + f"C{iteration}")
                    else:
                        cutoff_name = supply_name[:supply_name.index("C") + 1] + f"_{iteration}"
                        weights_columns.append(cutoff_name)
            if any_cutoff_found:
                weights_np = np.transpose(weights_np_trans)
                incidence_np = np.transpose(incidence_np_trans)

                #Editing the self-dataframes
                self.weights = pd.DataFrame(weights_np, index = weights_rows, columns = weights_columns)
                self.incidence = pd.DataFrame(incidence_np, index = weights_rows, columns = weights_columns)
                self.supply = pd.DataFrame(supply_np, index = weights_columns, columns = list(self.supply.columns))

                #Evaluate new possible matches and run Maximum Bipartite Matching once more
                self.add_graph()
                bipartite_matching = ig.Graph.maximum_bipartite_matching(self.graph, weights=self.graph.es["label"])
            
            iteration += 1
        #Reset the dataframes to the originals without any cutoff
        self.supply = original_supply
        self.demand = original_demand
        self.weights = original_weights
        self.incidence = original_incidence

        #Extract the matches
        for match_edge in bipartite_matching.edges():
            demand_index = match_edge.source_vertex["label"]
            supply_index = match_edge.target_vertex["label"]
            if "C" in supply_index:
                c_indx = supply_index.index("C")
                supply_index = supply_index[:c_indx] #remove the "C" from the cut-off-elements
            self.add_pair(demand_index, supply_index) 

    @_matching_decorator
    def match_genetic_algorithm(self):
        """Genetic algorithm with the initial population with random solutions (not necessary an actual solution)"""

        number_of_demand_elements = len(self.demand)
        weights_new = hm.transform_weights(self.weights) #Create a new weight matrix with only one column representing all new elements
        weights_1d_array = weights_new.to_numpy().flatten()
        weights = np.array_split(weights_1d_array, number_of_demand_elements)
        max_weight = np.max(weights_1d_array[~np.isnan(weights_1d_array)])
        supply_names = weights_new.columns
        chromosome_length = len(supply_names) * len(self.demand)
        requested_number_of_chromosomes = len(supply_names)**2

        #Initializing a random population
        initial_population = np.array(([[random.randint(0,1) for x in range(chromosome_length)] for y in range(requested_number_of_chromosomes)]))
        solutions_per_population = len(initial_population)

        def fitness_func(solution, solution_idx):
            """Fitness function to calculate fitness value of chromosomes
            Genetic algorithm expects a maximization fitness function => when we are minimizing lca we must divide by 1/LCA
            
            Args:
                solution (list): a list of integers representing the solution of the matching
                solution_idx (int): the index of the solution

            Returns:
                float: the fitness of the solution
            """
            fitness = 0
            reward = 0
            solutions = np.array_split(solution, number_of_demand_elements)
            penalty = -max_weight
            indexes_of_matches = []
            for i in range(len(solutions)):
                num_matches_in_bracket = 0
                for j in range(len(solutions[i])):
                    if solutions[i][j] == 1:
                        if np.isnan(weights[i][j]): #Element cannot be matched => penalty
                            fitness += penalty #Penalty
                        else:
                            reward += weights[i][j] #score of match
                            num_matches_in_bracket += 1
                            new_element_index = len(solutions[i])-1
                            if not j == new_element_index: #Means that a supply element (not a new element) is matched with a demand element
                                indexes_of_matches.append(j)
                            
                if num_matches_in_bracket > 1:
                    fitness += 10*penalty #Penalty for matching multiple supply elemenets to the same demand element
                elif num_matches_in_bracket < 1:
                    fitness += penalty #Penalty for not matching at all
            
            index_duplicates = {x for x in indexes_of_matches if indexes_of_matches.count(x) > 1}
            if len(index_duplicates) > 0: #Means some supply elements are assigned the same demand element
                fitness = -10e10
            elif reward != 0:
                fitness += 100/reward
            return fitness
           
        #Using pygad-module
        """Parameters are set by use of trial and error. These parameters have given a satisfactory solution"""
        ga_instance = pygad.GA(
            initial_population=initial_population,
            num_generations=int((len(self.demand) + len(self.supply))),
            num_parents_mating=int(np.ceil(solutions_per_population/2)),
            #fitness_func=fitness_func_matrix,
            fitness_func = fitness_func,
            # binary representation of the problem with help from: https://blog.paperspace.com/working-with-different-genetic-algorithm-representations-python/
            # (also possible with: gene_space=[0, 1])
            #mutation_by_replacement=True,
            gene_type=int,
            parent_selection_type="sss",    # steady_state_selection() https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#steady-state-selection
            keep_elitism= int(np.ceil(solutions_per_population/2)),
            #keep_parents=-1, #-1 => keep all parents, 0 => keep none
            crossover_type="single_point",
            mutation_type = "random",
            #mutation_num_genes=int(solutions_per_population/5), Not needed if mutation_probability is set
            mutation_probability = 0.1,
            mutation_by_replacement=True,
            random_mutation_min_val=0,
            random_mutation_max_val=1,   # upper bound exclusive, so only 0 and 1
            #save_best_solutions=True, #Needs a lot of memory
            )

        ga_instance.run()
        logging.debug(ga_instance.initial_population)
        logging.debug(ga_instance.population)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        extracted_results = hm.extract_genetic_solution(weights_new, solution, number_of_demand_elements)
        for index, row in extracted_results.iterrows():
            self.add_pair(index, row["Matches from genetic"])

    @_matching_decorator
    def match_genetic_algorithm_DEPRECIATED(self):
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
                    cp_model.LinearExpr.Term(x[i,j], (self.demand.Score[i]*m_fac+1 - coeff_array[i,j]))
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
        
  
def run_matching(demand, supply, score_function_string, constraints = None, add_new = True, solution_limit = 120,
                bipartite = False, greedy_single = False, greedy_plural = False, genetic = False, milp = False, sci_milp = False, brute=False, brutevol2 = False,brutevol3=False,brutevol4=False, bipartite_plural = False, bipartite_plural_multiple = False):

    """Run selected matching algorithms and returns results for comparison.
    By default, bipartite, and both greedy algorithms are run. Activate and deactivate as wished."""

    #TODO Can **kwargs be used instead of all these arguments
    # create matching object 
    matching = Matching(demand=demand, supply=supply, score_function_string=score_function_string,constraints=constraints, add_new=add_new, multi = True, solution_limit=solution_limit)

    matches =[] # results to return
    headers = []
    if greedy_single:
        matching.match_greedy(plural_assign=False)
        matches.append({'Name': 'Greedy Algorithm','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if greedy_plural:
        matching.match_greedy(plural_assign=True)
        matches.append({'Name': 'Greedy Algorithm Plural', 'Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if bipartite:
        matching.match_bipartite_graph()
        matches.append({'Name': 'MBM', 'Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if milp:
        matching.match_mixed_integer_programming()
        matches.append({'Name': 'MILP','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if sci_milp:
        matching.match_scipy_milp()
        matches.append({'Name': 'Scipy MILP','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if genetic:
        matching.match_genetic_algorithm()
        matches.append({'Name': 'Genetic Algorithm','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    if brute:
        matching.match_brute()
        matches.append({'Name': 'Brute Force Approach','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})

    if bipartite_plural:
        matching.match_bipartite_plural()
        matches.append({'Name': 'MBM Plural','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})

    if bipartite_plural_multiple:
        matching.match_bipartite_plural_multiple()
        matches.append({'Name': 'MBM Plural Multiple','Match object': copy(matching), 'Time': matching.solution_time, 'PercentNew': matching.pairs.isna().sum()})
    return matches


if __name__ == "__main__":
    #DEMAND_JSON = sys.argv[1]
    #SUPPLY_JSON = sys.argv[2]
    #RESULT_FILE = sys.argv[3]
    DEMAND_JSON = r"MatchingAlgorithms\sample_demand_input.json"
    SUPPLY_JSON = r"MatchingAlgorithms\sample_supply_input.json"
    RESULT_FILE = r"MatchingAlgorithms\result.csv"
    
    constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='} # dictionary of constraints to add to the method
    demand, supply = hm.create_random_data(demand_count=8, supply_count=8)
    score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"
    result = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=True, greedy_plural = False, bipartite=False, genetic=True)
    simple_pairs = hm.extract_pairs_df(result)
    simple_results = hm.extract_results_df(result)
    print("Simple pairs:")
    print(simple_pairs)
    print()
    print("Simple results")
    print(simple_results)