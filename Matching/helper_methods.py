import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import logging
import LCA as lca
import itertools
import random

# ==== HELPER METHODS ====
# This file contains various methods used for testing and development. 

def extract_pairs_df(dict_list):
    """Creates a dataframe with the matching pairs all evaluated matching methods. 
    input: list of dictionaries for each matching method
    output: dataframe with the matching pairs of all methods."""
    sub_df = []
    cols = []
    for run in dict_list:
        sub_df.append(run['Match object'].pairs)
        cols.append(run['Name'])
    df = pd.concat(sub_df, axis = 1)
    df.columns = cols
    return df

def extract_results_df(dict_list):
    """Creates a dataframe with the scores from each method"""
    sub_df = []
    cols = []
    for run in dict_list:
        sub_df.append(run['Match object'].result)
        cols.append(run['Name'])
    
    df = pd.DataFrame(sub_df, index= cols)    
    df=df.rename(columns={0:"LCA"})
    return df.round(3)

def remove_alternatives(x, y):
    if x > y:
        return np.nan
    else:
        return x

# def extract_LCA_new(dict_list):
#     matchobj=dict_list[0]["Match object"]
#     sum=matchobj.demand["LCA"].sum()
#     return sum

def transform_weights(weights):
    """Transform the weight matrix to only contain one column with new elements"""
    weights = weights.copy(deep = True)
    cols=list(weights.columns)[-len(weights):]
    weights["N"]=weights[cols].sum(axis=1)
    weights = weights.drop(columns=cols)
    return weights


def create_random_data(demand_count, supply_count, demand_gwp=lca.TIMBER_GWP, supply_gwp=lca.TIMBER_REUSE_GWP, length_min = 4, length_max = 15.0, area_min = 0.15, area_max = 0.25):
    """Create two dataframes for the supply and demand elements used to evaluate the different matrices"""
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    demand = pd.DataFrame()
    supply = pd.DataFrame()
    # create element lenghts
    demand['Length'] = ((length_max/2 + 1) - length_min) * np.random.random_sample(size = demand_count) + length_min
    supply['Length'] = ((length_max + 1) - length_min) * np.random.random_sample(size = supply_count) + length_min
    # create element areas independent of the length. Can change this back to Artur's method later, but I want to see the effect of even more randomness. 
    demand['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = demand_count) + area_min
    supply['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = supply_count) + area_min
    # intertia moment
    demand['Inertia_moment'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    # height - assuming square cross sections
    demand['Height'] = np.power(demand['Area'], 0.5)
    supply['Height'] = np.power(supply['Area'], 0.5)
    # gwp_factor
    demand['Gwp_factor'] = demand_gwp
    supply['Gwp_factor'] = supply_gwp
    # Change index names
    demand.index = map(lambda text: 'D' + str(text), demand.index)
    supply.index = map(lambda text: 'S' + str(text), supply.index)
    return demand.round(2), supply.round(2)


def display_graph(matching, graph_type='rows', show_weights=True, show_result=True):
    """Plot the graph and matching result"""
    if not matching.graph:
        matching.add_graph()
    weight = None
    if show_weights:
        # weight = list(np.absolute(np.array(self.graph.es["label"]) - 8).round(decimals=2)) 
        weight = list(np.array(matching.graph.es["label"]).round(decimals=2)) 
    edge_color = None
    edge_width = matching.graph.es["label"]
    if show_result and not matching.pairs.empty:
        edge_color = ["gray"] * len(matching.graph.es)
        edge_width = [0.7] * len(matching.graph.es)
        # TODO could be reformatted like this https://igraph.readthedocs.io/en/stable/tutorials/bipartite_matching.html#tutorials-bipartite-matching
        not_found = 0
        for index, pair in matching.pairs.iterrows():
            source = matching.graph.vs.find(label=index) 
            try:
                target = matching.graph.vs.find(label=pair['Supply_id'])
                edge = matching.graph.es.select(_between = ([source.index], [target.index]))
                edge_color[edge.indices[0]] = "black" #"red"
                edge_width[edge.indices[0]] = 2.5
            except ValueError:
                not_found+=1
        if not_found > 0:
            logging.error("%s elements not found - probably no new elements supplied.", not_found)
    vertex_color = []
    for v in matching.graph.vs:
        if 'D' in v['label']:
            vertex_color.append("lightgray")
        elif 'S' in v['label']:
            vertex_color.append("slategray")
        else:
            vertex_color.append("pink")
    layout = matching.graph.layout_bipartite()
    if graph_type == 'rows':
        layout = matching.graph.layout_bipartite()
    elif graph_type == 'circle':
        layout = matching.graph.layout_circle()
    if matching.graph:
        fig, ax = plt.subplots(figsize=(15, 10))
        ig.plot(
            matching.graph,
            target=ax,
            layout=layout,
            vertex_size=0.4,
            vertex_label=matching.graph.vs["label"],
            palette=ig.RainbowPalette(),
            vertex_color=vertex_color,
            edge_width=edge_width,
            edge_label=weight,
            edge_color=edge_color,
            edge_curved=0.15
        )
        plt.show()

def extract_genetic_solution(weights, best_solution, num_buckets):
    #columns = weights.columns.values.tolist()
    #index_first_new = weights.columns.values.tolist().index("N0")
    #supply_names_only_reuse = columns[:index_first_new]

    #result = weights[supply_names_only_reuse].copy(deep = True)
    result = weights.copy(deep = True)
    buckets = np.array_split(best_solution, num_buckets)
    demands = weights.index
    weight_cols = weights.columns.values.tolist()
    match_column = []
    for i in range(len(buckets)):
        index = np.where(buckets[i] == 1)[0] #Finding matches
        if len(index) == 0 or len(index) > 1: #This happens either if a match is found or multiple supply-elements are matched with the same demand elemend => invalid solution
            match = f"N{i}" #Set match to New element. 
            if len(index) > 1:
                logging.info("OBS: Multiple supply matched with one demand")
        else:
            match = weight_cols[index[0]]
            if match == "N":
                match = f"N{i}"
            #match = [weight_cols[x] for x in index]
        match_column.append(match)
    result["Matches from genetic"] = match_column
    return result
            
def print_genetic_solution(weights, best_solution, num_buckets):
    result = weights.copy(deep = True)
    buckets = np.array_split(best_solution, num_buckets)
    demands = weights.index
    weight_cols = weights.columns.values.tolist()
    weight_cols = list(map(lambda x: x.replace("N0", "N"), weight_cols))
    weight_cols.append("N0")
    match_column = []
    for i in range(len(buckets)):
        index = np.where(buckets[i] == 1)[0] #Finding matches
        if len(index) == 0:
            match = ["No match"]
        else:
            match = [weight_cols[x] for x in index]
        match_column.append(match)
    result["Matches from genetic"] = match_column
    return result

def create_initial_population_genetic(binary_incidence, weights, size_of_population):
    three_d_list=[]
    all_possible_solutions = []
    incidence_list=binary_incidence.values.tolist()
    count = 0
    #Creates a 3d list containing all possible locations of matches based on the incidence matrix
    for row in incidence_list:
        rowlist=[]
        for i in range(len(row)):
            if row[i]==1:
                newlist=[0]*len(row)
                newlist[i]=1
                rowlist.append(newlist)
        three_d_list.append(rowlist)
    for subset in itertools.product(*three_d_list):
            #subset_df=pd.DataFrame(data=list(subset),index=weights.index,columns=weights.columns)
            #sum=subset_df.sum()
            #invalid_solution=(sum>1).any()
            column_sum = np.sum(list(subset), axis = 0)[:-1] #All sums of columns except the "New"-column
            invalid_solution = len([*filter(lambda x: x > 1, column_sum)]) > 0
            if not invalid_solution:
                all_possible_solutions.append(sum(list(subset), []))
                count += 1
    random.shuffle(all_possible_solutions)
    return all_possible_solutions[:size_of_population]

def create_random_population_genetic(chromosome_length, requested_population_size):
    initial_population = []
    for i in range(requested_population_size):
        solution = [np.random.choice([0,1], p = [0.75, 0.25]) for x in range(chromosome_length)]
        if solution not in initial_population:
            initial_population.append(solution)
    return initial_population
    

    



                

                




print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")