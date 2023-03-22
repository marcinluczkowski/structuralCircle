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

def extract_results_df(dict_list, column_name):
    """Creates a dataframe with the scores from each method"""
    sub_df = []
    cols = []
    for run in dict_list:
        sub_df.append(run['Match object'].result)
        cols.append(run['Name'])
    df = pd.DataFrame(sub_df, index= cols)    
    df=df.rename(columns={0: column_name})
    return df.round(3)

def remove_alternatives(x, y):
    if x > y:
        return np.nan
    else:
        return x

def transform_weights(weights):
    """Transform the weight matrix to only contain one column with new elements in stead of one column for each new element"""
    weights = weights.copy(deep = True)
    cols=list(weights.columns)[-len(weights):]
    weights["N"]=weights[cols].sum(axis=1)
    weights = weights.drop(columns=cols)
    return weights

def create_random_data_demand(demand_count, demand_lat, demand_lon, new_lat, new_lon, demand_gwp=lca.TIMBER_GWP,gwp_price=lca.GWP_PRICE,new_price_per_m2=lca.NEW_ELEMENT_PRICE_TIMBER, length_min = 4, length_max = 15.0, area_min = 0.15, area_max = 0.25):
    """Create two dataframes for the supply and demand elements used to evaluate the different matrices"""
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    demand = pd.DataFrame()
   
    # create element lenghts
    demand['Length'] = ((length_max/2 + 1) - length_min) * np.random.random_sample(size = demand_count) + length_min
    # create element areas independent of the length. Can change this back to Artur's method later, but I want to see the effect of even more randomness. 
    demand['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = demand_count) + area_min
    # intertia moment
    demand['Inertia_moment'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    # height - assuming square cross sections
    demand['Height'] = np.power(demand['Area'], 0.5)
    # gwp_factor
    demand['Gwp_factor'] = demand_gwp
    demand["Demand_lat"]=demand_lat
    demand["Demand_lon"]=demand_lon
    demand["Supply_lat"]=new_lat
    demand["Supply_lon"]=new_lon
    demand["Price_per_m2"]=new_price_per_m2
    demand["Gwp_price"]=gwp_price

    # Change index names
    demand.index = map(lambda text: 'D' + str(text), demand.index)
    return demand.round(4)

def create_random_data_supply(supply_count,demand_lat, demand_lon,supply_coords,supply_gwp=lca.TIMBER_REUSE_GWP,gwp_price=lca.GWP_PRICE,reused_prise_per_m2=lca.REUSED_ELEMENT_PRICE_TIMBER, length_min = 4, length_max = 15.0, area_min = 0.15, area_max = 0.25):
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    supply = pd.DataFrame()
    supply['Length'] = ((length_max + 1) - length_min) * np.random.random_sample(size = supply_count) + length_min
    supply['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = supply_count) + area_min
    supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    supply['Height'] = np.power(supply['Area'], 0.5)
    supply['Gwp_factor'] = supply_gwp
    supply["Demand_lat"]=demand_lat
    supply["Demand_lon"]=demand_lon
    supply["Location"]=0
    supply["Supply_lat"]=0
    supply["Supply_lon"]=0
    supply["Price_per_m2"]=reused_prise_per_m2
    supply["Gwp_price"]=gwp_price
    
    for row in range(len(supply)):
        lokasjon=random.randint(0, len(supply_coords)-1)
        supply.loc[row,"Supply_lat"]=supply_coords.loc[lokasjon,"Lat"]
        supply.loc[row,"Supply_lon"]=supply_coords.loc[lokasjon,"Lon"]
        supply.loc[row,"Location"]=supply_coords.loc[lokasjon,"Place"]
    supply.index = map(lambda text: 'S' + str(text), supply.index)

    return supply.round(4)


def extract_brute_possibilities(incidence_matrix):
    """Extracts all demand matching possibilities from incidence matrix.
    
    returns a 3D list where each outer list contains possibilities for each row based on incidence matrix.
    """
    binary_incidence = incidence_matrix*1 #returnes incidence matrix with 1 and 0 instead od True/False
    
    three_d_list=[]
    incidence_list=binary_incidence.values.tolist()
    for row in incidence_list:
        rowlist=[]
        for i in range(len(row)):
            if row[i]==1:
                newlist=[0]*len(row)
                newlist[i]=1
                rowlist.append(newlist)
        three_d_list.append(rowlist)
    return three_d_list 

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

def extract_genetic_solution(weights, best_solution, number_of_demand_elements):
    """Converts the best solution a column containing the supply matches for each demand element. 
    This column is added to the weight-matrix in order to visually check if the matching makes sense
    - weights: Pandas Dafarame
    - best_solution: 1d-list with 0's and 1's
    - number_of_demand_elements: integer

    Returns a pandas dataframe
    """
    result = weights.copy(deep = True)
    buckets = np.array_split(best_solution, number_of_demand_elements)
    weight_cols = weights.columns.values.tolist()
    match_column = []
    for i in range(len(buckets)):
        index = np.where(buckets[i] == 1)[0] #Finding matches
        if len(index) == 0 or len(index) > 1: #This happens either if a match is not found or multiple supply-elements are matched with the same demand elemend => invalid solution
            match = f"N{i}" #Set match to New element. 
            if len(index) > 1:
                logging.info("OBS: Multiple supply matched with one demand")
        else:
            match = weight_cols[index[0]]
            if match == "N":
                match = f"N{i}"
        match_column.append(match)
    result["Matches from genetic"] = match_column
    return result
            
def print_genetic_solution(weights, best_solution, number_of_demand_elements):
    """Print the genetic solution in a readable way to visually evaluate if the solution makes sence. Used for debugging
    - weights: Pandas Dafarame
    - best_solution: 1d-list with 0's and 1's
    - number_of_demand_elements: integer

    Returns a pandas dataframe
    """
    result = weights.copy(deep = True)
    buckets = np.array_split(best_solution, number_of_demand_elements)
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


def create_initial_population_genetic(binary_incidence, size_of_population, include_invalid_combinations):
    """Creating initial population based valid solutions from the incidence matrix
    Good, but itertools.product has very long runtime!!!
    - binary incidence: Pandas dataframe
    - size_of_population: Integer
    - include_invalid_combinations: Boolean
    
    Returns an initial population as a nested list containing 0's and 1's
    """

    three_d_list=[]
    incidence_list=binary_incidence.values.tolist()
    valid_solutions = []
    #Creates a 3d-list containing all possible locations of matches based on the incidence matrix
    for row in incidence_list:
        rowlist=[]
        for i in range(len(row)):
            if row[i]==1:
                newlist=[0]*len(row)
                newlist[i]=1
                rowlist.append(newlist)
        three_d_list.append(rowlist)

    all_possible_solutions = list(itertools.product(*three_d_list)) #EXTREMELY LONG RUNTIME

    if include_invalid_combinations: #Include invalid solutions, such as one demand element is matched with multiple supply elements
        if len(all_possible_solutions) < size_of_population:
            samples = all_possible_solutions
        else:
            samples = random.sample(all_possible_solutions, size_of_population) #Takes n random elements from the list containing all possible solutions
        #initial_population = map(lambda x,y: x.append(y.flatten()), initial_population, samples)
        initial_population = [sum(list(x), []) for x in samples]

    else: #Only valid solutions are included in the initial population
        number_of_possible_solutions = len(all_possible_solutions)
        sample_size = min(int(np.sqrt(number_of_possible_solutions*np.log10(number_of_possible_solutions))), size_of_population*100)
        if len(all_possible_solutions) <= sample_size: #If number of possible solutions is smaller than the sample size
            solutions = all_possible_solutions
        else:
            solutions = random.sample(all_possible_solutions, sample_size) #Takes n random elements from the list containing all possible solutions
        
        #Evaluate if the samples are a valid solution or not
        for subset in solutions:
            column_sum = np.sum(list(subset), axis = 0)[:-1] #All sums of columns except the "New"-column
            invalid_solution = len([*filter(lambda x: x > 1, column_sum)]) > 0
            if not invalid_solution:
                valid_solutions.append(sum(list(subset), [])) #Appen valid solutions
            
        if len(valid_solutions) < size_of_population:
            initial_population = valid_solutions
        else:
            initial_population = random.sample(valid_solutions, size_of_population)#Takes n random elements from the list containing all possible solutions
    return initial_population

    

def create_random_population_genetic(chromosome_length, requested_population_size, probability_of_0, probability_of_1):
    """ Creates a random population with a given probability of having 0 or 1 for each gene
    - chromosome_length: integer
    - requested_population_size: integer
    - probability_of_0: float
    - probability_of_1: float

    NOTE: The probabilities must sum to 1!
    
    Returns an initial population as a nested list containing 0's and 1's
    """

    initial_population = []
    count = 0
    for i in range(requested_population_size*100):
        solution = [np.random.choice([0,1], p = [probability_of_0, probability_of_1]) for x in range(chromosome_length)]
        if solution not in initial_population:
            initial_population.append(solution)
            count += 1
        if count == requested_population_size:
            break
    return initial_population

def add_graph_plural(demand_matrix, supply_matrix, weight_matrix, incidence_matrix):
    """Add a graph notation based on incidence matrix"""
    vertices = [0]*len(demand_matrix.index) + [1]*len(supply_matrix.index)
    num_rows = len(demand_matrix.index)
    edges = np.transpose(np.where(incidence_matrix))
    edges = [[edge[0], edge[1]+num_rows] for edge in edges]
    edge_weights = weight_matrix.to_numpy().reshape(-1,)
    edge_weights = edge_weights[~np.isnan(edge_weights)]
    # We need to reverse the weights, so that the higher the better. Because of this edge weights are initial score minus the replacement score:
    edge_weights = (np.array([demand_matrix.Score[edge[0]] for edge in edges ])+0.0000001) - edge_weights 
    # assemble graph
    graph = ig.Graph.Bipartite(vertices, edges)
    graph.es["label"] = edge_weights
    graph.vs["label"] = list(demand_matrix.index)+list(supply_matrix.index) #vertice names
    return graph

#def add_cutt_off

def count_matches(matches, algorithm):
    return matches.pivot_table(index = [algorithm], aggfunc = 'size')


print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")