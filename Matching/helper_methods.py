import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import logging
import helper_methods_LCA as lca
import random
from datetime import date
import helper_methods_plotting as plot


import json
# Opening JSON file
with open('Matching\Data\LCA_data.json') as json_file:
    data = json.load(json_file)

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
    """Transform the weight matrix to only contain one column with new elements in stead of one column for each new element

    Args:
        DataFrame: weight matrix

    Returns:
        DataFrame: weight matrix
    """
    weights = weights.copy(deep = True)
    cols=list(weights.columns)[-len(weights):]
    weights["N"]=weights[cols].sum(axis=1)
    weights = weights.drop(columns=cols)
    return weights

def create_random_data_demand(demand_count, demand_lat, demand_lon, new_lat, new_lon, demand_gwp=data["TIMBER_GWP"],gwp_price=data["GWP_PRICE"],new_price=data["NEW_ELEMENT_PRICE_TIMBER"], length_min = 1, length_max = 15.0, area_min = 0.15, area_max = 0.15):
    """Create two dataframes for the supply and demand elements used to evaluate the different matrices"""
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    demand = pd.DataFrame()
   
    # create element lenghts
    demand['Length'] = ((length_max/2 + 1) - length_min) * np.random.random_sample(size = demand_count) + length_min
    # create element areas independent of the length. Can change this back to Artur's method later, but I want to see the effect of even more randomness. 
    demand['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = demand_count) + area_min
    # intertia moment
    demand['Moment of Inertia'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    # height - assuming square cross sections
    #demand['Height'] = np.power(demand['Area'], 0.5)
    # gwp_factor

    #TODO: Must be moved to matching script:
    demand['Gwp_factor'] = demand_gwp
    #################################
    demand["Demand_lat"]=demand_lat
    demand["Demand_lon"]=demand_lon
    demand["Supply_lat"]=new_lat
    demand["Supply_lon"]=new_lon

    #TODO: Must be integrated in the matching script
    demand["Price"]=new_price
    demand["Gwp_price"]=gwp_price
    ##############################

    # Change index names
    demand.index = map(lambda text: 'D' + str(text), demand.index)
    return demand.round(4)

def create_random_data_supply(supply_count,demand_lat, demand_lon,supply_coords,supply_gwp=data["TIMBER_REUSE_GWP"],gwp_price=data["GWP_PRICE"],reused_price=data["REUSED_ELEMENT_PRICE_TIMBER"], length_min = 1, length_max = 15.0, area_min = 0.15, area_max = 0.15):
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    supply = pd.DataFrame()
    supply['Length'] = ((length_max + 1) - length_min) * np.random.random_sample(size = supply_count) + length_min
    supply['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = supply_count) + area_min
    supply['Moment of Inertia'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    supply['Height'] = np.power(supply['Area'], 0.5)
    supply['Gwp_factor'] = supply_gwp
    supply["Demand_lat"]=demand_lat
    supply["Demand_lon"]=demand_lon
    supply["Location"]=0
    supply["Supply_lat"]=0
    supply["Supply_lon"]=0
    supply["Price"]=reused_price
    supply["Gwp_price"]=gwp_price
    
    for row in range(len(supply)):
        lokasjon=random.randint(0, len(supply_coords)-1)
        supply.loc[row,"Supply_lat"]=supply_coords.loc[lokasjon,"Lat"]
        supply.loc[row,"Supply_lon"]=supply_coords.loc[lokasjon,"Lon"]
        supply.loc[row,"Location"]=supply_coords.loc[lokasjon,"Place"]
    supply.index = map(lambda text: 'S' + str(text), supply.index)

    return supply.round(4)


def create_random_data_demand_conference(requested_tonnes, length_min, length_max):
    """Generates a demand dataset that has approximately the requested number of tonnes of steel

    Args:
        requested_tonnes (float): desired total number of tonnes steel in the dataset
        length_min (float): minimum element length
        length_max (float): maxmimum element length

    Returns:
        DataFrame: demand dataset
    """
    #Available steel sections with corresponding area and moment of inertia
    steel_cs = {"CHS 457x40": (5.2402e-2, 1.149e-3, 0.4114),# (area, moment of inertia, mass [tonne/m])
                "CHS 508x30": (3.7935e-2, 1.109e-3, 0.3536), 
                "CHS 610x30": (5.4664e-2, 2.305e-3, 0.4291),
                "CHS 813x30": (7.3789e-2, 5.664e-3, 0.5793),
                "CHS 1067x25": (8.1838e-2, 1.111e-2, 0.6424),
                "CHS 1219x25": (9.3777e-2, 1.6720e-2, 0.7361)
    }

    demand = pd.DataFrame(columns = ["Length", "Area", "Moment of Inertia", "Material", "Manufacturer", "Latitude", "Longitude"])
    material = "Steel"
    manufacturer = 0
    lat = 0
    lon = 0
    #Add random data
    tonne_used = 0.0
    while tonne_used < requested_tonnes:
        print(tonne_used)
        steel_idx = random.choice(list(steel_cs.keys()))
        length = np.random.uniform(length_min, length_max)
        area = steel_cs[steel_idx][0]
        moment = steel_cs[steel_idx][1]
        tonne = steel_cs[steel_idx][2] * length
        demand.loc[len(demand.index)] = [length, area, moment, material, manufacturer, lat, lon]
        tonne_used += tonne
    return demand

def create_random_data_supply_conference(requested_tonnes, length_min, length_max, supply_coords):
    """Generates a supply dataset that has approximately the requested number of tonnes of steel

    Args:
        requested_tonnes (float): desired total number of tonnes steel in the dataset
        length_min (float): minimum element length
        length_max (float): maxmimum element length
        supply_cords (DataFrame): contains the location name, latitude and longitude of the supply elements

    Returns:
        DataFrame: supply dataset
    """
    #Available steel sections with corresponding area and moment of inertia
    steel_cs = {"CHS 457x40": (5.2402e-2, 1.149e-3, 0.4114),# (area, moment of inertia, mass [tonne/m])
                "CHS 508x30": (3.7935e-2, 1.109e-3, 0.3536), 
                "CHS 610x30": (5.4664e-2, 2.305e-3, 0.4291),
                "CHS 813x30": (7.3789e-2, 5.664e-3, 0.5793),
                "CHS 1067x25": (8.1838e-2, 1.111e-2, 0.6424),
                "CHS 1219x25": (9.3777e-2, 1.6720e-2, 0.7361)
    }

    supply = pd.DataFrame(columns = ["Length", "Area", "Moment of Inertia", "Material", "Location", "Latitude", "Longitude"])
    material = "Steel"
    #Add random data
    tonne_used = 0.0
    while tonne_used < requested_tonnes:
        steel_idx = random.choice(list(steel_cs.keys()))
        loc_idx = lokasjon=random.randint(0, len(supply_coords)-1)
        loc = supply_coords.loc[loc_idx,"Location"]
        lat = supply_coords.loc[loc_idx,"Latitude"]
        lon = supply_coords.loc[lokasjon,"Longitude"]
        length = np.random.uniform(length_min, length_max)
        area = steel_cs[steel_idx][0]
        moment = steel_cs[steel_idx][1]
        tonne = steel_cs[steel_idx][2] * length
        supply.loc[len(supply.index)] = [length, area, moment, material, loc, lat, lon]
        tonne_used += tonne
    return supply

def extract_brute_possibilities(incidence_matrix):
    """Extracts all matching possibilities based on the incidence matrix.

    Args:
        Dataframe: incidence matrix

    Returns:
        list: three-dimensional list
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
    """Converts the best solution into a column containing the supply matches for each demand element. 
    This column is added to the weight-matrix in order to visually check if the matching makes sense

    Args:
        weights (DataFrame): weight matrix
        best_solution (list): list of integers containing the best solution from genetic algorithm
        number_of_demand_elements (int): number of demand elements

    Returns:
        DataFrame: The weight matrix with a new column containing the matches for each demand element
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
            weight_match = weights.loc[f"D{i}"][index[0]]
            match = weight_cols[index[0]]
            if match == "N" or np.isnan(weight_match): #New element is assigned if match is "N" or if the match is not allowed
                match = f"N{i}"
        match_column.append(match)
    result["Matches from genetic"] = match_column
    return result

def print_genetic_solution(weights, best_solution, number_of_demand_elements):
    """Prints the genetic solution in a readable way to visually evaluate if the solution makes sence. Used for debugging

    Args:
        weights (DataFrame): weight matrix
        best_solution (list): list of integers containing the best solution from genetic algorithm
        number_of_demand_elements (int): number of demand elements

    Returns:
        DataFrame: The weight matrix with a new column containing the matches for each demand element
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


def export_dataframe_to_xlsx(dataframe, file_location):
    """Exports a dataframe to a csv file

    Args:
        dataframe (DataFrame): dataframe
        file_location (string): location of the new file
    """
    dataframe.to_excel(file_location)

def export_dataframe_to_csv(dataframe, file_location):
    """Exports a dataframe to a csv file

    Args:
        dataframe (DataFrame): dataframe
        file_location (string): location of the new file
    """
    dataframe.to_csv(file_location)

def import_dataframe_from_file(file_location, index_replacer):
    """Creates a DataFrame from a file

    Args:
        file_location (string): filepath
        index_replacer (string): string to replace the index with, either "S" or "D"

    Returns:
        DataFrame: DataFrame generated from the file
    """

    if "xlsx" in file_location: #Excel file
        dataframe = pd.read_excel(file_location)
    else: #CSV file
        dataframe = pd.read_csv(file_location)
    if "Column1" in list(dataframe.columns):
        dataframe.drop(columns=["Column1"], inplace = True)
    if "Unnamed: 0" in list(dataframe.columns):
        dataframe.drop(columns=["Unnamed: 0"], inplace = True)
    dataframe.index = map(lambda text: index_replacer + str(text), dataframe.index)
    new_columns = {col: col.split('[')[0].strip() for col in dataframe.columns}
    dataframe = dataframe.rename(columns = new_columns)
    dataframe = dataframe.fillna(0.0) #Fill NaN-values with zeros
    return dataframe



def add_graph_plural(demand_matrix, supply_matrix, weight_matrix, incidence_matrix):
    """Add a graph notation based on incidence matrix
    
    THIS METHOD WAS NEVER USED
    """
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


def count_matches(matches, algorithm):
    """Counts the number of plural matches for each supply element

    Args:
        matches (Pandas Dataframe): return dataframe of hm.extract_pairs_df()
        algorithm (str): Name of algorithm in "matches"

    Returns:
        Pandas Dataframe: A count for each supply element
    """
    return matches.pivot_table(index = [algorithm], aggfunc = 'size')


def format_float(num):
    """
    Formats a float into a string with comma-separated thousands.
    Args:
        num (float): The number to format.
    Returns:
        str: The formatted string.
    """
    if isinstance(num, float):
        num = str(int(num)) if num.is_integer() else str(num)
    else:
        num = str(num)
    if len(num) <= 3:
        return num
    result = []
    for i, char in enumerate(reversed(num)):
        if i % 3 == 0 and i != 0:
            result.append(' ')
        result.append(char)
    return ''.join(reversed(result))

def generate_score_function_string(constants):
    """Generating the score function string for the matching

    Args:
        constants (dictionary): constants to use in the matching tool

    Returns:
        string: the score function string for evaluation of the weight matrix
    """
    metric = constants["Metric"]
    transportation = constants["Include transportation"]
    if metric == "GWP":
        score_function_string = f"@lca.calculate_lca(length=Length, area=Area, include_transportation={transportation}, distance = Distance, gwp_factor=Gwp_factor, transport_gwp = {constants['TRANSPORT_GWP']}, density = Density)"
    elif metric == "Combined":
        score_function_string = f"@lca.calculate_score(length=Length, area=Area, include_transportation = {transportation}, distance = Distance, gwp_factor = Gwp_factor, transport_gwp = {constants['TRANSPORT_GWP']}, price = Price, priceGWP = {constants['VALUATION_GWP']}, density = Density, price_transport = {constants['PRICE_TRANSPORTATION']})"
    elif metric == "Price":
        score_function_string = f"@lca.calculate_price(length=Length, area =Area, include_transportation = {transportation}, distance = Distance, price = Price, density = Density, price_transport= {constants['PRICE_TRANSPORTATION']})"
    return score_function_string

def fill_closest_manufacturer(dataframe, constants):
    """Fill the dataframe with the colsets manufacturer depending on the material

    Args:
        dataframe (DataFrame): dataframe (supply or demand)
        constants (dictionary): constants to use in the matching tool

    Returns:
        DataFrame: updated dataframe with coordinates of the closest manufacturer
    """

    manufacturers_df=pd.read_excel(r"./Data/CSV/manufacturers_locations.xlsx")

    shortest_distance_tree=10**10
    shortest_distance__steel=10**10

    shortest_tree={
        "Manufacturer": "place",
        "Latitude": "XX.XXX",
        "Longitude":"YY.YYY"
    }
    shortest_steel={
        "Manufacturer": "place",
        "Latitude": "XX.XXX",
        "Longitude":"YY.YYY"
    }
    for index,row in manufacturers_df.iterrows():
        driving_distance=lca.calculate_driving_distance(str(row["Latitude"]),str(row["Longitude"]),constants["Site latitude"],constants["Site longitude"])

        if str(row["Material"]) == "Timber" and driving_distance < shortest_distance_tree:
            shortest_distance_tree = driving_distance
            shortest_tree["Manufacturer"]=str(row["Manufacturer"])
            shortest_tree["Latitude"]=row["Latitude"]
            shortest_tree["Longitude"]=row["Longitude"]

        elif str(row["Material"])=="Steel" and driving_distance<shortest_distance__steel:
            shortest_distance__steel=driving_distance
            shortest_steel["Manufacturer"]=str(row["Manufacturer"])
            shortest_steel["Latitude"]=str(row["Latitude"])
            shortest_steel["Longitude"]=str(row["Longitude"])

    mask_tree=(dataframe["Material"]=="Timber") & (dataframe["Manufacturer"] == 0)
    dataframe.loc[mask_tree,["Manufacturer","Latitude","Longitude"]]=[shortest_tree["Manufacturer"],shortest_tree["Latitude"],shortest_tree["Longitude"]]
    mask_steel=(dataframe["Material"]=="Steel") & (dataframe["Manufacturer"] == 0)
    dataframe.loc[mask_steel,["Manufacturer","Latitude","Longitude"]]=[shortest_steel["Manufacturer"],shortest_steel["Latitude"],shortest_steel["Longitude"]]
    
    dataframe = dataframe.astype({'Latitude': float, 'Longitude': float})

    return dataframe


def generate_run_string(constants):
    """Generate the run-string required to use the method run_matching in matching.py

    Args:
        constants (dictionary): constants to use in the matching tool

    Returns:
        string: runstring
    """
    algorithms = constants["Algorithms"]
    run_string = "run_matching(demand, supply, score_function_string, constraints = constraint_dict, add_new = True"
    #Adding the user-selected algorithms to the run-string    
    if len(algorithms) != 0:
        for algorithm in algorithms:
            run_string += f", {algorithm} = True"
    run_string += ")"
    return run_string

print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")