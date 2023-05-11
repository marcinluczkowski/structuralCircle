import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import logging
import LCA as lca
import random
from fpdf import FPDF
from datetime import date
import plotting as plot


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

def extract_results_df_pdf(dict_list, constants):
    """Creates a dataframe with the scores from each method"""
    sub_df = {"Names": [], "Score": [], "Time": [], "Substitutions": [], "Sub_percent": []}
    cols = []
    used_constants = {"Density timber": (constants["TIMBER_DENSITY"], "kg/m^3"), "Density steel": (constants["STEEL_DENSITY"], "kg/m^3")}
    metric = constants["Metric"]
    include_transportation = constants["Include transportation"]
    #Get performance of all algorithms runned:
    for run in dict_list:
        sub_df["Score"].append(round(run['Match object'].result, 2))
        sub_df["Time"].append(run["Time"])
        num_subs = len(run['Match object'].pairs[run['Match object'].pairs["Supply_id"].str.startswith("S")])
        sub_df["Substitutions"].append(num_subs)
        sub_df["Sub_percent"].append(round(num_subs/len(run["Match object"].demand)*100, 2))
        sub_df["Names"].append(run["Name"])
        cols.append(run['Name'])
    algorithms_df = pd.DataFrame(sub_df, index= cols)   
    algorithms_df = algorithms_df.sort_values(by=["Score", 'Time'], ascending=[True, True])
    results_dict = algorithms_df.iloc[0].to_dict()
    results_dict["Algorithm"] = algorithms_df.iloc[0].name
    results_dict["Performance"] = algorithms_df
    #Get the pair of the best matching
    index_algorithm = next(filter(lambda i: dict_list[i]['Name'] == algorithms_df.iloc[0].name, range(len(dict_list))))
    match_object = dict_list[index_algorithm]["Match object"]
    all_new_score = match_object.demand["Score"].sum()
    all_new_transport = match_object.demand["Transportation"].sum()
    results_dict["All new score"] = round(all_new_score, 2)
    

    results_dict.update(constants)
    if metric == "GWP":
        results_dict["Unit"] = "kg CO2 equivalents"
        used_constants.update({"GWP new timber": (constants["TIMBER_GWP"],"kg C02 equivalents"), "GWP reused timber": (constants["TIMBER_REUSE_GWP"], "kg C02 equivalents"), "GWP new steel": (constants["STEEL_GWP"], "kg C02 equivalents"), "GWP reused steel": (constants["STEEL_REUSE_GWP"], "kg C02 equivalents")})
    elif metric == "Price":
        results_dict["Unit"] = "kr"
        used_constants.update({"Price new timber": (constants["TIMBER_PRICE"], "kr/m^3"), "Price reused timber": (constants["TIMBER_REUSE_PRICE"], "kr/m^3"), "Price new steel": (constants["STEEL_REUSE_PRICE"], "kr/m^3"), "Price reused steel": (constants["STEEL_REUSE_PRICE"], "kr/m^3")})
    elif metric == "Combined":
        results_dict["Unit"] = "kr"
        used_constants.update({"GWP new timber": (constants["TIMBER_GWP"],"kg C02 equivalents"), "GWP reused timber": (constants["TIMBER_REUSE_GWP"], "kg C02 equivalents"), "GWP new steel": (constants["STEEL_GWP"], "kg C02 equivalents"), "GWP reused steel": (constants["STEEL_REUSE_GWP"], "kg C02 equivalents"), "Valuation of GWP": (constants["VALUATION_GWP"], "kr/kg C02 equivalents")})
        used_constants.update({"Price new timber": (constants["TIMBER_PRICE"], "kr/m^3"), "Price reused timber": (constants["TIMBER_REUSE_PRICE"], "kr/m^3"), "Price new steel": (constants["STEEL_REUSE_PRICE"], "kr/m^3"), "Price reused steel": (constants["STEEL_REUSE_PRICE"], "kr/m^3")})
    results_dict["Savings"] =  round(results_dict["All new score"] - results_dict["Score"], 2)
    results_dict["Number_reused"] = len(match_object.supply) - len(match_object.demand)
    results_dict["Number_demand"] = len(match_object.demand)
    results_dict["Number of substitutions"] = len(match_object.pairs[match_object.pairs["Supply_id"].str.startswith("S")])
    results_dict["Number of substitutions"] = sub_df["Substitutions"][sub_df["Names"].index(results_dict["Algorithm"])]
    if include_transportation:
        results_dict["Transportation included"] = "Yes"
        results_dict["Transportation score"] = round(match_object.result_transport, 2)
        results_dict["Transportation percentage"] = round(match_object.result_transport/results_dict["Score"]*100, 2)
        results_dict["Transportation all new"] = round(all_new_transport, 2)
        if metric == "GWP":
            used_constants.update({"GWP transportation": (constants["TRANSPORT_GWP"],"kg/m^3 per tonne")})
        elif metric == "Combined":
            used_constants.update({"GWP transportation": (constants["TRANSPORT_GWP"],"kg/m^3 per tonne"), "Price of transportation": (constants["PRICE_TRANSPORTATION"], "kr/km/tonne")})
        elif metric == "Price":
            used_constants.update({"Price of transportation": (constants["PRICE_TRANSPORTATION"], "kr/km/tonne")})                 
    else:
        results_dict["Transportation included"] = "No"
        results_dict["Transportation percentage"] = 0
        results_dict["Transportation score"] = 0
        results_dict["Transportation all new"] = 0

    pairs = extract_pairs_df(dict_list)[results_dict["Algorithm"]]
    pairs.name = "Substitutions"
    results_dict["Pairs"] = pairs
    results_dict["Constants used"] = used_constants

    #TODO: More information about supply and demand elements
    return results_dict


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

def create_random_data_demand(demand_count, demand_lat, demand_lon, new_lat, new_lon, demand_gwp=lca.TIMBER_GWP,gwp_price=lca.GWP_PRICE,new_price=lca.NEW_ELEMENT_PRICE_TIMBER, length_min = 1, length_max = 15.0, area_min = 0.15, area_max = 0.15):
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

def create_random_data_supply(supply_count,demand_lat, demand_lon,supply_coords,supply_gwp=lca.TIMBER_REUSE_GWP,gwp_price=lca.GWP_PRICE,reused_price=lca.REUSED_ELEMENT_PRICE_TIMBER, length_min = 1, length_max = 15.0, area_min = 0.15, area_max = 0.15):
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

def create_random_data_supply_pdf_reports(supply_count, length_min, length_max, area_min, area_max, materials, supply_coords):
    steel_cs = {"IPE100": (1.03e-3, 1.71e-6),
                "IPE140": (1.64e-3, 5.41e-6),
                "IPE160": (2.01e-3, 8.69e-6),
                "IPE180": (2.39e-3, 13.20e-6),
                "IPE220": (3.34e-3, 27.7e-6),
                "IPE270": (4.59e-3, 57.9e-6),
                "IPE300": (5.38e-3, 83.6e-6)
    }

    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    supply = pd.DataFrame()
    supply['Length'] = np.round((length_max - length_min) * np.random.random_sample(size = supply_count) + length_min, 2)
    #supply['Length']
    supply['Area'] = 0 
    #TODO: Only works for squared sections! Not applicable for steel sections #
    supply['Moment of Inertia'] = 0
    supply['Material'] = ""
    supply["Location"]=0
    supply["Latitude"]=0
    supply["Longitude"]=0
    
    for row in range(len(supply)):
        material = materials[random.randint(0, len(materials)-1)]
        if material == "Timber":
            area = np.random.uniform(area_min, area_max)
            supply.loc[row, "Area"] = area
            supply.loc[row, "Moment of Inertia"] = area**2/12
        elif material == "Steel":
            cs = random.choice(list(steel_cs.keys()))
            supply.loc[row, "Area"] = steel_cs[cs][0]
            supply.loc[row, "Moment of Inertia"] = steel_cs[cs][1]
        supply.loc[row, "Material"] = material
        lokasjon=random.randint(0, len(supply_coords)-1)
        supply.loc[row,"Latitude"]=supply_coords.loc[lokasjon,"Latitude"]
        supply.loc[row,"Longitude"]=supply_coords.loc[lokasjon,"Longitude"]
        supply.loc[row,"Location"]=supply_coords.loc[lokasjon,"Location"]
    #supply.index = map(lambda text: 'S' + str(text), supply.index) 

    return supply

def create_random_data_demand_pdf_reports(demand_count, length_min, length_max, area_min, area_max, materials):
    steel_cs = {"IPE100": (1.03e-3, 1.71e-6),
                "IPE140": (1.64e-3, 5.41e-6),
                "IPE160": (2.01e-3, 8.69e-6),
                "IPE180": (2.39e-3, 13.20e-6),
                "IPE220": (3.34e-3, 27.7e-6),
                "IPE270": (4.59e-3, 57.9e-6),
                "IPE300": (5.38e-3, 83.6e-6)
    }
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    demand = pd.DataFrame()
    demand['Length'] = np.round((length_max - length_min) * np.random.random_sample(size = demand_count) + length_min, 2)
    demand['Area'] = 0
    #TODO: Only works for squared sections! Not applicable for steel sections
    demand['Moment of Inertia'] = 0
    demand['Material'] = ""
    demand["Manufacturer"]=0
    demand["Latitude"]=0
    demand["Longitude"]=0
    
    for row in range(len(demand)):
        material = materials[random.randint(0, len(materials)-1)]
        if material == "Timber":
            area = np.random.uniform(area_min, area_max)
            demand.loc[row, "Area"] = area
            demand.loc[row, "Moment of Inertia"] = area**2/12
        elif material == "Steel":
            cs = random.choice(list(steel_cs.keys()))
            demand.loc[row, "Area"] = steel_cs[cs][0]
            demand.loc[row, "Moment of Inertia"] = steel_cs[cs][1]
        demand.loc[row, "Material"] = material

    return demand

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
"""Converts the best solution a column containing the supply matches for each demand element. 
    This column is added to the weight-matrix in order to visually check if the matching makes sense
    - weights: Pandas Dafarame
    - best_solution: 1d-list with 0's and 1's
    - number_of_demand_elements: integer

    Returns a pandas dataframe
    """
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
    if "xlsx" in file_location:
        dataframe = pd.read_excel(file_location)
    else: #CSV file
        dataframe = pd.read_csv(file_location)
    if "Column1" in list(dataframe.columns):
        dataframe.drop(columns=["Column1"], inplace = True)
    dataframe.index = map(lambda text: index_replacer + str(text), dataframe.index)
    new_columns = {col: col.split('[')[0].strip() for col in dataframe.columns}
    dataframe = dataframe.rename(columns = new_columns)
    dataframe = dataframe.fillna(0.0)
    return dataframe



def add_graph_plural(demand_matrix, supply_matrix, weight_matrix, incidence_matrix):
    """Add a graph notation based on incidence matrix
    
    Not used
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

def generate_plots_pdf_report(supply, demand, pdf_results, transportation_included):
    plot.create_graph(supply, demand, target_column="Length", unit=r"[m]", number_of_intervals=5, fig_title = "", save_filename=r"length_plot.png")
    plot.create_graph(supply, demand, target_column="Area", unit=r"[m$^2$]", number_of_intervals=5, fig_title = "", save_filename=r"area_plot.png")
    plot.create_graph(supply, demand, target_column="Moment of Inertia", unit=r"[m$^4$]", number_of_intervals=5, fig_title = "", save_filename=r"inertia_plot.png")
    plot.plot_materials(supply, demand, "", save_filename=r"material_plot.png")

    if transportation_included:
        plot.create_map_substitutions(supply, pdf_results, "supply", color = "green", legend_text="Substitution locations", save_name=r"map_reused_subs")
        plot.create_map_substitutions(demand, pdf_results, "demand", color = "red", legend_text="Manufacturer locations", save_name=r"map_manu_subs")

def generate_pdf_report(results, projectname, supply, demand, filepath):
    def new_page():
        # Add a page to the PDF
        pdf.add_page()
        
        # Set the background color
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(0, 0, 210, 297, "F")
        
        # Add the image to the PDF
        pdf.image(r"./Local_files/NTNU-logo.png", x=10, y=10, w=30)

        # Add the date to the upper right corner of the PDF
        pdf.set_xy(200, 10)
        pdf.set_font("Times", size=10)
        pdf.cell(0, 10, str(date.today().strftime("%B %d, %Y")), 0, 1, "R")
    # Create a new PDF object
    # Create a new PDF object
    #Add CSV containing results to "Results"-folder
    export_dataframe_to_xlsx(results["Pairs"], filepath + (projectname+"_substitutions.xlsx"))
    if results["Transportation included"] == "No":
        transportation_included = False
    elif results["Transportation included"] == "Yes":
        transportation_included = True

    #Add relevant plots
    generate_plots_pdf_report(supply, demand, results, transportation_included)
    pdf = FPDF()
    new_page()
    ##################PAGE 1##################
    

    # Set the font and size for the title
    pdf.set_font("Times", size=36)
    #pdf.set_text_color(0, 64, 128)
    pdf.set_text_color(0, 80, 158)
    pdf.set_y(10)
    # Add the title to the PDF
    pdf.cell(0, 50, "Results from Element Matching", 0, 1, "C")
    pdf.set_left_margin(15)

    # Information about the project:
    ################################
    pdf.set_y(50)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Times", size=12, style = "B")
    pdf.cell(30, 10, f"Project name: ", 0, 0)
    pdf.set_font("Times", size=12, style = "")
    pdf.cell(0, 10, f"{results['Project name']}", 0, 1)

    pdf.set_font("Times", size=12, style = "B")
    pdf.cell(55, 10, f"Construction site located at: ", 0, 0)
    pdf.set_font("Times", size=12, style = "")
    site_lat = round(float(results['Site latitude']), 4)
    site_lon = round(float(results['Site longitude']), 4)
    pdf.cell(0, 10, f"{site_lat}, {site_lon}", 0, 1)


    # Set the font and size for the tables
    pdf.set_font("Times", size=12)
    pdf.set_left_margin(15)
    table_x = (pdf.w - 180) / 2
    table_y1 = 75
    table_y2 = 165

    #Summary
    ######################
    pdf.set_y(table_y1)
    pdf.set_font("Times", size=24, style ="")
    pdf.multi_cell(160, 7, txt="Summary of results")
    pdf.set_font("Times", size=10)
    pdf.set_left_margin(30)
    pdf.ln(5)
    pdf.set_fill_color(96, 150, 208)
    pdf.set_draw_color(204, 204, 204)
    pdf.cell(50, 10, f"Total score", 1, 0, "C", True)
    pdf.cell(50, 10, f"Score without reuse", 1, 0, "C", True)
    pdf.cell(25, 10, "Savings", 1, 0, "C", True)
    pdf.cell(25, 10, "Substitutions", 1, 1, "C", True)
    pdf.set_fill_color(247, 247, 247)
    substitutions = round(results['Number of substitutions']/results['Number_demand']*100, 2)
    savings = round(results['Savings']/results['All new score']*100, 2)
    pdf.cell(50, 10, f"{results['Score']} {results['Unit']}", 1, 0, "C", True)
    #TODO check this line: better? change from kr to NOK?
    pdf.cell(50, 10, f"{format_float(round(results['All new score'],0))} {results['Unit']}", 1, 0, "C", True)
    pdf.cell(25, 10, f"{savings}%", 1, 0, "C", True)
    pdf.cell(25, 10, f"{substitutions}%", 1, 1, "C", True) 
    pdf.ln()

    #Short text summary
    pdf.set_left_margin(15)
    pdf.set_y(110)
    pdf.set_font("Times", size=12, style ="")
    summary = f"The '{results['Algorithm']}' algorithm yields the best results, substituting {results['Number of substitutions']}/{results['Number_demand']} demand elements ({substitutions}%). Using '{results['Metric']}' as the optimization metric, a total score of {results['Score']} {results['Unit']} is achieved. For comparison, a score of {results['All new score']} {results['Unit']} would have been obtained by employing exclusively new materials. This results in a total saving of {savings}%."
    if transportation_included:
        summary += f" Note that impacts of transporting the materials to the construction site is accounted for and contributes to {results['Transportation percentage']}% of the total score. "
    else:
        summary += f" Note that impacts of transporting the materials to the construction site is not accounted for. "
    summary += f"Open the CSV-file \"{projectname}_substitutions.csv\" to examine the substitutions."

    pdf.multi_cell(pdf.w-2*15,8, summary, 0, "L", False)


    #Constants used in calculations:
    ###############
    pdf.set_font("Times", size=16, style ="")
    pdf.set_xy(table_x, table_y2)
    pdf.multi_cell(160, 7, txt="Constants used in calculations")
    pdf.set_font("Times", size=10)
    pdf.ln(5)
    pdf.set_fill_color(96, 150, 208)
    pdf.set_draw_color(204, 204, 204)
    pdf.set_left_margin(30)
    pdf.cell(50, 10, "Constant", 1, 0, "C", True)
    pdf.cell(50, 10, "Value", 1, 0, "C", True)
    pdf.cell(50, 10, "Unit", 1, 1, "C", True)
    pdf.set_fill_color(247, 247, 247)

    for key, values in results["Constants used"].items():
        pdf.cell(50, 10, f"{key}", 1, 0, "C", True)
        pdf.cell(50, 10, f"{values[0]}", 1, 0, "C", True)
        pdf.cell(50, 10, f"{values[1]}", 1, 0, "C", True)
        pdf.ln()

        
    
    ##################PAGE 2##################
    new_page()
    #Information about datasets:
    ###########################
    pdf.set_font("Times", size=16, style ="")
    pdf.set_xy(table_x, 30)
    pdf.multi_cell(160, 7, txt="Information about datasets")
    pdf.set_font("Times", size=10)
    pdf.ln(5)
    pdf.set_left_margin(30)
    pdf.set_fill_color(96, 150, 208)
    pdf.set_draw_color(204, 204, 204)
    pdf.cell(30, 10, "Elements", 1, 0, "C", True)
    pdf.cell(80, 10, "Filename", 1, 0, "C", True)
    pdf.cell(40, 10, "Number of elements", 1, 1, "C", True)
    pdf.set_fill_color(247, 247, 247)
    pdf.cell(30, 10, f"Reused", 1, 0, "C", True)
    pdf.cell(80, 10, f"{results['Supply file location'].split('/')[-1]}", 1, 0, "C", True)
    pdf.cell(40, 10, f"{results['Number_reused']}", 1, 0, "C", True)
    pdf.ln()
    pdf.cell(30, 10, f"Demand", 1, 0, "C", True)
    pdf.cell(80, 10, f"{results['Demand file location'].split('/')[-1]}", 1, 0, "C", True)
    pdf.cell(40, 10, f"{results['Number_demand']}", 1, 0, "C", True)

    pdf.set_left_margin(15)
    pdf.set_y(80)
    pdf.set_font("Times", size=12, style ="")
    summary_info = f"The files contains {results['Number_reused']} reuse elements and {results['Number_demand']} demand elements. The graphs below depicts some of the properties of the elements, including length, area, moment of inertia and the material distribution."
    pdf.multi_cell(pdf.w-2*15,8, summary_info, 0, "L", False)

    #Images, 6 of them
    plots = ["Plots/material_plot.png", "Plots/length_plot.png", "Plots/area_plot.png", "Plots/inertia_plot.png"]
    x = 7.5
    y = 102.5
    w = 95
    for i in range (len(plots)):
        pdf.image(r"" + f"./Local_files/GUI_files/Results/{plots[i]}", x, y, w)
        print("Included", plots[i])
        if i % 2 == 1:
            y += 72.5
            x = 7.5
        if i % 2 == 0:
            x += 100
    
    

    #pdf.set_y(110)
    
    
    #TODO: Add more information about the datasets. Some histograms and graphs!

    ##################PAGE 2.5 (Transportation)##################
    new_page()
    y_information = 30

    if transportation_included: #Add a page with information about only transportation
        pdf.set_left_margin(15)
        pdf.set_y(y_information) #prior 30
        pdf.set_font("Times", size=16, style ="")
        pdf.multi_cell(160, 7, txt="Impact of transportation")
        pdf.set_font("Times", size=10)
        pdf.set_left_margin(30)
        pdf.ln(5)
        pdf.set_fill_color(96, 150, 208)
        pdf.set_draw_color(204, 204, 204)
        pdf.cell(50, 10, f"Transportation score", 1, 0, "C", True)
        pdf.cell(50, 10, "Percentage of total score", 1, 0, "C", True)
        pdf.cell(50, 10, "Transportation all new", 1, 1, "C", True)
        pdf.set_fill_color(247, 247, 247)
        pdf.cell(50, 10, f"{results['Transportation score']} {results['Unit']}", 1, 0, "C", True)
        pdf.cell(50, 10, f"{results['Transportation percentage']}%", 1, 0, "C", True)
        pdf.cell(50, 10, f"{results['Transportation all new']} {results['Unit']}", 1, 1, "C", True)
        pdf.ln()
        y_information += 35
        #Short text summary
        pdf.set_left_margin(15)
        pdf.set_y(y_information)
        pdf.set_font("Times", size=12, style ="")
        summary = f"All calculations in this report take impacts of transportation of the materials to the construction site into consideration. Transportation itself is responsible for {results['Transportation score']} {results['Unit']}. This accounts for {results['Transportation percentage']}% of the total score of {results['Score']} {results['Unit']}. For comparison, the transportation impact for exclusively using new materials would have been {results['Transportation all new']} {results['Unit']}. Two maps are included to show the location of the suggested substitutions of reused elements and the manufacturer locations where new elements can be obtained. The numbers on the maps indicate the number of elements present at each location."
        
        pdf.multi_cell(pdf.w-2*15,8, summary, 0, "L", False)
        
        pdf.set_y(y_information)
        maps = ["Maps/map_reused_subs.png", "Maps/map_manu_subs.png"]
        x = 7.5
        y = y_information + 60
        w = 95
        for i in range (len(maps)):
            pdf.image(r"" + f"./Local_files/GUI_files/Results/{maps[i]}", x, y, w)
            print("Included", maps[i])
            if i % 2 == 1:
                y += 75
                x = 7.5
            if i % 2 == 0:
                x += 100

        y_information = 205

    pdf.set_left_margin(15)
    pdf.set_y(y_information)
    pdf.set_font("Times", size=16, style ="")
    pdf.multi_cell(160, 7, txt="Performance of algorithms")
    pdf.set_font("Times", size=10)
    pdf.set_left_margin(17)
    pdf.ln(5)
    pdf.set_fill_color(96, 150, 208)
    pdf.set_draw_color(204, 204, 204)
    pdf.cell(75, 10, "Name", 1, 0, "C", True)
    pdf.cell(51, 10, "Total score", 1, 0, "C", True)
    pdf.cell(25, 10, "Substitutions", 1, 0, "C", True)
    pdf.cell(25, 10, "Time", 1, 1, "C", True)

    pdf.set_fill_color(247, 247, 247)
    performance = results['Performance'] #Dataframe
    print_names = ""
    for i in range(len(performance)):
        y_information += 10
        pdf.cell(75, 10, f"{performance.iloc[i]['Names']}", 1, 0, "C", True)
        pdf.cell(51, 10, f"{performance.iloc[i]['Score']} {results['Unit']}", 1, 0, "C", True)
        pdf.cell(25, 10, f"{performance.iloc[i]['Sub_percent']}%", 1, 0, "C", True)
        pdf.cell(25, 10, f"{performance.iloc[i]['Time']}s", 1, 0, "C", True)
        if len(performance) == 1:
            print_names += performance.iloc[i]['Names']
        elif i != len(performance) - 1:
            print_names += performance.iloc[i]['Names']
            print_names += ", " 
        else:
            print_names += "and " + performance.iloc[i]['Names']
        pdf.ln()


    pdf.set_font("Times", size=12, style ="")
    pdf.set_left_margin(15)
    pdf.set_y(y_information+25)
    summary = f"The design tool is runned with {len(performance)} algorithms, namely: {print_names}. The {results['Algorithm']} yields the lowest score, as shown in the table. The substitutions by this algorithm are completed in {results['Time']} seconds."
    pdf.multi_cell(pdf.w-2*15,8, summary, 0, "L", False)
 
    # Save the PDF to a file
    pdf.output(filepath + projectname+"_report.pdf")


def generate_score_function_string(constants):
    metric = constants["Metric"]
    transportation = constants["Include transportation"]
    if metric == "GWP":
        score_function_string = f"@lca.calculate_lca(length=Length, area=Area, include_transportation={transportation}, distance = Distance, gwp_factor=Gwp_factor, transport_gwp = {constants['TRANSPORT_GWP']}, density = Density)"
    elif metric == "Combined":
        score_function_string = f"@lca.calculate_score(length=Length, area=Area, include_transportation = {transportation}, distance = Distance, gwp_factor = Gwp_factor, transport_gwp = {constants['TRANSPORT_GWP']}, price = Price, priceGWP = {constants['VALUATION_GWP']}, density = Density, price_transport = {constants['PRICE_TRANSPORTATION']})"
    elif metric == "Price":
        score_function_string = f"@lca.calculate_price(length=Length, area =Area, include_transportation = {transportation}, distance = Distance, price = Price, density = Density, price_transport= {constants['PRICE_TRANSPORTATION']})"
    return score_function_string

def fill_closest_manufacturer(dataframe,constants):
    manufacturers_df=pd.read_excel(r"./CSV/manufacturers_locations.xlsx")

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

        if str(row["Material"])=="Timber" and driving_distance<shortest_distance_tree:
            shortest_distance_tree=driving_distance
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

def add_necessary_columns_pdf(dataframe, constants):
    dataframe = dataframe.copy()
    metric = constants["Metric"]
    element_type = list(dataframe.index)[0][:1]
    dataframe["Density"] = 0
    dataframe["Site_lat"] = constants["Site latitude"]
    dataframe["Site_lon"] = constants["Site longitude"]
    if metric == "GWP":
        dataframe["Gwp_factor"] = 0 
    elif metric == "Combined":
        dataframe["Gwp_factor"] = 0 
        dataframe["Price"] = 0
    elif metric == "Price":
        dataframe["Price"] = 0

    #If dataframe is demand, fill in the location and corresponding coordinates and to the closet manufacturer.
    if element_type=="D" and constants["Include transportation"]:
        dataframe=fill_closest_manufacturer(dataframe,constants)

    for row in range(len(dataframe)):
        material = dataframe.iloc[row][dataframe.columns.get_loc("Material")].split()[0] #NOTE: Assumes that material-column has the material name as the first word, e.g. "Timber C14" or "Steel ASTM A992"
        dataframe.iloc[row, dataframe.columns.get_loc("Density")] = constants[f"{material.upper()}_DENSITY"]

        if element_type == "S":
            constant_name = f"{material.upper()}_REUSE"
        else:
            constant_name = f"{material.upper()}"

        if metric == "GWP" or metric == "Combined":
                dataframe.iloc[row, dataframe.columns.get_loc("Gwp_factor")] = constants[constant_name + "_GWP"]
        if metric == "Price" or metric == "Combined":
                if material.upper() == "STEEL":
                    price = constants[constant_name + "_PRICE"] * constants[f"{material.upper()}_DENSITY"]
                else:
                    price = constants[constant_name + "_PRICE"]
                dataframe.iloc[row, dataframe.columns.get_loc("Price")] = price
    return dataframe

def generate_run_string(constants):
    algorithms = constants["Algorithms"]
    run_string = "run_matching(demand, supply, score_function_string, constraints = constraint_dict, add_new = True"
    if len(algorithms) != 0:
        for algorithm in algorithms:
            run_string += f", {algorithm} = True"
    run_string += ")"
    return run_string

def extract_best_solution(result, metric):
    results = extract_results_df(result, column_name = f"{metric}")

print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")