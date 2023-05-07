import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching # Matching
import LCA as lca
import plotting as plot


#==========USER FILLS IN============#
#Constants
#TODO: FIND ALL DEFAULT VALUES FOR CONSTANTS, especially for price
constants = {
    "TIMBER_GWP": 28.9,       #kg CO2 eq per m^3, based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 2.25,        # 0.0778*28.9 = 2.25kg CO2 eq per m^3, based on Eberhardt
    "TRANSPORT_GWP": 89.6,    #gram per tonne per km, Engedal et. al. 
    "TIMBER_DENSITY": 491.0,  #kg/m^3, based on NEPD-3442-2053-EN
    "STEEL_GWP": 9263.0, #kg CO2 eq per m^3, Norsk stål + density of steel
    "STEEL_REUSE_GWP": 278.0, #kg CO2 eq per m^3, reduction of 97% from new according to Høydahl and Walter
    "VALUATION_GWP": 0.7, #NOK per kg CO2, based on OECD
    "TIMBER_PRICE": 3400.0, #Per m^3, Treindustrien 2023
    "TIMBER_REUSE_PRICE" : 3400.0, #Per m^3, assumes the price is the same is new elements
    "STEEL_PRICE": 500, #Per m^2, Random value TODO: ADD REAL VALUE
    "STEEL_REUSE_PRICE": 200, #Per m^2, Random value TODO: ADD REAL VALUE
    "PRICE_TRANSPORTATION": 0.3, #NOK per km per tonne, Grønland 2022 + Gran 2013
    "STEEL_DENSITY": 7850,
    ########################
    "Project name": "Campussamling Hesthagen",
    "Metric": "GWP",
    "Algorithms": ["bipartite_plural", "bipartite_plural_multiple"],
    "Include transportation": False,
    "Cite latitude": "63.4154171",
    "Cite longitude": "10.3994672",
    "Demand file location": r"./CSV/genetic_demand.csv",
    "Supply file location": r"./CSV/genetic_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}


def generate_datasets(d_counts, s_counts):
    supply_coords = pd.DataFrame(columns = ["Location", "Latitude", "Longitude"])

    tiller = ["Tiller", "63.3590272", "10.3751236"]
    storen = ["Støren", "63.033639", "10.286356"]
    orkanger = ["Orkanger", "63.3000", "9.8468"]
    storlien = ["Storlien", "63.3160", "12.1018"]
    hell = ["Hell", "63.4452539", "10.8971079"]
    melhus = ["Melhus", "63.2897753", "10.2934154"]


    supply_coords.loc[len(supply_coords)] = tiller
    supply_coords.loc[len(supply_coords)] = storen
    supply_coords.loc[len(supply_coords)] = orkanger
    supply_coords.loc[len(supply_coords)] = storlien
    supply_coords.loc[len(supply_coords)] = hell
    supply_coords.loc[len(supply_coords)] = melhus
    


    demand_coords = {"Steel": ("Norsk Stål Trondheim", "63.4384474", "10.40994"), "Timber": ("XL-BYGG Lade","63.4423683","10.4438836")}


    materials = ["Timber", "Steel"]

    #GENERATE FILE
    #============
    supply = hm.create_random_data_supply_pdf_reports(supply_count = s_counts, length_min = 1.0, length_max = 10.0, area_min = 0.004, area_max = 0.04, materials = materials, supply_coords = supply_coords)
    demand = hm.create_random_data_demand_pdf_reports(demand_count = d_counts, length_min = 1.0, length_max = 10.0, area_min = 0.004, area_max = 0.04, materials = materials, demand_coords = demand_coords)
    supply.index = map(lambda text: "S" + str(text), supply.index)
    demand.index = map(lambda text: "D" + str(text), demand.index)
    return demand, supply

# ========== Comparing bipartite plural vs bipartite plural multiple ============== 
var1 = 1
d_counts = np.linspace(4, 20, num = 2).astype(int)
s_counts = (d_counts * var1).astype(int)
internal_runs = 50
constraint_dict = constants["constraint_dict"]
score_function_string = hm.generate_score_function_string(constants)
run_string = hm.generate_run_string(constants)
results = [] #list of results for each iteration

hm.print_header("Starting Run")

dict_made = False
x_values = []
for d, s in zip(d_counts, s_counts):
    x_values.append(d+s)
    #create data
    temp_times = [[] for _ in range(len(constants["Algorithms"]))]
    temp_scores = [[] for _ in range(len(constants["Algorithms"]))]
    for i in range(internal_runs):
        demand, supply = generate_datasets(d, s)
        #Add necessary columns to run the algorithm
        supply = hm.add_necessary_columns_pdf(supply, constants)
        demand = hm.add_necessary_columns_pdf(demand, constants)
        result = eval(run_string)
        if dict_made == False:
            time_dict = {res[list(res.keys())[0]] : [] for res in result}
            score_dict = {res[list(res.keys())[0]] : [] for res in result}
            dict_made = True
        for i in range(len(result)):
            temp_times[i].append(result[i]["Match object"].solution_time)
            temp_scores[i].append(result[i]["Match object"].result)

    mean_time = np.mean(temp_times, axis = 1)
    mean_score = np.mean(temp_scores, axis = 1)
    for i in range(len(list(time_dict.keys()))):
        key = list(time_dict.keys())[i]
        time_dict[key].append(mean_time[i])
        score_dict[key].append(mean_score[i])




#pairs_df = pd.concat([res['Match object'].pairs for res in results[0]], axis = 1)
#pairs_df.columns = [res[list(res.keys())[0]] for res in results[0]]

plot.plot_algorithm(time_dict, x_values, xlabel = "Number of elements", ylabel = "Running time [s]", title = "", fix_overlapping=False, save_filename="bipartite_results_time.png")
plot.plot_algorithm(score_dict, x_values, xlabel = "Number of elements", ylabel = "Total score [kg CO2 equiv.]", title = "", fix_overlapping=True, save_filename="bipartite_results_score.png")