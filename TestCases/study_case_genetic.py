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
    "TIMBER_GWP": 28.9,       # based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 2.25,        # 0.0778*28.9 = 2.25 based on Eberhardt
    "TRANSPORT_GWP": 96.0,    # TODO kg/m3/t based on ????
    "TIMBER_DENSITY": 491.0,  # kg, based on NEPD-3442-2053-EN
    "STEEL_GWP": 800, #Random value
    "STEEL_REUSE_GWP": 4, #Random value
    "VALUATION_GWP": 0.6, #In kr:Per kg CO2, based on OECD
    "TIMBER_PRICE": 435, #Per m^3 https://www.landkredittbank.no/blogg/2021/prisen-pa-sagtommer-okte-20-prosent/
    "TIMBER_REUSE_PRICE" : 100, #Per m^3, Random value
    "STEEL_PRICE": 500, #Per m^2, Random value
    "STEEL_REUSE_PRICE": 200, #Per m^2, Random value
    "PRICE_TRANSPORTATION": 3.78, #Price per km per tonn. Derived from 2011 numbers on scaled t0 2022 using SSB
    "STEEL_DENSITY": 7850,
    ########################
    "Project name": "Sognsveien 17",
    "Metric": "GWP",
    "Algorithms": ["genetic", "greedy_single"],
    "Include transportation": False,
    "Cite latitude": "59.94161606",
    "Cite longitude": "10.72994518",
    #"Demand file location": r"./CSV/DEMAND_DATAFRAME_SVERRE.xlsx",
    #"Supply file location": r"./CSV/SUPPLY_DATAFRAME_SVERRE.xlsx",
    "Demand file location": r"./CSV/genetic_demand.csv",
    "Supply file location": r"./CSV/genetic_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}


def generate_datasets(d_counts, s_counts):
    supply_coords = pd.DataFrame(columns = ["Location", "Latitude", "Longitude"])

    tiller = ["Tiller", "63.3604", "10.4008"]
    gjovik = ["Gjovik", "60.8941", "10.5001"]
    orkanger = ["Orkanger", "63.3000", "9.8468"]
    storlien = ["Storlien", "63.3160", "12.1018"]

    supply_coords.loc[len(supply_coords)] = tiller
    supply_coords.loc[len(supply_coords)] = gjovik
    supply_coords.loc[len(supply_coords)] = orkanger
    supply_coords.loc[len(supply_coords)] = storlien


    demand_coords = {"Steel": ("Norsk Stål Trondheim", "63.4384474", "10.40994"), "Timber": ("XL-BYGG Lade","63.4423683","10.4438836")}


    materials = ["Timber", "Steel"]

    #GENERATE FILE
    #============
    supply = hm.create_random_data_supply_pdf_reports(supply_count = s_counts, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
    demand = hm.create_random_data_demand_pdf_reports(demand_count = d_counts, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, demand_coords = demand_coords)
    supply.index = map(lambda text: "S" + str(text), supply.index)
    demand.index = map(lambda text: "D" + str(text), demand.index)
    return demand, supply

# ========== SCENARIO 1 ============== 
var1 = 1
d_counts = np.linspace(5, 10, num = 4).astype(int)
s_counts = (d_counts * var1).astype(int)
internal_runs = 30
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
            AAAA = result[i]["Match object"].result
            if np.isnan(AAAA):
                match_object = result[i]["Match object"]
                weights = match_object.weights
                pairs = match_object.pairs
                test = 4
            temp_times[i].append(result[i]["Match object"].solution_time)
            temp_scores[i].append(result[i]["Match object"].result)

    mean_time = np.mean(temp_times, axis = 1)
    mean_score = np.mean(temp_scores, axis = 1)
    for i in range(len(list(time_dict.keys()))):
        key = list(time_dict.keys())[i]
        time_dict[key].append(mean_time[i])
        score_dict[key].append(mean_score[i])




test = 2

#pairs_df = pd.concat([res['Match object'].pairs for res in results[0]], axis = 1)
#pairs_df.columns = [res[list(res.keys())[0]] for res in results[0]]

plot.plot_algorithm(time_dict, x_values, xlabel = "Number of elements", ylabel = "Running time [s]", title = "", fix_overlapping=False, save_filename="genetic_results_time.png")
plot.plot_algorithm(score_dict, x_values, xlabel = "Number of elements", ylabel = "Score", title = "", fix_overlapping=False, save_filename="genetic_results_score.png")