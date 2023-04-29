import sys
sys.path.append('./Matching')
import matching
from matching import run_matching # Matching
import helper_methods as hm
import LCA as lca

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    "Algorithms": ["bipartite", "greedy_plural", "bipartite_plural", "bipartite_plural_multiple"],
    "Include transportation": False,
    "Cite latitude": "59.94161606",
    "Cite longitude": "10.72994518",
    #"Demand file location": r"./CSV/DEMAND_DATAFRAME_SVERRE.xlsx",
    #"Supply file location": r"./CSV/SUPPLY_DATAFRAME_SVERRE.xlsx",
    "Demand file location": r"./CSV/bipartite_plural_demand.csv",
    "Supply file location": r"./CSV/bipartite_plural_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}
#========================#
#========================#
#Generating dataset
#===================
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
    return supply, demand


# ========== SCENARIO 1 ============== 
var1 = 1
#d_counts = np.logspace(1, 3, num = 5).astype(int) Use this later when actually testing. Using the below for now to reduce time
d_counts = np.linspace(10, 250, num = 4).astype(int)
s_counts = (d_counts * var1).astype(int)

results = [] #list of results for each iteration

hm.print_header("Starting Run")

for d, s in zip(d_counts, s_counts):
    #create data
    print(f'\n*** Running for {d} demand and {s} supply elements.***\n')
    demand, supply = generate_datasets(d, s)
    constraint_dict = constants["constraint_dict"]
    score_function_string = hm.generate_score_function_string(constants)
    #Add necessary columns to run the algorithm
    supply = hm.add_necessary_columns_pdf(supply, constants)
    demand = hm.add_necessary_columns_pdf(demand, constants)
    run_string = hm.generate_run_string(constants)
    result = eval(run_string)
    results.append(result)
    
    
n_els = d_counts+s_counts # number of elements for each iteration

time_dict = {res[list(res.keys())[0]] : [] for res in results[0]} # create a dictionary for the time spent running each method with different number of elements
lca_dict = {res[list(res.keys())[0]] : [] for res in results[0]}

for iteration in results:
    for method in iteration: # iterate through all methods
        lca_dict[method['Name']].append(method['Match object'].result) 
        time_dict[method['Name']].append(method['Match object'].solution_time) 

pairs_df = pd.concat([res['Match object'].pairs for res in results[0]], axis = 1)
pairs_df.columns = [res[list(res.keys())[0]] for res in results[0]]

fig, ax = plt.subplots()
for key, items in time_dict.items():
    plt.plot(n_els, items, label = key)
plt.legend()
plt.xlabel('Number of elements')
plt.ylabel('Solution time [s]')
plt.yscale('log')
plt.plot()
plt.show()

fig, ax = plt.subplots()
for key, items in lca_dict.items():
    plt.plot(n_els, items, label = key)
plt.legend()
plt.xlabel('Number of elements')
plt.ylabel('LCA_score')
#plt.yscale('log')
plt.plot()
plt.show()