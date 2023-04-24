import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca


#==========USER FILLS IN============#
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>=', 'Material': '=='}

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
    "Project name": "Bod materialteknisk",
    "Metric": "Price",
    "Algorithms": ["bipartite", "greedy_plural", "greedy_single", "bipartite_plural"],
    "Include transportation": True,
    "Coordinates site": {"Latitude": "10.3969", "Longitude": "63.4269"},
    "Demand file location": r"./CSV/pdf_demand.csv",
    "Supply file location": r"./CSV/pdf_supply.csv"
}
#========================#

#Generating dataset
#===================
supply_coords = pd.DataFrame(columns = ["Location", "Latitude", "Longitude"])
tiller = ["Tiller", "10.4008", "63.3604"]
gjovik = ["Gjovik", "10.5001", "60.8941"]
orkanger = ["Orkanger", "9.8468", "63.3000"]
storlien = ["Storlien", "12.1018", "63.3160"]

supply_coords.loc[len(supply_coords)] = tiller
supply_coords.loc[len(supply_coords)] = gjovik
supply_coords.loc[len(supply_coords)] = orkanger
supply_coords.loc[len(supply_coords)] = storlien

demand_coords = {
    "Moelven": {"Latitude": "10.6956", "Longitude": "60.9298"},
    "Norsk stål": {"Latitude": "10.475", "Longitude": "59.8513"}
}


materials = ["Timber", "Steel"]

#GENERATE FILE
#============
#supply = hm.create_random_data_supply_pdf_reports(supply_count = 20, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
#demand = hm.create_random_data_demand_pdf_reports(demand_count = 20, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, demand_coords = demand_coords)
#hm.export_dataframe_to_csv(supply, r"" + constants["Supply file location"])
#hm.export_dataframe_to_csv(demand, r"" + constants["Demand file location"])
#========================================
score_function_string = hm.generate_score_function_string(constants)
supply = hm.import_dataframe_from_csv(r"" + constants["Supply file location"])
demand = hm.import_dataframe_from_csv(r"" + constants["Demand file location"])

#Add necessary columns to run the algorithm
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)

run_string = hm.generate_run_string(constants)

#Running the matching
result = eval(run_string)
simple_pairs = hm.extract_pairs_df(result)
pdf_results = hm.extract_results_df_pdf(result, constants["Metric"], constants["Include transportation"])
print("Simple pairs:")
print(simple_pairs)

#TODO: Add the information to the report!


