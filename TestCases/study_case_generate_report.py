import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca


#==========USER FILLS IN============#
project_name = "Bod materialteknisk"
metric = "GWP"
algorithms = ["bipartite", "greedy_plural", "greedy_single", "bipartite_plural"]
include_transportation = False
coordinates_site = {"Latitude": "10.3969", "Longitude": "63.4269"}
demand_file_location = r"./CSV/pdf_demand.csv"
supply_file_location = r"./CSV/pdf_supply.csv"
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>=', 'Material': '=='}

#Constants
constants = {
    "TIMBER_GWP": 28.9,       # based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 2.25,        # 0.0778*28.9 = 2.25 based on Eberhardt
    "TRANSPORT_GWP": 96.0,    # TODO kg/m3/t based on ????
    "TIMBER_DENSITY": 491.0,  # kg, based on NEPD-3442-2053-EN
    "STEEL_GWP": 800, #Random value
    "STEEL_REUSE_GWP": 4, #Random value
    "TRANSPORTATION_GWP": None,
    "VALUATION_GWP": None,
    "TIMBER_PRICE": None,
    "TIMBER_REUSE_PRICE" : None,
    "STEEL_PRICE": None,
    "STEEL_REUSE_PRICE": None,
    "PRICE_TRANSPORTATION": None,
    "STEEL_DENSITY": 7850,
    "PRICE_M2": None
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
supply = hm.create_random_data_supply_pdf_reports(supply_count = 20, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
demand = hm.create_random_data_demand_pdf_reports(demand_count = 20, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, demand_coords = demand_coords)
hm.export_dataframe_to_csv(supply, supply_file_location)
hm.export_dataframe_to_csv(demand, demand_file_location)
#========================================
score_function_string = hm.generate_score_function_string(metric, include_transportation, constants)
supply = hm.import_dataframe_from_csv(supply_file_location)
demand = hm.import_dataframe_from_csv(demand_file_location)

#Add necessary columns to run the algorithm
supply = hm.add_necessary_columns_pdf(supply, metric, constants, coordinates_site)
demand = hm.add_necessary_columns_pdf(demand, metric, constants, coordinates_site)

run_string = hm.generate_run_string(algorithms)

#Running the matching
result = eval(run_string)
simple_pairs = hm.extract_pairs_df(result)
pdf_results = hm.extract_results_df_pdf(result, metric, include_transportation)
print("Simple pairs:")
print(simple_pairs)

#TODO: Add the information to the report!


