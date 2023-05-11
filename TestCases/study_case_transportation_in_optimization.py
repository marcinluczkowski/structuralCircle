import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca

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
    "Algorithms": ["bipartite", "greedy_plural"],
    "Include transportation": False,
    "Site latitude": "59.94161606",
    "Site longitude": "10.72994518",
    #"Demand file location": r"./CSV/DEMAND_DATAFRAME_SVERRE.xlsx",
    #"Supply file location": r"./CSV/SUPPLY_DATAFRAME_SVERRE.xlsx",
    "Demand file location": r"./CSV/transportation_demand.csv",
    "Supply file location": r"./CSV/transportation_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}
#========================#
#Generating dataset
#===================
supply_coords = pd.DataFrame(columns = ["Location", "Latitude", "Longitude"])

tiller = ["Tiller", "63.3604", "10.4008"]
gjovik = ["Gjovik", "60.8941", "10.5001"]
orkanger = ["Orkanger", "63.3000", "9.8468"]
storlien = ["Storlien", "63.3160", "12.1018"]

supply_coords.loc[len(supply_coords)] = tiller
supply_coords.loc[len(supply_coords)] = gjovik
supply_coords.loc[len(supply_coords)] = orkanger
supply_coords.loc[len(supply_coords)] = storlien




materials = ["Timber", "Steel"]

#GENERATE FILE
#============
supply = hm.create_random_data_supply_pdf_reports(supply_count = 10, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
demand = hm.create_random_data_demand_pdf_reports(demand_count = 10, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials)
hm.export_dataframe_to_csv(supply, r"" + "./CSV/transportation_supply.csv")
hm.export_dataframe_to_csv(demand, r"" + "./CSV/transportation_demand.csv")
#========================================
score_function_string = hm.generate_score_function_string(constants)
supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")


constraint_dict = constants["constraint_dict"]
#Add necessary columns to run the algorithm
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)
run_string = hm.generate_run_string(constants)
result_wo_transportation = eval(run_string)

simple_pairs_wo_transportation = hm.extract_pairs_df(result_wo_transportation)
simple_results_wo_transportation = hm.extract_results_df(result_wo_transportation, column_name = "LCA")
print("Simple pairs without transportation LCA:")
print(simple_pairs_wo_transportation)
print()
print("Simple results without transportation LCA")
print(simple_results_wo_transportation)

#Include transporation
constants["Include transportation"] = True
score_function_string = hm.generate_score_function_string(constants) #New scorestring
run_string = hm.generate_run_string(constants) #New runstring
result_transportation = eval(run_string)

simple_pairs_transportation = hm.extract_pairs_df(result_transportation)
simple_results_transportation = hm.extract_results_df(result_transportation, column_name = "LCA")
print("Simple pairs WITH transportation LCA:")
print(simple_pairs_transportation)
print()
print("Simple results WITH transportation LCA")
print(simple_results_transportation)



