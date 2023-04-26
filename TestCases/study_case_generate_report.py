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
    "Project name": "Bod nidarosdomen",
    "Metric": "GWP",
    "Algorithms": ["bipartite", "greedy_plural", "greedy_single", "bipartite_plural"],
    "Include transportation": True,
    "Cite latitude": "63.4269",
    "Cite longitude": "10.3969",
    "Demand file location": r"./CSV/pdf_demand.csv",
    "Supply file location": r"./CSV/pdf_supply.csv"
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


demand_coords = {"Steel": ("Norsk Stål Trondheim", "63.4384474", "10.40994"), "Timber": ("XL-BYGG Lade","63.4423683","10.4438836")}


materials = ["Timber", "Steel"]

#GENERATE FILE
#============
supply = hm.create_random_data_supply_pdf_reports(supply_count = 10, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
demand = hm.create_random_data_demand_pdf_reports(demand_count = 10, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, demand_coords = demand_coords)
hm.export_dataframe_to_csv(supply, r"" + constants["Supply file location"])
hm.export_dataframe_to_csv(demand, r"" + constants["Demand file location"])
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
pdf_results = hm.extract_results_df_pdf(result, constants)
print("Simple pairs:")
print(simple_pairs)
simple_results = hm.extract_results_df(result, "Test")
print(simple_results)

#TODO: Add the information to the report!

pdf = hm.generate_pdf_report(pdf_results, filepath = r"./Results/")



