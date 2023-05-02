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
    "Project name": "Bod nidarosdomen",
    "Metric": "GWP",
    "Algorithms": ["bipartite", "greedy_plural", "greedy_single", "bipartite_plural"],
    "Include transportation": False,
    "Cite latitude": "63.4269097",
    "Cite longitude": "10.3969375",
    "Demand file location": r"./CSV/pdf_demand.csv",
    "Supply file location": r"./CSV/pdf_supply.csv",
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


demand_coords = {"Steel": ("Norsk Stål Trondheim", "63.4384474", "10.40994"), "Timber": ("XL-BYGG Lade","63.4423683","10.4438836")}


materials = ["Timber", "Steel"]

#GENERATE FILE
#============
#supply = hm.create_random_data_supply_pdf_reports(supply_count = 50, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
#demand = hm.create_random_data_demand_pdf_reports(demand_count = 50, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, demand_coords = demand_coords)
#hm.export_dataframe_to_csv(supply, r"" + "./CSV/pdf_supply.csv")
#hm.export_dataframe_to_csv(demand, r"" + "./CSV/pdf_demand.csv")
#========================================

score_function_string = hm.generate_score_function_string(constants)
#supply = hm.import_dataframe_from_csv(r"" + constants["Supply file location"])
#demand = hm.import_dataframe_from_csv(r"" + constants["Demand file location"])
supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")

#hm.create_graph(supply, demand, "Length", number_of_intervals= 2, save_filename = r"C:\Users\sigur\Downloads\test.png")

constraint_dict = constants["constraint_dict"]
#Add necessary columns to run the algorithm
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)

run_string = hm.generate_run_string(constants)
result = eval(run_string)
simple_pairs = hm.extract_pairs_df(result)
pdf_results = hm.extract_results_df_pdf(result, constants)
#hm.create_map_substitutions(supply, pdf_results)
#hm.create_map(supply)
#hm.create_map_manufacturer_matches(demand, pdf_results)
#hm.create_map_manufacturers(demand)
hm.create_map_dataframe(demand, color = "red", legend_text="Manufacturer locations", save_name="map_manufacturer_test")
#hm.create_map_dataframe(supply, color = "green", legend_text="")
#pdf = hm.generate_pdf_report(pdf_results, filepath = r"./Results/")
#print(hm.extract_pairs_df(result))