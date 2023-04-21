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
algorithms = ["Maximum Bipartite", "Greedy Plural"]
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
    "STEEL_GWP": None,
    "STEEL_REUSE_GWP": None,
    "TRANSPORTATION_GWP": None,
    "VALUATION_GWP": None,
    "TIMBER_PRICE": None,
    "TIMBER_REUSE_PRICE" : None,
    "STEEL_PRICE": None,
    "STEEL_REUSE_PRICE": None,
    "PRICE_TRANSPORTATION": None,
    "STEEL_DENSITY": None,
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
#supply = hm.create_random_data_supply_pdf_reports(supply_count = 10, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
#demand = hm.create_random_data_demand_pdf_reports(demand_count = 10, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, demand_coords = demand_coords)
#hm.export_dataframe_to_csv(supply, supply_file_location)
#hm.export_dataframe_to_csv(demand, demand_file_location)
#========================================
score_function_string = hm.generate_score_function_string(metric, include_transportation, constants)
supply = hm.import_dataframe_from_csv(supply_file_location)
demand = hm.import_dataframe_from_csv(demand_file_location)


supply = hm.add_necessary_columns(constants, coordinates_site)
demand = hm.add_necessary_columns(constants, coordinates_site)
"""
result_wo_transportation = run_matching(demand, supply, score_function_string_wo_transportation, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=False, greedy_plural = True, bipartite=False,genetic=False,brute=False, bipartite_plural = True)
simple_pairs_wo_transportation = hm.extract_pairs_df(result_wo_transportation)
simple_results_wo_transportation = hm.extract_results_df(result_wo_transportation, column_name = "LCA")
print("Simple pairs without transportation LCA:")
print(simple_pairs_wo_transportation)
print()
print("Simple results without transportation LCA")
print(simple_results_wo_transportation)

hm.create_report("LCA", 3)
"""
"""
print("Bipartite plural matches:")
print("\n",hm.count_matches(simple_pairs_wo_transportation, algorithm = "Bipartite plural"))
print("Bipartite plural multi matches:")
print("\n",hm.count_matches(simple_pairs_wo_transportation, algorithm = "Bipartite plural multi"))
print("Greedy plural matches:")
print("\n",hm.count_matches(simple_pairs_wo_transportation, algorithm = "Greedy_plural"))
"""
"""
score_function_string_transportation = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor,distance = Distance, include_transportation=True)"
result_transportation = run_matching(demand, supply, score_function_string_transportation, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=True, bipartite=True,genetic=False,brute=False)
simple_pairs_transportation = hm.extract_pairs_df(result_transportation)
simple_results_transportation = hm.extract_results_df(result_transportation, column_name = "LCA")
print("Simple pairs WITH transportation LCA:")
print(simple_pairs_transportation)
print()
print("Simple results WITH transportation LCA")
print(simple_results_transportation)
"""


