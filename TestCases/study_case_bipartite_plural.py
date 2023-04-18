import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca

#Where is the actual site where our elements must be transportet too
demand_coordinates = {"Latitude": "10.3969", "Longitude": "63.4269"}

#Defines the coordinates from where the NEW elementes are transported from, 
#Moelv:
#new_coordinates={"Latitude":"10.7005","Longitude":"60.9277"}
new_coordinates = {"Latitude": "10.3969", "Longitude": "63.4269"}


#Defines different coordinates from where REUSED elements can be transported from
supply_coords = pd.DataFrame(columns = ["Place", "Lat", "Lon"])
tiller = ["Tiller", "10.4008", "63.3604"]
gjovik = ["Gjovik", "10.5001", "60.8941"]
orkanger = ["Orkanger", "9.8468", "63.3000"]
storlien = ["Storlien", "12.1018", "63.3160"]

supply_coords.loc[len(supply_coords)] = tiller
supply_coords.loc[len(supply_coords)] = gjovik
supply_coords.loc[len(supply_coords)] = orkanger
supply_coords.loc[len(supply_coords)] = storlien


constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='} # dictionary of constraints to add to the method
demand = hm.create_random_data_demand(demand_count=100, demand_lat = demand_coordinates["Latitude"], demand_lon = demand_coordinates["Longitude"],new_lat = new_coordinates["Latitude"], new_lon = new_coordinates["Longitude"], length_min = 1, length_max = 10.0, area_min = 0.15, area_max = 0.30)
supply = hm.create_random_data_supply(supply_count=100,demand_lat = demand_coordinates["Latitude"], demand_lon = demand_coordinates["Longitude"],supply_coords = supply_coords, length_min = 1, length_max = 30.0, area_min = 0.15, area_max = 0.30)

score_function_string_wo_transportation = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, distance = Distance, include_transportation=False)"

result_wo_transportation = run_matching(demand, supply, score_function_string_wo_transportation, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=False, greedy_plural = True, bipartite=True,genetic=False,brute=False, bipartite_plural = True)
simple_pairs_wo_transportation = hm.extract_pairs_df(result_wo_transportation)
simple_results_wo_transportation = hm.extract_results_df(result_wo_transportation, column_name = "LCA")
print("Simple pairs without transportation LCA:")
print(simple_pairs_wo_transportation)
print()
print("Simple results without transportation LCA")
print(simple_results_wo_transportation)
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


