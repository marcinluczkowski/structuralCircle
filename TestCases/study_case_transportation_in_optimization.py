import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca

demand_coordinates = {"Latitude": "10.3969", "Longitude": "63.4269"}



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
demand = hm.create_random_data_demand(demand_count = 5, demand_lat = demand_coordinates["Latitude"], demand_lon = demand_coordinates["Longitude"])
supply = hm.create_random_data_supply(supply_count=8,demand_lat = demand_coordinates["Latitude"], demand_lon = demand_coordinates["Longitude"],supply_coords = supply_coords)
supply.head()
score_function_string_demand = "@lca.calculate_lca_demand(length=Length, area=Area, gwp_factor=Gwp_factor)"
score_function_string_supply_transportation = "@lca.calculate_lca_supply(length=Length, area=Area, gwp_factor=Gwp_factor,demand_lat=Demand_lat,demand_lon=Demand_lon,supply_lat=Supply_lat,supply_lon=Supply_lon,include_transportation=True)"

result_with_transportation = run_matching(demand, supply, score_function_string_demand,score_function_string_supply_transportation, constraints = constraint_dict, add_new = True, sci_milp=True, milp=True, greedy_single=True, bipartite=True)
score_function_string_supply_wo_transportation = "@lca.calculate_lca_supply(length=Length, area=Area, gwp_factor=Gwp_factor,demand_lat=Demand_lat,demand_lon=Demand_lon,supply_lat=Supply_lat,supply_lon=Supply_lon,include_transportation=False)"

result_wo_transportation = run_matching(demand, supply, score_function_string_demand,score_function_string_supply_wo_transportation, constraints = constraint_dict, add_new = True, sci_milp=True, milp=True, greedy_single=True, bipartite=True)



simple_pairs_transportation = hm.extract_pairs_df(result_with_transportation)
simple_results_transportation = hm.extract_results_df(result_with_transportation)
print("Simple pairs including transportation in LCA:")
print(simple_pairs_transportation)
print()
print("Simple results including transportation in LCA:")
print(simple_results_transportation)


simple_pairs_wo_transportation = hm.extract_pairs_df(result_wo_transportation)
simple_results_wo_transportation = hm.extract_results_df(result_wo_transportation)
print("Simple pairs without transportation LCA:")
print(simple_pairs_wo_transportation)
print()
print("Simple results without transportation LCA")
print(simple_results_wo_transportation)



