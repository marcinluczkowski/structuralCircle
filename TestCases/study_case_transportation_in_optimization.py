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

print(rd.randint(0, len(supply_coords)))




constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='} # dictionary of constraints to add to the method
demand, supply = hm.create_random_data(demand_count = 4, supply_count=5, demand_lat = demand_coordinates["Latitude"], demand_lon = demand_coordinates["Longitude"], supply_coords = supply_coords)
score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"
result = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True, sci_milp=True, milp=True, greedy_single=True, bipartite=True)
simple_pairs = hm.extract_pairs_df(result)
simple_results = hm.extract_results_df(result)
print("Simple pairs:")
print(simple_pairs)
print()
print("Simple results")
print(simple_results)


