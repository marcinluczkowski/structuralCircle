import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching # Matching
import LCA as lca

demand = hm.import_dataframe_from_csv(r"C:\Users\sigur\OneDrive - NTNU\Masteroppgave\CSV\genetic_demand.csv")
supply = hm.import_dataframe_from_csv(r"C:\Users\sigur\OneDrive - NTNU\Masteroppgave\CSV\genetic_supply.csv")

# create constraint dictionary
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}
# TODO add 'Material': '=='

hm.print_header('Genetic study case')

score_function_string_demand = "@lca.calculate_lca_demand(length=Length, area=Area, gwp_factor=Gwp_factor)"
score_function_string_supply = "@lca.calculate_lca_supply(length=Length, area=Area, gwp_factor=Gwp_factor,demand_lat=Demand_lat,demand_lon=Demand_lon,supply_lat=Supply_lat,supply_lon=Supply_lon,include_transportation=False)"
result_simple = run_matching(demand, supply, score_function_string_demand=score_function_string_demand, score_function_string_supply = score_function_string_supply, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=False, greedy_plural = True, bipartite=False, genetic =True)

#FIXME When testing with new elements. Why are the scores (LCA) identical even though we have different matching DataFrames. 

simple_pairs = hm.extract_pairs_df(result_simple)
simple_results = hm.extract_results_df(result_simple)

print("Simple pairs:")
print(simple_pairs)

print()
print("Simple results")
print(simple_results)
