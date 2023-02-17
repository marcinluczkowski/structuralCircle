import sys
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching # Matching
import LCA as lca

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Test from JSON files with Slettelokka data 
hm.print_header("SLETTELØKKA MATCHING")


DEMAND_JSON = r".\Data\Sample_JSON\sample_demand_input.json"
SUPPLY_JSON = r".\Data\Sample_JSON\sample_supply_input.json"
RESULT_FILE = r".\Data\Sample_JSON\result.csv"
#read and clean demand df
demand = pd.read_json(DEMAND_JSON)
demand_header = demand.iloc[0]
demand.columns = demand_header
demand.drop(axis = 1, index= 0, inplace=True)
demand.reset_index(drop = True, inplace = True)
demand.index = ['D' + str(num) for num in demand.index]

demand.Length *=0.01
demand.Area *=0.0001
demand.Inertia_moment *=0.00000001
demand.Height *=0.01
demand.Gwp_factor = lca.TIMBER_GWP

#read and clean supply df
supply = pd.read_json(SUPPLY_JSON)
supply_header = supply.iloc[0]
supply.columns = supply_header
supply.drop(axis = 1, index= 0, inplace=True)
supply.reset_index(drop = True, inplace = True)
supply.index = ['R' + str(num) for num in supply.index]
supply.Gwp_factor = lca.TIMBER_REUSE_GWP

# scale input from mm to m
supply.Length *=0.01
supply.Area *=0.0001
supply.Inertia_moment *=0.00000001
supply.Height *=0.01

constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}

#--- CREATE AND EVALUATE ---
score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"
result_slette = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True, sci_milp=True, milp=False, greedy_single=True, bipartite=True)


slette_pairs = hm.extract_pairs_df(result_slette)
slette_results = hm.extract_results_df(result_slette)
print(slette_pairs)
print(slette_results)
