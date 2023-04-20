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
demand_file_location = r"./CSV/pdf_demand.csv"
supply_file_location = r"./CSV/pdf_supply.csv"

#Constants
TIMBER_GWP = 28.9       # based on NEPD-3442-2053-EN
TIMBER_REUSE_GWP = 2.25        # 0.0778*28.9 = 2.25 based on Eberhardt
TRANSPORT_GWP = 96.0    # TODO kg/m3/t based on ????
TIMBER_DENSITY = 491.0  # kg, based on NEPD-3442-2053-EN
STEEL_GWP = None
STEEL_REUSE_GWP = None
TRANSPORTATION_GWP = None
VALUATION_GWP = None
TIMBER_PRICE = None
TIMBER_REUSE_PRICE = None
STEEL_PRICE = None
STEEL_REUSE_PRICE = None
PRICE_TRANSPORTATION = None
TIMBER_DENSITY = None
STEEL_DENSITY = None

#========================#
constants = [TIMBER_GWP, TIMBER_REUSE_GWP, TRANSPORT_GWP, TIMBER_DENSITY, STEEL_GWP, STEEL_REUSE_GWP, TRANSPORTATION_GWP, VALUATION_GWP, TIMBER_PRICE, TIMBER_REUSE_PRICE, STEEL_PRICE, STEEL_REUSE_PRICE,
PRICE_TRANSPORTATION,
TIMBER_DENSITY,
STEEL_DENSITY,

score_function_string = hm.generate_score_function_string(metric, include_transportation)


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


