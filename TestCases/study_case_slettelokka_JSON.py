import sys
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching # Matching
import LCA as lca

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    "Algorithms": ["bipartite", "greedy_plural", "bipartite_plural", "bipartite_plural_multiple"],
    "Include transportation": False,
    "Cite latitude": "59.94161606",
    "Cite longitude": "10.72994518",
    #"Demand file location": r"./CSV/DEMAND_DATAFRAME_SVERRE.xlsx",
    #"Supply file location": r"./CSV/SUPPLY_DATAFRAME_SVERRE.xlsx",
    "Demand file location": r"./CSV/bipartite_plural_demand.csv",
    "Supply file location": r"./CSV/bipartite_plural_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}
#========================#

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
demand["Moment of Inertia"] *=0.00000001
demand.Height *=0.01


#read and clean supply df
supply = pd.read_json(SUPPLY_JSON)
supply_header = supply.iloc[0]
supply.columns = supply_header
supply.drop(axis = 1, index= 0, inplace=True)
supply.reset_index(drop = True, inplace = True)
supply.index = ['S' + str(num) for num in supply.index]

# scale input from mm to m
supply.Length *=0.01
supply.Area *=0.0001
supply["Moment of Inertia"] *=0.00000001
supply.Height *=0.01

#Add material to datasets, only concidering timber
supply["Material"] = "Timber"
demand["Material"] = "Timber"

constraint_dict = constants["constraint_dict"]
score_function_string = hm.generate_score_function_string(constants)
#Add necessary columns to run the algorithm
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)
run_string = hm.generate_run_string(constants)
result_slette = eval(run_string)
slette_pairs = hm.extract_pairs_df(result_slette)
slette_results = hm.extract_results_df(result_slette, constants["Metric"])
print(slette_pairs)
print(slette_results)
