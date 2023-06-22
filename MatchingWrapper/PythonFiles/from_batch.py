import sys # import system to read arguments
import os
import numpy as np # import numpy for calculating the periods
import pandas as pd
import ast # for converting string input to dictionary
import re # regular expression operators
# Import relevant methods and information
from elementmatcher import (Matching, run_matching,
                            )
from elementmatcher.src.helper_methods import extract_results_df, generate_run_string
from elementmatcher.src.LCA import (calculate_lca, calculate_score, 
                                    TIMBER_DENSITY, TIMBER_GWP, TIMBER_REUSE_GWP)
from elementmatcher import Matching
from elementmatcher import LCA
from elementmatcher import helper_methods as hm

# ----- INPUT -----

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
    "Project name": "Sletteløkka",
    "Metric": "GWP",
    "Algorithms": ["bipartite", "greedy_plural", "bipartite_plural", "bipartite_plural_multiple"],
    "Include transportation": False,
    "Site latitude": "59.94161606", # Sletteløkka
    "Site longitude": "10.72994518", # Sletteløkka
    "Demand file location": r"",
    "Supply file location": r"",
    "constraint_dict": {'Area' : '>=', 'Length' : '>=', 'Material': '=='}, # Default dictionary
    "Method Number" : 4, # Greedy Plural by default
}

constants["Method Number"] = int(sys.argv[1]) # number of method to run for the problem

constants["Demand file location"] =  str(sys.argv[2]) # path to demand json
constants["Supply file location"] = str(sys.argv[3]) # path to supply json
constants["constraint_dict"] =  str(sys.argv[4]) # string representing the dictionary

# ----- METHODS -----

def clean_df(df, ind = ''):
    cols = df.loc[0].values
    df.drop(0,inplace = True)
    df.columns = cols
    df = df.apply(pd.to_numeric, axis = 0, **{'errors' : 'ignore'}) # Convert to numeric data if possible
    index_mapper =  {i : f'{ind}{i}' for i in range(df.shape[0]+1)}
    df.rename(index = index_mapper, inplace = True)
    return df

def prepare_dict_string(dict_string):
    # Use regular expressions to enclose keys and values in single quotes
    pattern = r"(\b\w+\b)\s*:\s*([^,}]+|'[^']+?')" # regular expression for modyfing the string
    modified_string = re.sub(pattern,  r"'\1': '\2'", dict_string)
    c_dict = ast.literal_eval(modified_string) # Construct the dictionary from a valid string
    return c_dict 

# ----- PREPARE DATA -----
building_latitude = 59.945757 # Latitude of Sletteløkka #TODO Make this user-input
building_longitude = 10.843786 # Longitude of Sletteløkka


# convert string constraints to dictionary 
constraint_dict = prepare_dict_string(constants["constraint_dict"])

# create dataframe from json
demand_df = pd.read_json(constants["Demand file location"])
supply_df = pd.read_json(constants["Supply file location"])
demand_df = clean_df(demand_df, ind = 'D') # clean dataframe to get it on the proper format
supply_df = clean_df(supply_df, ind = 'S') # clean dataframe to get it on the proper format

# Check for transportation columns in the input data. If found, we use Transportation, else, we don't
column_search = "Long.*|Lat.*"
if( (supply_df.columns.str.contains(column_search).sum() > 1) and (demand_df.columns.str.contains(column_search).sum() >1)):
    constants["Include transportation"] = True
    
supply_df = hm.add_necessary_columns_pdf(supply_df, constants)
demand_df = hm.add_necessary_columns_pdf(demand_df, constants)

constants["Include transportation"] = True
# How to evaluate. Create score function string without transportation and only considering gwp_factor
score_function_string = hm.generate_score_function_string(constants) #New scorestring


# create the mathching object 
#add_new = True # Add new elements to the matching problem
solution_limit = 200 # number of second before terminating current algorithm

matching = Matching(demand=demand_df, supply=supply_df, score_function_string=score_function_string,constraints=constraint_dict, add_new = True, multi = True, solution_limit=solution_limit)

# run matching
methods_dict = {0 : 'matching.match_greedy()',
                1 : 'matching.match_greedy(plural_assign = True)',
                2 : 'matching.match_bipartite_graph()',
                3 : 'matching.match_scipy_milp()',
                4 : 'matching.match_mixed_integer_programming()',
                5 : 'matching.match_bipartite_plural()',
                6 : 'matching.match_bipartite_plural_multiple()',}

#run matching based on user input method
method_string = methods_dict[constants["Method Number"]] # Select correct method to run
divided_name = re.findall(r"([\w]+)|([.()])", method_string)
parts = [match[0] or match[1] for match in divided_name if match[0] or match[1]] 

eval(method_string) # Run matching with input method

#Write result to file
write_path = '\\'.join(constants["Supply file location"].split('\\')[:-1] + ['result.json']) # write to the same path as the demand/supply files comes from.
matching.pairs.to_json(write_path) # write the matching pairs back to the result file for use in Grasshopper

# ------ OUTPUT -------
hm.print_header("Matching complete")
print(f"Matching problem solved using {parts[2]}")
print(f"Total runtime: {round(matching.solution_time,2)} s.")
print(f"Substituted {matching.pairs.Supply_id.count()} demand elements with {matching.pairs.Supply_id.str.contains('S').sum()} reclaimed supply elements ")
print(f"Total score after running the matching problem: {matching.result}\n\
From this, transportation contributed with {round(matching.result_transport, 2)}")
print(f'Matching pairs written to {write_path}')
print(matching.pairs)