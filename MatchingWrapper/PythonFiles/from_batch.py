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


# Recieve input from terminal arguments

method_number = int(sys.argv[1]) # number of method to run for the problem

demand_json = str(sys.argv[2]) # path to demand json
supply_json = str(sys.argv[3]) # path to supply json
constraint_string =  "{Area : >=, Length: >=}" #str(sys.argv[4]) # string representing the dictionary

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

# convert string constraints to dictionary 
constraint_dict = prepare_dict_string(constraint_string)

# create dataframe from json
demand_df = pd.read_json(demand_json)
supply_df = pd.read_json(supply_json)
demand_df = clean_df(demand_df, ind = 'D') # clean dataframe to get it on the proper format
supply_df = clean_df(supply_df, ind = 'S') # clean dataframe to get it on the proper format


# add imaginary coordinates at this point. Or a check. If no coordinates are present the demand and supply have the same locations
random_lat = 0.0; ramdom_lon = 0.0
demand_df['Latitude'] = random_lat; demand_df['Longitude'] = ramdom_lon; demand_df['Gwp_factor'] = LCA.TIMBER_GWP
supply_df['Latitude'] = random_lat; supply_df['Longitude'] = ramdom_lon; supply_df['Gwp_factor'] = LCA.TIMBER_REUSE_GWP



# How to evaluate. Create score function string without transportation and only considering gwp_factor
score_function_string = "@calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"

#set all methods to false except the one selected by the user. 
# create the mathching object 
add_new = True # Add new elements to the matching problem
solution_limit = 200 # number of second before terminating current algorithm
matching = Matching(demand=demand_df, supply=supply_df, score_function_string=score_function_string,constraints=constraint_dict, add_new=add_new, multi = True, solution_limit=solution_limit)

# run matching
methods_dict = {0 : 'matching.match_greedy(plural_assign = False)',
                1 : 'matching.match_greedy(plural_assign = True)',
                2 : 'matching.match_bipartite_graph()',
                3 : 'matching.match_scipy_milp()',}

#run matching based on user input method
method_string = methods_dict[method_number]
divided_name = re.findall(r"([\w]+)|([.()])", method_string)
parts = [match[0] or match[1] for match in divided_name if match[0] or match[1]]
eval(method_string)

#Write result to file
write_path = '\\'.join(supply_json.split('\\')[:-1] + ['result.json']) # write to the same path as the demand/supply files comes from.
matching.pairs.to_json(write_path) # write the matching pairs back to the result file for use in Grasshopper

# ------ OUTPUT -------
# ------ OUTPUT -------
hm.print_header("Matching complete")
print(f"Matching problem solved using {parts[2]}")
print(f"Total runtime: {round(matching.solution_time,2)} s.")
print(f"Substituted {matching.pairs.Supply_id.str.contains('S').sum()}/{matching.pairs.Supply_id.count()} demand elements")
print(f"Total score after running the matching problem: {matching.result}\n\
From this, transportation contributed with {round(matching.result_transport, 2)}")
print(f'Matching pairs written to {write_path}')
print(matching.pairs)

sys.stdout.flush()