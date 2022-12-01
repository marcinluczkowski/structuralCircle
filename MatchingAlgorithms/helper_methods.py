import pandas as pd
import numpy as np

# ==== HELPER METHODS ====
# This file contains various methods used for testing and development. 

def extract_pairs_df(dict_list):
    """Creates a dataframe with the matching pairs all evaluated matching methods. 
    input: list of dictionaries for each matching method
    output: dataframe with the matching pairs of all methods."""
    sub_df = []
    cols = []
    for run in dict_list:
        sub_df.append(run['Match object'].pairs)
        cols.append(run['Name'])
    df = pd.concat(sub_df, axis = 1)
    df.columns = cols
    return df

print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")