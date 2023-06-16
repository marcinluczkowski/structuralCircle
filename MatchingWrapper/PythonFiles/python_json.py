import sys # import system to read arguments
import os
import numpy as np # import numpy for calculating the periods
import pandas as pd
# Import relevant methods and information
#from elementmatcher.src.matching import Matching, run_matching
#from elementmatcher.src.helper_methods import extract_results_df, generate_run_string
#from elementmatcher.src.LCA import (calculate_lca, calculate_score, 
#                                    TIMBER_DENSITY, TIMBER_GWP, TIMBER_REUSE_GWP)


# Recieve input from terminal arguments
"""
method_number = int(sys.argv[1]) # number of method to run for the problem
demand_json = str(sys.argv[2]) # path to demand json
supply_json = str(sys.argv[3]) # path to supply json
"""
def clean_df(df):
    cols = df.loc[0].values
    df.drop(0,inplace = True)
    df.columns = cols

# create dataframe from json
demand_df = pd.read_json(r'C:\Users\sverremh\AppData\Roaming\Grasshopper\Libraries\MatchingWrapper\PythonFiles\demand.json')
supply_df = pd.read_json(r'C:\Users\sverremh\AppData\Roaming\Grasshopper\Libraries\MatchingWrapper\PythonFiles\supply.json')
clean_df(demand_df) # clean dataframe to get it on the proper format
clean_df(supply_df) # clean dataframe to get it on the proper format




# ------ OUTPUT -------
print("The file have sucessfully run...")
sys.stdout.flush() # flush print to terminal.  


"""
amplitude = float(sys.argv[1]) # first input number
periods = float(sys.argv[2]) # second input number

x_array = np.arange(0, 6 * np.pi, step = 0.1)
y_array = amplitude * np.sin(x_array * periods)

# This path should change as well
path = sys.path[0] # Path we're working from 
print("Working path in python: " + path)
#print(path)

# set working dir

filepath = os.path.join(path, "first_test.csv")
with open(filepath, 'w') as f:
    f.write('x_coordinate,y_coordinate\n')
    for x, y in zip(x_array, y_array):
        f.write('{0},{1}\n'.format(x,y))    

print("CSV-file named {0} created".format(filepath))
"""