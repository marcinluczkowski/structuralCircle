import sys
#import json
import pandas as pd
import numpy as np

path = sys.argv[1] # get the input json file
out_path = sys.argv[2] # get the output path
#dict_in = json.loads(string_in)
#print(string_in)
#print(string_in)
df = pd.read_json(path) # read the json file into a dataframe

# calculate the midpoint of each line
mid = lambda row, i, j : (row[i] + row[j]) / 2

new_cols = ['Xm', 'Ym', 'Zm']
cols = df.columns
for i, col in enumerate(new_cols):
    df[col] = np.add(df[cols[i]] , df[cols[i+3]] ) / 2

# return the new columns from the df. as json
df[new_cols].to_json(out_path)
#print(df[new_cols].to_json(out_path))
print(f'File saved to {out_path}')   