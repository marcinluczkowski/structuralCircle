# -*- coding: utf-8 -*-
"""
Python script for reading input json files and do some operations on them using
a mapping script.  
Created on Thu Sep 29 10:50:44 2022

@author: sverremh
"""

import pandas as pd
import numpy as np
import sys

# --- Retrieve input ---
#demand_json = "C:\\Users\\sverremh\\AppData\\Roaming\\Grasshopper\\Libraries\\FirstPython\\Debug\\net48\\Files\\demand_input.json"  #sys.argv[2] # path to json demand file
#supply_json= "C:\\Users\\sverremh\\AppData\\Roaming\\Grasshopper\\Libraries\\FirstPython\\Debug\\net48\Files\\supply_input.json"#sys.argv[3] # path to json supply file
#result_path = "C:\\Users\\sverremh\\AppData\\Roaming\\Grasshopper\\Libraries\\FirstPython\\Debug\\net48\\Files\\result_mapping.json"#sys.argv[4] # path to write result file to

demand_json = sys.argv[1] # path to json demand file
supply_json= sys.argv[2] # path to json supply file
result_path = sys.argv[3] # path to write result file to
constraints = sys.argv[4] # constraint string to use for the matching object
method = sys.argv[5] # method to use for the matching object



# --- Prepare data ---
df_demand = pd.read_json(demand_json)
demand_header = df_demand.iloc[0]
df_demand.columns = demand_header
df_demand.drop(axis = 1, index= 0, inplace=True)


# supply dataframe
df_supply = pd.read_json(supply_json)
supply_header = df_supply.iloc[0]
df_supply.columns = supply_header
df_supply.drop(axis = 1, index= 0, inplace=True)





# --- Method ---
# add unique id similar to Artur's python script
df_demand['ID'] =  np.arange(df_demand.shape[0])
df_supply['ID'] =  np.arange(df_supply.shape[0])

# convert to numpy array
demand_np = df_demand.to_numpy(dtype = float)
supply_np = df_supply.to_numpy(dtype = float)

mapping_id = []
logs = []
plural_assign = True

outer_count = demand_np.shape[0]
inner_count = supply_np.shape[0]

print(list(range(outer_count)))
print(range(inner_count))


print(f'Some number from array: {demand_np[0][0]}')
for i in range(outer_count):
    match = False
    for j in range(inner_count): 
        if demand_np[i][0] <= supply_np[j][0] and demand_np[i][1] <= supply_np[j][1] and demand_np[i][2] <= supply_np[j][2] and demand_np[i][3] <= supply_np[j][3]:
            match = True
            mapping_id.append(supply_np[j][-1])
            break
    if match: 
        if plural_assign:
            # shorten the supply element:
            supply_np[j][0] = supply_np[j][0] - demand_np[i][0]
            # sort the supply list
            supply_np = sorted(supply_np, key=lambda x: x[0]) # TODO move this element instead of sorting whole list
            logs.append("#"+str(i)+" Found element #"+str(j)+" and utilized only "+str(supply_np[j][0]/1000)+"m of "+str(demand_np[i][0]/1000)+"m. Demand: L="+str(demand_np[i][0]/1000)+"m, A="+str(demand_np[i][1]/100)+"cm2, I="+str(demand_np[i][2]/10000)+"cm4, H="+str(demand_np[i][3]/10)+"cm.")

        else:
            del supply_np[j]
            logs.append("#"+str(i)+" Found element #"+str(j)+" and utilized fully. Demand: L="+str(demand_np[i][0]/1000)+"m, A="+str(demand_np[i][1]/100)+"cm2, I="+str(demand_np[i][2]/10000)+"cm4, H="+str(demand_np[i][3]/10)+"cm.")
                        
    else:
        mapping_id.append(None)
        logs.append("#"+str(i)+" Not found. Demand: L="+str(demand_np[i][0]/1000)+"m, A="+str(demand_np[i][1]/100)+"cm2, I="+str(demand_np[i][2]/10000)+"cm4, H="+str(demand_np[i][3]/10)+"cm.")  


result_df = pd.DataFrame(mapping_id, columns = ["MappingIndex"])

print("Demand dataframe:")
print(df_demand.head(10))
print("\nSupply Dataframe:")
print(df_supply.head(10))

print('----- RESULTS -----')
#print(result_df.head(19))

# --- Write results --- 
result_df.to_json(result_path)