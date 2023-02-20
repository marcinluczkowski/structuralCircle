import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching # Matching
import LCA as lca

### Test with just few elements

demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Gwp_factor'])
supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Gwp_factor'])

# Add a perfect matching pair
demand.loc['D1'] = {'Material': 1, 'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Gwp_factor':lca.TIMBER_GWP}
supply.loc['S1'] = {'Material': 1, 'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Gwp_factor':lca.TIMBER_REUSE_GWP}

# Add non-matchable demand
# demand.loc['D2'] = {'Material': 1, 'Length': 13.00, 'Area': 0.001, 'Inertia_moment':0.00001, 'Height': 0.05}
# TODO new inertia moment
demand.loc['D2'] = {'Material': 1, 'Length': 13.00, 'Area': 0.02, 'Inertia_moment':0.00001, 'Height': 0.05, 'Gwp_factor':lca.TIMBER_GWP}

# Add non-matchable supply
# supply.loc['R2'] = {'Material': 1, 'Length': 0.1, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}
supply.loc['S2'] = {'Material': 1, 'Length': 1.2, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Gwp_factor':lca.TIMBER_REUSE_GWP}

# Add element with two good matches, where second slighlty better
demand.loc['D3'] = {'Material': 1, 'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Gwp_factor':lca.TIMBER_GWP}
supply.loc['S3'] = {'Material': 1, 'Length': 5.20, 'Area': 0.042, 'Inertia_moment':0.00015, 'Height': 0.22, 'Gwp_factor':lca.TIMBER_REUSE_GWP}
supply.loc['S4'] = {'Material': 1, 'Length': 5.10, 'Area': 0.041, 'Inertia_moment':0.00014, 'Height': 0.21, 'Gwp_factor':lca.TIMBER_REUSE_GWP}

# Add element with much bigger match
demand.loc['D4'] = {'Material': 1, 'Length': 8.00, 'Area': 0.1, 'Inertia_moment':0.0005, 'Height': 0.50, 'Gwp_factor':lca.TIMBER_GWP}
supply.loc['S5'] = {'Material': 1, 'Length': 12.00, 'Area': 0.2, 'Inertia_moment':0.0008, 'Height': 0.8, 'Gwp_factor':lca.TIMBER_REUSE_GWP}

# Add supply that can after cut fits perfectly
#demand.loc['D5'] = {'Material': 1, 'Length': 3.50, 'Area': 0.19, 'Inertia_moment':0.0008, 'Height': 0.80, 'Gwp_factor':lca.TIMBER_GWP}
#demand.loc['D6'] = {'Material': 1, 'Length': 5.50, 'Area': 0.18, 'Inertia_moment':0.00076, 'Height': 0.75, 'Gwp_factor':lca.TIMBER_GWP}
#supply.loc['R6'] = {'Material': 1, 'Length': 9.00, 'Area': 0.20, 'Inertia_moment':0.0008, 'Height': 0.8, 'Gwp_factor':lca.TIMBER_REUSE_GWP}

# Add element that fits the cut from D4 when allowing multiple assignment
demand.loc['D5'] = {'Material': 1, 'Length': 4.00, 'Area': 0.1, 'Inertia_moment':0.0005, 'Height': 0.50, 'Gwp_factor':lca.TIMBER_GWP}

# create constraint dictionary
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}
# TODO add 'Material': '=='

hm.print_header('Simple Study Case')

score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"
result_simple = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=True, bipartite=True)

#FIXME When testing with new elements. Why are the scores (LCA) identical even though we have different matching DataFrames. 

simple_pairs = hm.extract_pairs_df(result_simple)
simple_results = hm.extract_results_df(result_simple)

print("Simple pairs:")
print(simple_pairs)

print()
print("Simple results")
print(simple_results)

# Calculate volumes
dem = result_simple[0]['Match object'].demand
sup = result_simple[0]['Match object'].supply


indices = list(dem.index)
simple_el_ids = simple_pairs.mask(simple_pairs.isna(), indices, axis = 0) # replace NaN values with the intital index.
areas = simple_el_ids.applymap(lambda el : dem.Area[el] if 'D' in el else sup.Area[el]) # Find the correct areas for each matching.
volumes = areas.apply(lambda row: row * dem.Length.to_numpy(), axis = 0) # Get the volume by calculating the correct area with the length of demand elements.
total_volumes = volumes.sum() # Sum each column to get total volumes. 
initial_volume = sum(dem.Area*dem.Length)

ratios = (total_volumes - initial_volume) / initial_volume


print(ratios)
# result_simple[0]['Match object'].display_graph()

### Add scatter plot:

# import matplotlib.pyplot as plt
# plt.scatter(demand.Length, demand.Area, s=50, c='b', marker="X", label='Demand')
# plt.scatter(supply.Length, supply.Area, s=50, c='r', marker="X", label='Supply') 
# plt.legend()
# plt.xlabel("Length")
# plt.ylabel("Area")
# for i, row in demand.iterrows():
#     plt.annotate(i, (row['Length']-0.6, row['Area']-0.004))
# for i, row in supply.iterrows():
#     if i != "R4":
#         plt.annotate(i, (row['Length']+0.2, row['Area']-0.003))
# plt.show()

# simple_pairs = hm.extract_pairs_df(result_simple)
# print(simple_pairs)