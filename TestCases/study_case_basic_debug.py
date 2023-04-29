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
demand.loc['D1'] = {'Material': 1, 'Length': 9.90, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Gwp_factor':lca.TIMBER_GWP}
demand.loc['D2'] = {'Material': 1, 'Length': 0.0999999999, 'Area': 0.04, 'Inertia_moment':0.000013, 'Height': 0.05, 'Gwp_factor':lca.TIMBER_GWP}
supply.loc['S1'] = {'Material': 1, 'Length': 10.00, 'Area': 0.40, 'Inertia_moment':0.00013, 'Height': 0.20, 'Gwp_factor':lca.TIMBER_REUSE_GWP}

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

assignment_summary_df = hm.get_assignment_df(simple_pairs, sup.index) # create overview of element assignments


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