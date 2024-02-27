# import relevant packages
import pandas as pd
import sys
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import helper_methods_LCA as lca


# Create two datasets with two elements in each - demand D and supply S:
demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Gwp_factor'])
demand.loc['D1'] = {'Length': 4.00, 'Area': 0.06, 'Inertia_moment':0.00030, 'Height': 0.30, 'Gwp_factor':lca.data["TIMBER_GWP"]}
demand.loc['D2'] = {'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00010, 'Height': 0.20, 'Gwp_factor':lca.data["TIMBER_GWP"]}
supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Gwp_factor'])
supply.loc['S1'] = {'Length': 5.50, 'Area': 0.045, 'Inertia_moment':0.00010, 'Height': 0.20, 'Gwp_factor':lca.data["TIMBER_REUSE_GWP"]}
supply.loc['S2'] = {'Length': 4.50, 'Area': 0.065, 'Inertia_moment':0.00035, 'Height': 0.35, 'Gwp_factor':lca.data["TIMBER_REUSE_GWP"]}

# create constraint dictionary
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}

# create optimization formula
score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"

# run the matching
result_simple = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=True, bipartite=True)

# display results - matching table
print(hm.extract_pairs_df(result_simple))
# display results - the score
print(hm.extract_results_df(result_simple, "Score"))