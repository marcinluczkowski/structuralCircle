import sys
sys.path.append('./Matching')
from matching import Matching
import pandas as pd
import numpy as np
import sys
import LCA as lca

# read input argument from console
method_name = sys.argv[1]
demand_path = sys.argv[2]
supply_path = sys.argv[3]
result_path = sys.argv[4]
constraint_string = sys.argv[5]

# read and clean demand df
#read and clean demand df
demand = pd.read_json(demand_path)
demand_header = demand.iloc[0]
demand.columns = demand_header
demand.drop(axis = 1, index= 0, inplace=True)
demand.reset_index(drop = True, inplace = True)
demand.Length *=0.01
demand.Area *=0.0001
demand.Inertia_moment *=0.00000001
demand.Height *=0.01
demand.Gwp_factor = lca.TIMBER_GWP
#read and clean supply df
supply = pd.read_json(supply_path)
supply_header = supply.iloc[0]
supply.columns = supply_header
supply.drop(axis = 1, index= 0, inplace=True)
supply.reset_index(drop = True, inplace = True)
supply.Length *=0.01
supply.Area *=0.0001
supply.Inertia_moment *=0.00000001
supply.Height *=0.01
supply.Gwp_factor = lca.TIMBER_REUSE_GWP

# constraints: these are added 

# create matching object
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>=', 'Height': '>='} # make this as dynamic dictionary based on input string.

matching = Matching( demand, supply, add_new=False, constraints = constraint_dict)
matching.evaluate()
matching.get_weights() #TODO Move into methods which needs weighting
# do the matching
if method_name == "nestedList":
    matching.match_nested_loop(plural_assign=True)

elif method_name == "bipartiteGraph":
    matching.match_bipartite_graph()

elif method_name == "milp":
    matching.match_cp_solver()

else:
    matching.match_nested_loop(plural_assign=True)


# write result file
matching.pairs.to_json(result_path)