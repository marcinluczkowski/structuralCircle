from igraph import Matrix
from matching import Matching
import pandas as pd
import random

### Test with just few elements

demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height'])
supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Is_new'])
# Add a perfect matching pair
demand.loc['D1'] = {'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R1'] = {'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}
# Add non-matchable demand
demand.loc['D2'] = {'Length': 13.00, 'Area': 0.001, 'Inertia_moment':0.00001, 'Height': 0.05}
# Add non-matchable supply
supply.loc['R2'] = {'Length': 0.1, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}
# Add element with two good matches, where second slighlty better
demand.loc['D3'] = {'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R3'] = {'Length': 5.20, 'Area': 0.042, 'Inertia_moment':0.00015, 'Height': 0.22, 'Is_new':False}
supply.loc['R4'] = {'Length': 5.10, 'Area': 0.041, 'Inertia_moment':0.00014, 'Height': 0.21, 'Is_new':False}
# Add element with much bigger match
demand.loc['D4'] = {'Length': 8.00, 'Area': 0.1, 'Inertia_moment':0.0005, 'Height': 0.50}
supply.loc['R5'] = {'Length': 12.00, 'Area': 0.2, 'Inertia_moment':0.0008, 'Height': 0.8, 'Is_new':False}

# create constraint dictionary
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>=', 'Height': '>='}


# --- Old incidence matrix --- 
# create matching object
matching0 = Matching(demand, supply, add_new=True, multi=False, constraints = constraint_dict)
matching0.evaluate2() #TODO Delete this method if the above methods work
incidence0 = matching0.incidence.copy()
matching0.match_bipartite_graph()
matching0.match_nested_loop(plural_assign=False)
matching0.match_nested_loop(plural_assign=True)



# --- New incidence analysis matrix ---
matching1 = Matching(demand, supply, add_new=True, multi=False, constraints = constraint_dict)
matching1.evaluate() #TODO Delete this method if the above methods work
matching1.weigth_incidence()
incidence1 = matching1.incidence.copy()
matching1.match_bipartite_graph()
matching1.match_nested_loop(plural_assign=False)
matching1.match_nested_loop(plural_assign=True)


print(incidence0 == incidence1)
