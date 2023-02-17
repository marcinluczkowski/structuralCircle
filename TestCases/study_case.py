import sys
sys.path.append('./Matching')
import matching
import helper_methods as hm
import LCA as lca

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='} # dictionary of constraints to add to the method


# ========== SCENARIO 1 ============== 
var1 = 1
#d_counts = np.logspace(1, 3, num = 5).astype(int) Use this later when actually testing. Using the below for now to reduce time
d_counts = np.linspace(10, 250, num = 4).astype(int)
s_counts = (d_counts * var1).astype(int)

results = [] #list of results for each iteration

hm.print_header("Starting Run")

for d, s in zip(d_counts, s_counts):
    #create data
    print(f'\n*** Running for {d} demand and {s} supply elements.***\n')
    demand, supply = hm.create_random_data(demand_count=d, supply_count=s)
    results.append(matching.run_matching(demand, supply, constraints = constraint_dict, add_new = True, sci_milp=True, milp=True, greedy_single=False, bipartite=True))
    
    
n_els = d_counts+s_counts # number of elements for each iteration

time_dict = {res[list(res.keys())[0]] : [] for res in results[0]} # create a dictionary for the time spent running each method with different number of elements
lca_dict = {res[list(res.keys())[0]] : [] for res in results[0]}

for iteration in results:
    for method in iteration: # iterate through all methods
        lca_dict[method['Name']].append(method['Match object'].result) 
        time_dict[method['Name']].append(method['Match object'].solution_time) 

pairs_df = pd.concat([res['Match object'].pairs for res in results[0]], axis = 1)
pairs_df.columns = [res[list(res.keys())[0]] for res in results[0]]

fig, ax = plt.subplots()
for key, items in time_dict.items():
    plt.plot(n_els, items, label = key)
plt.legend()
plt.xlabel('Number of elements')
plt.ylabel('Solution time [s]')
plt.yscale('log')
plt.plot()
plt.show()

fig, ax = plt.subplots()
for key, items in lca_dict.items():
    plt.plot(n_els, items, label = key)
plt.legend()
plt.xlabel('Number of elements')
plt.ylabel('LCA_score')
#plt.yscale('log')
plt.plot()
plt.show()