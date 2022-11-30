import matching
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MIN_LENGTH = 2 # m
MAX_LENGTH = 15.0 # m
MIN_AREA = 0.25 # m^2
MAX_AREA = 0.25 # m^2

constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='} # dictionary of constraints to add to the method

def create_random_data(demand_count, supply_count, seed = 2):
    """Create two dataframes for the supply and demand elements used to evaluate the different matrices"""
    np.random.RandomState(seed) #TODO not sure if this is the right way to do it. Read documentation
    demand = pd.DataFrame()
    supply = pd.DataFrame()

    # create element lenghts
    demand['Length'] = ((MAX_LENGTH/2 + 1) - MIN_LENGTH) * np.random.random_sample(size = demand_count) + MIN_LENGTH
    supply['Length'] = ((MAX_LENGTH + 1) - MIN_LENGTH) * np.random.random_sample(size = supply_count) + MIN_LENGTH

    # create element areas independent of the length. Can change this back to Artur's method later, but I want to see the effect of even more randomness. 
    #demand['Area'] = ((MAX_AREA + .001) - MIN_AREA) * np.random.random_sample(size = demand_count) + MIN_AREA
    #supply['Area'] = ((MAX_AREA + .001) - MIN_AREA) * np.random.random_sample(size = supply_count) + MIN_AREA

    # constraints
    demand['Area'] = np.full((demand_count,), MIN_AREA)
    supply['Area'] = np.full((supply_count,), MIN_AREA)


    # intertia moment
    demand['Inertia_moment'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section

    # height - assuming square cross sections
    demand['Height'] = np.power(demand['Area'], 0.5)
    supply['Height'] = np.power(supply['Area'], 0.5)

    supply['Is_new'] = False
    
    # Change index names
    demand.index = map(lambda text: 'D' + str(text), demand.index)
    supply.index = map(lambda text: 'R' + str(text), supply.index)
    
    return demand.round(2), supply.round(2)


# ========== SCENARIO 1 ============== 
var1 = 1
#d_counts = np.logspace(1, 3, num = 5).astype(int) Use this later when actually testing. Using the below for now to reduce time
d_counts = np.linspace(10, 2000, num = 4).astype(int)
s_counts = (d_counts * var1).astype(int)

results = [] #list of results for each iteration

print('======== Starting RUN ============')

for d, s in zip(d_counts, s_counts):
    #create data
    print(f'\n*** Running for {d} demand and {s} supply elements.***\n')
    demand, supply = create_random_data(demand_count=d, supply_count=s)
    results.append(matching.run_matching(demand, supply, constraints = constraint_dict, add_new = False, sci_milp=True))
    
n_els = d_counts*s_counts # number of elements for each iteration

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