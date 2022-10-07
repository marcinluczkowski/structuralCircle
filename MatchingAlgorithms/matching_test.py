from matching import *

demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height'])
supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Is_new'])

# Add a perfect matching pair
demand.loc['D1'] = {'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R1'] = {'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}

# Add non-matchable demand
demand.loc['D2'] = {'Length': 13.00, 'Area': 0.001, 'Inertia_moment':0.00001, 'Height': 0.05}

# Add non-matchable supply
supply.loc['R2'] = {'Length': 0.1, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}

# Add one with two good matches, where second slighlty better
demand.loc['D3'] = {'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R3'] = {'Length': 5.20, 'Area': 0.042, 'Inertia_moment':0.00015, 'Height': 0.22, 'Is_new':False}
supply.loc['R4'] = {'Length': 5.10, 'Area': 0.041, 'Inertia_moment':0.00014, 'Height': 0.21, 'Is_new':False}


import time

matching = Matching(demand, supply, add_new=True, multi=False)

start = time.time()
matching.evaluate()
end = time.time()
print("Weight evaluation execution time: "+str(round(end - start,3))+"sec")

start = time.time()
matching.match_bipartite_graph()
end = time.time()
print(f"Matched: {len(matching.pairs['Supply_id'].unique())} to {matching.pairs['Supply_id'].count()} elements ({100*matching.pairs['Supply_id'].count()/len(demand)}%), resulting in LCA (GWP): {round(matching.result, 2)}kgCO2eq, in: {round(end - start,3)}sec.")

RESULT_FILE = r"C:\Code\structuralCircle\MatchingAlgorithms\result.csv"
matching.pairs.to_csv(RESULT_FILE)
# matching.display_graph()