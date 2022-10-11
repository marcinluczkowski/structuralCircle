from matching import Matching, logging
import pandas as pd
import time


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
# Add one with two good matches, where second slighlty better
demand.loc['D3'] = {'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R3'] = {'Length': 5.20, 'Area': 0.042, 'Inertia_moment':0.00015, 'Height': 0.22, 'Is_new':False}
supply.loc['R4'] = {'Length': 5.10, 'Area': 0.041, 'Inertia_moment':0.00014, 'Height': 0.21, 'Is_new':False}
# Define the problem to solve
matching = Matching(demand, supply, add_new=True, multi=False)
# evaluate weights
start = time.time()
matching.evaluate()
end = time.time()
logging.info("Weight evaluation execution time: "+str(round(end - start,3))+"sec")
# apply matching
start = time.time()
matching.match_bipartite_graph()
end = time.time()
logging.info(f"Matched: {len(matching.pairs['Supply_id'].unique())} to {matching.pairs['Supply_id'].count()} elements ({100*matching.pairs['Supply_id'].count()/len(demand)}%), resulting in LCA (GWP): {round(matching.result, 2)}kgCO2eq, in: {round(end - start,3)}sec.")

# matching.display_graph()


### Test from JSON files with Slettelokka data 

DEMAND_JSON = r".\sample_demand_input.json"
SUPPLY_JSON = r".\sample_supply_input.json"
RESULT_FILE = r".\result.csv"
#read and clean demand df
demand = pd.read_json(DEMAND_JSON)
demand_header = demand.iloc[0]
demand.columns = demand_header
demand.drop(axis = 1, index= 0, inplace=True)
demand.reset_index(drop = True, inplace = True)
demand.Length *=0.01
demand.Area *=0.0001
demand.Inertia_moment *=0.00000001
demand.Height *=0.01
#read and clean supply df
supply = pd.read_json(SUPPLY_JSON)
supply_header = supply.iloc[0]
supply.columns = supply_header
supply.drop(axis = 1, index= 0, inplace=True)
supply['Is_new'] = False
supply.reset_index(drop = True, inplace = True)
supply.Length *=0.01
supply.Area *=0.0001
supply.Inertia_moment *=0.00000001
supply.Height *=0.01
# Define the problem to solve
matching = Matching(demand, supply, add_new=True, multi=False)

# evaluate weights
start = time.time()
matching.evaluate()
end = time.time()
logging.info("Weight evaluation execution time: "+str(round(end - start,3))+"sec")
# apply matching
start = time.time()
matching.match_bipartite_graph()
end = time.time()
logging.info(f"Matched: {len(matching.pairs['Supply_id'].unique())} to {matching.pairs['Supply_id'].count()} elements ({100*matching.pairs['Supply_id'].count()/len(demand)}%), resulting in LCA (GWP): {round(matching.result, 2)}kgCO2eq, in: {round(end - start,3)}sec.")

# matching.display_graph()


### Test with random generated elements

import random

random.seed(3)

DEMAND_COUNT = 100
SUPPLY_COUNT = 2000
MIN_LENGTH = 1.0
MAX_LENGTH = 10.0
MIN_AREA = 0.0025   # 5x5cm
MAX_AREA = 0.25     # 50x50cm

demand = pd.DataFrame()
demand['Length'] = [x/10 for x in random.choices(range(int(MIN_LENGTH*10), int(MAX_LENGTH*10)), k=DEMAND_COUNT)]        # [m], random between the range
demand['Area'] = demand.apply(lambda row: round((random.choice(range(0, int(MAX_AREA*10000)-int(MIN_AREA*10000))) /10000 /MAX_LENGTH * row['Length'] + MIN_AREA) * 10000)/10000, axis=1)        # [m2], random between the range but dependent on the length of the element
demand['Inertia_moment'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
demand['Height'] = demand.apply(lambda row: row['Area']**(0.5), axis=1)   # derived from area assuming square section
supply = pd.DataFrame()
supply['Length'] = [x/10 for x in random.choices(range(int(MIN_LENGTH*10), int(MAX_LENGTH*10)), k=SUPPLY_COUNT)]        # [m], random between the range
supply['Area'] = supply.apply(lambda row: round((random.choice(range(0, int(MAX_AREA*10000)-int(MIN_AREA*10000))) /10000 /MAX_LENGTH * row['Length'] + MIN_AREA) * 10000)/10000, axis=1)        # [m2], random between the range but dependent on the length of the element
supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
supply['Height'] = supply.apply(lambda row: row['Area']**(0.5), axis=1)   # derived from area assuming square section
supply['Is_new'] = [False for i in range(SUPPLY_COUNT)]
# Define the problem to solve
matching = Matching(demand, supply, add_new=True, multi=False)
# evaluate weights
start = time.time()
matching.evaluate()
end = time.time()
logging.info("Weight evaluation execution time: "+str(round(end - start,3))+"sec")
# apply matching
start = time.time()
matching.match_bipartite_graph()
end = time.time()
logging.info(f"Matched: {len(matching.pairs['Supply_id'].unique())} to {matching.pairs['Supply_id'].count()} elements ({100*matching.pairs['Supply_id'].count()/len(demand)}%), resulting in LCA (GWP): {round(matching.result, 2)}kgCO2eq, in: {round(end - start,3)}sec.")

# matching.display_graph()