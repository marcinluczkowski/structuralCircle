from matching import Matching
import pandas as pd
import random
import time


print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")
### Test with just few elements


demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height'])
supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Is_new'])
# Add a perfect matching pair
demand.loc['D1'] = {'Material': 1, 'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R1'] = {'Material': 1, 'Length': 7.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}
# Add non-matchable demand
demand.loc['D2'] = {'Material': 1, 'Length': 13.00, 'Area': 0.001, 'Inertia_moment':0.00001, 'Height': 0.05}
# Add non-matchable supply
supply.loc['R2'] = {'Material': 1, 'Length': 0.1, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20, 'Is_new':False}
# Add element with two good matches, where second slighlty better
demand.loc['D3'] = {'Material': 1, 'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00013, 'Height': 0.20}
supply.loc['R3'] = {'Material': 1, 'Length': 5.20, 'Area': 0.042, 'Inertia_moment':0.00015, 'Height': 0.22, 'Is_new':False}
supply.loc['R4'] = {'Material': 1, 'Length': 5.10, 'Area': 0.041, 'Inertia_moment':0.00014, 'Height': 0.21, 'Is_new':False}
# Add element with much bigger match
demand.loc['D4'] = {'Material': 1, 'Length': 8.00, 'Area': 0.1, 'Inertia_moment':0.0005, 'Height': 0.50}
supply.loc['R5'] = {'Material': 1, 'Length': 12.00, 'Area': 0.2, 'Inertia_moment':0.0008, 'Height': 0.8, 'Is_new':False}
# Add supply that can after cut fits perfectly
#demand.loc['D5'] = {'Material': 1, 'Length': 3.50, 'Area': 0.19, 'Inertia_moment':0.0008, 'Height': 0.80}
#demand.loc['D6'] = {'Material': 1, 'Length': 5.50, 'Area': 0.18, 'Inertia_moment':0.00076, 'Height': 0.75}
#supply.loc['R6'] = {'Material': 1, 'Length': 9.00, 'Area': 0.20, 'Inertia_moment':0.0008, 'Height': 0.8, 'Is_new':False}
# Add element that fits the cut from D4 when allowing multiple assignment
demand.loc['D5'] = {'Length': 4.00, 'Area': 0.1, 'Inertia_moment':0.0005, 'Height': 0.50}


# create constraint dictionary
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}
# TODO add 'Material': '=='

print_header("Simple Study Case")

matching = Matching(demand, supply, add_new=True, multi=False, constraints = constraint_dict)
matching.evaluate()
matching.weight_incidence()
matching.match_bipartite_graph()
weight_bi = matching.weights.copy(deep = True).sum().sum()
pairs_bi = matching.pairs.copy(deep = True)

matching.match_greedy_algorithm(plural_assign=False)
weight_g0 = matching.weights.copy(deep = True).sum().sum()
greedy0 = matching.pairs.copy(deep = True)

matching.match_greedy_algorithm(plural_assign=True)
weight_g1 = matching.weights.copy(deep = True).sum().sum()
greedy1 = matching.pairs.copy(deep = True)
#matching.match_genetic_algorithm()
#matching.match_mixed_integer_programming() #TODO Make the "pairs" df similar to the other methods, Now it is integers
matching.match_cp_solver()
weight_cp = matching.weights.copy(deep = True).sum().sum()
cp = matching.pairs.copy(deep = True)
#milp = matching.pairs.copy(deep=True)

weight_g1 = matching.weights.copy(deep = True).sum().sum()
greedy1 = matching.pairs.copy(deep = True)
#matching.match_mixed_integer_programming() #TODO Make the "pairs" df similar to the other methods, Now it is integers
#milp = matching.pairs.copy(deep=True)
# matching.match_genetic_algorithm()
test = pd.concat([pairs_bi, greedy0, greedy1, cp], axis = 1) # look at how all the assignments are working.
test.columns = ["Bipartite", "Greedy_single", "Greedy_multiple", "MILP"]

"""
### Test from JSON files with Slettelokka data 
print_header("SLETTELØKKA MATCHING")
matching = Matching(demand, supply, add_new=True, multi=False, constraints = constraint_dict)

DEMAND_JSON = r"MatchingAlgorithms\sample_demand_input.json"
SUPPLY_JSON = r"MatchingAlgorithms\sample_supply_input.json"
RESULT_FILE = r"MatchingAlgorithms\result.csv"
#read and clean demand df
demand = pd.read_json(DEMAND_JSON)
demand_header = demand.iloc[0]
demand.columns = demand_header
demand.drop(axis = 1, index= 0, inplace=True)
demand.reset_index(drop = True, inplace = True)
demand.index = ['D' + str(num) for num in demand.index]

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
supply.index = ['R' + str(num) for num in supply.index]

# scale input from mm to m
supply.Length *=0.01
supply.Area *=0.0001
supply.Inertia_moment *=0.00000001
supply.Height *=0.01

#--- CREATE AND EVALUATE ---
incidence_shapes = []
matching = Matching(demand, supply, add_new=True, multi=False, constraints = constraint_dict)
matching.evaluate()
matching.weight_incidence()
matching.match_bipartite_graph()
incidence_shapes.append(matching.incidence.shape)
weight_bi = matching.weights.copy(deep = True).sum().sum()
pairs_bi = matching.pairs.copy(deep = True)

matching.match_greedy_algorithm(plural_assign=False)
weight_g0 = matching.weights.copy(deep = True).sum().sum()
greedy0 = matching.pairs.copy(deep = True)
incidence_shapes.append(matching.incidence.shape)

matching.match_greedy_algorithm(plural_assign=True)
weight_g1 = matching.weights.copy(deep = True).sum().sum()
greedy1 = matching.pairs.copy(deep = True)
incidence_shapes.append(matching.incidence.shape)

matching.match_cp_solver()
"""
test = pd.concat([pairs_bi, greedy0, greedy1], axis = 1) # look at how all the assignments are working.
# matching.match_cp_solver()
# ERROR matching.match_mixed_integer_programming()


### Test with random generated elements
print_header("RANDOM ELEMENTS n_D = 100, n_S = 200")
random.seed(3)

DEMAND_COUNT = 100
SUPPLY_COUNT = 200
MIN_LENGTH = 1.0
MAX_LENGTH = 10.0
MIN_AREA = 0.0025   # 5x5cm
MAX_AREA = 0.25     # 50x50cm

demand = pd.DataFrame()
demand['Length'] = [x/10 for x in random.choices(range(int(MIN_LENGTH*10), int(MAX_LENGTH*10)), k=DEMAND_COUNT)]        # [m], random between the range
demand['Area'] = demand.apply(lambda row: round((random.choice(range(0, int(MAX_AREA*10000)-int(MIN_AREA*10000))) /10000 /MAX_LENGTH * row['Length'] + MIN_AREA) * 10000)/10000, axis=1)        # [m2], random between the range but dependent on the length of the element
demand['Inertia_moment'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
demand['Height'] = demand.apply(lambda row: row['Area']**(0.5), axis=1)   # derived from area assuming square section
demand.index = ['D' + str(num) for num in demand.index]

supply = pd.DataFrame()
supply['Length'] = [x/10 for x in random.choices(range(int(MIN_LENGTH*10), int(MAX_LENGTH*10)), k=SUPPLY_COUNT)]        # [m], random between the range
supply['Area'] = supply.apply(lambda row: round((random.choice(range(0, int(MAX_AREA*10000)-int(MIN_AREA*10000))) /10000 /MAX_LENGTH * row['Length'] + MIN_AREA) * 10000)/10000, axis=1)        # [m2], random between the range but dependent on the length of the element
supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
supply['Height'] = supply.apply(lambda row: row['Area']**(0.5), axis=1)   # derived from area assuming square section
supply['Is_new'] = [False for i in range(SUPPLY_COUNT)]
supply.index = ['R' + str(num) for num in supply.index]

matching = Matching(demand, supply, add_new=True, multi=False)
matching.evaluate()
matching.weight_incidence()
matching.match_bipartite_graph()
matching.match_greedy_algorithm(plural_assign=False)
matching.match_greedy_algorithm(plural_assign=True)
matching.match_cp_solver()
# ERROR matching.match_mixed_integer_programming()

"""
### Test with random generated elements
print_header("RANDOM ELEMENTS n_D = 200, n_S = 10000")
random.seed(3)

DEMAND_COUNT = 100
SUPPLY_COUNT = 10000
MIN_LENGTH = 1.0
MAX_LENGTH = 10.0
MIN_AREA = 0.0025   # 5x5cm
MAX_AREA = 0.25     # 50x50cm

demand = pd.DataFrame()
demand['Length'] = [x/10 for x in random.choices(range(int(MIN_LENGTH*10), int(MAX_LENGTH*10)), k=DEMAND_COUNT)]        # [m], random between the range
demand['Area'] = demand.apply(lambda row: round((random.choice(range(0, int(MAX_AREA*10000)-int(MIN_AREA*10000))) /10000 /MAX_LENGTH * row['Length'] + MIN_AREA) * 10000)/10000, axis=1)        # [m2], random between the range but dependent on the length of the element
demand['Inertia_moment'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
demand['Height'] = demand.apply(lambda row: row['Area']**(0.5), axis=1)   # derived from area assuming square section
demand.index = ['D' + str(num) for num in demand.index]

supply = pd.DataFrame()
supply['Length'] = [x/10 for x in random.choices(range(int(MIN_LENGTH*10), int(MAX_LENGTH*10)), k=SUPPLY_COUNT)]        # [m], random between the range
supply['Area'] = supply.apply(lambda row: round((random.choice(range(0, int(MAX_AREA*10000)-int(MIN_AREA*10000))) /10000 /MAX_LENGTH * row['Length'] + MIN_AREA) * 10000)/10000, axis=1)        # [m2], random between the range but dependent on the length of the element
supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
supply['Height'] = supply.apply(lambda row: row['Area']**(0.5), axis=1)   # derived from area assuming square section
supply['Is_new'] = [False for i in range(SUPPLY_COUNT)]
supply.index = ['R' + str(num) for num in supply.index]

matching = Matching(demand, supply, add_new=True, multi=False)
matching.evaluate()
matching.weight_incidence()
matching.match_bipartite_graph()
matching.match_greedy_algorithm(plural_assign=False)
matching.match_greedy_algorithm(plural_assign=True)
# ERROR matching.match_mixed_integer_programming()


"""