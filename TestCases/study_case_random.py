import sys
sys.path.append('./Matching')
from matching import Matching, run_matching #, TIMBER_GWP, REUSE_GWP_RATIO
import pandas as pd
import random
import time
import helper_methods as hm
import LCA as lca

# create constraint dictionary
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}

# ====  Test with randomly generated elements ====
hm.print_header("RANDOM ELEMENTS n_D = 100, n_S = 200")
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
demand['Gwp_factor'] = lca.TIMBER_GWP
demand.index = ['D' + str(num) for num in demand.index]

supply = pd.DataFrame()
supply['Length'] = [x/10 for x in random.choices(range(int(MIN_LENGTH*10), int(MAX_LENGTH*10)), k=SUPPLY_COUNT)]        # [m], random between the range
supply['Area'] = supply.apply(lambda row: round((random.choice(range(0, int(MAX_AREA*10000)-int(MIN_AREA*10000))) /10000 /MAX_LENGTH * row['Length'] + MIN_AREA) * 10000)/10000, axis=1)        # [m2], random between the range but dependent on the length of the element
supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
supply['Height'] = supply.apply(lambda row: row['Area']**(0.5), axis=1)   # derived from area assuming square section
supply['Gwp_factor'] = lca.TIMBER_REUSE_GWP
supply.index = ['R' + str(num) for num in supply.index]

result_rndm1 = run_matching(demand=demand, supply = supply, constraints=constraint_dict, add_new=False, 
            milp=True, sci_milp = False)


### Test with random generated elements
hm.print_header("RANDOM ELEMENTS n_D = 200, n_S = 200")
random.seed(3)

DEMAND_COUNT = 200
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


score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"
result = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True, sci_milp=True, milp=False, greedy_single=True, bipartite=True)

