import sys
sys.path.append('./Matching')
from matching import Matching, run_matching   #, TIMBER_GWP, REUSE_GWP_RATIO
import helper_methods as hm
import numpy as np
import csv
import json
import ast
import numexpr as ne
import pandas as pd
import random

class truss():
    def __init__(self) -> None:
        self.type = None
        self.length = 0.0
        self.angle = 0.0
        self.distr = 0.0
        self.elements = []

def create_trusses_from_JSON(csv_path):
    # type_length_angle_distribution_span, [[L,(w,h)],[L,(w,h)],[L,(w,h)]]
    trusses = []
    with open(csv_path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if row[0] == 'Name':
                continue
            #TODO FIX THIS
            parts = row[0].split('_')            
            new_truss = truss()
            new_truss.type = parts[0]
            new_truss.length = parts[1]
            new_truss.angle = parts[2]
            new_truss.distr = parts[3]

            elements = ast.literal_eval(row[1])
            for element in elements:
                new_truss.elements.append(element)
            trusses.append(new_truss)
    return trusses


if __name__ == "__main__":
    
    # Generate a set of unique trusses from CSV file:
    #PATH = "MatchingAlgorithms/study_case_data.csv"
    PATH = "Data\\CSV files trusses\\truss_all_types_beta.csv"
    trusses = create_trusses_from_JSON(PATH)

    # Initiate the demand and supply sets
    demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height'])
    supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Is_new'])
    constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}
    
    
    # From that set, distinguish N_D demand and N_S supply elements, based on the desired number and ratios:
    # e.g. N_D, N_S = 100, 50   means ratio 1:0.5 with 100 designed and 50 available elements
    N_D, N_S = 8, 300 #TODO I'm now assuming that this is the number of trusses, and not elements that we "reclaim". Let's discuss when time :) 
    
    # create demand elements
    np.random.seed(2022)
    count = 0
    while count < N_D:
        truss = np.random.choice(trusses) # take a random truss for list of available trusses
        lengths = np.array([el[0] for el in truss.elements], dtype=float) # get all element lengths and store them in numpy array        
        widths = np.array([el[1][0] for el in truss.elements], dtype=float) # get all element widths and store them in numpy array
        widths = ne.evaluate('widths*0.001')
        heights = np.array([el[1][1] for el in truss.elements], dtype=float) # get all element heights and store them in numpy array
        heights = ne.evaluate('heights * 0.001')
        moments_inertia = ne.evaluate('(widths * heights**3) / 12')
        areas = ne.evaluate('widths*heights')
        sub_df = pd.DataFrame({'Length' : lengths, 'Area' : areas, 'Inertia_moment' : moments_inertia, 'Height' : heights, 'Width' : widths})
        demand = pd.concat([demand, sub_df], ignore_index=True)
        count += 1
    
        
    # create supply elements
    count = 0
    np.random.seed(2023)
    while count <N_S:
        truss = np.random.choice(trusses) # take a random truss for list of available trusses
        lengths = np.array([el[0] for el in truss.elements], dtype=float) # get all element lengths and store them in numpy array        
        widths = np.array([el[1][0] for el in truss.elements], dtype=float) # get all element widths and store them in numpy array
        widths = ne.evaluate('widths*0.001')
        heights = np.array([el[1][1] for el in truss.elements], dtype=float) # get all element heights and store them in numpy array
        heights = ne.evaluate('heights * 0.001')
        moments_inertia = ne.evaluate('(widths * heights**3) / 12')
        areas = ne.evaluate('widths*heights')

        sub_df = pd.DataFrame({'Length' : lengths, 'Area' : areas, 'Inertia_moment' : moments_inertia, 'Height' : heights, 'Width' : widths})
        supply = pd.concat([supply, sub_df], ignore_index=True)
        count += 1
    supply['Is_new'] = False
    
    # while demand.shape[0] < N_D:
    #     truss = np.random.choice(trusses)
    #     # print(truss.__dict__)
    #     for e in truss.elements:
    #         i = 0
    #         while demand.shape[0] < N_D and i < len(truss.elements):
    #             i += 1
    #             l = e[0]
    #             # print(l)
    #             b = e[1][0]
    #             h = e[1][1]
    #             new_elem = pd.DataFrame({'Length': l, 'Area': b*h, 'Inertia_moment': b*(h**3)/12, 'Height': h}, index=[0])
    #             demand = pd.concat([demand, new_elem], ignore_index=True)
    demand.index = ['D' + str(num) for num in demand.index]

    # np.random.seed(2023)
    # while supply.shape[0] < N_S:
    #     truss = np.random.choice(trusses)
    #     # print(truss.__dict__)
    #     for e in truss.elements:
    #         i = 0
    #         while supply.shape[0] < N_S and i < len(truss.elements):
    #             i += 1   
    #             l = e[0]
    #             # print(l)
    #             b = e[1][0]
    #             h = e[1][1]
    #             new_elem = pd.DataFrame({'Length': l, 'Area': b*h, 'Inertia_moment': b*(h**3)/12, 'Height': h, 'Is_new':False}, index=[0])
    #             supply = pd.concat([supply, new_elem], ignore_index=True)

    supply.reset_index(drop = True, inplace = True)
    supply.index = ['S' + str(num) for num in supply.index]

    # Run the matching
    result = run_matching(demand=demand, supply=supply, constraints=constraint_dict, add_new=False, greedy_single=True, bipartite=True,
            milp=False, sci_milp=True)

    pairs = hm.extract_pairs_df(result)

    # Print results
    print(pairs)
    for res in result:
        print(f"Name: {res['Name']}\t\t*Result: {res['Match object'].result} kg, time: {res['Match object'].solution_time} s")
