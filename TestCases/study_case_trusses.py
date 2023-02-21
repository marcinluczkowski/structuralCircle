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
import matplotlib.pyplot as plt
import LCA as lca

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


def elements_from_trusses(trusses):
    es = []
    ts = []
    for truss in trusses:
        t = []
        for elem in truss.elements:
            e = {
                "Length": elem[0]*0.001,
                "Width": elem[1][0]*0.001,
                "Height": elem[1][1]*0.001,
                "Inertia_moment": elem[1][0]*0.001 * elem[1][1]*0.001**3 / 12,
                "Area": elem[1][0]*0.001 * elem[1][1]*0.001,
            }
            t.append(e)
            es.append(e)
        ts.append(t)
    return ts


def pick_random(n_a, n_b, elements, whole_trusses=True):
    
    random.seed(2023)  # Preserve the same seed to replicate the results

    if not whole_trusses:
        elements = [item for sublist in elements for item in sublist]

    if n_a + n_b > len(elements):
        raise Exception("You can't pick more elements than are available in the set!")

    selected = random.sample(elements, n_a + n_b)
    
    set_a = selected[0:n_a]
    set_b = selected[n_a:len(selected)]

    if whole_trusses:
        # flatten the list of elements
        flat_set_a = [item for sublist in set_a for item in sublist]
        flat_set_b = [item for sublist in set_b for item in sublist]
    else:
        flat_set_a = set_a[:]
        flat_set_b = set_b[:]
    return flat_set_a, flat_set_b


if __name__ == "__main__":
    
    # Generate a set of unique trusses from CSV file:
    PATH = "Data\\CSV files trusses\\truss_all_types_beta_4.csv"
    trusses = create_trusses_from_JSON(PATH)
    truss_elements = elements_from_trusses(trusses)

    all_elements = [item for sublist in truss_elements for item in sublist]
    all_elem_df = pd.DataFrame(all_elements)

    all_elem_df['Section'] = (round(all_elem_df['Width']*100,2)).map(str) + 'x' + (round(all_elem_df['Height']*100,2)).map(str)

    test_plot_args = {'font.family':'serif','font.serif':['Times New Roman'], 'xtick.bottom' : 'False'}
    hm.plot_hexbin(all_elem_df, **test_plot_args)

    constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}
    score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"

    result_table = []

    # From that set, distinguish N_D demand and N_S supply elements, based on the desired number and ratios:
    # e.g. N_D, N_S = 100, 50   means ratio 1:0.5 with 100 designed and 50 available elements
    
    amounts = [
        # test
        [15,10],
        [25,20],
        # [35,30],
        # variable ratios
        # [980,20],
        # [909,91],
        # [833,167],
        # [667,333],
        # [500,500],
        # [333,667],
        # [167,833],
        # [91,909],
        # [20,980],
        # variable count
        # [1,9],
        # [10,90],
        # [20,180],
        # [50,450],
        # [100,900],
        # [200,1800],
        # [500,4500],
        # [1000,9000],
        ]    

    results_df = pd.DataFrame(columns = ["Greedy_single", "Greedy_plural", "Bipartite", "Scipy_MILP"])
    results_time_df = pd.DataFrame(columns = ["Greedy_single", "Greedy_plural", "Bipartite", "Scipy_MILP"])
    
    for x in amounts:
        N_D, N_S = x

        print(f"DEMANDxSUPPLY: {N_D}x{N_S}")

        set_a, set_b = pick_random(N_D, N_S, truss_elements, whole_trusses=False)
        demand = pd.DataFrame(set_a)
        demand.index = ['D' + str(num) for num in demand.index]
        demand["Gwp_factor"] = lca.TIMBER_GWP

        supply = pd.DataFrame(set_b)
        supply.index = ['S' + str(num) for num in supply.index]
        supply["Gwp_factor"] = lca.TIMBER_REUSE_GWP
        
        # hm.plot_hexbin(demand, supply)
        
        # Run the matching
        result = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = False,
        milp=False, sci_milp=True, greedy_single=True, bipartite=True) 

        pairs = hm.extract_pairs_df(result)
        # Print results
        # print(pairs)

        new_row = {}
        new_time_row = {}
        for res in result:
            # score saved result:
            new_row[res['Name']] = round(100 - 100*res['Match object'].result/res['Match object'].demand.Score.sum(), 2)
            new_time_row[res['Name']] = round(res['Time'], 2)
            # actual result:
            # new_row[res['Name']] = round(res['Match object'].result, 3)

        results_df.loc[f"{N_D}x{N_S}"] = new_row
        results_time_df.loc[f"{N_D}x{N_S}"] = new_time_row
    
    hm.plot_savings(results_df, **test_plot_args)
    hm.plot_time(results_time_df, **test_plot_args)
    
    print(results_df)
    print(results_time_df)

    hm.plot_savings(result_table, **test_plot_args)
    print(results_df.transpose())

    #hm.plot_histograms(all_elem_df)
    # hm.plot_scatter(all_elem_df)
    # hm.plot_bubble(demand, supply)
    # hm.plot_hexbin(demand, supply)

    pass