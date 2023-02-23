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
import math

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


def pick_random(n_a, n_b, elements, whole_trusses=True, duplicates=False):
    
    random.seed(2023)  # Preserve the same seed to replicate the results

    if not whole_trusses:
        elements = [item for sublist in elements for item in sublist]

    if n_a + n_b < len(elements):
        selected = random.sample(elements, n_a + n_b)
    else:
        # more than in the set - allow duplicates
        selected = random.sample(elements, n_a + n_b, counts=[math.ceil((n_a + n_b)/len(elements))]*len(elements))

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
    save_figs = True
    save_csv = True
    #PATH =  "Data\\CSV files trusses\\truss_all_types_beta_second_version.csv" Test with another dataset
    trusses = create_trusses_from_JSON(PATH)
    truss_elements = elements_from_trusses(trusses)

    all_elements = [item for sublist in truss_elements for item in sublist]
    all_elem_df = pd.DataFrame(all_elements)

    all_elem_df['Section'] = (round(all_elem_df['Width']*100,2)).map(str) + 'x' + (round(all_elem_df['Height']*100,2)).map(str)


    plot_kwargs = {'font.family':'serif','font.serif':['Times New Roman'], 'axes.labelsize' : 15}
    
    #hm.plot_hexbin(all_elem_df, font_scale=1)
    #hm.plot_hexbin_remap(all_elem_df, set(all_elem_df.Area), font_scale=1, save_fig = save_figs, **plot_kwargs)

    constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}
    score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"

    result_table = []

    # From that set, distinguish N_D demand and N_S supply elements, based on the desired number and ratios:
    # e.g. N_D, N_S = 100, 50   means ratio 1:0.5 with 100 designed and 50 available elements
    
    amounts = [
        # test
        # [15,10],
        # [25,20],
        # [35,30]
        # variable ratios
        [985,15],
        [970,30],
        [941,59],
        [889,111],
        [800,200],
        [667,333],
        [500,500],
        [333,667],
        [200,800],
        [111,889],
        [59,941],
        [30,970],
        [15,985],
        # variable count
        # [1,10],
        # [2,20],
        # [4,40],
        # [8,80],
        # [16,160],
        # [32,320],
        # [64,640],
        # [128,1280],
        # [256,2560],        
        # [512,5120],
        # [1024,10240],
        #only without MIP
        #[2048,20480],
        #[4096,40960],
        ]

    results_score_df = pd.DataFrame(columns = ["Greedy_single", "Greedy_plural", "Bipartite", "MILP"])
    results_time_df = pd.DataFrame(columns = ["Greedy_single", "Greedy_plural", "Bipartite", "MILP"])
    results_old_df = pd.DataFrame(columns = ["Greedy_single", "Greedy_plural", "Bipartite", "MILP"])

    supply_ass_df_list = []
    supply_ass_names = []
    for x in amounts:
        N_D, N_S = x

        print(f"DEMAND-SUPPLY: {N_D}:{N_S}")

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
                            milp=True, sci_milp=False, greedy_single=True, greedy_plural=True, bipartite=True, solution_limit= 400) 

        pairs = hm.extract_pairs_df(result) # get matching pairs
        supply_assignments_df = hm.get_assignment_df(pairs, supply_ids= supply.index)
        supply_ass_df_list.append(supply_assignments_df) # append assignments for solution to list. 
        supply_ass_names.append(f'{N_D}_{N_S}')
        #supply_assignments_df['Length'] = supply.Length
        # Print results
        # print(pairs)

        new_score_row = {}
        new_old_row = {}
        new_time_row = {}
        for res in result:
            res['Match object'].pairs['Supply_id'].fillna('none', inplace=True)
            # score saved result:
            new_score_row[res['Name']] = round(res['Match object'].demand.Score.sum() - res['Match object'].result, 2)
            new_old_row[res['Name']] = round( 100* res['Match object'].pairs['Supply_id'].map(lambda x: x[0] == 'S').sum() / res['Match object'].pairs.count()[0] , 2)
            new_time_row[res['Name']] = round(res['Time'], 2)
            # actual result:
            # new_row[res['Name']] = round(res['Match object'].result, 3)


        results_score_df.loc[f"{N_D}:{N_S}"] = new_score_row
        results_old_df.loc[f"{N_D}:{N_S}"] = new_old_row
        results_time_df.loc[f"{N_D}:{N_S}"] = new_time_row
    
    normalised_score_df = results_score_df.apply(lambda row: row / max(row), axis = 1).multiply(100).round(2)
    normalised_old_df = results_old_df.apply(lambda row: row / max(row), axis = 1).multiply(100).round(2)

    
    #hm.plot_savings(results_score_df, **plot_kwargs)
    hm.plot_savings(normalised_score_df, save_fig = save_figs, **plot_kwargs) # Added by Sverre
    #hm.plot_old(results_old_df, **plot_kwargs)
    hm.plot_old(normalised_old_df, save_fig = save_figs, **plot_kwargs) # Added by Sverre to see effect of normalising numbers
    hm.plot_time(results_time_df, save_fig = save_figs, **plot_kwargs)

    
    print(results_score_df)
    print(results_old_df)
    print(results_time_df)


    # Save to CSV:

    if save_csv:
        #name = "var_amount_less_5k"
        name = 'var_ratio'
        time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')
        results_score_df.to_csv(f'Results/CSV_Matching/{time}_Result_{name}_score.csv', index=True)
        results_old_df.to_csv(f'Results/CSV_Matching/{time}_Result_{name}_substituted.csv', index=True)
        results_time_df.to_csv(f'Results/CSV_Matching/{time}_Result_{name}_time.csv', index=True)


    # hm.plot_savings(result_table)
    # print(result_df.transpose())


    #hm.plot_histograms(all_elem_df)
    # hm.plot_scatter(all_elem_df)
    # hm.plot_bubble(demand, supply)
    # hm.plot_hexbin(demand, supply)

    # --- write supply assignment dfs to Excel
    time_1 = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')
    name_1 = "Assignments"
    assignment_path = f'Results/Supply Assignments/{time_1}_{name_1}_score.xlsx'
    write_assignments = True
    if write_assignments:
        with pd.ExcelWriter(assignment_path) as writer:
            for i, df_sheet in enumerate(supply_ass_df_list):
                df_sheet.to_excel(writer, sheet_name = f'Elements {supply_ass_names[i]}')
    


    pass