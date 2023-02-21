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


### ADD PLOTS

def plot_histograms(df):
    
    # csfont = {'fontname':'Times New Roman'}
    # plt.rcParams.update({'font.size': 22}) # must set in top
    plt.rcParams['font.size'] = 12
    plt.rcParams["font.family"] = "Times New Roman"

    ### List unique values of width/height:
    # TODO redo the histogram so that names are displayed, not area.
    df['Cross-sections'] = df['Width'].astype(str) + "x" + df['Height'].astype(str)
    
    ### Plot the histogram of truss elements:
    df.hist(column=['Length', 'Area'], bins=20)
 
    # plt.Axes.set_axisbelow(b=True)
    plt.title('Area')
    plt.show()


def plot_scatter(df):

    ### Scatter plot of all elements width/height:
    df.plot.scatter(x='Width', y='Height')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()


def plot_hexbin(demand, supply):
    # TODO try https://seaborn.pydata.org/examples/hexbin_marginals.html
    pass


def plot_bubble(demand, supply):

    # if close to one another, don't add but increase size:
    demand_chart = pd.DataFrame(columns = ['Length', 'Area', 'dot_size'])
    tolerance_length = 2.5
    tolerance_area = 0.002
    dot_size = 70

    for index, row in demand.iterrows():
        if demand_chart.empty:
            # add first bubble
            demand_chart = pd.concat([demand_chart, pd.DataFrame({'Length': row['Length'], 'Area': row['Area'], 'dot_size': 1}, index=[index])])
        # check if similiar bubble already present:
        elif demand_chart.loc[  (abs(demand_chart['Length'] - row['Length']) < tolerance_length) & (abs(demand_chart['Area'] - row['Area']) < tolerance_area) ].empty:
            # not, so add new bubble
            demand_chart = pd.concat([demand_chart, pd.DataFrame({'Length': row['Length'], 'Area': row['Area'], 'dot_size': 1}, index=[index])])
        else:
            # already present, so increase the bubble size:
            ind = demand_chart.loc[  (abs(demand_chart['Length'] - row['Length']) < tolerance_length) & (abs(demand_chart['Area'] - row['Area']) < tolerance_area) ].index[0]
            demand_chart.at[ind,'dot_size'] = demand_chart.at[ind,'dot_size'] +1

    demand_chart['dot_size_scaled'] = dot_size * (demand_chart['dot_size']**0.5)

    supply_chart = pd.DataFrame(columns = ['Length', 'Area', 'dot_size'])
    for index, row in supply.iterrows():
        if supply_chart.empty:
            # add first bubble
            supply_chart = pd.concat([supply_chart, pd.DataFrame({'Length': row['Length'], 'Area': row['Area'], 'dot_size': 1}, index=[index])])
        # check if similiar bubble already present:
        elif supply_chart.loc[  (abs(supply_chart['Length'] - row['Length']) < tolerance_length) & (abs(supply_chart['Area'] - row['Area']) < tolerance_area) ].empty:
            # not, so add new bubble
            supply_chart = pd.concat([supply_chart, pd.DataFrame({'Length': row['Length'], 'Area': row['Area'], 'dot_size': 1}, index=[index])])
        else:
            # already present, so increase the bubble size:
            ind = supply_chart.loc[  (abs(supply_chart['Length'] - row['Length']) < tolerance_length) & (abs(supply_chart['Area'] - row['Area']) < tolerance_area) ].index[0]
            supply_chart.at[ind,'dot_size'] = supply_chart.at[ind,'dot_size'] +1

    supply_chart['dot_size_scaled'] = dot_size * (supply_chart['dot_size']**0.5)

    plt.scatter(demand_chart.Length, demand_chart.Area, s=list(demand_chart.dot_size_scaled), c='b', alpha=0.5, label='Demand')
    plt.scatter(supply_chart.Length, supply_chart.Area, s=list(supply_chart.dot_size_scaled), c='g', alpha=0.5, label='Supply')

    lgnd = plt.legend(loc="lower right")
    lgnd.legendHandles[0]._sizes = [50]
    lgnd.legendHandles[1]._sizes = [50]

    plt.xlabel("Length", size=16)
    plt.ylabel("Area", size=16)

    for i, row in demand_chart.iterrows():
        if row['dot_size'] < 10:
           plt.annotate(str(row['dot_size']), (row['Length']-0.19, row['Area']-0.0002))
        else:
           plt.annotate(str(row['dot_size']), (row['Length']-0.34, row['Area']-0.0002))
    for i, row in supply_chart.iterrows():
        if row['dot_size'] < 10:
           plt.annotate(str(row['dot_size']), (row['Length']-0.19, row['Area']-0.0002))
        else:
           plt.annotate(str(row['dot_size']), (row['Length']-0.34, row['Area']-0.0002))

    plt.show()


if __name__ == "__main__":
    
    # Generate a set of unique trusses from CSV file:
    PATH = "Data\\CSV files trusses\\truss_all_types_beta_4.csv"
    trusses = create_trusses_from_JSON(PATH)
    truss_elements = elements_from_trusses(trusses)

    all_elements = [item for sublist in truss_elements for item in sublist]
    all_elem_df = pd.DataFrame(all_elements)

    constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}
    score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"

    result_table = []

    # From that set, distinguish N_D demand and N_S supply elements, based on the desired number and ratios:
    # e.g. N_D, N_S = 100, 50   means ratio 1:0.5 with 100 designed and 50 available elements
    
    amounts = [
         [980,20],
         [909,91]#,
        #  [833,167],
        #  [667,333],
        #  [500,500],
        #  [333,667],
        #  [167,833],
        #  [91,909],
        #  [20,980]
        ]    
    
    for x in amounts:
        N_D, N_S = x
        set_a, set_b = pick_random(int(N_D/10), int(N_S/10), truss_elements, whole_trusses=True)
        demand = pd.DataFrame(set_a)
        demand.index = ['D' + str(num) for num in demand.index]

        demand["Gwp_factor"] = lca.TIMBER_GWP

        supply = pd.DataFrame(set_b)
        supply.index = ['S' + str(num) for num in supply.index]
        supply["Gwp_factor"] = lca.TIMBER_REUSE_GWP
        # Run the matching
        plot_histograms(demand)
        #plot_histograms(supply)
        """
        result = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True,
        milp=False, sci_milp=True, greedy_single=True, bipartite=True) 

        pairs = hm.extract_pairs_df(result)
        # Print results
        # print(pairs)
        for res in result:
            result_table.append([
            res['Name'],
            0.0, #res['PercentNew']
            round(res['Match object'].result, 2),
            round(res['Time'], 2)
            ])
        
        print(f"{N_D}x{N_S}")
        """

    result_df = pd.DataFrame(result_table)

    print(result_df.transpose())

    #plot_histograms(all_elem_df)
    # plot_scatter(all_elem_df)
    # plot_bubble(demand, supply)
    # plot_hexbin(demand, supply)
    
    pass