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

#==========USER FILLS IN============#
#Constants
#TODO: FIND ALL DEFAULT VALUES FOR CONSTANTS, especially for price
constants = {
    "TIMBER_GWP": 28.9,       # based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 2.25,        # 0.0778*28.9 = 2.25 based on Eberhardt
    "TRANSPORT_GWP": 96.0,    # TODO kg/m3/t based on ????
    "TIMBER_DENSITY": 491.0,  # kg, based on NEPD-3442-2053-EN
    "STEEL_GWP": 800, #Random value
    "STEEL_REUSE_GWP": 4, #Random value
    "VALUATION_GWP": 0.6, #In kr:Per kg CO2, based on OECD
    "TIMBER_PRICE": 435, #Per m^3 https://www.landkredittbank.no/blogg/2021/prisen-pa-sagtommer-okte-20-prosent/
    "TIMBER_REUSE_PRICE" : 100, #Per m^3, Random value
    "STEEL_PRICE": 500, #Per m^2, Random value
    "STEEL_REUSE_PRICE": 200, #Per m^2, Random value
    "PRICE_TRANSPORTATION": 3.78, #Price per km per tonn. Derived from 2011 numbers on scaled t0 2022 using SSB
    "STEEL_DENSITY": 7850,
    ########################
    "Project name": "Sognsveien 17",
    "Metric": "GWP",
    "Algorithms": ["bipartite", "greedy_plural", "bipartite_plural", "bipartite_plural_multiple"],
    "Include transportation": False,
    "Cite latitude": "59.94161606",
    "Cite longitude": "10.72994518",
    #"Demand file location": r"./CSV/DEMAND_DATAFRAME_SVERRE.xlsx",
    #"Supply file location": r"./CSV/SUPPLY_DATAFRAME_SVERRE.xlsx",
    "Demand file location": r"./CSV/bipartite_plural_demand.csv",
    "Supply file location": r"./CSV/bipartite_plural_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}
#========================#

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
                "Moment of Inertia": elem[1][0]*0.001 * elem[1][1]*0.001**3 / 12,
                "Area": elem[1][0]*0.001 * elem[1][1]*0.001,
            }
            t.append(e)
            es.append(e)
        ts.append(t)
    return ts


def pick_random(n_a, n_b, elements, whole_trusses=True, duplicates=False):
    
    #random.seed(2023)  # Preserve the same seed to replicate the results
    random.seed(4)  # Preserve the same seed to replicate the results

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


### ADD PLOTS

def plot_histograms(df):
    
    # Generate a set of unique trusses from CSV file:
    PATH = "Data\\CSV files trusses\\truss_all_types_beta_4.csv"
    trusses = create_trusses_from_JSON(PATH)
    truss_elements = elements_from_trusses(trusses)
    all_elements = [item for sublist in truss_elements for item in sublist]
    all_elem_df = pd.DataFrame(all_elements)

    constraint_dict = {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>='}
    
    # From that set, distinguish N_D demand and N_S supply elements, based on the desired number and ratios:
    # e.g. N_D, N_S = 100, 50   means ratio 1:0.5 with 100 designed and 50 available elements
    N_D, N_S = 50, 50
    
    set_a, set_b = pick_random(N_D, N_S, truss_elements, whole_trusses=False)
    
    demand = pd.DataFrame(set_a)
    demand.index = ['D' + str(num) for num in demand.index]
    supply = pd.DataFrame(set_b)
    supply.index = ['S' + str(num) for num in supply.index]
    supply.insert(5, "Is_new", False)

    #Add material timber:
    supply["Material"] = "Timber"
    demand["Material"] = "Timber"
    constraint_dict = constants["constraint_dict"]
    score_function_string = hm.generate_score_function_string(constants)
    #Add necessary columns to run the algorithm
    supply = hm.add_necessary_columns_pdf(supply, constants)
    demand = hm.add_necessary_columns_pdf(demand, constants)
    run_string = hm.generate_run_string(constants)
    result = eval(run_string)
    # Run the matching
    pairs = hm.extract_pairs_df(result)

    # Print results
    print(pairs)
    for res in result:
        print(f"Name: {res['Name']}\t\t*Result: {res['Match object'].result} kg, time: {res['Match object'].solution_time} s")


    ### ADD PLOTS
    # csfont = {'fontname':'Times New Roman'}
    # plt.rcParams.update({'font.size': 22}) # must set in top
    plt.rcParams['font.size'] = 12
    plt.rcParams["font.family"] = "Times New Roman"

    ### List unique values of width/height:
    # TODO redo the histogram so that names are displayed, not area.
    df['Cross-sections'] = df['Width'].astype(str) + "x" + df['Height'].astype(str)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    # X-tick labels taken from Artur-table, can find from df later by introducing information to DataFrame
    new_ticks = ['36x36', '36x48', '36x148', '36x198', 
                '48x148', '48x198', '61x198', '73x198', '73x223']


    
    ### Test JointPlot from Seaborn
    sns.set_theme(style = 'ticks')
    jg = sns.jointplot(x='Length', y='Area', data = all_elem_df, kind = 'hex', color = "#4CB391")
    
    # Histogram
    sns.set_theme(style='ticks')
    fig, ax = plt.subplots(figsize = (15, 8))
    sns.despine(fig = fig)
    sns.histplot(x = 'Area', palette='light:m_r', edgecolor = '0.3', linewidth = .5, ax=ax, data = all_elem_df)
    ax.set_title('Cross-section Histogram')
    

    plt.show()
    #fig, axs = plt.subplots(1,2, sharex=False, sharey=False)
    
    ### Plot the histogram of truss elements:
    df.hist(column=['Length', 'Area'], bins=20)
 
    # plt.Axes.set_axisbelow(b=True)
    plt.title('Area')
    plt.show()

    #fig, axs = plt.subplots(1,2, sharex=False, sharey=False)
    
    ### Plot the histogram of truss elements:
    #all_elem_df.hist(column=['Length', 'Area'], bins=20)#, ax = axs[0])
    #plt.show()



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
    #ticks = axs[1].get_xticks()
    #axs[0].set_xticks(ticks)
    #plt.sca(axs[1])
    #plt.xticks(xticks = ticks, xticklabels = new_ticks)
    # fig, ax = plt.subplots(1,1)
    # all_elem_df.plot.scatter(x='Width', y='Height', ax = ax)

    # # set x_ticks
    # new_tick_pos = sorted(list(set(all_elem_df.Area)))
    # #ax.set_xticks(new_tick_pos)
    # ax.set_xticklabels(new_ticks, rotation = 45)
    # #ax.xticks = new_ticks
    # plt.show()

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
    save_figs = False
    save_csv = False
    #PATH =  "Data\\CSV files trusses\\truss_all_types_beta_second_version.csv" Test with another dataset
    trusses = create_trusses_from_JSON(PATH)
    truss_elements = elements_from_trusses(trusses)

    all_elements = [item for sublist in truss_elements for item in sublist]
    all_elem_df = pd.DataFrame(all_elements)

    all_elem_df['Section'] = (round(all_elem_df['Width']*100,2)).map(str) + 'x' + (round(all_elem_df['Height']*100,2)).map(str)


    plot_kwargs = {'font.family':'serif','font.serif':['Times New Roman'], 'axes.labelsize' : 15}
    
    #hm.plot_hexbin(all_elem_df, font_scale=1)
    #hm.plot_hexbin_remap(all_elem_df, set(all_elem_df.Area), font_scale=1, save_fig = save_figs, **plot_kwargs)

    constraint_dict = {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>='}
    score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"

    result_table = []

    # From that set, distinguish N_D demand and N_S supply elements, based on the desired number and ratios:
    # e.g. N_D, N_S = 100, 50   means ratio 1:0.5 with 100 designed and 50 available elements
    
    amounts = [
        # test
        # [15,10],
        # [25,20],
        # [25,20],
        # [35,30],
        # variable ratios
        # [985,15],
        # [970,30],
        # [941,59],
        # [889,111],
        # [800,200],
        # [667,333],
        # [500,500],
        # [333,667],
        # [200,800],
        # [111,889],
        # [59,941],
        # [30,970],
        # [15,985],
        # variable count
        [1,10],
        [2,20],
        [4,40],
        [8,80],
        [16,160],
        [32,320],
        [64,640],
        [128,1280],
        [256,2560],        
        [512,5120],
        [1024,10240],
        #only without MIP
        # [2048,20480],
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


        supply = pd.DataFrame(set_b)
        supply.index = ['S' + str(num) for num in supply.index]
        supply["Material"] = "Timber"
        demand["Material"] = "Timber"
        # Run the matching
        constraint_dict = constants["constraint_dict"]
        score_function_string = hm.generate_score_function_string(constants)
        #Add necessary columns to run the algorithm
        supply = hm.add_necessary_columns_pdf(supply, constants)
        demand = hm.add_necessary_columns_pdf(demand, constants)
        run_string = hm.generate_run_string(constants)
        result = eval(run_string)

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

    
    
    print(results_score_df)
    print(results_old_df)
    print(results_time_df)


    # Save to CSV:

    if save_csv:
        name = "var_amount_10k"
        #name = 'var_ratio'
        time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')
        results_score_df.to_csv(f'Local Files/Results/CSV_Matching/{time}_Result_{name}_score.csv', index=True)
        results_old_df.to_csv(f'Local Files/Results/CSV_Matching/{time}_Result_{name}_substituted.csv', index=True)
        results_time_df.to_csv(f'Local Files/Results/CSV_Matching/{time}_Result_{name}_time.csv', index=True)


    # print(result_df.transpose())


    # --- write supply assignment dfs to Excel
    time_1 = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')
    name_1 = "Assignments"
    assignment_path = f'Local Files/Results/Supply Assignments var amount/{time_1}_{name_1}_score.xlsx'
    write_assignments = False
    if write_assignments:
        with pd.ExcelWriter(assignment_path) as writer:
            for i, df_sheet in enumerate(supply_ass_df_list):
                sum_row = df_sheet.applymap(lambda x : len(x) > 0).sum().to_frame().T
                sum_row.rename({0: 'Total support els'}, inplace = True)
                df_sheet = df_sheet.append(sum_row)
                df_sheet.to_excel(writer, sheet_name = f'Elements {supply_ass_names[i]}')
    


    pass