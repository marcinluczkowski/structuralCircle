import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import igraph as ig
import logging
import LCA as lca
import itertools
import random
import seaborn as sns


# ==== HELPER METHODS ====
# This file contains various methods used for testing and development. 

def extract_pairs_df(dict_list):
    """Creates a dataframe with the matching pairs all evaluated matching methods. 
    input: list of dictionaries for each matching method
    output: dataframe with the matching pairs of all methods."""
    sub_df = []
    cols = []
    for run in dict_list:
        sub_df.append(run['Match object'].pairs)
        cols.append(run['Name'])
    df = pd.concat(sub_df, axis = 1)
    df.columns = cols
    return df

def extract_results_df(dict_list):
    """Creates a dataframe with the scores from each method"""
    sub_df = []
    cols = []
    for run in dict_list:
        sub_df.append(run['Match object'].result)
        cols.append(run['Name'])
    df = pd.DataFrame(sub_df, index= cols)    
    df=df.rename(columns={0:"LCA"})
    return df.round(3)

def get_assignment_df(matching_pairs_df, supply_ids, only_old = True):
    """Takes the matching pairs df and returns a new df with all supply element ids as index and the list of demand element
    elements assigned to it in columns. One column per method used to match"""
    df = pd.DataFrame(data = None, index = supply_ids, columns= matching_pairs_df.columns)
    columns = matching_pairs_df.columns
    for s in supply_ids:
        new_row_lst = [matching_pairs_df[col].index[list(map(lambda s_id: s_id == s, matching_pairs_df[col]))].to_list() for col in columns]
        df.loc[s, columns] = new_row_lst
        # for col in columns:
        #     bool_array = list(map(lambda s_id: s_id == s, matching_pairs_df[col]))
        #     df.loc[s, col] = matching_pairs_df[col].index[bool_array].to_list()
    if only_old: # remove new elements from df. 
        df = df[df.index.str.contains('S')]
    return df


def remove_alternatives(x, y):
    if x > y:
        return np.nan
    else:
        return x

# def extract_LCA_new(dict_list):
#     matchobj=dict_list[0]["Match object"]
#     sum=matchobj.demand["LCA"].sum()
#     return sum


### ADD PLOTS

def plot_histograms(df, save_fig = False, **kwargs):
    
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

    if save_fig:
        f_name = 'histogram'
        plt.savefig(f'Results\\Figures\\{f_name}.png', dpi = 400, transparent = True)

    plt.show()


def plot_scatter(df, **kwargs):
    ### Scatter plot of all elements width/height:
    df.plot.scatter(x='Width', y='Height')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()



def plot_hexbin(df, style = 'ticks', font_scale = 1.1, save_fig = False,  **kwargs):
    # Based on https://seaborn.pydata.org/examples/hexbin_marginals.html
    #plt.figure()    
    # TODO Sverre, try with section names: sns.jointplot(x=df['Length'], y=df['Section'], kind="hex", color="#4CB391")
    
    sns.set_theme(style = style, font_scale = font_scale, rc = kwargs)
    g = sns.jointplot(x=df['Length'], y=df['Area'], kind="hex", color="#4CB391")
    
    #g.set_axis_labels(**kwargs)
    # sns.jointplot(x=supply['Length'], y=supply['Area'], kind="hex", color="#eb4034")
        
    if save_fig:
        f_name = 'hexbin'
        plt.savefig(f'Results\\Figures\\{f_name}.png', dpi = 400, transparent = True)

    plt.show()

def plot_hexbin_remap(df, unique_values, style = 'ticks', font_scale = 1.1, save_fig = False,  **kwargs):
    """Plot the Cross Section and length histogram using remapped values"""
    sns.set_theme(style = style, font_scale = font_scale, height = 20, rc = kwargs) # set styling configuration
    
    # get all unique areas
    cross_secs = ['(36x36)', '(36x48)', '(36x148)', '(36x198)', '(48x148)', '(48x198)', '(61x198)', '(73x198)', '(73x223)']
     
    map_dict = {a:cs for a, cs in zip(sorted(unique_values), cross_secs)}
    map_dict2 = {a:(i+1) for i, a in enumerate(sorted(unique_values))}
    df['Cross Sections [mm]'] = df.Area.map(map_dict2).astype(int)
    g = sns.jointplot(x=df['Length'], y=df['Cross Sections [mm]'], kind="hex", color="#4CB391")
    g.ax_joint.set_yticks(list(map_dict2.values()))
    g.ax_joint.set_yticklabels(cross_secs)

    if save_fig:
        f_name = 'hexbin_mapped'
        plt.savefig(f'Results\\Figures\\{f_name}.png', dpi = 400, transparent = True)

    plt.show()

def barplot_sns(result_df, normalize = True, style = 'ticks', font_scale = 1.1, save_fig = False,
                show_fig = False, **kwargs):
    """Add docstring""" 
    if normalize:
        result_df = result_df.div(result_df.max(axis = 1), axis = 0).mul(100).round(2)

def plot_savings(result_df, normalize = True, style = 'ticks', font_scale = 1.1, save_fig = False, **kwargs):
    #plt.figure()
    if normalize: # normalize the dataframe according to best score in each row
        result_df = result_df.div(result_df.max(axis = 1), axis = 0).mul(100).round(2)

    # Setting for the plot    
    sns.set_theme(style = style, font_scale = font_scale, rc = kwargs)

    # data = pd.DataFrame(result_list, columns=['GreedyS','GreedyP','MaxBM','MIP'])
    plot = sns.lineplot(data=result_df, palette="tab10", linewidth=2.5, markers=True)
    plot.set(xlabel='Elements (Demand : Supply)', ylabel='Normalised score(GWP) savings')
    plt.xticks(rotation=30)

    if save_fig:
        f_name = "score saved"
        plt.savefig(f'Results\\Figures\\{f_name}.png', dpi = 400, transparent = True)

    plt.show()

def plot_old(result_df, style = 'ticks', font_scale = 1.1, save_fig = False, **kwargs):
    plt.figure()
    sns.set_theme(style = style, font_scale = font_scale, rc = kwargs)
    # data = pd.DataFrame(result_list, columns=['GreedyS','GreedyP','MaxBM','MIP'])
    plot = sns.lineplot(data=result_df, palette="tab10", linewidth=2.5, markers=True)
    plot.set(xlabel='Elements (Demand : Supply)', ylabel='Normalise % of substitutions')
    plt.xticks(rotation=30)

    if save_fig:
        f_name = 'reused elements'
        plt.savefig(f'Results\\Figures\\{f_name}.png', dpi = 400, transparent = True)


    plt.show()

def plot_time(result_df, style = 'ticks', font_scale = 1.1, save_fig = False, **kwargs):
    #plt.figure()
    sns.set_theme(style = style, font_scale = font_scale, rc = kwargs)
    plot = sns.lineplot(data=result_df, palette="tab10", linewidth=2.5, markers=True)
    plot.set(yscale="log", xlabel='Elements (Demand : Supply)', ylabel='Time [s]')
    plt.xticks(rotation=30)
    if save_fig:
        f_name = 'time plot'
        plt.savefig(f'Results\\Figures\\{f_name}.png', dpi = 400, transparent = True)

    plt.show()

def plot_bubble(demand, supply, **kwargs):

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


def create_random_data_demand(demand_count, demand_lat, demand_lon, demand_gwp=lca.TIMBER_GWP, length_min = 4, length_max = 15.0, area_min = 0.15, area_max = 0.25):
    """Create two dataframes for the supply and demand elements used to evaluate the different matrices"""
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    demand = pd.DataFrame()
   
    # create element lenghts
    demand['Length'] = ((length_max/2 + 1) - length_min) * np.random.random_sample(size = demand_count) + length_min
    # create element areas independent of the length. Can change this back to Artur's method later, but I want to see the effect of even more randomness. 
    demand['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = demand_count) + area_min
    # intertia moment
    demand['Inertia_moment'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    # height - assuming square cross sections
    demand['Height'] = np.power(demand['Area'], 0.5)
    # gwp_factor
    demand['Gwp_factor'] = demand_gwp
    demand["Demand_lat"]=demand_lat
    demand["Demand_lon"]=demand_lon

    # Change index names
    demand.index = map(lambda text: 'D' + str(text), demand.index)
    return demand.round(4)

def create_random_data_supply(supply_count,demand_lat, demand_lon,supply_coords,supply_gwp=lca.TIMBER_REUSE_GWP, length_min = 4, length_max = 15.0, area_min = 0.15, area_max = 0.25):
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    supply = pd.DataFrame()
    supply['Length'] = ((length_max + 1) - length_min) * np.random.random_sample(size = supply_count) + length_min
    supply['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = supply_count) + area_min
    supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    supply['Height'] = np.power(supply['Area'], 0.5)
    supply['Gwp_factor'] = supply_gwp
    supply["Demand_lat"]=demand_lat
    supply["Demand_lon"]=demand_lon
    supply["Location"]=0
    supply["Supply_lat"]=0
    supply["Supply_lon"]=0
    
    for row in range(len(supply)):
        lokasjon=random.randint(0, len(supply_coords)-1)
        supply.loc[row,"Supply_lat"]=supply_coords.loc[lokasjon,"Lat"]
        supply.loc[row,"Supply_lon"]=supply_coords.loc[lokasjon,"Lon"]
        supply.loc[row,"Location"]=supply_coords.loc[lokasjon,"Place"]
    supply.index = map(lambda text: 'S' + str(text), supply.index)

    return supply.round(4)


def extract_brute_possibilities(incidence_matrix):
    """Extracts all demand matching possibilities from incidence matrix.
    
    returns a 3D list where each outer list contains possibilities for each row based on incidence matrix.
    """
    binary_incidence = incidence_matrix*1 #returnes incidence matrix with 1 and 0 instead od True/False
    
    three_d_list=[]
    incidence_list=binary_incidence.values.tolist()
    for row in incidence_list:
        rowlist=[]
        for i in range(len(row)):
            if row[i]==1:
                newlist=[0]*len(row)
                newlist[i]=1
                rowlist.append(newlist)
        three_d_list.append(rowlist)
    return three_d_list 

def display_graph(matching, graph_type='rows', show_weights=True, show_result=True):
    """Plot the graph and matching result"""
    if not matching.graph:
        matching.add_graph()
    weight = None
    if show_weights:
        # weight = list(np.absolute(np.array(self.graph.es["label"]) - 8).round(decimals=2)) 
        weight = list(np.array(matching.graph.es["label"]).round(decimals=2)) 
    edge_color = None
    edge_width = matching.graph.es["label"]
    if show_result and not matching.pairs.empty:
        edge_color = ["gray"] * len(matching.graph.es)
        edge_width = [0.7] * len(matching.graph.es)
        # TODO could be reformatted like this https://igraph.readthedocs.io/en/stable/tutorials/bipartite_matching.html#tutorials-bipartite-matching
        not_found = 0
        for index, pair in matching.pairs.iterrows():
            source = matching.graph.vs.find(label=index) 
            try:
                target = matching.graph.vs.find(label=pair['Supply_id'])
                edge = matching.graph.es.select(_between = ([source.index], [target.index]))
                edge_color[edge.indices[0]] = "black" #"red"
                edge_width[edge.indices[0]] = 2.5
            except ValueError:
                not_found+=1
        if not_found > 0:
            logging.error("%s elements not found - probably no new elements supplied.", not_found)
    vertex_color = []
    for v in matching.graph.vs:
        if 'D' in v['label']:
            vertex_color.append("lightgray")
        elif 'S' in v['label']:
            vertex_color.append("slategray")
        else:
            vertex_color.append("pink")
    layout = matching.graph.layout_bipartite()
    if graph_type == 'rows':
        layout = matching.graph.layout_bipartite()
    elif graph_type == 'circle':
        layout = matching.graph.layout_circle()
    if matching.graph:
        fig, ax = plt.subplots(figsize=(15, 10))
        ig.plot(
            matching.graph,
            target=ax,
            layout=layout,
            vertex_size=0.4,
            vertex_label=matching.graph.vs["label"],
            palette=ig.RainbowPalette(),
            vertex_color=vertex_color,
            edge_width=edge_width,
            edge_label=weight,
            edge_color=edge_color,
            edge_curved=0.15
        )
        plt.show()

def extract_genetic_solution(weights, best_solution, number_of_demand_elements):
    """Converts the best solution a column containing the supply matches for each demand element. 
    This column is added to the weight-matrix in order to visually check if the matching makes sense
    - weights: Pandas Dafarame
    - best_solution: 1d-list with 0's and 1's
    - number_of_demand_elements: integer

    Returns a pandas dataframe
    """
    result = weights.copy(deep = True)
    buckets = np.array_split(best_solution, number_of_demand_elements)
    weight_cols = weights.columns.values.tolist()
    match_column = []
    for i in range(len(buckets)):
        index = np.where(buckets[i] == 1)[0] #Finding matches
        if len(index) == 0 or len(index) > 1: #This happens either if a match is not found or multiple supply-elements are matched with the same demand elemend => invalid solution
            match = f"N{i}" #Set match to New element. 
            if len(index) > 1:
                logging.info("OBS: Multiple supply matched with one demand")
        else:
            match = weight_cols[index[0]]
            if match == "N":
                match = f"N{i}"
        match_column.append(match)
    result["Matches from genetic"] = match_column
    return result
            
def print_genetic_solution(weights, best_solution, number_of_demand_elements):
    """Print the genetic solution in a readable way to visually evaluate if the solution makes sence. Used for debugging
    - weights: Pandas Dafarame
    - best_solution: 1d-list with 0's and 1's
    - number_of_demand_elements: integer

    Returns a pandas dataframe
    """
    result = weights.copy(deep = True)
    buckets = np.array_split(best_solution, number_of_demand_elements)
    weight_cols = weights.columns.values.tolist()
    weight_cols = list(map(lambda x: x.replace("N0", "N"), weight_cols))
    weight_cols.append("N0")
    match_column = []
    for i in range(len(buckets)):
        index = np.where(buckets[i] == 1)[0] #Finding matches
        if len(index) == 0:
            match = ["No match"]
        else:
            match = [weight_cols[x] for x in index]
        match_column.append(match)
    result["Matches from genetic"] = match_column
    return result


def create_initial_population_genetic(binary_incidence, size_of_population, include_invalid_combinations):
    """Creating initial population based valid solutions from the incidence matrix
    Good, but itertools.product has very long runtime!!!
    - binary incidence: Pandas dataframe
    - size_of_population: Integer
    - include_invalid_combinations: Boolean
    
    Returns an initial population as a nested list containing 0's and 1's
    """

    three_d_list=[]
    incidence_list=binary_incidence.values.tolist()
    valid_solutions = []
    #Creates a 3d-list containing all possible locations of matches based on the incidence matrix
    for row in incidence_list:
        rowlist=[]
        for i in range(len(row)):
            if row[i]==1:
                newlist=[0]*len(row)
                newlist[i]=1
                rowlist.append(newlist)
        three_d_list.append(rowlist)

    all_possible_solutions = list(itertools.product(*three_d_list)) #EXTREMELY LONG RUNTIME

    if include_invalid_combinations: #Include invalid solutions, such as one demand element is matched with multiple supply elements
        if len(all_possible_solutions) < size_of_population:
            samples = all_possible_solutions
        else:
            samples = random.sample(all_possible_solutions, size_of_population) #Takes n random elements from the list containing all possible solutions
        #initial_population = map(lambda x,y: x.append(y.flatten()), initial_population, samples)
        initial_population = [sum(list(x), []) for x in samples]

    else: #Only valid solutions are included in the initial population
        number_of_possible_solutions = len(all_possible_solutions)
        sample_size = min(int(np.sqrt(number_of_possible_solutions*np.log10(number_of_possible_solutions))), size_of_population*100)
        if len(all_possible_solutions) <= sample_size: #If number of possible solutions is smaller than the sample size
            solutions = all_possible_solutions
        else:
            solutions = random.sample(all_possible_solutions, sample_size) #Takes n random elements from the list containing all possible solutions
        
        #Evaluate if the samples are a valid solution or not
        for subset in solutions:
            column_sum = np.sum(list(subset), axis = 0)[:-1] #All sums of columns except the "New"-column
            invalid_solution = len([*filter(lambda x: x > 1, column_sum)]) > 0
            if not invalid_solution:
                valid_solutions.append(sum(list(subset), [])) #Appen valid solutions
            
        if len(valid_solutions) < size_of_population:
            initial_population = valid_solutions
        else:
            initial_population = random.sample(valid_solutions, size_of_population)#Takes n random elements from the list containing all possible solutions
    return initial_population

    

def create_random_population_genetic(chromosome_length, requested_population_size, probability_of_0, probability_of_1):
    """ Creates a random population with a given probability of having 0 or 1 for each gene
    - chromosome_length: integer
    - requested_population_size: integer
    - probability_of_0: float
    - probability_of_1: float

    NOTE: The probabilities must sum to 1!
    
    Returns an initial population as a nested list containing 0's and 1's
    """

    initial_population = []
    count = 0
    for i in range(requested_population_size*100):
        solution = [np.random.choice([0,1], p = [probability_of_0, probability_of_1]) for x in range(chromosome_length)]
        if solution not in initial_population:
            initial_population.append(solution)
            count += 1
        if count == requested_population_size:
            break
    return initial_population

print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")