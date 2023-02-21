import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import igraph as ig
import logging
import LCA as lca
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

def plot_histograms(df, **kwargs):
    
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


def plot_scatter(df, **kwargs):
    ### Scatter plot of all elements width/height:
    df.plot.scatter(x='Width', y='Height')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()



def plot_hexbin(df, style = 'ticks', font_scale = 1.1,  **kwargs):
    # Based on https://seaborn.pydata.org/examples/hexbin_marginals.html
    #plt.figure()    
    # TODO Sverre, try with section names: sns.jointplot(x=df['Length'], y=df['Section'], kind="hex", color="#4CB391")
    
    sns.set_theme(style = style, font_scale = font_scale, rc = kwargs)
    g = sns.jointplot(x=df['Length'], y=df['Area'], kind="hex", color="#4CB391")
    
    #g.set_axis_labels(**kwargs)
    # sns.jointplot(x=supply['Length'], y=supply['Area'], kind="hex", color="#eb4034")
    plt.show()

def plot_hexbin_remap(df, unique_values, style = 'ticks', font_scale = 1.1,  **kwargs):
    """Plot the Cross Section and length histogram using remapped values"""
    sns.set_theme(style = style, font_scale = font_scale, rc = kwargs) # set styling configuration
    
    # get all unique areas
    cross_secs = ['36x36', '36x48', '36x72', '36x98', '48x48', '48x98', '98x136', '98x148', '148x223']
     
    map_dict = {a:cs for a, cs in zip(sorted(unique_values), cross_secs)}
    map_dict2 = {a:(i+1) for i, a in enumerate(sorted(unique_values))}
    df['Cross Sections'] = df.Area.map(map_dict2).astype(int)
    g = sns.jointplot(x=df['Length'], y=df['Cross Sections'], kind="hex", color="#4CB391")
    g.ax_joint.set_yticks(list(map_dict2.values()))
    g.ax_joint.set_yticklabels(cross_secs)
    plt.show()
    pass


def plot_savings(result_df, style = 'ticks', font_scale = 1.1, **kwargs):
    #plt.figure()
    sns.set_theme(style = style, font_scale = font_scale, rc = kwargs)

    # data = pd.DataFrame(result_list, columns=['GreedyS','GreedyP','MaxBM','MIP'])
    plot = sns.lineplot(data=result_df, palette="tab10", linewidth=2.5, markers=True)
    plot.set(xlabel='Test case', ylabel='% of score saved')
    plt.xticks(rotation=20)
    plt.show()

def plot_time(result_df, style = 'ticks', font_scale = 1.1, **kwargs):
    #plt.figure()
    sns.set_theme(style = style, font_scale = font_scale, rc = kwargs)
    plot = sns.lineplot(data=result_df, palette="tab10", linewidth=2.5, markers=True)
    plot.set(yscale="log", xlabel='Test case', ylabel='Time [s]')
    plt.xticks(rotation=20)
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



def create_random_data(demand_count, supply_count, demand_gwp=lca.TIMBER_GWP, supply_gwp=lca.TIMBER_REUSE_GWP, length_min = 4, length_max = 15.0, area_min = 0.15, area_max = 0.25):
    """Create two dataframes for the supply and demand elements used to evaluate the different matrices"""
    np.random.RandomState(2023) #TODO not sure if this is the right way to do it. Read documentation
    demand = pd.DataFrame()
    supply = pd.DataFrame()
    # create element lenghts
    demand['Length'] = ((length_max/2 + 1) - length_min) * np.random.random_sample(size = demand_count) + length_min
    supply['Length'] = ((length_max + 1) - length_min) * np.random.random_sample(size = supply_count) + length_min
    # create element areas independent of the length. Can change this back to Artur's method later, but I want to see the effect of even more randomness. 
    demand['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = demand_count) + area_min
    supply['Area'] = ((area_max + .001) - area_min) * np.random.random_sample(size = supply_count) + area_min
    # intertia moment
    demand['Inertia_moment'] = demand.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    supply['Inertia_moment'] = supply.apply(lambda row: row['Area']**(2)/12, axis=1)   # derived from area assuming square section
    # height - assuming square cross sections
    demand['Height'] = np.power(demand['Area'], 0.5)
    supply['Height'] = np.power(supply['Area'], 0.5)
    # gwp_factor
    demand['Gwp_factor'] = demand_gwp
    supply['Gwp_factor'] = supply_gwp
    # Change index names
    demand.index = map(lambda text: 'D' + str(text), demand.index)
    supply.index = map(lambda text: 'S' + str(text), supply.index)
    return demand.round(2), supply.round(2)


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

print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")