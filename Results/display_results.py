import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('./Matching/')
from helper_methods import (plot_hexbin_remap, plot_histograms, plot_old, 
                            plot_savings, plot_time)
sys.path.append('./TestCases/')
from study_case_trusses import create_trusses_from_JSON, elements_from_trusses

# Create all elements
trusses = create_trusses_from_JSON('Data\\CSV files trusses\\truss_all_types_beta_4.csv')
truss_elements = elements_from_trusses(trusses)

all_elements = [item for sublist in truss_elements for item in sublist]
all_elem_df = pd.DataFrame(all_elements)



var_score_df =  pd.read_csv('Results\CSV_Matching\\2023-02-23_08-45_Result_var_amount_less_5k_score.csv', index_col=0).astype(float)
var_sub_df =  pd.read_csv('Results\\CSV_Matching\\2023-02-23_08-45_Result_var_amount_less_5k_substituted.csv', index_col=0).astype(float)
var_time_df =  pd.read_csv('Results\\CSV_Matching\\2023-02-23_08-45_Result_var_amount_less_5k_time.csv', index_col=0).astype(float)

# Create global settings for all plots
plot_kwargs = {'font.family':'serif','font.serif':['Times New Roman'], 'axes.labelsize' : 28,
                'axes.spines.right' : False, 'axes.spines.top' : False ,
                'figure.figsize' : [13,8], 'xtick.major.size': 8.0, 'ytick.major.size': 8.0, 
                'xtick.labelsize': 20.0, 'ytick.labelsize': 20.0,
                'legend.fontsize' : 'large', 'lines.linewidth' : 3}
save_figs = True
show_figs = False
#figsize = (10,10)
# plot hexbin of all elements
plot_kwargs_hex = {'font.family':'serif','font.serif':['Times New Roman'], 'axes.labelsize' : 12,
                'axes.spines.right' : False, 'axes.spines.top' : False ,
                'figure.figsize' : [12,8], 'ytick.major.size': 6.0, 'xtick.major.size': 6.0, 
                'xtick.labelsize': 12.0, 'ytick.labelsize': 12.0,
                'legend.fontsize' : 'large'}
plot_hexbin_remap(all_elem_df, set(all_elem_df.Area), font_scale=1, save_fig = save_figs, show_fig= show_figs, **plot_kwargs_hex)

"""
ratio_or_amount = 'amount'
plot_savings(var_score_df, save_fig = save_figs, show_fig= show_figs, ratio_amount= ratio_or_amount, **plot_kwargs) 
plot_old(var_sub_df, save_fig = save_figs,  show_fig= show_figs, ratio_amount= ratio_or_amount, **plot_kwargs) 
plot_time(var_time_df, save_fig = save_figs,  show_fig= show_figs, ratio_amount= ratio_or_amount, **plot_kwargs)


#read the csv files into dataframes
var_score_ratio_df =  pd.read_csv('Results\\CSV_Matching\\2023-02-23_11-29_Result_var_ratio_score.csv', index_col=0).astype(float)
var_sub_ratio_df =  pd.read_csv('Results\\CSV_Matching\\2023-02-23_11-29_Result_var_ratio_substituted.csv', index_col=0).astype(float)
var_time_ratio_df =  pd.read_csv('Results\\CSV_Matching\\2023-02-23_11-29_Result_var_ratio_time.csv', index_col=0).astype(float)

amount = 'ratio'
plot_savings(var_score_ratio_df, save_fig = save_figs, show_fig= show_figs, ratio_amount= amount, **plot_kwargs) 
plot_old(var_sub_ratio_df, save_fig = save_figs,  show_fig= show_figs, ratio_amount= amount, **plot_kwargs) 
plot_time(var_time_ratio_df, save_fig = save_figs,  show_fig= show_figs, ratio_amount= amount, **plot_kwargs)

"""