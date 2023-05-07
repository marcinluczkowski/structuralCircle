import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca
import plotting as plot


#==========USER FILLS IN============#

#Constants
#TODO: FIND ALL DEFAULT VALUES FOR CONSTANTS, especially for price
constants = {
    "TIMBER_GWP": 28.9,       #kg CO2 eq per m^3, based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 2.25,        # 0.0778*28.9 = 2.25kg CO2 eq per m^3, based on Eberhardt
    "TRANSPORT_GWP": 89.6,    #gram per tonne per km, Engedal et. al. 
    "TIMBER_DENSITY": 491.0,  #kg/m^3, based on NEPD-3442-2053-EN
    "STEEL_GWP": 9263.0, #kg CO2 eq per m^3, Norsk stål + density of steel
    "STEEL_REUSE_GWP": 278.0, #kg CO2 eq per m^3, reduction of 97% from new according to Høydahl and Walter
    "VALUATION_GWP": 0.7, #NOK per kg CO2, based on OECD
    "TIMBER_PRICE": 3400.0, #Per m^3, Treindustrien 2023
    "TIMBER_REUSE_PRICE" : 3400.0, #Per m^3, assumes the price is the same is new elements
    "STEEL_PRICE": 67, #NOK per kg, ENTRA 2021
    "STEEL_REUSE_PRICE": 100, #NOK per kg, ENTRA 2021
    "PRICE_TRANSPORTATION": 0.3, #NOK per km per tonne, Grønland 2022 + Gran 2013
    "STEEL_DENSITY": 7850, #kg/m^3 EUROCODE
    ########################
    "Project name": "Sognsveien 17",
    "Metric": "Price",
    "Algorithms": ["bipartite", "greedy_plural"],
    "Include transportation": True,
    "Cite latitude": "59.94161606",
    "Cite longitude": "10.72994518",
    "Demand file location": r"./CSV/DEMAND_DATAFRAME_SVERRE.xlsx",
    "Supply file location": r"./CSV/SUPPLY_DATAFRAME_SVERRE.xlsx",
    #"Demand file location": r"./CSV/pdf_demand.csv",
    #"Supply file location": r"./CSV/pdf_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}
#========================#
#Generating dataset
#===================
supply_coords = pd.DataFrame(columns = ["Location", "Latitude", "Longitude"])

tiller = ["Tiller", "63.3604", "10.4008"]
gjovik = ["Gjovik", "60.8941", "10.5001"]
orkanger = ["Orkanger", "63.3000", "9.8468"]
storlien = ["Storlien", "63.3160", "12.1018"]

supply_coords.loc[len(supply_coords)] = tiller
supply_coords.loc[len(supply_coords)] = gjovik
supply_coords.loc[len(supply_coords)] = orkanger
supply_coords.loc[len(supply_coords)] = storlien


demand_coords = {"Steel": ("Norsk Stål Trondheim", "63.4384474", "10.40994"), "Timber": ("XL-BYGG Lade","63.4423683","10.4438836")}


materials = ["Timber", "Steel"]

#GENERATE FILE
#============
#supply = hm.create_random_data_supply_pdf_reports(supply_count = 10, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
#demand = hm.create_random_data_demand_pdf_reports(demand_count = 10, length_min = 1.0, length_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, demand_coords = demand_coords)
#hm.export_dataframe_to_csv(supply, r"" + "./CSV/pdf_supply.csv")
#hm.export_dataframe_to_csv(demand, r"" + "./CSV/pdf_demand.csv")
#========================================

score_function_string = hm.generate_score_function_string(constants)
supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")
#Add necessary columns to run the algorithm
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)

#plot.create_graph(supply, demand, target_column="Length", unit=r"[m]", number_of_intervals=5, fig_title = "", save_filename=r"length_plot.png")
#plot.create_graph(supply, demand, target_column="Area", unit=r"[m$^2$]", number_of_intervals=5, fig_title = "", save_filename=r"area_plot.png")
#plot.create_graph(supply, demand, target_column="Moment of Inertia", unit=r"[m$^4$]", number_of_intervals=5, fig_title = "", save_filename=r"inertia_plot.png")
plot.plot_materials(supply, demand, "", save_filename=r"material_plot.png")


constraint_dict = constants["constraint_dict"]

run_string = hm.generate_run_string(constants)
result = eval(run_string)
""""""
simple_pairs = hm.extract_pairs_df(result)
pdf_results = hm.extract_results_df_pdf(result, constants)
#plot.create_map_substitutions(supply, pdf_results, "supply", color = "green", legend_text="Substitution locations", save_name=r"map_reuse_subs")
#plot.create_map_substitutions(demand, pdf_results, "demand", color = "red", legend_text="Manufacturer locations", save_name=r"map_manu_subs")
#pdf = hm.generate_pdf_report(pdf_results, projectname = constants["Project name"], filepath = r"./Local_files/GUI_files/Results/")
#print(hm.extract_pairs_df(result))



