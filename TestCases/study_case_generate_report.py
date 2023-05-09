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
    "STEEL_PRICE": 67.0, #NOK per kg, ENTRA 2021
    "STEEL_REUSE_PRICE": 100.0, #NOK per kg, ENTRA 2021
    "PRICE_TRANSPORTATION": 0.3, #NOK per km per tonne, Grønland 2022 + Gran 2013
    "STEEL_DENSITY": 7850.0, #kg/m^3 EUROCODE
    ########################
    "Project name": "Testing PDF",
    "Metric": "GWP",
    "Algorithms": ["greedy_plural", "milp", "bipartite_plural"],
    "Include transportation": True,
    "Cite latitude": "63.4154171",
    "Cite longitude": "10.3994672",
    "Demand file location": r"./CSV/testing_pdf_demand.csv",
    "Supply file location": r"./CSV/testing_pdf_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}
#========================#
#Generating dataset
#===================
supply_coords = pd.DataFrame(columns = ["Location", "Latitude", "Longitude"])

steinkjer = ["Steinkjer", "64.024861", "11.4891085"]
storen = ["Støren", "63.033639", "10.286356"]
orkanger = ["Orkanger", "63.3000", "9.8468"]
meraker = ["Meråker", "63.415312", "11.747262"]
berkak = ["Berkåk", "62.8238946","9.9934341"]
melhus = ["Melhus", "63.2897753", "10.2934154"]

supply_coords.loc[len(supply_coords)] = steinkjer
supply_coords.loc[len(supply_coords)] = storen
supply_coords.loc[len(supply_coords)] = orkanger
supply_coords.loc[len(supply_coords)] = meraker
supply_coords.loc[len(supply_coords)] = berkak
supply_coords.loc[len(supply_coords)] = melhus

materials = ["Timber", "Steel"]

#GENERATE FILE
#============
supply = hm.create_random_data_supply_pdf_reports(supply_count = 100, length_min = 1.0, length_max = 10.0, area_min = 0.004, area_max = 0.04, materials = materials, supply_coords = supply_coords)
demand = hm.create_random_data_demand_pdf_reports(demand_count = 100, length_min = 1.0, length_max = 10.0, area_min = 0.004, area_max = 0.04, materials = materials)
hm.export_dataframe_to_csv(supply, r"" + "./CSV/testing_pdf_supply.csv")
hm.export_dataframe_to_csv(demand, r"" + "./CSV/testing_pdf_demand.csv")
#supply.to_excel(r"" + "./CSV/study_case_supply.xlsx")
#demand.to_excel(r"" + "./CSV/study_case_demand.xlsx")
#========================================

score_function_string = hm.generate_score_function_string(constants)
supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")
#Add necessary columns to run the algorithm
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)
constraint_dict = constants["constraint_dict"]
#Run matching
run_string = hm.generate_run_string(constants)
result = eval(run_string)
pdf_results = hm.extract_results_df_pdf(result, constants)

plot.create_graph(supply, demand, target_column="Length", unit=r"[m]", number_of_intervals=5, fig_title = "", save_filename=r"length_plot.png")
plot.create_graph(supply, demand, target_column="Area", unit=r"[m$^2$]", number_of_intervals=5, fig_title = "", save_filename=r"area_plot.png")
plot.create_graph(supply, demand, target_column="Moment of Inertia", unit=r"[m$^4$]", number_of_intervals=5, fig_title = "", save_filename=r"inertia_plot.png")
plot.plot_materials(supply, demand, "", save_filename=r"material_plot.png")

if constants["Include transportation"]:
    plot.create_map_substitutions(supply, pdf_results, "supply", color = "green", legend_text="Substitution locations", save_name=r"map_reused_subs")
    plot.create_map_substitutions(demand, pdf_results, "demand", color = "red", legend_text="Manufacturer locations", save_name=r"map_manu_subs")
hm.generate_pdf_report(pdf_results, constants["Project name"],supply,demand, filepath = r"./Local_files/GUI_files/Results/")


#TODO: Create map with the random supply locations and the cite location

#Case 1: GWP UTEN TRANSPORT
#CASE 2: GWP Med t
#CASE 3: Combined med transport



#plot.create_graph(supply, demand, target_column="Length", unit=r"[m]", number_of_intervals=5, fig_title = "", save_filename=r"length_plot.png")
#plot.create_graph(supply, demand, target_column="Area", unit=r"[m$^2$]", number_of_intervals=5, fig_title = "", save_filename=r"area_plot.png")
#plot.create_graph(supply, demand, target_column="Moment of Inertia", unit=r"[m$^4$]", number_of_intervals=5, fig_title = "", save_filename=r"inertia_plot.png")
#plot.plot_materials(supply, demand, "", save_filename=r"material_plot.png")


#plot.create_map_substitutions(supply, pdf_results, "supply", color = "green", legend_text="Substitution locations", save_name=r"map_reuse_subs")
#plot.create_map_substitutions(demand, pdf_results, "demand", color = "red", legend_text="Manufacturer locations", save_name=r"map_manu_subs")
#pdf = hm.generate_pdf_report(pdf_results, projectname = constants["Project name"], filepath = r"./Local_files/GUI_files/Results/")
#print(hm.extract_pairs_df(result))



