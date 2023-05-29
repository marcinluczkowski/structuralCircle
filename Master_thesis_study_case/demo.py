import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca
import plotting as plot

"""
Code needed to generate datasets for the demonstration of the design tool
"""

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
    "STEEL_REUSE_PRICE": 67.0, #NOK per kg, ENTRA 2021
    "PRICE_TRANSPORTATION": 4.0, #NOK per km per tonne, Grønland 2022 + Gran 2013
    "STEEL_DENSITY": 7850.0, #kg/m^3 EUROCODE
    ########################
    "Project name": "Demo",
    "Metric": "GWP",
    "Algorithms": ["greedy_single", "greedy_plural", "bipartite_plural"],
    "Include transportation": True,
    "Site latitude": "63.4154171",
    "Site longitude": "10.3994672",
    "Demand file location": r"./CSV/demo_demand.xlsx",
    "Supply file location": r"./CSV/demo_supply.xlsx",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}
#========================#
#Generating dataset
#===================
supply_coords = pd.DataFrame(columns = ["Location", "Latitude", "Longitude"])

steinkjer = ["Steinkjer", "64.024861", "11.4891085"]
meraker = ["Meråker", "63.415312", "11.747262"]
berkak = ["Berkåk", "62.8238946","9.9934341"]
vinjeora = ["Vinjeøra", "63.206995", "8.997224"]
namsos = ["Namsos", "64.4675884", "11.501161"]
dombas = ["Dombås", "62.073208", "9.121031"]

supply_coords.loc[len(supply_coords)] = steinkjer
supply_coords.loc[len(supply_coords)] = namsos
supply_coords.loc[len(supply_coords)] = vinjeora
supply_coords.loc[len(supply_coords)] = meraker
supply_coords.loc[len(supply_coords)] = berkak
supply_coords.loc[len(supply_coords)] = dombas


materials = ["Timber","Steel"]

# GENERATE FILE
# ============
#supply = hm.create_random_data_supply_pdf_reports(supply_count = 500, length_min = 1.0, length_max = 10.0, area_min = 0.004, area_max = 0.04, materials = materials, supply_coords = supply_coords)
#demand = hm.create_random_data_demand_pdf_reports(demand_count = 500, length_min = 1.0, length_max = 10.0, area_min = 0.004, area_max = 0.04, materials = materials)
#hm.export_dataframe_to_csv(supply, r"" + "./CSV/demo_supply.csv")
#hm.export_dataframe_to_csv(demand, r"" + "./CSV/demo_demand.csv")
#hm.export_dataframe_to_xlsx(supply, r"" + "./CSV/demo_supply.xlsx")
#hm.export_dataframe_to_xlsx(demand, r"" + "./CSV/demo_demand.xlsx")
#supply.to_excel(r"" + "./CSV/demo_supply.xlsx")
#demand.to_excel(r"" + "./CSV/demo_supply.xlsx")
#========================================

supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")
hm.export_dataframe_to_csv(supply, r"" + "./CSV/demo_supply.csv")
hm.export_dataframe_to_csv(demand, r"" + "./CSV/demo_demand.csv")


#PRE-PROSESSING DATA
supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)
constraint_dict = constants["constraint_dict"]

########### DEMO: GWP with transportation ###########
score_function_string = hm.generate_score_function_string(constants)
run_string = hm.generate_run_string(constants)
result_case_demo = eval(run_string)
pdf_results_case_demo = hm.extract_results_df_pdf(result_case_demo, constants)
hm.generate_pdf_report(pdf_results_case_demo, constants["Project name"], supply, demand, filepath = r"./Local_files/GUI_files/Results/")


#PLOTS FOR OVERLEAF
#plot.create_map_supply_locations(supply_coords, constants["Site latitude"], constants["Site longitude"], save_name="supply_locations", include_site=False)
#plot.create_map_manufacturer_location(timber_lat="63.368923", timber_lon="10.3749681", steel_lat="63.438445", steel_lon="10.40994", site_lat=constants["Site latitude"], site_lon=constants["Site longitude"], save_name = "manu_locations")