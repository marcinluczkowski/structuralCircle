import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca
import plotting as plot


##### LOWEST BENCHMARK ####
#==========USER FILLS IN============#

#Constants
NOK_TO_EURO = 0.085
constants = {
    "TIMBER_GWP": 28.9,       #kg CO2 eq per m^3, based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 2.25,        # 0.0778*28.9 = 2.25kg CO2 eq per m^3, based on Eberhardt
    "TRANSPORT_GWP": 89.6,    #gram per tonne per km, Engedal et. al. 
    "TIMBER_DENSITY": 491.0,  #kg/m^3, based on NEPD-3442-2053-EN
    "STEEL_GWP": 9263.0, #kg CO2 eq per m^3, Norsk stål + density of steel
    "STEEL_REUSE_GWP": 278.0, #kg CO2 eq per m^3, reduction of 97% from new according to Høydahl and Walter
    "VALUATION_GWP": 0.7 *0.5 * NOK_TO_EURO, #NOK per kg CO2, based on OECD
    "TIMBER_PRICE": 3400.0 * NOK_TO_EURO, #Per m^3, Treindustrien 2023
    "TIMBER_REUSE_PRICE" : 3400.0 * NOK_TO_EURO, #Per m^3, assumes the price is the same is new elements
    "STEEL_PRICE": 67.0 * NOK_TO_EURO, #NOK per kg, ENTRA 2021
    "STEEL_REUSE_PRICE": 67.0 * NOK_TO_EURO, #NOK per kg, ENTRA 2021
    "PRICE_TRANSPORTATION": 4.0 * NOK_TO_EURO, #NOK per km per tonne, Grønland 2022 + Gran 2013
    "STEEL_DENSITY": 7850.0, #kg/m^3 EUROCODE
    ########################
    "Project name": "Con_lowest_benchmark",
    "Metric": "Combined",
    "Algorithms": ["greedy_plural", "bipartite_plural"],
    "Include transportation": True,
    "Site latitude": "53.4630014",
    "Site longitude": "-2.2950054",
    "Demand file location": r"./CSV/con_new_demand.xlsx",
    "Supply file location": r"./CSV/con_new_supply.xlsx",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}


### Generating dataset ###

supply_coords = pd.DataFrame(columns = ["Location", "Latitude", "Longitude"])

liverpool = ["Liverpool", "53.4308365", "-2.9633863"]
sheffield = ["Sheffield", "53.3705246", "-1.4747101"]
birmingham = ["Birmingham", "52.4757188","-1.8681831"]
newcastle = ["Newcastle", "54.9750507", "-1.621952"]
london = ["London", "51.5548885", "-0.108438"]

supply_coords.loc[len(supply_coords)] = liverpool
supply_coords.loc[len(supply_coords)] = sheffield
supply_coords.loc[len(supply_coords)] = birmingham
supply_coords.loc[len(supply_coords)] = newcastle
supply_coords.loc[len(supply_coords)] = london

#demand = hm.create_random_data_demand_conference(9000, 2.5, 12.1)
#supply = hm.create_random_data_supply_conference(9000, 2.5, 12.1, supply_coords)
#demand.to_excel(r"" + "./CSV/con_new_demand.xlsx")
#supply.to_excel(r"" + "./CSV/con_new_supply.xlsx")

supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)
constraint_dict = constants["constraint_dict"]

score_function_string = hm.generate_score_function_string(constants)
run_string = hm.generate_run_string(constants)
result_case2 = eval(run_string)
pdf_results_case2 = hm.extract_results_df_pdf(result_case2, constants)
hm.generate_pdf_report(pdf_results_case2, constants["Project name"], supply, demand, filepath = r"./Local_files/GUI_files/Results/")

##### Highest BENCHMARK ####
#==========USER FILLS IN============#

#Constants
NOK_TO_EURO = 0.085
constants = {
    "TIMBER_GWP": 28.9,       #kg CO2 eq per m^3, based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 2.25,        # 0.0778*28.9 = 2.25kg CO2 eq per m^3, based on Eberhardt
    "TRANSPORT_GWP": 89.6,    #gram per tonne per km, Engedal et. al. 
    "TIMBER_DENSITY": 491.0,  #kg/m^3, based on NEPD-3442-2053-EN
    "STEEL_GWP": 9263.0, #kg CO2 eq per m^3, Norsk stål + density of steel
    "STEEL_REUSE_GWP": 278.0, #kg CO2 eq per m^3, reduction of 97% from new according to Høydahl and Walter
    "VALUATION_GWP": 0.7 *2 * NOK_TO_EURO, #NOK per kg CO2, based on OECD
    "TIMBER_PRICE": 3400.0 * NOK_TO_EURO, #Per m^3, Treindustrien 2023
    "TIMBER_REUSE_PRICE" : 3400.0 * NOK_TO_EURO, #Per m^3, assumes the price is the same is new elements
    "STEEL_PRICE": 67.0 * NOK_TO_EURO, #NOK per kg, ENTRA 2021
    "STEEL_REUSE_PRICE": 67.0 * NOK_TO_EURO, #NOK per kg, ENTRA 2021
    "PRICE_TRANSPORTATION": 4.0 * NOK_TO_EURO, #NOK per km per tonne, Grønland 2022 + Gran 2013
    "STEEL_DENSITY": 7850.0, #kg/m^3 EUROCODE
    ########################
    "Project name": "Con_highest_benchmark",
    "Metric": "Combined",
    "Algorithms": ["greedy_plural", "bipartite_plural"],
    "Include transportation": True,
    "Site latitude": "53.4630014",
    "Site longitude": "-2.2950054",
    "Demand file location": r"./CSV/con_new_demand.xlsx",
    "Supply file location": r"./CSV/con_new_supply.xlsx",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}

supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)
constraint_dict = constants["constraint_dict"]

score_function_string = hm.generate_score_function_string(constants)
run_string = hm.generate_run_string(constants)
result_case2 = eval(run_string)
pdf_results_case2 = hm.extract_results_df_pdf(result_case2, constants)
hm.generate_pdf_report(pdf_results_case2, constants["Project name"], supply, demand, filepath = r"./Local_files/GUI_files/Results/")