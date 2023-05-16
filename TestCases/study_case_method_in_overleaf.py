import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching # Matching
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
    "STEEL_REUSE_PRICE": 67.0, #NOK per kg, ENTRA 2021
    "PRICE_TRANSPORTATION": 4.0, #NOK per km per tonne, Grønland 2022 + Gran 2013
    "STEEL_DENSITY": 7850.0, #kg/m^3 EUROCODE
    ########################
    "Project name": "Campussamling Hesthagen",
    "Metric": "GWP",
    "Algorithms": ["greedy_plural","bipartite_plural"],
    "Include transportation": False,
    "Site latitude": "63.4154171",
    "Site longitude": "10.3994672",
    "Demand file location": r"./CSV/genetic_demand.csv",
    "Supply file location": r"./CSV/genetic_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}


def generate_datasets(d_counts, s_counts):
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


    materials = ["Timber"]

    #GENERATE FILE
    #============
    supply = hm.create_random_data_supply_pdf_reports(supply_count = s_counts, length_min = 1.0, length_max = 10.0, area_min = 0.004, area_max = 0.04, materials = materials, supply_coords = supply_coords)
    demand = hm.create_random_data_demand_pdf_reports(demand_count = d_counts, length_min = 1.0, length_max = 10.0, area_min = 0.004, area_max = 0.04, materials = materials)
    supply.index = map(lambda text: "S" + str(text), supply.index)
    demand.index = map(lambda text: "D" + str(text), demand.index)
    return demand, supply

demand, supply = generate_datasets(10, 3)
supply = hm.add_necessary_columns_pdf(supply, constants)
demand = hm.add_necessary_columns_pdf(demand, constants)

constraint_dict = constants["constraint_dict"]
score_function_string = hm.generate_score_function_string(constants)
run_string = hm.generate_run_string(constants)
result = eval(run_string)
pdf_results = hm.extract_results_df_pdf(result, constants)

test = 4

