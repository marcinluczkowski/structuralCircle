import sys
sys.path.append('./app/matchingTool/src/Matching')
from matching import Matching
import pandas as pd
import numpy as np
import sys
import helper_methods_LCA as lca
import helper_methods as hm
import helper_methods_PDF as hmpdf
from matching import run_matching
# read input argument from console
#NOTE TO SVERRE: DONT KNOW IF THIS IS NEEDED?
"""
method_name = sys.argv[1]
demand_path = sys.argv[2]
supply_path = sys.argv[3]
result_path = sys.argv[4]
constraint_string = sys.argv[5]
"""
######################

#==========USER FILLS IN============#
#Constants
#TODO: FIND ALL DEFAULT VALUES FOR CONSTANTS, especially for price
constants = {


    "WINDOW_GWP": 28.9,      
    "WINDOW_REUSE_GWP": 2.25,        
    "WINDOW_DENSITY": 491.0,  
    "WINDOW_PRICE": 435, 
    "WINDOW_REUSE_PRICE" : 100,

    "DOOR_GWP": 10,      
    "DOOR_REUSE_GWP": 3,        
    "DOOR_DENSITY": 200,  
    "DOOR_PRICE": 250, 
    "DOOR_REUSE_PRICE" : 80,

    "TIMBER_GWP": 15,       # based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 1.5,        # 0.0778*28.9 = 2.25 based on Eberhardt
    "TIMBER_DENSITY": 200,  # kg, based on NEPD-3442-2053-EN
    "TIMBER_PRICE": 100, #Per m^3 https://www.landkredittbank.no/blogg/2021/prisen-pa-sagtommer-okte-20-prosent/
    "TIMBER_REUSE_PRICE" : 70, #Per m^3, Random value

    "CONCRETE_GWP": 400,      
    "CONCRETE_DENSITY": 491.0,  
    "CONCRETE_PRICE": 435, 
    "CONCRETE_REUSE_PRICE" : 100,
    "CONCRETE_REUSE_GWP": 100,  

    "N/A_GWP": 400,      
    "N/A_DENSITY": 491.0,  
    "N/A_PRICE": 435, 
    "N/A_REUSE_PRICE" : 100,
    "N/A_REUSE_GWP": 100,  
 
    "STEEL_GWP": 9263, # [kg CO2 pr m^3] Taken from Trine and Elise 
    "STEEL_REUSE_GWP": 200, #[kg CO2 pr m^3] Taken from Trine and Elise 
    "STEEL_PRICE": 6, #[NOK pr m^3] Taken from Trine and Elise 
    "STEEL_REUSE_PRICE": 67, #[NOK pr m^3] Taken from Trine and Elise 
    "STEEL_DENSITY": 100, #[kg pr m^3] Taken from Trine and Elise 

    "VALUATION_GWP": 0.7, #In kr:Per kg CO2, based on OECD
    "PRICE_TRANSPORTATION": 4, #Price per km per tonn. Derived from 2011 numbers on scaled t0 2022 using SSB
    "TRANSPORT_GWP": 81,    # TODO kg/m3/t based on ????
    "Price" : 100,



    ########################
    "Project name": "Sognsveien 17",
    "Metric": "GWP",
    #"Algorithms": ["bipartite", "greedy_plural", "bipartite_plural", "bipartite_plural_multiple"],
    "Algorithms": ["greedy_plural"],
    "Include transportation": False,
    "Site latitude": "59.94161606",
    "Site longitude": "10.72994518",
    "Demand file location": r"./app/matchingTool/src/TestCases/Data/CSV/Demand_homemade.xlsx",
    "Supply file location": r"./app/matchingTool/src/TestCases/Data/CSV/Supply_homemade.xlsx",
    #"Demand file location": r"./app/matchingTool/src/TestCases/Data/CSV/test_demand_IFC.csv",
    #"Supply file location": r"./app/matchingTool/src/TestCases/Data/CSV/test_supply_IFC.csv",

    "tol_1D_area" : "0.9",
    "tol_1D_moment_of_inertia" : "0.9",
    "tol_1D_length" : "0.9",
    "tol_1D_width" : "0.9",
    "tol_1D_height" : "0.9",  
    "tol_1D_quality" : "0.9",

    "tol_2D_width" : "0.9",
    "tol_2D_height" : "0.9",
    "tol_2D_quality" : "0.9",

    "tol_3D_length" : "0.9",
    "tol_3D_width" : "0.9",
    "tol_3D_height" : "0.9",
    "tol_3D_quality" : "0.9",

    "constraint_dict": {'Area' : '>=', 'Iy e-6' : '>=', 'Length' : '>=', 'Width' : '>=', 'Height' : '>=', 'Material': '==', 'Quality' : '>='},
    "constraint2D_dict" :  {'Width' : '==', 'Height' : '==', 'Material' : '==', 'Quality' : '>='},
    "constraint3D_dict" :  {'Length' : '>=','Width' : '==', 'Height' : '==', 'Material' : '==', 'Quality' : '>='},

    "Name" : {"Timber", "Steel", "Window", "Door", "Slab", "Concrete - B35", "Steel - S355", "Steel - S235"},
    "element_linear" : {"IfcBeam", "IfcColumn"},
    "element_2d" : {"IfcWindow", "IfcDoor"},
    "element_3d" : {"IfcSlab"}
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






#GENERATE FILE
#============
"""supply = hmpdf.create_random_data_supply_pdf_reports(supply_count = 10, length_min = 1.0, length_max = 10.0, width_min = 1.0, width_max = 10.0, height_min = 1.0, height_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials, supply_coords = supply_coords)
demand = hmpdf.create_random_data_demand_pdf_reports(demand_count = 10, length_min = 1.0, length_max = 10.0, width_min = 1.0, width_max = 10.0, height_min = 1.0, height_max = 10.0, area_min = 0.15, area_max = 0.30, materials = materials)
hm.export_dataframe_to_csv(supply, r"" + "./TestCases/Data/CSV/test_supply.csv")
hm.export_dataframe_to_csv(demand, r"" + "./TestCases/Data/CSV/test_demand.csv")
"""
#========================================
score_function_string = hm.generate_score_function_string(constants)
score_function_string_2d = hm.generate_score_function_string_2d(constants)

supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")

#hm.create_graph(supply, demand, "Length", number_of_intervals= 2, save_filename = r"C:\Users\sigur\Downloads\test.png")

constraint_dict = constants["constraint_dict"]
constraint2D_dict = constants["constraint2D_dict"]
constraint3D_dict = constants["constraint3D_dict"]


#Add necessary columns to run the algorithm
supply = hmpdf.add_necessary_columns_pdf(supply, constants)
demand = hmpdf.add_necessary_columns_pdf(demand, constants)


m= Matching(demand, supply, score_function_string, score_function_string_2d, constants, constraints = constraint_dict, constraints2D = constraint2D_dict, constraints3D = constraint3D_dict, solution_limit=60)

m.match_greedy(plural_assign=True)
simple_pairs = m.pairs
m.calculate_result()
simple_results = m.result
print("------------------------- Results -------------------------")
print("Weights matrix ")
print(m.weights)
print()
print("Algorithm used; " + ", ".join(constants["Algorithms"]))
print()
print("Simple pairs:")
print(simple_pairs)
print()
if (constants["Metric"] == "Price"):
    print("Simple " + constants["Metric"] + " score;")
if (constants["Metric"] == "GWP"):
    print("Simple " + constants["Metric"] + " score;")
if (constants["Metric"] == "Combined"):
    print("Simple " + constants["Metric"] + " score;")
print(simple_results)



#hm.export_dataframe_to_xlsx(pd.DataFrame(m.weights, r"" + "./app/matchingTool/src/TestCases/Data/CSV/Results/weights_before"))

constants["Metric"] = "Price"
score_function_string = hm.generate_score_function_string(constants)
score_function_string = score_function_string.replace(" ", "")


if tuple(m.demand['Element'].tolist()) in constants["element_linear"] or tuple(m.demand['Element'].tolist()) in constants["element_3d"]:
    m.demand['Price pr element'], m.demand['Transportation'] = m.demand.eval(m.score_function_string)
else:
    m.demand['Price pr element'], m.demand['Transportation'] = m.demand.eval(m.score_function_string_2d)

if tuple(m.supply['Element'].tolist()) in constants["element_linear"] or tuple(m.supply['Element'].tolist()) in constants["element_3d"]:
    m.supply['Price pr element'], m.supply['Transportation'] = m.supply.eval(m.score_function_string)
else:
    m.supply['Price pr element'], m.supply['Transportation'] = m.supply.eval(m.score_function_string_2d)   


constants["Metric"] = "GWP"
score_function_string = hm.generate_score_function_string(constants)
score_function_string = score_function_string.replace(" ", "")


if tuple(m.demand['Element'].tolist()) in constants["element_linear"] or tuple(m.demand['Element'].tolist()) in constants["element_3d"]:
    m.demand['GWP pr element'], m.demand['Transportation'] = m.demand.eval(m.score_function_string)
else:
    m.demand['GWP pr element'], m.demand['Transportation'] = m.demand.eval(m.score_function_string_2d)

if tuple(m.supply['Element'].tolist()) in constants["element_linear"] or tuple(m.supply['Element'].tolist()) in constants["element_3d"]:
    m.supply['GWP pr element'], m.supply['Transportation'] = m.supply.eval(m.score_function_string)
else:
    m.supply['GWP pr element'], m.supply['Transportation'] = m.supply.eval(m.score_function_string_2d)   


#hm.export_dataframe_to_xlsx(m.weights, r"" + "./app/matchingTool/src/TestCases/Data/CSV/Results/weights")
hm.export_dataframe_to_xlsx(m.supply, r"" + "./app/matchingTool/src/TestCases/Data/CSV/Results/test_supply_IFC_result1.xlsx")
hm.export_dataframe_to_xlsx(m.demand, r"" + "./app/matchingTool/src/TestCases/Data/CSV/Results/test_demand_IFS_result1.xlsx")
hm.export_dataframe_to_xlsx(m.pairs, r"" + "./app/matchingTool/src/TestCases/Data/CSV/Results/pairs.xlsx")





print(m.demand)
print(m.supply)