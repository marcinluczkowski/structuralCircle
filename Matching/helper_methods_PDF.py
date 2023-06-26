import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import logging
import helper_methods_LCA as lca
import random
from fpdf import FPDF
from datetime import date
import helper_methods_plotting as plot


def extract_results_df_pdf(dict_list, constants):
    """Post-processesing of the results from the matching-tool. Generates needed information for the PDF report

    Args:
        dict_list (dictionary): the dictionary received after running "run-matching" in matching.py
        constants (dictionary): all the user inputs used for calulations in the matching process

    Returns:
        dictionary: all the information needed to generate the PDF report
    """

    sub_df = {"Names": [], "Score": [], "Time": [], "Substitutions": [], "Sub_percent": []}
    cols = []
    used_constants = {"Density timber": (constants["TIMBER_DENSITY"], "kg/m^3"), "Density steel": (constants["STEEL_DENSITY"], "kg/m^3")}
    metric = constants["Metric"]
    include_transportation = constants["Include transportation"]
    #Get performance of all algorithms runned:
    for run in dict_list:
        sub_df["Score"].append(round(run['Match object'].result, 2))
        sub_df["Time"].append(run["Time"])
        num_subs = len(run['Match object'].pairs[run['Match object'].pairs["Supply_id"].str.startswith("S")])
        sub_df["Substitutions"].append(num_subs)
        sub_df["Sub_percent"].append(round(num_subs/len(run["Match object"].demand)*100, 2))
        sub_df["Names"].append(run["Name"])
        cols.append(run['Name'])
    algorithms_df = pd.DataFrame(sub_df, index= cols)   
    algorithms_df = algorithms_df.sort_values(by=["Score", 'Time'], ascending=[True, True]) #Sorting the algorithms to find the best one
    results_dict = algorithms_df.iloc[0].to_dict()
    results_dict["Algorithm"] = algorithms_df.iloc[0].name
    results_dict["Performance"] = algorithms_df

    #Get more information about the best algorithm
    index_algorithm = next(filter(lambda i: dict_list[i]['Name'] == algorithms_df.iloc[0].name, range(len(dict_list))))
    match_object = dict_list[index_algorithm]["Match object"]
    all_new_score = match_object.demand["Score"].sum()
    all_new_transport = match_object.demand["Transportation"].sum()
    results_dict["All new score"] = round(all_new_score, 2)
    results_dict.update(constants)

    #Adding the constants used in the matching tool
    if metric == "GWP":
        results_dict["Unit"] = "kgCO2eq"
        used_constants.update({"GWP new timber": (constants["TIMBER_GWP"],"kgCO2eq/m^3"), "GWP reusable timber": (constants["TIMBER_REUSE_GWP"], "kgCO2eq/m^3"), "GWP new steel": (constants["STEEL_GWP"], "kgCO2eq/m^3"), "GWP reusable steel": (constants["STEEL_REUSE_GWP"], "kgCO2eq/m^3")})
    elif metric == "Price":
        results_dict["Unit"] = "NOK"
        used_constants.update({"Price new timber": (constants["TIMBER_PRICE"], "NOK/m^3"), "Price reusable timber": (constants["TIMBER_REUSE_PRICE"], "NOK/m^3"), "Price new steel": (constants["STEEL_REUSE_PRICE"], "NOK/kg"), "Price reusable steel": (constants["STEEL_REUSE_PRICE"], "NOK/kg")})
    elif metric == "Combined":
        results_dict["Unit"] = "NOK"
        used_constants.update({"GWP new timber": (constants["TIMBER_GWP"],"kgCO2eq/m^3"), "GWP reusable timber": (constants["TIMBER_REUSE_GWP"], "kgCO2eq/m^3"), "GWP new steel": (constants["STEEL_GWP"], "kgCO2eq/m^3"), "GWP reusable steel": (constants["STEEL_REUSE_GWP"], "kgCO2eq/m^3"), "Valuation of GWP": (constants["VALUATION_GWP"], "NOK/kgCO2eq")}) 
        used_constants.update({"Price new timber": (constants["TIMBER_PRICE"], "NOK/m^3"), "Price reusable timber": (constants["TIMBER_REUSE_PRICE"], "NOK/m^3"), "Price new steel": (constants["STEEL_PRICE"], "NOK/kg"), "Price reusable steel": (constants["STEEL_REUSE_PRICE"], "NOK/kg")})
    
    #Information about the savings and substitutions
    results_dict["Savings"] =  round(results_dict["All new score"] - results_dict["Score"], 2)
    results_dict["Number_reused"] = len(match_object.supply) - len(match_object.demand)
    results_dict["Number_demand"] = len(match_object.demand)
    results_dict["Number of substitutions"] = len(match_object.pairs[match_object.pairs["Supply_id"].str.startswith("S")])
    results_dict["Number of substitutions"] = sub_df["Substitutions"][sub_df["Names"].index(results_dict["Algorithm"])]

    #Adding information about transportation
    if include_transportation:
        results_dict["Transportation included"] = "Yes"
        results_dict["Transportation score"] = round(match_object.result_transport, 2)
        results_dict["Transportation percentage"] = round(match_object.result_transport/results_dict["Score"]*100, 2)
        results_dict["Transportation all new"] = round(all_new_transport, 2)
        if metric == "GWP":
            used_constants.update({"GWP transportation": (constants["TRANSPORT_GWP"],"g/tonne/km")})
        elif metric == "Combined":
            used_constants.update({"GWP transportation": (constants["TRANSPORT_GWP"],"g/tonne/km"), "Price of transportation": (constants["PRICE_TRANSPORTATION"], "NOK/tonne/km")})
        elif metric == "Price":
            used_constants.update({"Price of transportation": (constants["PRICE_TRANSPORTATION"], "NOK/tonne/km")})                 
    else:
        results_dict["Transportation included"] = "No"
        results_dict["Transportation percentage"] = 0
        results_dict["Transportation score"] = 0
        results_dict["Transportation all new"] = 0

    #Adding the pairs of the best algorithm
    pairs = extract_pairs_df(dict_list)[results_dict["Algorithm"]]
    pairs.name = "Substitutions"
    results_dict["Pairs"] = pairs
    results_dict["Constants used"] = used_constants
    return results_dict

def create_random_data_supply_pdf_reports(supply_count, length_min, length_max, area_min, area_max, materials, supply_coords):
    """Creates random data for the case studies in the master thesis

    Args:
        supply_count (int): number of supply elements
        length_min (float): minimum length
        length_max (float): maximum length
        area_min (float): minimum area
        area_max (_float): maximum area
        materials (list): list of avaiable materials as string => materials = ["Timber", "Steel"]
        supply_coords (DataFrame): available supply locations with name of location and the corresponding latitude and longitude

    Returns:
        DataFrame: supply dataset
    """

    #Available steel sections with corresponding area and moment of inertia
    steel_cs = {"IPE100": (1.03e-3, 1.71e-6), #(area, moment of inertia)
                "IPE140": (1.64e-3, 5.41e-6),
                "IPE160": (2.01e-3, 8.69e-6),
                "IPE180": (2.39e-3, 13.20e-6),
                "IPE220": (3.34e-3, 27.7e-6),
                "IPE270": (4.59e-3, 57.9e-6),
                "IPE300": (5.38e-3, 83.6e-6)
    }

    np.random.RandomState(2023)
    supply = pd.DataFrame()
    supply['Length'] = np.round((length_max - length_min) * np.random.random_sample(size = supply_count) + length_min, 2)
    supply['Area'] = 0 
    supply['Moment of Inertia'] = 0
    supply['Material'] = ""
    supply["Location"]=0
    supply["Latitude"]=0
    supply["Longitude"]=0
    
    #Add data
    for row in range(len(supply)):
        #Random material from the material list
        material = materials[random.randint(0, len(materials)-1)]
        if material == "Timber":
            area = np.random.uniform(area_min, area_max)
            supply.loc[row, "Area"] = area
            supply.loc[row, "Moment of Inertia"] = area**2/12
        elif material == "Steel":
            cs = random.choice(list(steel_cs.keys()))
            supply.loc[row, "Area"] = steel_cs[cs][0]
            supply.loc[row, "Moment of Inertia"] = steel_cs[cs][1]
        supply.loc[row, "Material"] = material
        #Random location from the location dataframe
        lokasjon=random.randint(0, len(supply_coords)-1)
        supply.loc[row,"Latitude"]=supply_coords.loc[lokasjon,"Latitude"]
        supply.loc[row,"Longitude"]=supply_coords.loc[lokasjon,"Longitude"]
        supply.loc[row,"Location"]=supply_coords.loc[lokasjon,"Location"]
    return supply

def create_random_data_demand_pdf_reports(demand_count, length_min, length_max, area_min, area_max, materials):
    """Creates random data for the case studies in the master thesis

    Args:
        demand_count (int): number of supply elements
        length_min (float): minimum length
        length_max (float): maximum length
        area_min (float): minimum area
        area_max (_float): maximum area
        materials (list): list of avaiable materials as string => materials = ["Timber", "Steel"]

    Returns:
        DataFrame: demand dataset
    """
    #Available steel sections with corresponding area and moment of inertia
    steel_cs = {"IPE100": (1.03e-3, 1.71e-6), # (area, moment of inertia)
                "IPE140": (1.64e-3, 5.41e-6),
                "IPE160": (2.01e-3, 8.69e-6),
                "IPE180": (2.39e-3, 13.20e-6),
                "IPE220": (3.34e-3, 27.7e-6),
                "IPE270": (4.59e-3, 57.9e-6),
                "IPE300": (5.38e-3, 83.6e-6)
    }
    np.random.RandomState(2023)
    demand = pd.DataFrame()
    demand['Length'] = np.round((length_max - length_min) * np.random.random_sample(size = demand_count) + length_min, 2)
    demand['Area'] = 0
    demand['Moment of Inertia'] = 0
    demand['Material'] = ""
    demand["Manufacturer"]=0
    demand["Latitude"]=0
    demand["Longitude"]=0
    
    #Add random data
    for row in range(len(demand)):
        #Random material from the material list
        material = materials[random.randint(0, len(materials)-1)]
        if material == "Timber":
            area = np.random.uniform(area_min, area_max)
            demand.loc[row, "Area"] = area
            demand.loc[row, "Moment of Inertia"] = area**2/12
        elif material == "Steel":
            cs = random.choice(list(steel_cs.keys()))
            demand.loc[row, "Area"] = steel_cs[cs][0]
            demand.loc[row, "Moment of Inertia"] = steel_cs[cs][1]
        demand.loc[row, "Material"] = material
    return demand

def generate_plots_pdf_report(supply, demand, pdf_results, transportation_included):
    """Generates the required plots for the automatically generated PDF report

    Args:
        supply (DataFrame): supply DataFrame
        demand (DataFrame): demand DataFrame
        pdf_results (dictionary): dictionary from extract_results_df_pdf()
        transportation_included (boolean): True or False depending on if transportation is included or not
    """
    plot.create_graph(supply, demand, target_column="Length", unit=r"[m]", number_of_intervals=5, fig_title = "", save_filename=r"length_plot.png")
    plot.create_graph(supply, demand, target_column="Area", unit=r"[m$^2$]", number_of_intervals=5, fig_title = "", save_filename=r"area_plot.png")
    plot.create_graph(supply, demand, target_column="Moment of Inertia", unit=r"[m$^4$]", number_of_intervals=5, fig_title = "", save_filename=r"inertia_plot.png")
    plot.plot_materials(supply, demand, "", save_filename=r"material_plot.png")

    if transportation_included:
        plot.create_map_substitutions(supply, pdf_results, "supply", color = "green", legend_text="Reusable elements", save_name=r"map_reused_subs")
        plot.create_map_substitutions(demand, pdf_results, "demand", color = "red", legend_text="Manufacturers", save_name=r"map_manu_subs")

def generate_pdf_report(results, projectname, supply, demand, filepath):
    """Automatically generating a PDF report that visualizes the results from the desing tool

    Args:
        results (dictionary): dictionary from extract_results_df_pdf()
        projectname (string): name of the project
        supply (DataFrame): supply dataframe
        demand (DataFrame): demand dataframe
        filepath (string): the filepath where the PDF should be saved
    """
    def new_page():
        """Creates a new page
        """
        pdf.add_page()
        
        # Set the background color
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(0, 0, 210, 297, "F")
        
        # Add the image to the PDF
        pdf.image(r"./Local_files/NTNU-logo.png", x=10, y=10, w=30)

        # Add the date to the upper right corner of the PDF
        pdf.set_xy(200, 10)
        pdf.set_font("Times", size=10)
        pdf.cell(0, 10, str(date.today().strftime("%B %d, %Y")), 0, 1, "R")

    #Add CSV containing results to "Results"-folder
    save_name = projectname.replace(" ", "_") + "_substitutions.xlsx"
    export_dataframe_to_xlsx(results["Pairs"], filepath + save_name)

    if results["Transportation included"] == "No":
        transportation_included = False
    elif results["Transportation included"] == "Yes":
        transportation_included = True
    #Add relevant plots
    generate_plots_pdf_report(supply, demand, results, transportation_included)
    pdf = FPDF()
    new_page()

    ################## SUMMARY OF THE RESULTS ##################
    # Set the font and size for the title
    pdf.set_font("Times", size=36)
    pdf.set_text_color(0, 80, 158)
    pdf.set_y(10)
    # Add the title to the PDF
    pdf.cell(0, 50, "Results from the Design Tool", 0, 1, "C")
    pdf.set_left_margin(15)

    # Information about the project:
    ################################
    pdf.set_y(50)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Times", size=12, style = "B")
    pdf.cell(30, 10, f"Project name: ", 0, 0)
    pdf.set_font("Times", size=12, style = "")
    pdf.cell(0, 10, f"{results['Project name']}", 0, 1)

    pdf.set_font("Times", size=12, style = "B")
    pdf.cell(55, 10, f"Construction site located at: ", 0, 0)
    pdf.set_font("Times", size=12, style = "")
    site_lat = round(float(results['Site latitude']), 4)
    site_lon = round(float(results['Site longitude']), 4)
    pdf.cell(0, 10, f"{site_lat}, {site_lon}", 0, 1)

    # Set the font and size for the tables
    pdf.set_font("Times", size=12)
    pdf.set_left_margin(15)
    table_x = (pdf.w - 180) / 2
    table_y1 = 75
    table_y2 = 180
    pdf.set_y(table_y1)
    pdf.set_font("Times", size=24, style ="")
    pdf.multi_cell(160, 7, txt="Summary of results")
    pdf.set_font("Times", size=10)
    pdf.set_left_margin(30)
    pdf.ln(5)
    pdf.set_fill_color(96, 150, 208)
    pdf.set_draw_color(204, 204, 204)
    pdf.cell(50, 10, f"Total score", 1, 0, "C", True)
    pdf.cell(50, 10, f"Score without reuse", 1, 0, "C", True)
    pdf.cell(25, 10, "Savings", 1, 0, "C", True)
    pdf.cell(25, 10, "Substitutions", 1, 1, "C", True)
    pdf.set_fill_color(247, 247, 247)
    #Extract information for the table
    score = format_float(round(results['Score'],0))
    unit = results["Unit"]
    new_score = format_float(round(results['All new score'],0))
    substitutions = round(results['Number of substitutions']/results['Number_demand']*100, 2)
    savings = round(results['Savings']/results['All new score']*100, 2)
    savings_value = format_float(round(results['Savings'],0))

    #Determine the format of the scores
    if unit == "NOK":
        score_text = f"{unit} {score}"
        score_new_text = f"{unit} {new_score}"
        savings_value_text = f"{unit} {savings_value}"
    else:
        score_text = f"{score} {unit}"
        score_new_text = f"{new_score} {unit}"
        savings_value_text = f"{savings_value} {unit}"

    pdf.cell(50, 10, score_text, 1, 0, "C", True)
    pdf.cell(50, 10, score_new_text, 1, 0, "C", True)
    pdf.cell(25, 10, f"{savings}%", 1, 0, "C", True)
    pdf.cell(25, 10, f"{substitutions}%", 1, 1, "C", True) 
    pdf.ln()

    #Short text summary
    pdf.set_left_margin(15)
    pdf.set_y(110)
    pdf.set_font("Times", size=12, style ="")
    summary = f"The best results was obtained by the following algorithm: {results['Algorithm']}. This algorithm sucessfully substituted {results['Number of substitutions']}/{results['Number_demand']} ({substitutions}%) of the demand elements with reusable elements. Using '{results['Metric']}' as the optimization metric, a total score of {score_text} was achieved. For comparison, a score of {score_new_text} would have been obtained by employing exclusively new materials. This resulted in a total saving of {savings}%, which corresponds to {savings_value_text}."
    if results["Metric"] == "GWP":
        summary += f" The savings is equivalent to {int(np.floor((results['All new score']-results['Score'])/206))*2} flights for one person between Oslo and Trondheim."
    if transportation_included:
        summary += f" Note that impacts of transporting the materials to the construction site was accounted for and contributed to {results['Transportation percentage']}% of the total score. "
    else:
        summary += f" Note that impacts of transporting the materials to the construction site was not accounted for. "
    summary += f"Open the Excel file \"{save_name}\" to examine the substitutions."

    pdf.multi_cell(pdf.w-2*15,8, summary, 0, "L", False)

 
    #Constants used in calculations:
    ###############
    if len(list(results["Constants used"].keys())) > 8:
        new_page()
        pdf.set_xy(table_x, 30)
    else:
        pdf.set_xy(table_x, table_y2)
    pdf.set_font("Times", size=16, style ="")
    pdf.multi_cell(160, 7, txt="Constants used in the calculations")
    pdf.set_font("Times", size=10)
    pdf.ln(5)
    pdf.set_fill_color(96, 150, 208)
    pdf.set_draw_color(204, 204, 204)
    pdf.set_left_margin(30)
    pdf.cell(50, 10, "Constant", 1, 0, "C", True)
    pdf.cell(50, 10, "Value", 1, 0, "C", True)
    pdf.cell(50, 10, "Unit", 1, 1, "C", True)
    pdf.set_fill_color(247, 247, 247)

    for key, values in results["Constants used"].items():
        pdf.cell(50, 10, f"{key}", 1, 0, "C", True)
        pdf.cell(50, 10, f"{values[0]}", 1, 0, "C", True)
        pdf.cell(50, 10, f"{values[1]}", 1, 0, "C", True)
        pdf.ln()

        
    
    ################## Information about the datasets ##################
    new_page()
    pdf.set_font("Times", size=16, style ="")
    pdf.set_xy(table_x, 30)
    pdf.multi_cell(160, 7, txt="Information about the datasets")
    pdf.set_font("Times", size=10)
    pdf.ln(5)
    pdf.set_left_margin(30)
    pdf.set_fill_color(96, 150, 208)
    pdf.set_draw_color(204, 204, 204)
    pdf.cell(30, 10, "Elements", 1, 0, "C", True)
    pdf.cell(80, 10, "Filename", 1, 0, "C", True)
    pdf.cell(40, 10, "Number of elements", 1, 1, "C", True)
    pdf.set_fill_color(247, 247, 247)
    pdf.cell(30, 10, f"Supply", 1, 0, "C", True)
    pdf.cell(80, 10, f"{results['Supply file location'].split('/')[-1]}", 1, 0, "C", True)
    pdf.cell(40, 10, f"{results['Number_reused']}", 1, 0, "C", True)
    pdf.ln()
    pdf.cell(30, 10, f"Demand", 1, 0, "C", True)
    pdf.cell(80, 10, f"{results['Demand file location'].split('/')[-1]}", 1, 0, "C", True)
    pdf.cell(40, 10, f"{results['Number_demand']}", 1, 0, "C", True)

    pdf.set_left_margin(15)
    pdf.set_y(80)
    pdf.set_font("Times", size=12, style ="")
    summary_info = f"The datasets contains {results['Number_reused']} supply elements and {results['Number_demand']} demand elements. The graphs below depicts the distribution of some of the properties of the elements, including the material, length, area, and moment of inertia."
    pdf.multi_cell(pdf.w-2*15,8, summary_info, 0, "L", False)

    #Plots to include
    plots = ["Plots/material_plot.png", "Plots/length_plot.png", "Plots/area_plot.png", "Plots/inertia_plot.png"]
    x = 7.5
    y = 102.5
    w = 95
    for i in range (len(plots)):
        pdf.image(r"" + f"./Local_files/GUI_files/Results/{plots[i]}", x, y, w)
        print("Included", plots[i])
        if i % 2 == 1:
            y += 72.5
            x = 7.5
        if i % 2 == 0:
            x += 100

    ################## Impact of transportation and Performance of algorithms ##################
    new_page()
    y_information = 30

    if transportation_included: #Add a page with information about only transportation
        pdf.set_left_margin(15)
        pdf.set_y(y_information) #prior 30
        pdf.set_font("Times", size=16, style ="")
        pdf.multi_cell(160, 7, txt="Impact of transportation")
        pdf.set_font("Times", size=10)
        pdf.set_left_margin(30)
        pdf.ln(5)
        pdf.set_fill_color(96, 150, 208)
        pdf.set_draw_color(204, 204, 204)
        pdf.cell(50, 10, f"Utilizing reusable elements", 1, 0, "C", True)
        pdf.cell(50, 10, "Percentage of total score", 1, 0, "C", True)
        pdf.cell(50, 10, "Only manufactured elements", 1, 1, "C", True)
        pdf.set_fill_color(247, 247, 247)
        transportation_score = format_float(round(results['Transportation score'], 0))
        new_transportation_score = format_float(round(results['Transportation all new'], 0))
        
        #Determine the format of the scores
        if unit == "NOK":
            trans_text = f"{unit} {transportation_score}"
            trans_new_text = f"{unit} {new_transportation_score}"
        else:
            trans_text = f"{transportation_score} {unit}"
            trans_new_text = f"{new_transportation_score} {unit}"

        pdf.cell(50, 10, trans_text, 1, 0, "C", True)
        pdf.cell(50, 10, f"{results['Transportation percentage']}%", 1, 0, "C", True)
        pdf.cell(50, 10, trans_new_text, 1, 1, "C", True)
        pdf.ln()
        y_information += 35
        #Short text summary
        pdf.set_left_margin(15)
        pdf.set_y(y_information)
        pdf.set_font("Times", size=12, style ="")
        summary = f"All calculations in this report accouned for the effects of material transportation to the construction site. Transportation itself was responsible for {trans_text}. This accounts for {results['Transportation percentage']}% of the total score of {score_text}. For comparison, the transportation impact for exclusively using new materials would have been {trans_new_text}. Two maps are included to show the locations of the suggested element substitutions from the design tool. The numbers on the maps indicate the number of elements transported from each location."
        pdf.multi_cell(pdf.w-2*15,8, summary, 0, "L", False)
        pdf.set_y(y_information)
        #Maps to include
        maps = ["Maps/map_reused_subs.png", "Maps/map_manu_subs.png"]
        x = 7.5
        y = y_information + 60
        w = 95
        for i in range (len(maps)):
            pdf.image(r"" + f"./Local_files/GUI_files/Results/{maps[i]}", x, y, w)
            print("Included", maps[i])
            if i % 2 == 1:
                y += 75
                x = 7.5
            if i % 2 == 0:
                x += 100

        new_page() #Create a new page for the performance

    #Performance of algorithms
    ##########################
    y_information = 30
    pdf.set_left_margin(15)
    pdf.set_y(y_information)
    pdf.set_font("Times", size=16, style ="")
    pdf.multi_cell(160, 7, txt="Performance of the optimization algorithms")
    pdf.set_font("Times", size=10)
    pdf.set_left_margin(17)
    pdf.ln(5)
    pdf.set_fill_color(96, 150, 208)
    pdf.set_draw_color(204, 204, 204)
    pdf.cell(75, 10, "Algorithm name", 1, 0, "C", True)
    pdf.cell(51, 10, "Total score", 1, 0, "C", True)
    pdf.cell(25, 10, "Substitutions", 1, 0, "C", True)
    pdf.cell(25, 10, "Time", 1, 1, "C", True)

    pdf.set_fill_color(247, 247, 247)
    performance = results['Performance'] #Dataframe
    
    print_names = ""
    for i in range(len(performance)):
        y_information += 10
        performance_score = format_float(round(performance.iloc[i]['Score'], 0)) 
        performance_time = round(performance.iloc[i]['Time'], 2)
        if unit == "NOK":
            performance_text = f"{unit} {performance_score}"
        else:
            performance_text = f"{performance_score} {unit}"
        pdf.cell(75, 10, f"{performance.iloc[i]['Names']}", 1, 0, "C", True)
        pdf.cell(51, 10, performance_text, 1, 0, "C", True)
        pdf.cell(25, 10, f"{performance.iloc[i]['Sub_percent']}%", 1, 0, "C", True)
        pdf.cell(25, 10, f"{performance_time}s", 1, 0, "C", True)
        if len(performance) == 1:
            print_names += performance.iloc[i]['Names']
        elif i != len(performance) - 1:
            print_names += performance.iloc[i]['Names']
            print_names += ", " 
        else:
            print_names += "and " + performance.iloc[i]['Names']
        pdf.ln()

    if len(performance) == 1:
        summary = f"The design tool achieved a score of {score_text} with the following algorithm: {performance.iloc[0]['Names']}. The substitutions by this algorithm are completed in {results['Time']} seconds"
    else:
        summary = f"The design tool was executed with {len(performance)} algorithms, namely: {print_names}. The {results['Algorithm']} yielded the lowest score, as shown in the table. The substitutions by this algorithm was completed in {results['Time']} seconds."
    pdf.set_font("Times", size=12, style ="")
    pdf.set_left_margin(15)
    pdf.set_y(y_information+25)
    pdf.multi_cell(pdf.w-2*15,8, summary, 0, "L", False)
    save_string = filepath+projectname
    save_string = save_string.replace(" ", "_")
    # Save the PDF to a file
    pdf.output(save_string+"_report.pdf")

def add_necessary_columns_pdf(dataframe, constants):
    """Pre-processing of the imported dataframes from a CSV or Excel file. Fill the dataframes with necessary columns based on the user-selected metric

    Args:
        dataframe (DataFrame): supply or demand dataframe
        constants (dictionary): constants to use in the matching tool

    Returns:
        DataFrame: updated dataframe with necessary columns
    """

    dataframe = dataframe.copy()
    metric = constants["Metric"]
    element_type = list(dataframe.index)[0][:1]
    dataframe["Density"] = 0
    dataframe["Site_lat"] = constants["Site latitude"]
    dataframe["Site_lon"] = constants["Site longitude"]

    if metric == "GWP":
        dataframe["Gwp_factor"] = 0 
    elif metric == "Combined":
        dataframe["Gwp_factor"] = 0 
        dataframe["Price"] = 0
    elif metric == "Price":
        dataframe["Price"] = 0

    #If dataframe is demand, fill in the location and corresponding coordinates and to the closet manufacturer.
    if element_type=="D" and constants["Include transportation"]:
        dataframe=fill_closest_manufacturer(dataframe,constants)

    #Adding necessary columns based on the chosen metric
    for row in range(len(dataframe)):
        material = dataframe.iloc[row][dataframe.columns.get_loc("Material")].split()[0] #NOTE: Assumes that material-column has the material name as the first word, e.g. "Timber C14" or "Steel ASTM A992"
        dataframe.iloc[row, dataframe.columns.get_loc("Density")] = constants[f"{material.upper()}_DENSITY"]

        if element_type == "S":
            constant_name = f"{material.upper()}_REUSE"
        else:
            constant_name = f"{material.upper()}"

        if metric == "GWP" or metric == "Combined":
                dataframe.iloc[row, dataframe.columns.get_loc("Gwp_factor")] = constants[constant_name + "_GWP"]
        if metric == "Price" or metric == "Combined":
                if material.upper() == "STEEL":
                    price = constants[constant_name + "_PRICE"] * constants[f"{material.upper()}_DENSITY"]
                else:
                    price = constants[constant_name + "_PRICE"]
                dataframe.iloc[row, dataframe.columns.get_loc("Price")] = price
    return dataframe

print_header = lambda matching_name: print("\n"+"="*(len(matching_name)+8) + "\n*** " + matching_name + " ***\n" + "="*(len(matching_name)+8) + "\n")