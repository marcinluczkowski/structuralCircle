import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import logging
import helper_methods_LCA as lca
import helper_methods as hm
import itertools
import random
import matplotlib.ticker as ticker
import folium
from selenium import webdriver
import time
import os
import platform
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns

""" Code for plotting """

color_palette = ["#EF8114", "#00509E", "#2E933C", "#CC2936", "#56203D"] #Orange, Blue, Green, Red, Purple 

def plot_algorithm(alg_dict, x_values, xlabel, ylabel, fix_overlapping, title, save_filename):
    """Plotting the performance of a given set of algorithms

    Args:
        alg_dict (dictonary): dictionary containing perfromance (either time or score) to be plotted
        x_values (list): list containing floats for each datapoint corresponding to the sum of the number of supply and number of demand elements
        xlabel (string): label for the x-axis
        ylabel (string): label for the y-axis
        fix_overlapping (boolean): set to True if you are worried that the lines will overlap, false if not
        title (string): title of the plot
        save_filename (string): filename for the saved plot
    """
    plt.rcParams["font.family"] = "Times new roman"
    fig, ax = plt.subplots(figsize = (7, 5))
    values = list(alg_dict.values())

    #Fixing overlapping if wanted
    if fix_overlapping:
        styles = ["dashdot", "dashed", "dotted"]
    else:
        styles = ["solid"]
    count = 0
    color_count = 0
    plotted_items = []
    for key, items in alg_dict.items():
        #Plotting each algorithm
        plt.plot(x_values, items, label = key, linestyle = styles[count], color = color_palette[color_count])
        count += 1
        color_count += 1
        if count == len(styles):
            count = 0
        if color_count == len(color_palette):
            color_count = 0
        plotted_items.append(list(items))
    plt.legend()
    plt.title(title, fontsize = 16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    ax.set_facecolor("white")
    ax.grid(visible=True, color="lightgrey", axis="y", zorder=1)
    #Removing the border lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    ax.tick_params(bottom=False, left=False)
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.savefig(r"Local_files/Plots_overleaf/" + save_filename, dpi=300, bbox_inches='tight', pad_inches=0.01)


def create_graph_specific_material(supply, demand, target_column, unit, number_of_intervals, material_string, fig_title, save_filename):
    """Create graph for specific material

    Args:
        supply (DataFrame): supply dataframe
        demand (DataFrame): demand dataframe
        target_column (string): column-name containing the requested information
        unit (string): unit of the values
        number_of_intervals (int): number of interals on the x-axis
        material_string (string): material name
        fig_title (string): title for the plot
        save_filename (string): filename of the saved figure
    """
    requested_supply = supply.loc[supply["Material"] == material_string]
    requested_demand = demand.loc[demand["Material"] == material_string]
    create_graph(requested_supply, requested_demand, target_column, unit, number_of_intervals, fig_title, save_filename)

def plot_materials(supply, demand, fig_title, save_filename):
    """Plot the material distribution

    Args:
        supply (DataFrame): supply dataframe
        demand (DataFrame): demand dataframe
        fig_title (string): title of the plot
        save_filename (string): filename of the saved figure
    """

    supply_counts = supply["Material"].value_counts().to_dict()
    demand_counts = demand["Material"].value_counts().to_dict()
    unique_keys = list(set(supply_counts.keys()) | set(demand_counts.keys()))

    for key in unique_keys:
        if key not in supply_counts:
            supply_counts[key] = 0
        if key not in demand_counts:
            demand_counts[key] = 0

    sorted_supply = {key: supply_counts[key] for key in unique_keys}
    sorted_demand = {key: demand_counts[key] for key in unique_keys}

    label = list(sorted_supply.keys())
    supply_values = sorted_supply.values()
    demand_values = sorted_demand.values()
    x = np.arange(len(label))
    plt.rcParams["font.family"] = "Times new roman"
    boxplot, ax = plt.subplots(figsize=(7, 5))
    plt.xlabel("Materials", fontsize=14)
    plt.ylabel("Number of elements", fontsize=14)
    plt.title(fig_title)
    ax.yaxis.get_major_locator().set_params(integer=True)
    width = 0.25
    ax.set_xlim([x[0]-0.60, x[-1]+0.60])
    bar1 = ax.bar(x - width / 2, supply_values, width, label="Supply", zorder=2, color = color_palette[0])
    bar2 = ax.bar(x + width / 2, demand_values, width, label="Demand", zorder=2, color = color_palette[1])
    ax.set_xticks(x)
    ax.set_xticklabels(label, fontsize=12)
    ax.legend(loc = "upper right", bbox_to_anchor=(1.10, 1.12))
    ax.set_facecolor("white")
    ax.grid(visible=True, color="lightgrey", axis="y", zorder=1)
    #for position in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[position].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    # set x-axis limits to reduce space between groups of bars
    save_name = r"./Local_files/GUI_files/Results/Plots/" + save_filename
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.01)

def create_graph(supply, demand, target_column, unit, number_of_intervals, fig_title, save_filename):
    """Create a distribution graph for a given property (given by target_column), for instance "Length", "Area" or "Moment of Inertia"

    Args:
        supply (DataFrame): supply dataframe
        demand (DataFrame): demand dataframe
        target_column (string): column_name containing the values to be plotted
        unit (string): the unit of the column_name
        number_of_intervals (int): number of intervals on the x-axis
        fig_title (string): title of the figure
        save_filename (string): filename to save the figure with
    """

    def count_leading_zeros(num):
        """Counts the number of leading zeros in a float

        Args:
            num (float): number

        Returns:
            int: number of leading zeros
        """
        count = 0
        num_str = str(num)
        if "e" in num_str:
            neg_pos = num_str.find("-")
            neg_num = num_str[neg_pos+1:]
            return float(neg_num) - 1
        decimal_pos = num_str.find(".")
        for i in range(decimal_pos+1, len(num_str)):
            if num_str[i] != "0":
                break
            else:
                count += 1
        return count

    supply_lengths = supply[target_column].to_numpy()
    demand_lengths = demand[target_column].to_numpy()
    min_length_pre = np.min([np.min(supply_lengths), np.min(demand_lengths)])
    if min_length_pre < 1:
        num_zeros = count_leading_zeros(min_length_pre)
        unit_change = int(num_zeros+1)
        unit_pure = unit.replace("[", "").replace("]", "")
        unit_change_pure = r"10$^{-"+ str(unit_change) + r"}$"
        unit = f"x {unit_change_pure} [{unit_pure}]"
        dec_format = 2
        supply_lengths = supply_lengths*10**unit_change
        demand_lengths = demand_lengths*10**unit_change
        min_length_pre = min_length_pre *10**unit_change
    else:
        dec_format = 1
    min_length = np.floor(min_length_pre * 10**dec_format)/10**dec_format
    max_length_pre = np.max([np.max(supply_lengths), np.max(demand_lengths)])
    max_length = np.ceil(max_length_pre*10**dec_format)/10**dec_format
   
    #Creating the itnervals
    interval_size = (max_length - min_length) / number_of_intervals
    supply_counts = {}
    demand_counts = {}
    start = min_length
    for i in range(number_of_intervals):
        end = start + interval_size
        supply_counts[f"{start:.{dec_format}f}-{end:.{dec_format}f}"] = 0
        demand_counts[f"{start:.{dec_format}f}-{end:.{dec_format}f}"] = 0
        start = end

    #Counting the number of supply lengts in each interval
    for length in supply_lengths:
        for interval in supply_counts:
            start, end = map(float, interval.split("-"))
            if start <= length <= end:
                supply_counts[interval] += 1
                break
    #Counting the number of demand lengts in each interval
    for length in demand_lengths:
        for interval in demand_counts:
            start, end = map(float, interval.split("-"))
            if start <= length <= end:
                demand_counts[interval] += 1
                break

    #Plotting the figure
    label = list(supply_counts.keys())
    supply_values = supply_counts.values()
    demand_values = demand_counts.values()
    x = np.arange(len(label))
    plt.rcParams["font.family"] = "Times new roman"
    boxplot, ax = plt.subplots(figsize=(7, 5))
    plt.xlabel(f"{target_column} {unit}", fontsize=14)
    plt.ylabel("Number of elements", fontsize=14)
    plt.title(fig_title)
    ax.yaxis.get_major_locator().set_params(integer=True)
    width = 0.25
    bar1 = ax.bar(x - width / 2, supply_values, width, label="Supply", zorder=2, color = color_palette[0])
    bar2 = ax.bar(x + width / 2, demand_values, width, label="Demand", zorder=2, color = color_palette[1])
    ax.set_xticks(x, label, fontsize=12)
    ax.legend(loc = "upper right", bbox_to_anchor=(1.10, 1.12))
    ax.set_facecolor("white")
    ax.grid(visible=True, color="lightgrey", axis="y", zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    save_name = r"./Local_files/GUI_files/Results/Plots/" + save_filename
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.01)

def create_map_substitutions(df, pdf_results, df_type, color, legend_text, save_name):
    """Creates a map containing the locations matched elements

    Args:
        df (DataFrame): dataframe, either supply or demand
        pdf_results (dictinary): the returned dictionary from extract_results_df_pdf()
        df_type (string): "supply" or "demand"
        color (string): color of the legend
        legend_text (string): legend text
        save_name (string): filename of the figure
    """
    #Finding the indexes of the matches
    if df_type == "supply":
        indexes = list(pdf_results["Pairs"][pdf_results["Pairs"].str.contains("S")])
    elif df_type == "demand":
        matches = list(pdf_results["Pairs"][pdf_results["Pairs"].str.contains("N")])
        indexes = list(map(lambda x: x.replace("N", "D"), matches))
    
    
    if len(indexes) == 0: #Creates an empty map if there is no matches
        create_empty_map(df, color, legend_text, save_name)
    else: #Creates map
        df = df.copy().loc[indexes]
        create_map_dataframe(df, color, legend_text, save_name)

def create_map_dataframe(df, color, legend_text, save_name):
    """Creates map

    Args:
        df (DataFrame): dataframe, either supply or demand
        color (string): color of the legend
        legend_text (string): legend text
        save_name (string): filename of the figure
    """
    df = df.copy()
    df_locations = df[["Latitude", "Longitude"]]
    site_coords = (df.iloc[0]["Site_lat"], df.iloc[0]["Site_lon"])

    #Counting the number of elements at each unique location
    coordinates_count = df_locations.groupby(['Latitude', 'Longitude']).size().reset_index(name='Count')
    coordinates_dict = dict(zip(coordinates_count[['Latitude', 'Longitude']].apply(tuple, axis=1), coordinates_count['Count']))

    #Create a map
    m = folium.Map(location=[df_locations.Latitude.mean(), df_locations.Longitude.mean()], control_scale=True)
    #Marker for the construction site
    folium.Marker([site_coords[0], site_coords[1]], icon=folium.Icon(prefix="fa", icon="fa-circle")).add_to(m)
    
    fit_view_coordinates = [site_coords]
    # Adding markers with numbers for the elements in the dataframe
    for coord, count in coordinates_dict.items():
        fit_view_coordinates.append(coord)
        marker_number = coordinates_dict[coord]
        location = [coord[0],coord[1]]
        icon_html = f'<div style="font-size: 12px; font-weight: bold; color: white; background-color: {color}; border-radius: 50%; padding: 10px 5px; height: 35px; width: 35px; text-align: center; line-height: 1.5;">{marker_number}</div>'
        folium.Marker(
        location=location,
        icon=folium.DivIcon(
            html=icon_html)
        ).add_to(m)

    #Fit the map the to coordinates present
    m.fit_bounds(fit_view_coordinates)

    # Create a custom legend with the marker colors and labels
    legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 180px; height: 45px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white;text-align:center;font-family: "Times New Roman", Times, serif;">
        <i class="fa-solid fa-circle" style="color:{color};font-size=0.5px;"></i> {legend_text}<br>
        <i class="fa-solid fa-location-dot" style="color:#38AADD;"></i> Construction site
        </div>
        '''

    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    file_dir = r"./Local_files/GUI_files/Results/Maps/"
    m.save(file_dir+f"{save_name}.html")

    #Take screenshot of the map
    if platform.system()=="Windows":
        file_dir = r"./Local_files/GUI_files/Results/Maps/"
        m.save(file_dir+f"{save_name}.html")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument("--headless")
        driver = webdriver.Chrome(chrome_options=options)
        filepath = os.getcwd() + file_dir+f"{save_name}.html"
        driver.get("file:///" + filepath)
        driver.maximize_window()
        time.sleep(3)
        driver.save_screenshot(file_dir+f"{save_name}.png")
        driver.quit
    else:
        file_dir = r"./Local_files/GUI_files/Results/Maps/"
        m.save(file_dir+f"{save_name}.html")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument("--headless")
        driver = webdriver.Chrome(chrome_options=options)
        filepath = os.getcwd() + file_dir[1:]+f"{save_name}.html"
        driver.get("file:///" + filepath)
        driver.maximize_window()
        time.sleep(3)
        driver.save_screenshot(file_dir+f"{save_name}.png")
        driver.quit

def create_empty_map(df, color, legend_text, save_name):
    """Creates an empty map with only a marker for the construction site

    Args:
        df (DataFrame): dataframe, either supply or demand
        color (string): color of the legend
        legend_text (string): legend text
        save_name (string): filename of the figure
    """
    df = df.copy()
    site_coords = (df.iloc[0]["Site_lat"], df.iloc[0]["Site_lon"])
    #Create a map
    m = folium.Map(location=[site_coords[0], site_coords[1]], control_scale=True)
    folium.Marker([site_coords[0], site_coords[1]], icon=folium.Icon(prefix="fa", icon="fa-circle")).add_to(m)
    
    # Create a custom legend with the marker colors and labels
    legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 180px; height: 50px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white;text-align:center;font-family: "Times New Roman", Times, serif;">
        <i class="fa-solid fa-circle" style="color:{color};font-size=0.5px;"></i> {legend_text}<br>
        <i class="fa-solid fa-location-dot" style="color:#38AADD;"></i> Construction site
        </div>
        '''

    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    file_dir = r"./Local_files/GUI_files/Results/Maps/"
    m.save(file_dir+f"{save_name}.html")

    #Take screenshot of the map
    if platform.system()=="Windows":
        file_dir = r"./Local_files/GUI_files/Results/Maps/"
        m.save(file_dir+f"{save_name}.html")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument("--headless")
        driver = webdriver.Chrome(chrome_options=options)
        filepath = os.getcwd() + file_dir+f"{save_name}.html"
        driver.get("file:///" + filepath)
        driver.maximize_window()
        time.sleep(3)
        driver.save_screenshot(file_dir+f"{save_name}.png")
        driver.quit
    else:
        file_dir = r"./Local_files/GUI_files/Results/Maps/"
        m.save(file_dir+f"{save_name}.html")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument("--headless")
        driver = webdriver.Chrome(chrome_options=options)
        filepath = os.getcwd() + file_dir[1:]+f"{save_name}.html"
        driver.get("file:///" + filepath)
        driver.maximize_window()
        time.sleep(3)
        driver.save_screenshot(file_dir+f"{save_name}.png")
        driver.quit

def create_map_supply_locations(supply_cords_df, site_lat, site_lon, include_site, save_name):
    """Creates a map of the unique supply locations

    Args:
        supply_cords_df (DataFrame): supply coordinates dataframe
        site_lat (float): latitude of the construction site
        site_lon (float): longitude of the construction site
        include_site (boolean): if a marker of the construction site should be included or not
        save_name (string): filename to save
    """
    df = supply_cords_df.copy()
    #Create a map
    m = folium.Map(location=[site_lat, site_lon], control_scale=True) 

    if include_site:
        folium.Marker([site_lat, site_lon], icon=folium.Icon(prefix="fa", icon="fa-circle")).add_to(m) #Marker for site location
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 180px; height: 45px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white;text-align:center;font-family: "Times New Roman", Times, serif;">
        <i class="fa-solid fa-location-dot" style="color:#6BA524;"></i> Reusable elements<br>
        <i class="fa-solid fa-location-dot" style="color:#38AADD;"></i> Construction site  
        </div>
        '''
    else:
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 180px; height: 25px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white;text-align:center;font-family: "Times New Roman", Times, serif;">
        <i class="fa-solid fa-location-dot" style="color:#6BA524;"></i> Reusable elements
        </div>
        '''


    fit_view_coordinates = [(site_lat, site_lon)]
    for index, row in df.iterrows():
        coord = (row["Latitude"], row["Longitude"])
        fit_view_coordinates.append(coord)
        location = [coord[0],coord[1]]
        folium.Marker([coord[0], coord[1]], icon=folium.Icon(prefix="fa", icon="fa-circle", color="green")).add_to(m) #Marker for reused locations

    m.fit_bounds(fit_view_coordinates)
    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    file_dir = r"./Local_files/GUI_files/Results/Maps/"
    m.save(file_dir+f"{save_name}.html")

    #Take screenshot of the maps
    if platform.system()=="Windows":
        file_dir = r"./Local_files/GUI_files/Results/Maps/"
        m.save(file_dir+f"{save_name}.html")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument("--headless")
        driver = webdriver.Chrome(chrome_options=options)
        filepath = os.getcwd() + file_dir+f"{save_name}.html"
        driver.get("file:///" + filepath)
        driver.maximize_window()
        time.sleep(3)
        driver.save_screenshot(file_dir+f"{save_name}.png")
        driver.quit
    else:
        file_dir = r"./Local_files/GUI_files/Results/Maps/"
        m.save(file_dir+f"{save_name}.html")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument("--headless")
        driver = webdriver.Chrome(chrome_options=options)
        filepath = os.getcwd() + file_dir[1:]+f"{save_name}.html"
        driver.get("file:///" + filepath)
        driver.maximize_window()
        time.sleep(3)
        driver.save_screenshot(file_dir+f"{save_name}.png")
        driver.quit

def create_map_manufacturer_location(timber_lat, timber_lon, steel_lat, steel_lon, site_lat, site_lon, save_name):
    """Crate map for the locations of the manufacturers

    Args:
        timber_lat (float): latitude of timber manufacturer
        timber_lon (float): longitude of timber manufacturer
        steel_lat (float): latitude of steel manufacturer
        steel_lon (float): longitude of steel manufacturer
        site_lat (float): latitude of construction site
        site_lon (float): longitude of construction site
        save_name (_type_): filename to save the figure
    """
    #Create a map
    m = folium.Map(location=[site_lat, site_lon], control_scale=True)

    #Marker for site location
    folium.Marker([site_lat, site_lon], icon=folium.Icon(prefix="fa", icon="fa-circle")).add_to(m) 

    #Custom legend
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 180px; height: 45px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white;text-align:center;font-family: "Times New Roman", Times, serif;">
    <i class="fa-solid fa-location-dot" style="color:#D53E2A;"></i> Manufacturers<br>
    <i class="fa-solid fa-location-dot" style="color:#38AADD;"></i> Construction site  
    </div>
    '''

    folium.Marker([steel_lat, steel_lon], icon=folium.Icon(prefix="fa", icon="fa-circle", color="red")).add_to(m) #Marker for steel manufacturer
    folium.Marker([timber_lat, timber_lon], icon=folium.Icon(prefix="fa", icon="fa-circle", color="red")).add_to(m) #Marker for timber manufacturer

    fit_view_coordinates = [(site_lat, site_lon), (timber_lat, timber_lon), (steel_lat, steel_lon)]
    m.fit_bounds(fit_view_coordinates)
    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    file_dir = r"./Local_files/GUI_files/Results/Maps/"
    m.save(file_dir+f"{save_name}.html")
    #Take screenshot of map
    if platform.system()=="Windows":
        file_dir = r"./Local_files/GUI_files/Results/Maps/"
        m.save(file_dir+f"{save_name}.html")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument("--headless")
        driver = webdriver.Chrome(chrome_options=options)
        filepath = os.getcwd() + file_dir+f"{save_name}.html"
        driver.get("file:///" + filepath)
        driver.maximize_window()
        time.sleep(3)
        driver.save_screenshot(file_dir+f"{save_name}.png")
        driver.quit
    else:
        file_dir = r"./Local_files/GUI_files/Results/Maps/"
        m.save(file_dir+f"{save_name}.html")
        options = webdriver.ChromeOptions()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches",["enable-automation"])
        options.add_argument("--headless")
        driver = webdriver.Chrome(chrome_options=options)
        filepath = os.getcwd() + file_dir[1:]+f"{save_name}.html"
        driver.get("file:///" + filepath)
        driver.maximize_window()
        time.sleep(3)
        driver.save_screenshot(file_dir+f"{save_name}.png")
        driver.quit

def plot_substitutions_matrix(save_name):
    """Plot the similarity matrix of the substitutions from different case studies

    Args:
        save_name (string): filename to save the figure
    """
    def calculate_fraction(column1, column2):
        """Calculate the similarity between the values in two columns

        Args:
            column1 (Pandas Series): values of column1
            column2 (Pandas Series): values of column2

        Returns:
            float: the fraction of similarities
        """

        equal_count = sum(column1 == column2)
        total_count = len(column1)
        return (equal_count / total_count)
    
    plt.figure(figsize=(7, 5))
    #The Excel-files containing the substitutions for each case study is manually inserted
    subs1 = hm.import_dataframe_from_file(file_location=r"./Local_files/GUI_files/Results/Case_Study_1_substitutions.xlsx", index_replacer="D").rename(columns={"Substitutions": "Case Study 1"})
    subs2 = hm.import_dataframe_from_file(file_location=r"./Local_files/GUI_files/Results/Case_Study_2_substitutions.xlsx", index_replacer="D").rename(columns={"Substitutions": "Case Study 2"})
    subs3 = hm.import_dataframe_from_file(file_location=r"./Local_files/GUI_files/Results/Case_Study_3_substitutions.xlsx", index_replacer="D").rename(columns={"Substitutions": "Case Study 3"})
    subs4 = hm.import_dataframe_from_file(file_location=r"./Local_files/GUI_files/Results/Case_Study_4_substitutions.xlsx", index_replacer="D").rename(columns={"Substitutions": "Case Study 4"})
    all_subs = pd.concat([subs1, subs2, subs3, subs4], axis = 1)
    columns = all_subs.columns

    #Create matrix
    M = pd.DataFrame(index=columns, columns=columns, dtype=float)

    #Calculate the fractions of similarities between the columns
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            col1 = all_subs[columns[i]]
            col2 = all_subs[columns[j]]
            percentage = calculate_fraction(col1, col2)
            M.at[columns[i], columns[j]] = percentage
            M.at[columns[j], columns[i]] = percentage
    
    #Create plot 
    sns.set(font='Times New Roman')
    ax = sns.heatmap(M, annot=True, cmap='YlGnBu')
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.yticks(rotation=0)
    file_path = r"./Local_files/Plots_overleaf/" + save_name
    plt.savefig(file_path, dpi = 300, bbox_inches='tight', pad_inches=0.01)