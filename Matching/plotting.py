import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import logging
import LCA as lca
import itertools
import random
import matplotlib.ticker as ticker
import folium
from selenium import webdriver
import time
import os

color_palette = ["#EF8114", "#00509E", "#2E933C", "#CC2936", "#56203D"] #Orange, Blue, Green, Red, Purple 

def plot_algorithm(alg_dict, x_values, xlabel, ylabel, fix_overlapping, title, save_filename):
    plt.rcParams["font.family"] = "Times new roman"
    fig, ax = plt.subplots(figsize = (7, 5))
    values = list(alg_dict.values())
    #Check if the plots are the same:
    min_value = np.min(values)
    if fix_overlapping:
        styles = ["dashdot", "dashed", "dotted"]
    else:
        styles = ["solid"]
    count = 0
    color_count = 0
    plotted_items = []
    for key, items in alg_dict.items():
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.xaxis.get_major_locator().set_params(integer=True)
    #plt.yscale('log')
    plt.savefig(r"Local_files/Plots_overleaf/" + save_filename, dpi=300)


def create_graph_specific_material(supply, demand, target_column, unit, number_of_intervals, material_string, fig_title, save_filename):
    requested_supply = supply.loc[supply["Material"] == material_string]
    requested_demand = demand.loc[demand["Material"] == material_string]
    create_graph(requested_supply, requested_demand, target_column, unit, number_of_intervals, fig_title, save_filename)

def plot_materials(supply, demand, fig_title, save_filename):
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
    width = 0.15
    bar1 = ax.bar(x - width / 2, supply_values, width, label="Reuse", zorder=2, color = color_palette[0])
    bar2 = ax.bar(x + width / 2, demand_values, width, label="Demand", zorder=2, color = color_palette[1])
    ax.set_xticks(x, label, fontsize=12)
    ax.legend()
    ax.set_facecolor("white")
    ax.grid(visible=True, color="lightgrey", axis="y", zorder=1)
    #for position in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[position].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    save_name = r"./Local_files/GUI_files/Results/Plots/" + save_filename
    plt.savefig(save_name, dpi=100)



def create_graph(supply, demand, target_column, unit, number_of_intervals, fig_title, save_filename):
    def count_leading_zeros(num):
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
   
    
    interval_size = (max_length - min_length) / number_of_intervals
    #dec_format_max = len(str(max_length_pre).split('.')[1])
    #dec_format_min = len(str(min_length_pre).split('.')[1])
    #dec_format = shifter
    supply_counts = {}
    demand_counts = {}
    start = min_length
    for i in range(number_of_intervals):
        end = start + interval_size
        #intervals.append("{:.1f}-{:.1f}".format(start, end))
        #supply_counts["{:.1f}-{:.1f}".format(start, end)] = 0
        #intervals.append(f"{start}:.{dec_format}f-{end}:.{dec_format}f")
        supply_counts[f"{start:.{dec_format}f}-{end:.{dec_format}f}"] = 0
        demand_counts[f"{start:.{dec_format}f}-{end:.{dec_format}f}"] = 0
        #demand_counts["{:.1f}-{:.1f}".format(start, end)] = 0
        start = end

    for length in supply_lengths:
        for interval in supply_counts:
            start, end = map(float, interval.split("-"))
            if start <= length <= end:
                supply_counts[interval] += 1
                break
    for length in demand_lengths:
        for interval in demand_counts:
            start, end = map(float, interval.split("-"))
            if start <= length <= end:
                demand_counts[interval] += 1
                break

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
    bar1 = ax.bar(x - width / 2, supply_values, width, label="Reuse", zorder=2, color = color_palette[0])
    bar2 = ax.bar(x + width / 2, demand_values, width, label="Demand", zorder=2, color = color_palette[1])
    ax.set_xticks(x, label, fontsize=12)
    ax.legend()
    ax.set_facecolor("white")
    ax.grid(visible=True, color="lightgrey", axis="y", zorder=1)
    #for position in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[position].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    save_name = r"./Local_files/GUI_files/Results/Plots/" + save_filename
    plt.savefig(save_name, dpi=100)



def create_map_substitutions(df, pdf_results, df_type, color, legend_text, save_name):
    if df_type == "supply":
        indexes = list(pdf_results["Pairs"][pdf_results["Pairs"].str.contains("S")])
    elif df_type == "demand":
        matches = list(pdf_results["Pairs"][pdf_results["Pairs"].str.contains("N")])
        indexes = list(map(lambda x: x.replace("N", "D"), matches))
    
    if len(indexes) == 0:
        create_empty_map(df, color, legend_text, save_name)
    else:
        df = df.copy().loc[indexes]
        create_map_dataframe(df, color, legend_text, save_name)

def create_map_dataframe(df, color, legend_text, save_name):
    df = df.copy()
    df_locations = df[["Latitude", "Longitude"]]
    cite_coords = (df.iloc[0]["Cite_lat"], df.iloc[0]["Cite_lon"])
    coordinates_count = df_locations.groupby(['Latitude', 'Longitude']).size().reset_index(name='Count')
    coordinates_dict = dict(zip(coordinates_count[['Latitude', 'Longitude']].apply(tuple, axis=1), coordinates_count['Count']))
    m = folium.Map(location=[df_locations.Latitude.mean(), df_locations.Longitude.mean()], control_scale=True)
    folium.Marker([cite_coords[0], cite_coords[1]], icon=folium.Icon(prefix="fa", icon="fa-circle")).add_to(m)
    # Create a custom legend with the marker colors and labels
    fit_view_coordinates = [cite_coords]
    for coord, count in coordinates_dict.items():
        fit_view_coordinates.append(coord)
        marker_number = coordinates_dict[coord]
        location = [coord[0],coord[1]]
        icon_html = f'<div style="font-size: 12px; font-weight: bold; color: white; background-color: {color}; border-radius: 50%; padding: 5px 5px; height: 25px; width: 25px; text-align: center; line-height: 1.5;">{marker_number}</div>'
        folium.Marker(
        location=location,
        icon=folium.DivIcon(
            html=icon_html)
        ).add_to(m)

    m.fit_bounds(fit_view_coordinates)

    legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 180px; height: 50px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white;text-align:center;font-family: "Times New Roman", Times, serif;">
        <i class="fa-solid fa-circle" style="color:{color};font-size=0.5px;"></i> {legend_text}<br>
        <i class="fa-solid fa-location-dot" style="color:#38AADD;"></i> Cite location  
        </div>
        '''

    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    #img = map._to_png(5)
    #mg.save(r"./Results/map.png")
    # Display the map
    #map.show_in_browser()
    file_dir = r"./Local_files/GUI_files/Results/Maps/"
    m.save(file_dir+f"{save_name}.html")
    options = webdriver.ChromeOptions()
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches",["enable-automation"])
    options.add_argument("--headless")
    driver = webdriver.Chrome(chrome_options=options)
    #driver.get(r"./Results/map.html")
    filepath = os.getcwd() + file_dir+f"{save_name}.html"
    driver.get("file:///" + filepath)
    driver.maximize_window()
    time.sleep(3)
    driver.save_screenshot(file_dir+f"{save_name}.png")
    driver.quit()

def create_empty_map(df, color, legend_text, save_name):
    df = df.copy()
    cite_coords = (df.iloc[0]["Cite_lat"], df.iloc[0]["Cite_lon"])
    m = folium.Map(location=[cite_coords[0], cite_coords[1]], control_scale=True)
    folium.Marker([cite_coords[0], cite_coords[1]], icon=folium.Icon(prefix="fa", icon="fa-circle")).add_to(m)
    # Create a custom legend with the marker colors and labels

    legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 180px; height: 50px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white;text-align:center;font-family: "Times New Roman", Times, serif;">
        <i class="fa-solid fa-circle" style="color:{color};font-size=0.5px;"></i> {legend_text}<br>
        <i class="fa-solid fa-location-dot" style="color:#38AADD;"></i> Cite location  
        </div>
        '''

    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    file_dir = r"./Local_files/GUI_files/Results/Maps/"
    m.save(file_dir+f"{save_name}.html")
    options = webdriver.ChromeOptions()
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches",["enable-automation"])
    options.add_argument("--headless")
    driver = webdriver.Chrome(chrome_options=options)
    #driver.get(r"./Results/map.html")
    filepath = os.getcwd() + file_dir+f"{save_name}.html"
    driver.get("file:///" + filepath)
    driver.maximize_window()
    time.sleep(3)
    driver.save_screenshot(file_dir+f"{save_name}.png")
    driver.quit()