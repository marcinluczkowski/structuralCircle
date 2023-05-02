import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import logging
import LCA as lca
import itertools
import random
from fpdf import FPDF
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import folium
from selenium import webdriver
import time
import os

def create_graph(supply, demand, target_column, number_of_intervals, save_filename):
    supply_lengths = supply[target_column].to_numpy()
    demand_lengths = demand[target_column].to_numpy()
    max_length = np.ceil(np.max([np.max(supply_lengths), np.max(demand_lengths)]))
    min_length = np.floor(np.min([np.min(supply_lengths), np.min(demand_lengths)]))
    interval_size = (max_length - min_length) / number_of_intervals
    supply_counts = {}
    demand_counts = {}
    start = min_length
    for i in range(number_of_intervals):
        end = start + interval_size
        #intervals.append("{:.1f}-{:.1f}".format(start, end))
        supply_counts["{:.1f}-{:.1f}".format(start, end)] = 0
        demand_counts["{:.1f}-{:.1f}".format(start, end)] = 0
        start = end

    for length in supply_lengths:
        for interval in supply_counts:
            start, end = map(float, interval.split("-"))
            if start <= length < end:
                supply_counts[interval] += 1
                break
    for length in supply_lengths:
        for interval in demand_counts:
            start, end = map(float, interval.split("-"))
            if start <= length < end:
                demand_counts[interval] += 1
                break

    
    boxplot,ax=plt.subplots(figsize = (7, 5))
    label = list(supply_counts.keys())
    supply_values = supply_counts.values()
    demand_values = demand_counts.values()
    x=np.arange(len(label))
    width=0.25
    plt.rcParams["font.family"] = "Sans Serif"
    plt.grid(visible = True, color = "lightgrey", axis = "y")
    plt.xlabel("Lengths [m]", fontsize = 14)
    plt.ylabel("Number of elements", fontsize = 14)
    bar1=ax.bar(x-width,supply_values,width,label="Reuse")
    bar2=ax.bar(x,demand_values,width,label="Demand")
    ax.set_facecolor("white")
    ax.set_xticks(x,label, fontsize = 12)
    ax.legend()
    plt.plot()
    plt.savefig(save_filename, dpi = 300)

def create_map_substitutions(df, pdf_results, df_type, color, legend_text, save_name):
    if df_type == "supply":
        indexes = list(pdf_results["Pairs"][pdf_results["Pairs"].str.contains("S")])
    elif df_type == "demand":
        matches = list(pdf_results["Pairs"][pdf_results["Pairs"].str.contains("N")])
        indexes = list(map(lambda x: x.replace("N", "D"), matches))
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
    m.save(r""+f"./Results/{save_name}.html")
    options = webdriver.ChromeOptions()
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches",["enable-automation"])
    options.add_argument("--headless")
    driver = webdriver.Chrome(chrome_options=options)
    #driver.get(r"./Results/map.html")
    filepath = os.getcwd() + r"" + f"/Results/{save_name}.html"
    driver.get("file:///" + filepath)
    driver.maximize_window()
    time.sleep(5)
    driver.save_screenshot(r""+ f"./Results/{save_name}.png")
    driver.quit()