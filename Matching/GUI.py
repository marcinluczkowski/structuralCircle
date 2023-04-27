import sys
import random as rd
import pandas as pd
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import webbrowser
import time
import subprocess
import platform
import os

constants = {
    "TIMBER_GWP": 28.9,       # based on NEPD-3442-2053-EN
    "TIMBER_REUSE_GWP": 2.25,        # 0.0778*28.9 = 2.25 based on Eberhardt
    "TRANSPORT_GWP": 96.0,    # TODO kg/m3/t based on ????
    "TIMBER_DENSITY": 491.0,  # kg, based on NEPD-3442-2053-EN
    "STEEL_GWP": 800, #Random value
    "STEEL_REUSE_GWP": 4, #Random value
    "VALUATION_GWP": 0.6, #In kr:Per kg CO2, based on OECD
    "TIMBER_PRICE": 435, #Per m^3 https://www.landkredittbank.no/blogg/2021/prisen-pa-sagtommer-okte-20-prosent/
    "TIMBER_REUSE_PRICE" : 100, #Per m^3, Random value
    "STEEL_PRICE": 500, #Per m^2, R'andom value'
    "STEEL_REUSE_PRICE": 200, #Per m^2, Random value
    "PRICE_TRANSPORTATION": 3.78, #Price per km per tonn. Derived from 2011 numbers on scaled t0 2022 using SSB
    "STEEL_DENSITY": 7850,
    ########################
    "Project name": "Bod nidarosdomen",
    "Metric": "GWP",
    "Algorithms": ["bipartite", "greedy_plural", "greedy_single", "bipartite_plural"],
    "Include transportation": True,
    "Cite latitude": "63.4269",
    "Cite longitude": "10.3969",
    "Demand file location": r"./CSV/pdf_demand.csv",
    "Supply file location": r"./CSV/pdf_supply.csv",
    "constraint_dict": {'Area' : '>=', 'Moment of Inertia' : '>=', 'Length' : '>=', 'Material': '=='}
}

def checkalgos():
    One_or_more_choosen=False
    possibleAlgos=[greedy_var,greedy_plural_var,bipartite_var,MILP_var,genetic_var,brute_var,bipartite_plural_var,bipartite_multi_plural]

    for algo in possibleAlgos:
        if algo.get():
            One_or_more_choosen=True
    return One_or_more_choosen

def giveFileName():
    projectname=Projectname_entry.get()
    projectname=projectname.replace(" ","_")
    filename=projectname+"_report.pdf"
    filename_tk.set(filename)
    return projectname


def checkinputfields():
    One_or_more_missing=False
    InputfieldsCombined_Inc_Transport=[Projectname_entry,ProjectLatitude_entry,ProjectLongitude_entry,Timber_reused_gwp_entry,Timber_new_gwp_entry,Timber_price_entry,Timber_reused_price_entry,Steel_reused_gwp_entry,Steel_new_gwp_entry,Steel_price_entry,Steel_reused_price_entry,GWP_valuation_entry,Transport_GWP_entry,Transport_price_entry]
    InputfieldsCombined_No_Transport=[Projectname_entry,ProjectLatitude_entry,ProjectLongitude_entry,Timber_reused_gwp_entry,Timber_new_gwp_entry,Timber_price_entry,Timber_reused_price_entry,Steel_reused_gwp_entry,Steel_new_gwp_entry,Steel_price_entry,Steel_reused_price_entry,GWP_valuation_entry]
    InputfieldsGWP_Inc_Transport=[Projectname_entry,ProjectLatitude_entry,ProjectLongitude_entry,Timber_reused_gwp_entry,Timber_new_gwp_entry,Steel_reused_gwp_entry,Steel_new_gwp_entry,Transport_GWP_entry,Transport_price_entry]
    InputfieldsGWP_NO_Transport=[Projectname_entry,ProjectLatitude_entry,ProjectLongitude_entry,Timber_reused_gwp_entry,Timber_new_gwp_entry,Steel_reused_gwp_entry,Steel_new_gwp_entry]
    InputfieldsPrice_Inc_Transport=[Projectname_entry,ProjectLatitude_entry,ProjectLongitude_entry,Timber_price_entry,Timber_reused_price_entry,Steel_price_entry,Steel_reused_price_entry,Transport_GWP_entry,Transport_price_entry]
    InputfieldsPrice_NO_Transport=[Projectname_entry,ProjectLatitude_entry,ProjectLongitude_entry,Timber_price_entry,Timber_reused_price_entry,Steel_price_entry,Steel_reused_price_entry]
    inputfields=[Projectname_entry,ProjectLatitude_entry,ProjectLongitude_entry]
    
    if matching_metric_var.get() == "Combined" and include_transportation_var.get() == 1:
        inputfields=InputfieldsCombined_Inc_Transport
    elif matching_metric_var.get() == "Combined" and not include_transportation_var.get() == 1:
        inputfields=InputfieldsCombined_No_Transport
    
    elif matching_metric_var.get() == "GWP" and include_transportation_var.get() == 1:
        inputfields=InputfieldsGWP_Inc_Transport
        
    elif matching_metric_var.get() == "GWP" and not include_transportation_var.get() == 1:
        inputfields=InputfieldsGWP_NO_Transport

    elif matching_metric_var.get() == "Price" and include_transportation_var.get() == 1:        
        inputfields=InputfieldsPrice_Inc_Transport
    elif matching_metric_var.get() == "Price" and not include_transportation_var.get() == 1:        
        inputfields=InputfieldsPrice_NO_Transport

    for inputfield in inputfields:
        if inputfield.get()=="":
            One_or_more_missing=True
    return One_or_more_missing

def calculate():
    one_or_more_choosen=checkalgos()
    one_or_more_missing=checkinputfields()
    if matching_metric_var.get() == "Sverre":
        OpenUrl()
    
    elif not demand_filepath_bool.get() or not supply_filepath_bool.get():
        messagebox.showerror("Invalid input", "Please select input files")

    elif matching_metric_var.get() not in ["Price","GWP","Combined"]:
        messagebox.showerror("Invalid input", "Please select a valid metric. \"Combined\", \"GWP\" or \"Price\". ")

    elif one_or_more_missing:
        messagebox.showerror("Invalid input", "One or more input field is empty")

    elif not one_or_more_choosen:
        messagebox.showerror("Invalid input", "Please select at least one algorithm")
    
    else:        
        result_label.config(text="Running program...", foreground="green")
        root.update()
        #messagebox.showinfo("Status", "Running program...")
        #messagebox.showerror("Invalid input", "Running program....")
        updateconstants()
        score_function_string = hm.generate_score_function_string(constants)

        supply = hm.import_dataframe_from_file(r"" + constants["Supply file location"], index_replacer = "S")
        demand = hm.import_dataframe_from_file(r"" + constants["Demand file location"], index_replacer = "D")
        constraint_dict = constants["constraint_dict"]
        #Add necessary columns to run the algorithm
        supply = hm.add_necessary_columns_pdf(supply, constants)
        demand = hm.add_necessary_columns_pdf(demand, constants)
        run_string = hm.generate_run_string(constants)
        
        result = eval(run_string)
        simple_pairs = hm.extract_pairs_df(result)
        pdf_results = hm.extract_results_df_pdf(result, constants)
        
        projectname=giveFileName()
        pdf = hm.generate_pdf_report(pdf_results,projectname, filepath = r"./Local Files/GUI Files/Results/")
        result_label.config(text="Report generated", foreground="green")
        result_label.after(10000, clear_error_message)
        open_report_button.place(relx=0.5,rely=0.95,anchor="center")


def updateconstants():
    constants["TIMBER_GWP"]=float(Timber_new_gwp_entry.get())
    constants["TIMBER_REUSE_GWP"]=float(Timber_reused_gwp_entry.get())
    constants["TRANSPORT_GWP"]=float(Transport_GWP_entry.get())
    constants["STEEL_GWP"]=float(Steel_new_gwp_entry.get())
    constants["STEEL_REUSE_GWP"]=float(Steel_reused_gwp_entry.get())
    constants["VALUATION_GWP"]=float(GWP_valuation_entry.get())
    constants["TIMBER_PRICE"]=float(Timber_price_entry.get())
    constants["TIMBER_REUSE_PRICE"]=float(Timber_reused_price_entry.get())
    constants["STEEL_PRICE"]=float(Steel_price_entry.get())
    constants["STEEL_REUSE_PRICE"]=float(Steel_reused_price_entry.get())
    constants["PRICE_TRANSPORTATION"]=float(Transport_price_entry.get())
    constants["Project name"]=Projectname_entry.get()
    constants["Metric"]=matching_metric_var_constant.get()
    constants["Algorithms"]=get_list_algos()
    constants["Include transportation"]=getIncludeTranportYesNo()
    constants["Cite latitude"]=ProjectLatitude_entry.get()
    constants["Cite longitude"]=ProjectLongitude_entry.get()
    constants["Demand file location"]=r""+demand_filepath_string.get()
    constants["Supply file location"]=r""+supply_filepath_string.get()

def getIncludeTranportYesNo():
    if include_transportation_var.get() == 1:
        return True
    else:
        return False

def get_list_algos():
    Algorithms=[]
    if greedy_var.get()==1:
        Algorithms.append("greedy_single")
    if greedy_plural_var.get()==1:
        Algorithms.append("greedy_plural")
    if bipartite_var.get()==1:
        Algorithms.append("bipartite")
    if MILP_var.get()==1:
        Algorithms.append("milp")
    if genetic_var.get()==1:
        Algorithms.append("genetic")
    if brute_var.get()==1:
        Algorithms.append("brute")
    if bipartite_plural_var.get()==1:
        Algorithms.append("bipartite_plural")
    if bipartite_multi_plural.get()==1:
        Algorithms.append("bipartite_plural_multiple")
    return Algorithms


def browse_supply_file():
    global supply_filepath
    supply_filepath = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx"),("CSV Files", "*.csv"), ("all", "*")],
        title="Select a .xlsx or .csv file"
    )
    if supply_filepath:
        supply_filename = supply_filepath.split("/")[-1]
        supply_file_label.config(text=supply_filename,foreground="green")
        #TODO: Support both excel and csv file
        supply_df=pd.read_excel(supply_filepath)
        num_supply=int(len(supply_df.index))
        num_supply_elements.set(num_supply)
        num_supply_label.config(text=f"Number of supply elements: {num_supply}")
        supply_filepath_bool.set(True)
        print(supply_filepath)
        supply_filepath_string.set(supply_filepath)
        return supply_filepath

def browse_demand_file():
    global demand_filepath
    demand_filepath = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx"),("CSV Files", "*.csv")],
        title="Select a .xlsx or .csv file"
    )

    if demand_filepath:
        demand_filename = demand_filepath.split("/")[-1]
        demand_file_label.config(text=demand_filename,foreground="green")
        demand_df=pd.read_excel(demand_filepath)
        num_demand=int(len(demand_df.index))
        num_demand_elements.set(num_demand)
        num_demand_label.config(text=f"Number of demand elements: {num_demand}")
        demand_filepath_bool.set(True)
        demand_filepath_string.set(demand_filepath)
        return demand_filepath
    
def on_matching_metric_change(*args):
    matching_metric_var_constant.set(matching_metric_var.get())
    if matching_metric_var.get() == "Combined" and not include_transportation_var.get() == 1:
        #GWP
        Timber_reused_gwp_label.place(relx=0.03,rely=0.32,anchor="w")
        Timber_reused_gwp_entry.place(relx=0.23,rely=0.32,anchor="w")
        Timber_new_gwp_label.place(relx=0.28,rely=0.32,anchor="w")
        Timber_new_gwp_entry.place(relx=0.47,rely=0.32,anchor="w")
        
        Steel_reused_gwp_label.place(relx=0.03,rely=0.37,anchor="w")
        Steel_reused_gwp_entry.place(relx=0.23,rely=0.37,anchor="w")
        Steel_new_gwp_label.place(relx=0.28,rely=0.37,anchor="w")
        Steel_new_gwp_entry.place(relx=0.47,rely=0.37,anchor="w")
        
        #PRICE
        Timber_price_label.place(relx=0.52,rely=0.32,anchor="w")
        Timber_price_entry.place(relx=0.67,rely=0.32,anchor="w")
        Steel_price_label.place(relx=0.52,rely=0.37,anchor="w")
        Steel_price_entry.place(relx=0.67,rely=0.37,anchor="w")
        Timber_reused_price_label.place(relx=0.72,rely=0.32,anchor="w")
        Timber_reused_price_entry.place(relx=0.89,rely=0.32,anchor="w")
        Steel_reused_price_label.place(relx=0.72,rely=0.37,anchor="w")
        Steel_reused_price_entry.place(relx=0.89,rely=0.37,anchor="w")

        GWP_valuation_label.place(relx=0.495,rely=0.42,anchor="e")
        GWP_valuation_entry.place(relx=0.505,rely=0.42,anchor="w")
        include_transportation_checkbutton.place(relx=0.5,rely=0.48,anchor="center")

        Choose_algorithms_label.place(relx=0.5,rely=0.6,anchor="center")
        greedy_checkbutton.place(relx=0.15,rely=0.65,anchor="w")
        greedy_plural_checkbutton.place(relx=0.35,rely=0.65,anchor="w")
        bipartite_checkbutton.place(relx=0.55,rely=0.65,anchor="w")
        bipartite_plural_checkbutton.place(relx=0.75,rely=0.65,anchor="w")
        bipartite_multi_plural_checkbutton.place(relx=0.15,rely=0.7,anchor="w")
        MILP_checkbutton.place(relx=0.35,rely=0.7,anchor="w")
        genetic_checkbutton.place(relx=0.55,rely=0.7,anchor="w")
        brute_checkbutton.place(relx=0.75,rely=0.7,anchor="w")
    
    elif matching_metric_var.get() == "Combined" and include_transportation_var.get() == 1:
        Timber_reused_gwp_label.place(relx=0.03,rely=0.32,anchor="w")
        Timber_reused_gwp_entry.place(relx=0.23,rely=0.32,anchor="w")
        Timber_new_gwp_label.place(relx=0.28,rely=0.32,anchor="w")
        Timber_new_gwp_entry.place(relx=0.47,rely=0.32,anchor="w")
        
        Steel_reused_gwp_label.place(relx=0.03,rely=0.37,anchor="w")
        Steel_reused_gwp_entry.place(relx=0.23,rely=0.37,anchor="w")
        Steel_new_gwp_label.place(relx=0.28,rely=0.37,anchor="w")
        Steel_new_gwp_entry.place(relx=0.47,rely=0.37,anchor="w")
        
        #PRICE
        Timber_price_label.place(relx=0.52,rely=0.32,anchor="w")
        Timber_price_entry.place(relx=0.67,rely=0.32,anchor="w")
        Steel_price_label.place(relx=0.52,rely=0.37,anchor="w")
        Steel_price_entry.place(relx=0.67,rely=0.37,anchor="w")
        Timber_reused_price_label.place(relx=0.72,rely=0.32,anchor="w")
        Timber_reused_price_entry.place(relx=0.89,rely=0.32,anchor="w")
        Steel_reused_price_label.place(relx=0.72,rely=0.37,anchor="w")
        Steel_reused_price_entry.place(relx=0.89,rely=0.37,anchor="w")

        GWP_valuation_label.place(relx=0.495,rely=0.42,anchor="e")
        GWP_valuation_entry.place(relx=0.505,rely=0.42,anchor="w")

        include_transportation_checkbutton.place(relx=0.5,rely=0.48,anchor="center")
        Transport_GWP_label.place(relx=0.30,rely=0.53,anchor="w")
        Transport_GWP_entry.place(relx=0.44,rely=0.53,anchor="w")
        Transport_price_label.place(relx=0.50,rely=0.53,anchor="w")
        Transport_price_entry.place(relx=0.71,rely=0.53,anchor="w")

        Choose_algorithms_label.place(relx=0.5,rely=0.6,anchor="center")
        greedy_checkbutton.place(relx=0.15,rely=0.65,anchor="w")
        greedy_plural_checkbutton.place(relx=0.35,rely=0.65,anchor="w")
        bipartite_checkbutton.place(relx=0.55,rely=0.65,anchor="w")
        bipartite_plural_checkbutton.place(relx=0.75,rely=0.65,anchor="w")
        bipartite_multi_plural_checkbutton.place(relx=0.15,rely=0.7,anchor="w")
        MILP_checkbutton.place(relx=0.35,rely=0.7,anchor="w")
        genetic_checkbutton.place(relx=0.55,rely=0.7,anchor="w")
        brute_checkbutton.place(relx=0.75,rely=0.7,anchor="w")

    elif matching_metric_var.get() == "GWP" and not include_transportation_var.get() == 1:
        #GWP
        Timber_reused_gwp_label.place(relx=0.24,rely=0.32,anchor="w")
        Timber_reused_gwp_entry.place(relx=0.44,rely=0.32,anchor="w")
        Timber_new_gwp_label.place(relx=0.50,rely=0.32,anchor="w")
        Timber_new_gwp_entry.place(relx=0.69,rely=0.32,anchor="w")
        
        Steel_reused_gwp_label.place(relx=0.24,rely=0.37,anchor="w")
        Steel_reused_gwp_entry.place(relx=0.44,rely=0.37,anchor="w")
        Steel_new_gwp_label.place(relx=0.50,rely=0.37,anchor="w")
        Steel_new_gwp_entry.place(relx=0.69,rely=0.37,anchor="w")
        

        #Transport
        include_transportation_checkbutton.place(relx=0.5,rely=0.48,anchor="center")
        
        #ALGORITHMS CHECKBOXES
        Choose_algorithms_label.place(relx=0.5,rely=0.6,anchor="center")
        greedy_checkbutton.place(relx=0.15,rely=0.65,anchor="w")
        greedy_plural_checkbutton.place(relx=0.35,rely=0.65,anchor="w")
        bipartite_checkbutton.place(relx=0.55,rely=0.65,anchor="w")
        bipartite_plural_checkbutton.place(relx=0.75,rely=0.65,anchor="w")
        bipartite_multi_plural_checkbutton.place(relx=0.15,rely=0.7,anchor="w")
        MILP_checkbutton.place(relx=0.35,rely=0.7,anchor="w")
        genetic_checkbutton.place(relx=0.55,rely=0.7,anchor="w")
        brute_checkbutton.place(relx=0.75,rely=0.7,anchor="w")
     
        #Timber
        Timber_price_label.place_forget()
        Timber_price_entry.place_forget()
        Timber_reused_price_label.place_forget()
        Timber_reused_price_entry.place_forget()
        #Steel
        Steel_price_label.place_forget()
        Steel_price_entry.place_forget()
        Steel_reused_price_label.place_forget()
        Steel_reused_price_entry.place_forget()
        #Valuation
        GWP_valuation_label.place_forget()
        GWP_valuation_entry.place_forget()

        #Transport
        Transport_price_label.place_forget()
        Transport_price_entry.place_forget()
        
    elif matching_metric_var.get() == "GWP" and include_transportation_var.get() == 1:
        #GWP
        Timber_reused_gwp_label.place(relx=0.24,rely=0.32,anchor="w")
        Timber_reused_gwp_entry.place(relx=0.44,rely=0.32,anchor="w")
        Timber_new_gwp_label.place(relx=0.50,rely=0.32,anchor="w")
        Timber_new_gwp_entry.place(relx=0.69,rely=0.32,anchor="w")
        
        Steel_reused_gwp_label.place(relx=0.24,rely=0.37,anchor="w")
        Steel_reused_gwp_entry.place(relx=0.44,rely=0.37,anchor="w")
        Steel_new_gwp_label.place(relx=0.50,rely=0.37,anchor="w")
        Steel_new_gwp_entry.place(relx=0.69,rely=0.37,anchor="w")
        
        #Transport
        include_transportation_checkbutton.place(relx=0.5,rely=0.48,anchor="center")
        Transport_GWP_label.place(relx=0.41,rely=0.53,anchor="w")
        Transport_GWP_entry.place(relx=0.55,rely=0.53,anchor="w")
        
        #ALGORITHMS CHECKBOXES
        Choose_algorithms_label.place(relx=0.5,rely=0.6,anchor="center")
        greedy_checkbutton.place(relx=0.15,rely=0.65,anchor="w")
        greedy_plural_checkbutton.place(relx=0.35,rely=0.65,anchor="w")
        bipartite_checkbutton.place(relx=0.55,rely=0.65,anchor="w")
        bipartite_plural_checkbutton.place(relx=0.75,rely=0.65,anchor="w")
        bipartite_multi_plural_checkbutton.place(relx=0.15,rely=0.7,anchor="w")
        MILP_checkbutton.place(relx=0.35,rely=0.7,anchor="w")
        genetic_checkbutton.place(relx=0.55,rely=0.7,anchor="w")
        brute_checkbutton.place(relx=0.75,rely=0.7,anchor="w")
     
        #Timber
        Timber_price_label.place_forget()
        Timber_price_entry.place_forget()
        Timber_reused_price_label.place_forget()
        Timber_reused_price_entry.place_forget()
        #Steel
        Steel_price_label.place_forget()
        Steel_price_entry.place_forget()
        Steel_reused_price_label.place_forget()
        Steel_reused_price_entry.place_forget()
        #Valuation
        GWP_valuation_label.place_forget()
        GWP_valuation_entry.place_forget()

        #Transport
        Transport_price_label.place_forget()
        Transport_price_entry.place_forget()


    elif matching_metric_var.get() == "Price" and not include_transportation_var.get() == 1:  
        #PRICE
        Timber_price_label.place(relx=0.29,rely=0.32,anchor="w")
        Timber_price_entry.place(relx=0.44,rely=0.32,anchor="w")
        Steel_price_label.place(relx=0.29,rely=0.37,anchor="w")
        Steel_price_entry.place(relx=0.44,rely=0.37,anchor="w")
        Timber_reused_price_label.place(relx=0.52,rely=0.32,anchor="w")
        Timber_reused_price_entry.place(relx=0.69,rely=0.32,anchor="w")
        Steel_reused_price_label.place(relx=0.52,rely=0.37,anchor="w")
        Steel_reused_price_entry.place(relx=0.69,rely=0.37,anchor="w")

        #Transport 
        include_transportation_checkbutton.place(relx=0.5,rely=0.48,anchor="center")

        #Algorithms buttons
        Choose_algorithms_label.place(relx=0.5,rely=0.6,anchor="center")
        greedy_checkbutton.place(relx=0.15,rely=0.65,anchor="w")
        greedy_plural_checkbutton.place(relx=0.35,rely=0.65,anchor="w")
        bipartite_checkbutton.place(relx=0.55,rely=0.65,anchor="w")
        bipartite_plural_checkbutton.place(relx=0.75,rely=0.65,anchor="w")
        bipartite_multi_plural_checkbutton.place(relx=0.15,rely=0.7,anchor="w")
        MILP_checkbutton.place(relx=0.35,rely=0.7,anchor="w")
        genetic_checkbutton.place(relx=0.55,rely=0.7,anchor="w")
        brute_checkbutton.place(relx=0.75,rely=0.7,anchor="w")
        #Timber
        Timber_reused_gwp_label.place_forget()
        Timber_reused_gwp_entry.place_forget()
        Timber_new_gwp_label.place_forget()
        Timber_new_gwp_entry.place_forget()
        #Steel
        Steel_reused_gwp_label.place_forget()
        Steel_reused_gwp_entry.place_forget()
        Steel_new_gwp_label.place_forget()
        Steel_new_gwp_entry.place_forget()

        #Valuation
        GWP_valuation_label.place_forget()
        GWP_valuation_entry.place_forget()

        #Transport
        Transport_GWP_label.place_forget()
        Transport_GWP_entry.place_forget()
    
    elif matching_metric_var.get() == "Price" and include_transportation_var.get() == 1:        
        #PRICE
        Timber_price_label.place(relx=0.29,rely=0.32,anchor="w")
        Timber_price_entry.place(relx=0.44,rely=0.32,anchor="w")
        Steel_price_label.place(relx=0.29,rely=0.37,anchor="w")
        Steel_price_entry.place(relx=0.44,rely=0.37,anchor="w")
        Timber_reused_price_label.place(relx=0.52,rely=0.32,anchor="w")
        Timber_reused_price_entry.place(relx=0.69,rely=0.32,anchor="w")
        Steel_reused_price_label.place(relx=0.52,rely=0.37,anchor="w")
        Steel_reused_price_entry.place(relx=0.69,rely=0.37,anchor="w")

        #Transport 
        include_transportation_checkbutton.place(relx=0.5,rely=0.48,anchor="center")
        Transport_price_label.place(relx=0.37,rely=0.53,anchor="w")
        Transport_price_entry.place(relx=0.58,rely=0.53,anchor="w")
        
        #Algorithms buttons
        Choose_algorithms_label.place(relx=0.5,rely=0.6,anchor="center")
        greedy_checkbutton.place(relx=0.15,rely=0.65,anchor="w")
        greedy_plural_checkbutton.place(relx=0.35,rely=0.65,anchor="w")
        bipartite_checkbutton.place(relx=0.55,rely=0.65,anchor="w")
        bipartite_plural_checkbutton.place(relx=0.75,rely=0.65,anchor="w")
        bipartite_multi_plural_checkbutton.place(relx=0.15,rely=0.7,anchor="w")
        MILP_checkbutton.place(relx=0.35,rely=0.7,anchor="w")
        genetic_checkbutton.place(relx=0.55,rely=0.7,anchor="w")
        brute_checkbutton.place(relx=0.75,rely=0.7,anchor="w")
        #Timber
        Timber_reused_gwp_label.place_forget()
        Timber_reused_gwp_entry.place_forget()
        Timber_new_gwp_label.place_forget()
        Timber_new_gwp_entry.place_forget()
        #Steel
        Steel_reused_gwp_label.place_forget()
        Steel_reused_gwp_entry.place_forget()
        Steel_new_gwp_label.place_forget()
        Steel_new_gwp_entry.place_forget()

        #Valuation
        GWP_valuation_label.place_forget()
        GWP_valuation_entry.place_forget()

        #Transport
        Transport_GWP_label.place_forget()
        Transport_GWP_entry.place_forget()

    elif matching_metric_var.get() == "Sverre":
        #messagebox.showerror("Invalid input", "Please enter a valid number for Timber reused GWP")
        messagebox.showerror("Invalid input","As a tribute to Sverre, this project supervisor, you can now click \"Calculate\" to see where some of our knowledge orginates from!")
        include_transportation_checkbutton.place_forget()
        
        #GWP
        Timber_reused_gwp_label.place_forget()
        Timber_reused_gwp_entry.place_forget()
        Timber_new_gwp_label.place_forget()
        Timber_new_gwp_entry.place_forget()
        Steel_reused_gwp_label.place_forget()
        Steel_reused_gwp_entry.place_forget()
        Steel_new_gwp_label.place_forget()
        Steel_new_gwp_entry.place_forget()
        #Price
        Timber_price_label.place_forget()
        Timber_price_entry.place_forget()
        Timber_reused_price_label.place_forget()
        Timber_reused_price_entry.place_forget()
        Steel_price_label.place_forget()
        Steel_price_entry.place_forget()
        Steel_reused_price_label.place_forget()
        Steel_reused_price_entry.place_forget()
        GWP_valuation_label.place_forget()
        GWP_valuation_entry.place_forget()
        Choose_algorithms_label.place_forget()

    else:
        include_transportation_checkbutton.place_forget()
        #GWP
        Timber_reused_gwp_label.place_forget()
        Timber_reused_gwp_entry.place_forget()
        Timber_new_gwp_label.place_forget()
        Timber_new_gwp_entry.place_forget()
        Steel_reused_gwp_label.place_forget()
        Steel_reused_gwp_entry.place_forget()
        Steel_new_gwp_label.place_forget()
        Steel_new_gwp_entry.place_forget()
        #Price
        Timber_price_label.place_forget()
        Timber_price_entry.place_forget()
        Timber_reused_price_label.place_forget()
        Timber_reused_price_entry.place_forget()
        Steel_price_label.place_forget()
        Steel_price_entry.place_forget()
        Steel_reused_price_label.place_forget()
        Steel_reused_price_entry.place_forget()
        GWP_valuation_label.place_forget()
        GWP_valuation_entry.place_forget()
        Transport_GWP_entry.place_forget()
        Transport_GWP_label.place_forget()
        Transport_price_label.place_forget()
        Transport_price_entry.place_forget()
        Choose_algorithms_label.place_forget()
        greedy_checkbutton.place_forget()
        greedy_plural_checkbutton.place_forget()
        bipartite_checkbutton.place_forget()
        bipartite_plural_checkbutton.place_forget()
        bipartite_multi_plural_checkbutton.place_forget()
        MILP_checkbutton.place_forget()
        genetic_checkbutton.place_forget()
        brute_checkbutton.place_forget()

def include_transportation_checked():
    if include_transportation_var.get() == 1 and matching_metric_var.get() == "Combined":
        Transport_GWP_label.place(relx=0.30,rely=0.53,anchor="w")
        Transport_GWP_entry.place(relx=0.44,rely=0.53,anchor="w")
        Transport_price_label.place(relx=0.50,rely=0.53,anchor="w")
        Transport_price_entry.place(relx=0.71,rely=0.53,anchor="w")

    elif include_transportation_var.get() == 1 and matching_metric_var.get() == "GWP":
        Transport_GWP_label.place(relx=0.41,rely=0.53,anchor="w")
        Transport_GWP_entry.place(relx=0.55,rely=0.53,anchor="w")
        Transport_price_label.place(relx=0.50,rely=0.53,anchor="w")
        Transport_price_entry.place(relx=0.71,rely=0.53,anchor="w")
        Transport_price_label.place_forget()
        Transport_price_entry.place_forget()

    elif include_transportation_var.get() == 1 and matching_metric_var.get() == "Price":
        Transport_GWP_label.place(relx=0.41,rely=0.53,anchor="w")
        Transport_GWP_entry.place(relx=0.55,rely=0.53,anchor="w")
        Transport_price_label.place(relx=0.37,rely=0.53,anchor="w")
        Transport_price_entry.place(relx=0.58,rely=0.53,anchor="w")
        Transport_GWP_label.place_forget()
        Transport_GWP_entry.place_forget()

    else:
        Transport_GWP_label.place_forget()
        Transport_GWP_entry.place_forget()
        Transport_price_label.place_forget()
        Transport_price_entry.place_forget()

def none():
    pass

def validate_input(input_str):
    if input_str == "":
        return True
    try:
        float(input_str)
        return True
    except ValueError:
        #messagebox.showerror("Invalid input", "Please enter a valid number for Timber reused GWP")
        result_label.configure(text="Invalid input. Please enter numbers only.", foreground="red")
        result_label.after(2000,clear_error_message)
        return False

def warning_longruntime_brute():
    print("supply elements: ",num_supply_elements)
    if brute_var.get() and num_supply_elements.get()>14 and num_demand_elements.get()>14:
        result_label.configure(text="Warning! Brute force will take almost forever due to the datasets size. ("+str(num_demand_elements.get())+" * "+str(num_supply_elements.get())+")", foreground="red")
        result_label.after(6000,clear_error_message)
    elif not brute_var.get() and num_supply_elements.get()>14 and num_demand_elements.get()>14:
        result_label.configure(text="Smart choice!", foreground="green")
        result_label.after(4000,clear_error_message)

def warning_longruntime_genetic():
    if genetic_var.get() and num_supply_elements.get()>50 or num_demand_elements.get()>50:
        result_label.configure(text="Warning! Genetic algorithm requires a very long runtime to completedue to the datasets size. ("+str(num_demand_elements.get())+" * "+str(num_supply_elements.get())+")", foreground="red")
        result_label.after(6000,clear_error_message)
    elif not genetic_var.get() and num_supply_elements.get()>50 and num_demand_elements.get()>50:
        result_label.configure(text="")
        result_label.after(0,clear_error_message)

    
def clear_error_message():
    result_label.config(text="")
# Define the function that will be executed when the Calculate button is pressed   



def on_general_entry_click(event,entry,variabel):
    if float(entry.get())==constants[variabel]:
        entry.delete(0, tk.END)
        entry.config(text="",fg="black")

def on_general_entry_string_click(event,entry,variabel):
    if str(entry.get())==constants[variabel]:
        entry.delete(0, tk.END)
        entry.config(text="",fg="black")

def OpenUrl():
    url="https://nrksuper.no/serie/newton/DMPP21001720/sesong-2020/episode-19"
    webbrowser.open_new(url)

def open_report():
    filename=filename_tk.get()
    filename=r"./Local Files/GUI Files/Results/"+filename
    print("filename",filename)
    filepath=r""+filename

    if platform.system()=="Windows":
        current_directory = os.getcwd()
        file = current_directory + filepath
        os.startfile(file)
    elif platform.system() == "Darwin":
        subprocess.call(["open", filepath])
    else:
        subprocess.call(["xdg-open", filepath])

if __name__ == "__main__":
    # Create the main window and configure it to fill the whole screen
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    #Create title
    root.title("Element Matching Machine")
    root.title_label = ttk.Label(root, text="Element Matching Program", font=("Montserrat", 34, "bold"), foreground="#00509e", background="#1F1F1F")
    root.title_label.place(relx=0.5,y=50,anchor="center")
    #Create describtion
    root.description_label = tk.Label(root, text="Choose demand and supply files (.XLSX) and fill in the below variables, then click \"Calculate\" to generate a report with the results from the desired matching algorithms.",font=("Montserrat", 12, "bold"), foreground="#00509e")
    root.description_label.place(relx=0.5,rely=0.105,anchor="center")


    ##VARIABLES
    num_supply_elements = tk.IntVar()
    num_demand_elements = tk.IntVar()
    supply_filepath_bool=tk.BooleanVar()
    demand_filepath_bool=tk.BooleanVar()
    supply_filepath_string=tk.StringVar()
    demand_filepath_string=tk.StringVar()
    matching_metric_var_constant=tk.StringVar()
    filename_tk=tk.StringVar()


    ###LABELS,BUTTONS and ENTRYS###

    #Create Supply file label and browse button
    supply_file_label = tk.Label(root, text="No supply file selected",foreground="red")
    supply_file_label.place(relx=0.18, rely=0.21,anchor="center")
    supply_file_button = ttk.Button(root, text="Browse Supply File", command=browse_supply_file)
    supply_file_button.place(relx=0.18, rely=0.18,anchor="center")
    num_supply_label=tk.Label(root)
    num_supply_label.place(relx=0.18, rely=0.235,anchor="center")

    #Create Demand file label browse button
    demand_file_label = tk.Label(root, text="No demand file selected",foreground="red")
    demand_file_label.place(relx=0.80, rely=0.21,anchor="center")
    demand_file_button = ttk.Button(root, text="Browse Demand File", command=browse_demand_file)
    demand_file_button.place(relx=0.80, rely=0.18,anchor="center")
    num_demand_label=tk.Label(root)
    num_demand_label.place(relx=0.80, rely=0.235,anchor="center")
    #Create construction_site detalils label
    Construction_site_label = tk.Label(root, text="Construction site details:",font=("Montserrat",12,"bold"),foreground="#000000")
    Construction_site_label.place(relx=0.5,rely=0.14,anchor="center")

    #Create projectname label and entry
    Projectname_label = tk.Label(root, text="Project name:")
    Projectname_value_prefilled = tk.StringVar(value=constants["Project name"])
    Projectname_entry = tk.Entry(root,textvariable=Projectname_value_prefilled,fg="grey",widt=16)
    Projectname_label.place(relx=0.315,rely=0.18,anchor="center")
    Projectname_entry.place(relx=0.41,rely=0.18,anchor="center")
    Projectname_entry.bind('<FocusIn>', lambda event,entry=Projectname_entry,variabel="Project name":on_general_entry_string_click(event,entry,variabel))

    #Create projec latitude label and entry
    ProjectLatitude_label = tk.Label(root, text="Latitude:")
    ProjectLatitude_value_prefilled = tk.StringVar(value=constants["Cite latitude"])
    ProjectLatitude_entry = tk.Entry(root,textvariable=ProjectLatitude_value_prefilled,fg="grey",width=7)
    ProjectLatitude_label.place(relx=0.50,rely=0.18,anchor="center")
    ProjectLatitude_entry.place(relx=0.55,rely=0.18,anchor="center")
    ProjectLatitude_entry.bind('<FocusIn>', lambda event,entry=ProjectLatitude_entry,variabel="Cite latitude":on_general_entry_string_click(event,entry,variabel))

    #Create project longitude label and entry
    ProjectLongitude_label = tk.Label(root, text="Longitude:")
    ProjectLongitude_value_prefilled = tk.StringVar(value=constants["Cite longitude"])
    ProjectLongitude_entry = tk.Entry(root,textvariable=ProjectLongitude_value_prefilled,fg="grey",width=7)
    ProjectLongitude_label.place(relx=0.61,rely=0.18,anchor="center")
    ProjectLongitude_entry.place(relx=0.665,rely=0.18,anchor="center")
    ProjectLongitude_entry.bind('<FocusIn>', lambda event,entry=ProjectLongitude_entry,variabel="Cite longitude":on_general_entry_string_click(event,entry,variabel))


    # Create the Matching metric dropdown menu
    matching_metric_var = tk.StringVar()
    matching_metric_label = ttk.Label(root, text="Matching metric:",font=("Montserrat",12,"bold"),foreground="#000000")
    matching_metric_label.place(relx=0.5,rely=0.24,anchor="center")
    matching_metric_dropdown = ttk.Combobox(root, textvariable=matching_metric_var, values=["Price", "GWP", "Combined"])
    matching_metric_dropdown.place(relx=0.5,rely=0.27,anchor="center")
    matching_metric_var.trace("w", on_matching_metric_change)

    ###TIMBER
    #Create the timber reused GWP input field (not shown before matching metric is choosen)
    Timber_reused_gwp_label = tk.Label(root, text="Reused timber GWP [kgCO2eq per m^3]:")
    Timber_reused_GWP_prefilled=tk.DoubleVar(value=constants["TIMBER_REUSE_GWP"])
    Timber_reused_gwp_entry = tk.Entry(root,textvariable=Timber_reused_GWP_prefilled,fg="grey",widt=5)
    Timber_reused_gwp_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Timber_reused_gwp_entry.bind('<FocusIn>', lambda event,entry=Timber_reused_gwp_entry,variabel="TIMBER_REUSE_GWP":on_general_entry_click(event,entry,variabel))

    #Create the new timber GWP input field (not shown before matching metric is choosen)
    Timber_new_gwp_label = tk.Label(root, text="New timber GWP [kgCO2eq per m^3]:")
    Timber_new_prefilled=tk.DoubleVar(value=constants["TIMBER_GWP"])
    Timber_new_gwp_entry = tk.Entry(root,textvariable=Timber_new_prefilled,fg="grey",widt=5)
    Timber_new_gwp_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Timber_new_gwp_entry.bind('<FocusIn>', lambda event,entry=Timber_new_gwp_entry,variabel="TIMBER_GWP":on_general_entry_click(event,entry,variabel))

    #Create the new timber price input field (not shown before matching metric is choosen)
    Timber_price_label = tk.Label(root, text="New timber price [kr per m^3]:")
    Timber_price_prefilled=tk.DoubleVar(value=constants["TIMBER_PRICE"])
    Timber_price_entry = tk.Entry(root,textvariable=Timber_price_prefilled,fg="grey",widt=5)
    Timber_price_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Timber_price_entry.bind('<FocusIn>', lambda event,entry=Timber_price_entry,variabel="TIMBER_PRICE":on_general_entry_click(event,entry,variabel))

    #Create the new timber price input field (not shown before matching metric is choosen)
    Timber_reused_price_label = tk.Label(root, text="Reused timber price [kr per m^3]:")
    Timber_reused_price_prefilled=tk.DoubleVar(value=constants["TIMBER_REUSE_PRICE"])
    Timber_reused_price_entry = tk.Entry(root,textvariable=Timber_reused_price_prefilled,fg="grey",widt=5)
    Timber_reused_price_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Timber_reused_price_entry.bind('<FocusIn>', lambda event,entry=Timber_reused_price_entry,variabel="TIMBER_REUSE_PRICE":on_general_entry_click(event,entry,variabel))

    ###STEEL
    #Create the steel reused GWP input field (not shown before matching metric is choosen)
    Steel_reused_gwp_label = tk.Label(root, text="Reused steel GWP [kgCO2eq per m^3]:")
    Steel_reused_GWP_prefilled=tk.DoubleVar(value=constants["STEEL_REUSE_GWP"])
    Steel_reused_gwp_entry = tk.Entry(root,textvariable=Steel_reused_GWP_prefilled,fg="grey",widt=5)
    Steel_reused_gwp_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Steel_reused_gwp_entry.bind('<FocusIn>', lambda event,entry=Steel_reused_gwp_entry,variabel="STEEL_REUSE_GWP":on_general_entry_click(event,entry,variabel))

    #Create the new steel GWP input field (not shown before matching metric is choosen)
    Steel_new_gwp_label = tk.Label(root, text="New steel GWP [kgCO2eq per m^3]:")
    Steel_new_GWP_prefilled=tk.DoubleVar(value=constants["STEEL_GWP"])
    Steel_new_gwp_entry = tk.Entry(root,textvariable=Steel_new_GWP_prefilled,fg="grey",widt=5)
    Steel_new_gwp_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Steel_new_gwp_entry.bind('<FocusIn>', lambda event,entry=Steel_new_gwp_entry,variabel="STEEL_GWP":on_general_entry_click(event,entry,variabel))

    #Create the new steel price input field (not shown before matching metric is choosen)
    Steel_price_label = tk.Label(root, text="New steel price [kr per m^3]:")
    Steel_price_prefilled=tk.DoubleVar(value=constants["STEEL_PRICE"])
    Steel_price_entry = tk.Entry(root,textvariable=Steel_price_prefilled,fg="grey",widt=5)
    Steel_price_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Steel_price_entry.bind('<FocusIn>', lambda event,entry=Steel_price_entry,variabel="STEEL_PRICE":on_general_entry_click(event,entry,variabel))

    #Create the reused steel price input field (not shown before matching metric is choosen)
    Steel_reused_price_label = tk.Label(root, text="Reused steel price [kr per m^3]:")
    Steel_reused_price_prefilled=tk.DoubleVar(value=constants["STEEL_REUSE_PRICE"])
    Steel_reused_price_entry = tk.Entry(root,textvariable=Steel_reused_price_prefilled,fg="grey",widt=5)
    Steel_reused_price_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Steel_reused_price_entry.bind('<FocusIn>', lambda event,entry=Steel_reused_price_entry,variabel="STEEL_REUSE_PRICE":on_general_entry_click(event,entry,variabel))

    #Create the validation GWP label and field
    GWP_valuation_label = tk.Label(root, text="Valuation of GWP [kr per kgCO2]:")
    GWP_valuation_prefilled=tk.DoubleVar(value=constants["VALUATION_GWP"])
    GWP_valuation_entry = tk.Entry(root,textvariable=GWP_valuation_prefilled,fg="grey",widt=5)
    GWP_valuation_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    GWP_valuation_entry.bind('<FocusIn>', lambda event,entry=GWP_valuation_entry,variabel="VALUATION_GWP":on_general_entry_click(event,entry,variabel))

    ###TRANSPORT
    # Create the Include transportation checkbox
    include_transportation_var = tk.IntVar()
    include_transportation_checkbutton = tk.Checkbutton(root, text="Include transportation",font=("Montserrat", 12, "bold"), foreground="#000000", variable=include_transportation_var,command=include_transportation_checked)

    # Create the TransportGWP label and entry
    Transport_GWP_label = tk.Label(root, text="GWP transportation [factor]:")
    Transport_GWP_prefilled=tk.DoubleVar(value=constants["TRANSPORT_GWP"])
    Transport_GWP_entry = tk.Entry(root,textvariable=Transport_GWP_prefilled,fg="grey",widt=5)
    Transport_GWP_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Transport_GWP_entry.bind('<FocusIn>', lambda event,entry=Transport_GWP_entry,variabel="TRANSPORT_GWP":on_general_entry_click(event,entry,variabel))

    #Create the TransportPrice label and entry
    Transport_price_label = tk.Label(root, text="Transportation price [kr per km per tonne]:")
    Transport_price_prefilled=tk.DoubleVar(value=constants["PRICE_TRANSPORTATION"])
    Transport_price_entry = tk.Entry(root,textvariable=Transport_price_prefilled,fg="grey",widt=5)
    Transport_price_entry.config(validate='key', validatecommand=(root.register(validate_input), '%P'))
    Transport_price_entry.bind('<FocusIn>', lambda event,entry=Transport_price_entry,variabel="PRICE_TRANSPORTATION":on_general_entry_click(event,entry,variabel))

    ###ALGORITHMS

    #Create choose algorithms label
    Choose_algorithms_label = tk.Label(root, text="Choose the desidered algorithms:",font=("Montserrat",12,"bold"),foreground="#000000")

    greedy_var = tk.IntVar()
    greedy_checkbutton = tk.Checkbutton(root, text="Greedy",font=("Montserrat", 12), foreground="#000000", variable=greedy_var,command=none)

    greedy_plural_var = tk.IntVar()
    greedy_plural_checkbutton = tk.Checkbutton(root, text="Greedy plural",font=("Montserrat", 12), foreground="#000000", variable=greedy_plural_var,command=none)

    bipartite_var = tk.IntVar()
    bipartite_checkbutton = tk.Checkbutton(root, text="Bipartite",font=("Montserrat", 12), foreground="#000000", variable=bipartite_var,command=none)

    bipartite_plural_var = tk.IntVar()
    bipartite_plural_checkbutton = tk.Checkbutton(root, text="Bipartite plural",font=("Montserrat", 12), foreground="#000000", variable=bipartite_plural_var,command=none)

    bipartite_multi_plural = tk.IntVar()
    bipartite_multi_plural_checkbutton = tk.Checkbutton(root, text="Bipartite multi plural",font=("Montserrat", 12), foreground="#000000", variable=bipartite_multi_plural,command=none)

    MILP_var = tk.IntVar()
    MILP_checkbutton = tk.Checkbutton(root, text="MILP",font=("Montserrat", 12), foreground="#000000", variable=MILP_var,command=none)

    brute_var = tk.IntVar()
    brute_checkbutton = tk.Checkbutton(root, text="Brute force",font=("Montserrat", 12), foreground="#000000", variable=brute_var,command=warning_longruntime_brute)

    genetic_var = tk.IntVar()
    genetic_checkbutton = tk.Checkbutton(root, text="Genetic",font=("Montserrat", 12), foreground="#000000", variable=genetic_var,command=warning_longruntime_genetic)


    result_frame = ttk.Frame(root)
    result_frame.pack(side=tk.BOTTOM, padx=10, pady=10)
    #result_label = ttk.Label(result_frame, text="Result:")
    #result_label.pack(side=tk.BOTTOM)

    #Create the caulculate button
    calculate_button = ttk.Button(root, text="Calculate", command=calculate)
    calculate_button.place(relx=0.5,rely=0.8,anchor="center",relheight=0.05,relwidth=0.10)
    open_report_button = ttk.Button(root, text="Open report", command=open_report)
    result_label = ttk.Label(root, text="")
    result_label.place(relx=0.5,rely=0.9,anchor="center")
    
    # Run the program
    root.mainloop()
