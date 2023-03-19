import logging
import requests
import pandas as pd

TIMBER_GWP = 28.9       # based on NEPD-3442-2053-EN
TIMBER_REUSE_GWP = 2.25        # 0.0778*28.9 = 2.25 based on Eberhardt
TRANSPORT_GWP = 96.0    # TODO kg/m3/t based on ????
TIMBER_DENSITY = 491.0  # kg, based on NEPD-3442-2053-EN

#Inlcuding price:
NEW_ELEMENT_PRICE_TIMBER=435 #Per m^2 https://www.landkredittbank.no/blogg/2021/prisen-pa-sagtommer-okte-20-prosent/
REUSED_ELEMENT_PRICE_TIMBER=100 #Per m^2
GWP_PRICE=0.6 #In kr:Per kg CO2, based on OECD

#TODO include in matching

def calculate_lca(length, area,demand_lat,demand_lon,supply_lat,supply_lon,include_transportation,gwp_factor, density=TIMBER_DENSITY, ):
    """ Calculate Life Cycle Assessment """
    # TODO add processing
    # TODO add other impact categories than GWP
    volume = length * area
    lca = volume * gwp_factor
    if include_transportation:
        coords=[demand_lat,demand_lon,supply_lat,supply_lon]
        coordinates=pd.concat(coords,axis=1)
        coordinates["Distance"]=coordinates.apply(lambda row: calculate_driving_distance(row.Supply_lat,row.Supply_lon,row.Demand_lat,row.Demand_lon),axis=1)
        distance=coordinates["Distance"]
        transportation_LCA = calculate_transportation_LCA(volume, density, distance)
        logging.debug(f"Transportation LCA:", transportation_LCA)
        lca += transportation_LCA
    return lca


def calculate_score(length, area,demand_lat,demand_lon,supply_lat,supply_lon,include_transportation,gwp_factor, price_per_m2,priceGWP, density=TIMBER_DENSITY):
    """ Calculates a score, based on GWP and price for new elements. The score is total price for kg CO2 eq and price for elements. """
    # TODO add processing
    # TODO add other impact categories than GWP
    
    volume = length * area
    score = volume * gwp_factor
  
    if not include_transportation:
        score=score*priceGWP

    if include_transportation:
        coords=[demand_lat,demand_lon,supply_lat,supply_lon]
        coordinates=pd.concat(coords,axis=1)
        coordinates["Distance"]=coordinates.apply(lambda row: calculate_driving_distance(row.Supply_lat,row.Supply_lon,row.Demand_lat,row.Demand_lon),axis=1)
        distance=coordinates["Distance"]
        transportation_LCA = calculate_transportation_LCA(volume, density, distance)
        transportation_cost= calcultate_price_transport(volume,density,distance)
        logging.debug(f"Transportation LCA:", transportation_LCA)
        score += transportation_LCA
        score=score*priceGWP
        score+=transportation_cost
   
    price_element=volume*price_per_m2
    score+=price_element
    
    return score


def calculate_driving_distance(A_lat, A_lon, B_lat, B_lon):
    """Calculates the driving distance between two coordinates and returns the result in meters
    - Coordinates as a String
    """

    # TODO (Sigurd) Check if A or B should be first
    try:
        url = f"http://router.project-osrm.org/route/v1/car/{A_lat},{A_lon};{B_lat},{B_lon}?overview=false"
        req = requests.get(url)
        driving_distance_meter = req.json()["routes"][0]["distance"]
        distance = driving_distance_meter / 1000 #driving distance in km
    except:  # TODO need to define exception, not ALL.
        logging.error("Was not able to get the driving distance from OSRM-API")
        distance = 0
    return  distance

def calculate_transportation_LCA(volume, density, distance, factor = TRANSPORT_GWP):
    """Calculates the CO2 equivalents of driving one element a specific distance
    - volume in float
    - density in float
    - distance in float
    - factor in float
    """
    density = density / 1000 #convert kg/m^3 to tonne/m^3
    factor = factor / 1000 #convert gram to kg
    return volume * density * distance * factor #C02 equivalents in kg

def calcultate_price_transport(volume,density,distance):
    """
    distance in km
    """
    density = density / 1000 #convert kg/m^3 to tonne/m^3
    tonn=density*volume
    price=3.78 #Price per km per tonn. Derived from 2011 numbers on scaled t0 2022 using SSB
    return price*tonn*distance


