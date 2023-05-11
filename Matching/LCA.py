import logging
import requests
import pandas as pd

TIMBER_GWP = 28.9       # based on NEPD-3442-2053-EN
TIMBER_REUSE_GWP = 2.25        # 0.0778*28.9 = 2.25 based on Eberhardt
TRANSPORT_GWP = 96.0    # TODO kg/m3/t based on ????
TIMBER_DENSITY = 491.0  # kg, based on NEPD-3442-2053-EN

#Inlcuding price:
NEW_ELEMENT_PRICE_TIMBER=435 #Per m^3 https://www.landkredittbank.no/blogg/2021/prisen-pa-sagtommer-okte-20-prosent/
REUSED_ELEMENT_PRICE_TIMBER=100 #Per m^2
GWP_PRICE=0.6 #In kr:Per kg CO2, based on OECD
PRICE_TRANSPORT = 3.78 #Price per km per tonn. Derived from 2011 numbers on scaled t0 2022 using SSB

def calculate_lca(length, area, include_transportation, distance, gwp_factor, transport_gwp, density):
    """ Calculate Life Cycle Assessment """
    # TODO add processing
    # TODO add other impact categories than GWP
    volume = length * area
    lca = volume * gwp_factor
    transportation_LCA = lca.copy()
    transportation_LCA[:] = 0
    if include_transportation:
        transportation_LCA = calculate_transportation_LCA(volume, density, distance, transport_gwp)
        logging.debug(f"Transportation LCA:", transportation_LCA)
        lca += transportation_LCA
    return lca, transportation_LCA


def calculate_score(length, area, include_transportation, distance, gwp_factor, transport_gwp, price, priceGWP, density, price_transport):
    """ Calculates a score, based on GWP and price for new elements. The score is total price for kg CO2 eq and price for elements. """
    # TODO add processing
    # TODO add other impact categories than GWP and price?
    
    #TODO: Store information about GWP and Price
    volume = length * area
    score = volume * gwp_factor
    transportation_score = score.copy()
    transportation_score[:] = 0
  
    if not include_transportation:
        score=score*priceGWP

    if include_transportation:
        transportation_LCA = calculate_transportation_LCA(volume, density, distance, transport_gwp)
        transportation_cost= calcultate_price_transport(volume,density,distance, price_transport)
        logging.debug(f"Transportation LCA:", transportation_LCA)
        score += transportation_LCA
        score=score*priceGWP
        score+=transportation_cost
        transportation_score += transportation_LCA*priceGWP + transportation_cost
   
    price_element=volume*price
    score+=price_element
    
    return score, transportation_score

def calculate_price(length, area, include_transportation, distance, price, density, price_transport):
    volume = length * area
    score = volume * price #In kr
    transportation_score = score.copy()
    transportation_score[:] = 0

    if include_transportation:
        transportation_score = calcultate_price_transport(volume,density,distance, price_transport)
        score += transportation_score
    
    return score, transportation_score

def calculate_driving_distance(A_lat, A_lon, B_lat, B_lon):
    """Calculates the driving distance between two coordinates and returns the result in meters
    - Coordinates as a String
    """
    # TODO (Sigurd) Check if A or B should be first
    try:
        url = f"http://router.project-osrm.org/route/v1/car/{A_lon},{A_lat};{B_lon},{B_lat}?overview=false"
        req = requests.get(url)
        driving_distance_meter = req.json()["routes"][0]["distance"]
        distance = driving_distance_meter / 1000 #driving distance in km
    except:  # TODO need to define exception, not ALL.
        #logging.error(f"Was not able to get the driving distance from OSRM-API, URL:{url}") 
        distance = 0
    return  distance

def calculate_transportation_LCA(volume, density, distance, factor):
    """Calculates the CO2 equivalents of driving one element a specific distance
    - volume in float
    - density in float
    - distance in float
    - factor in float
    """
    density = density / 1000 #convert kg/m^3 to tonne/m^3
    factor = factor / 1000 #convert gram to kg
    return volume * density * distance * factor #C02 equivalents in kg

def calcultate_price_transport(volume, density, distance, price):
    """
    distance in km
    price per tonn per km in NOK
    """
    density = density / 1000 #convert kg/m^3 to tonne/m^3
    tonn=density*volume
    return price*tonn*distance


