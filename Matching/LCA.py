import logging
import requests

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
    """ Calculates the GWP of the elements

    Args:
        length (float): length of element
        area (float): area of element
        include_transportation (boolean): if transportation should be included
        distance (float): driving distance to the construction site
        gwp_factor (float): GWP factor for the element
        transport_gwp (float): GWP factor for transportation
        density (float): density of the element

    Returns:
        lca (float): the GWP of the element
        transportation_LCA (float): the GWP of transporting the element
    """
    volume = length * area
    lca = volume * gwp_factor
    transportation_LCA = lca.copy()
    transportation_LCA[:] = 0
    if include_transportation:
        transportation_LCA = calculate_transportation_LCA(volume, density, distance, transport_gwp)
        lca += transportation_LCA
    return lca, transportation_LCA


def calculate_score(length, area, include_transportation, distance, gwp_factor, transport_gwp, price, priceGWP, density, price_transport):
    """ Method for evaluatating the scores corresponding to the "Combined" metric (both price and GWP)

    Args:
        length (float): length of element
        area (float): area of element
        include_transportation (boolean): if transportation should be included
        distance (float): driving distance to the construction site
        gwp_factor (float): GWP factor for the element
        transport_gwp (float): GWP factor for transportation
        price (float): price of the element
        priceGWP (float): price factor for GWP
        density (float): density of the element
        price_transport (float): price factor for transportation

    Returns:
        score (float): the score of the element concidering both GWP and Price (combined)
        transportation_score (float): the score of transporting the element
    """
    CO2_impact = 0
    volume = length * area
    score = volume * gwp_factor
    transportation_score = score.copy()
    transportation_score[:] = 0

    CO2_impact += score
  
    if not include_transportation:
        score=score*priceGWP

    if include_transportation:
        transportation_LCA = calculate_transportation_LCA(volume, density, distance, transport_gwp)
        transportation_cost= calcultate_price_transport(volume,density,distance, price_transport)
        logging.debug(f"Transportation LCA:", transportation_LCA)
        score += transportation_LCA
        CO2_impact += transportation_LCA
        score=score*priceGWP
        score+=transportation_cost
        transportation_score += transportation_LCA*priceGWP + transportation_cost
   
    price_element=volume*price
    score+=price_element
    
    return score, transportation_score, CO2_impact

def calculate_price(length, area, include_transportation, distance, price, density, price_transport):
    """ Method for evaluatating the price of an element

    Args:
        length (float): length of element
        area (float): area of element
        include_transportation (boolean): if transportation should be included
        distance (float): driving distance to the construction site
        price (float): price of the element
        density (float): density of the element
        price_transport (float): price factor for transportation

    Returns:
        score (float): the price of the element 
        transportation_score (float): the price of transporting the element
    """
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
    """Using the OSRM API to calculate the driving distance between two locations A and B

    Args:
        A_lat (float): Latitude of location A
        A_lon (float): Longitude of location A
        B_lat (float): Latitude of location B
        B_lon (float): Longitude of location B

    Returns:
        float: driving distance
    """
    try:
        url = f"http://router.project-osrm.org/route/v1/car/{A_lon},{A_lat};{B_lon},{B_lat}?overview=false"
        req = requests.get(url)
        driving_distance_meter = req.json()["routes"][0]["distance"]
        distance = driving_distance_meter / 1000 #driving distance in km
    except: 
        #Mean the input coordinates was on the wrong format
        logging.error(f"Was not able to get the driving distance from OSRM-API, URL:{url}") 
        distance = 0
    return  distance

def calculate_transportation_LCA(volume, density, distance, factor):
    """Calculates the GWP of transporting the element a given distance

    Args:
        volume (float): volume of element
        density (float): density of element
        distance (float): driving distance
        factor (float): GWP transportation factor

    Returns:
        float: GWP of transportation
    """
    density = density / 1000 #convert kg/m^3 to tonne/m^3
    factor = factor / 1000 #convert gram to kg
    return volume * density * distance * factor #C02 equivalents in kg

def calcultate_price_transport(volume, density, distance, price):
    """Calculates the price of transporting the element a given distance

    Args:
        volume (float): volume of element
        density (float): density of element
        distance (float): driving distance
        price (float): price transportation factor

    Returns:
        float: price of transportation
    """
    density = density / 1000 #convert kg/m^3 to tonne/m^3
    tonn=density*volume
    return price*tonn*distance


