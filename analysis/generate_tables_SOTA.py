'''
This script does the comparison between our approach and the SOTA
'''

import logging
from scipy import spatial
import pandas as pd
from PIL import Image
from argparse import  ArgumentParser
from pathlib import Path
from tqdm import tqdm
from os.path import isfile, join
import math
import numpy as np


# python3 generate_tables_SOTA.py --set closed --diameter_km 50 

# python3 generate_tables_SOTA.py --set open --diameter_km 50 

def parse_args():
    args = ArgumentParser()

    args.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/data/omran/cities_SOTA/imgs_results"),
        help="Results CSVs folder.",
    )

    args.add_argument(
        "--GPS_database",
        type=Path,
        default=Path("/data/omran/siamese_cities/dataset_meta/gps_cities.csv"), 
        help="CSV file contains GPS coordinates for each city in the dataset.",
    )

    args.add_argument(
        "--diameter_km",
        type=int,
        default=50,
        help="Test",
    )

    args.add_argument(
        "--set",
        type=str,
        help="Open or closed set",
    )


    return args.parse_args()

import math

def is_gps_point_inside_city(city_lat, city_lng, diameter_km, point_lat, point_lng):
    """
    Checks if a GPS point falls inside the circle of the given diameter centered at a city's GPS location.
    
    Args:
        city_lat (float): Latitude of the city in degrees.
        city_lng (float): Longitude of the city in degrees.
        diameter_km (float): Diameter of the circle in kilometers.
        point_lat (float): Latitude of the GPS point in degrees.
        point_lng (float): Longitude of the GPS point in degrees.
    
    Returns:
        bool: True if the GPS point falls inside the circle, False otherwise.
    """
    # Convert latitude and longitude from degrees to radians
    city_lat_rad = math.radians(city_lat)
    city_lng_rad = math.radians(city_lng)
    point_lat_rad = math.radians(point_lat)
    point_lng_rad = math.radians(point_lng)
    
    # Calculate the great-circle distance between the city and the point using Haversine formula
    d_lat = point_lat_rad - city_lat_rad
    d_lng = point_lng_rad - city_lng_rad
    a = math.sin(d_lat/2)**2 + math.cos(city_lat_rad) * math.cos(point_lat_rad) * math.sin(d_lng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance_km = 6371 * c
    
    # Check if the distance is less than or equal to half of the diameter
    return distance_km <= (diameter_km / 2)

import csv

def get_city_coordinates(city_name,gps_database):
    """
    Reads the citygps.csv file and returns the latitude and longitude of a city given its name.
    
    Args:
        city_name (str): Name of the city.
    
    Returns:
        tuple: A tuple containing the latitude and longitude of the city in degrees.
              Returns None if city name is not found in the citygps.csv file.
    """
    with open(gps_database, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            if row[0].lower() == city_name.lower():
                city_lat = float(row[1])
                city_lng = float(row[2])
                return city_lat, city_lng
    # Return None if city name is not found
    return None


if __name__ == '__main__':


    args = parse_args()

    if(args.set == 'closed'):
        city_list = ["Amsterdam", "Barcelona", "Berlin", "London", "NewYork", "LosAngeles", "Rome", "Milan", "Paris", "Tokyo"]

    if(args.set == 'open'):
        city_list = ["Amman", "Istanbul", "Mexico_city", "Singapore", "Quebec", "Vancouver","Florence", "Rome", "Rio_de_Janeiro","Delhi"] 

    N = len(city_list)

    print('                ',end='')
    for city in city_list:
        print(f'{city.center(12)}', end='')

    print('')    

    diag = 0
    off_diag = 0
    for city_test in city_list:
        print(f'{city_test:<14} ', end=' ')
        for city_database in city_list:

            city_lat, city_lng = get_city_coordinates(city_database,args.GPS_database)

            df = pd.read_csv(f'{args.results_dir}/inference_{city_test}.csv')

            img_list = df["img_id"].unique()
            len_images = len(img_list)

            #print(len_images)
            df.query('p_key == "hierarchy"', inplace=True)

    
            df['in_city'] = df.apply(lambda row:is_gps_point_inside_city(city_lat, city_lng, args.diameter_km, row['pred_lat'], row['pred_lng']) , axis=1)


            # Filter the DataFrame to keep only rows where the distance is within the specified radius
            filtered_df = df[df['in_city'] == True]

            # Count the number of images inside the circle
            num_images_inside_circle = len(filtered_df)

            if(city_test==city_database):
                diag+=num_images_inside_circle/len_images
            else:
                off_diag+=num_images_inside_circle/len_images    


           
            acc_single_city = num_images_inside_circle/len_images

            truncated_accuracy = float(f'{acc_single_city}'[:4])
            print(f'&   {truncated_accuracy:.2f}   ', end='')

        print("\\\\")

    print(f'Diagonal acc :  {diag/N } and off-diagonal as {off_diag/(N * N - N)}')
        
    print(f'Acc total    : {0.5 * (diag/N ) + (0.5) * (1 - off_diag/(N * N - N))}')
