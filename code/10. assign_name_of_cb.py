#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:23:21 2024

@author: byeonghwa
"""

import os
import geopandas as gpd
from geopy.geocoders import Nominatim
from datetime import datetime

# Function to reverse-geocode coordinates and extract street-level addresses
def get_street_name(row):
    try:
        location = geolocator.reverse(row['coordinates'])  # Use geolocator to get address from lat/lon
        address = location.raw.get('address', {})
        country = address['country']

        # Build an address string based on country-specific keys
        if country == 'Canada':
            address_string = ', '.join([address[key] for key in keys_ca if key in address])
        elif country == 'United States':
            address_string = ', '.join([address[key] for key in keys_us if key in address])
        else:
            address_string = 'error'

        return address_string
    except Exception as e:
        return None  # Return None if geocoding fails

# Set working directory where input/output files are located
os.chdir('put_your_working_directory')

# Define address fields to extract for each country
keys_us = ['road', 'hamlet', 'city', 'state', 'country']
keys_ca = ['road', 'neighbourhood', 'city', 'state', 'country']

# Load commercial boundary polygons
base_df = gpd.read_file('put_your_commercial_boundary_file')
target_df = base_df.copy()

# Replace polygon geometries with their centroids for reverse geocoding
target_df['geometry'] = target_df['geometry'].centroid
target_df = target_df.to_crs('EPSG:4326')  # Convert coordinates to lat/lon (WGS84)

# Add a column with latitude/longitude tuples for geocoding
target_df['coordinates'] = target_df.apply(lambda x: (x['geometry'].y, x['geometry'].x), axis=1)

start = datetime.now()  # Start timing the process

# Initialize geocoder with user agent and timeout settings
geolocator = Nominatim(user_agent='by', timeout=10)

# Apply the reverse geocoding function to each centroid
target_df['address'] = target_df.apply(get_street_name, axis=1)
    
end = datetime.now()
print('Duration: {}'.format(end - start))  # Print processing duration

# Add the retrieved addresses back into the original GeoDataFrame
base_df['address'] = target_df['address']

# Save the commercial boundary data with appended addresses as a GeoPackage
base_df.to_file('put_your_output_name', driver='GPKG')