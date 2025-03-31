#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import geopandas as gpd
import pandas as pd
from datetime import datetime
from tools import *  # Custom module containing helper functions (such as assign_variable)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set working directory (replace with your actual path)
os.chdir('put your path')

# Define paths to input data
job_path = './jobs'              # Directory with job density data
bound_path = './bound'           # (Not used here but likely boundary data)
grid_path = './MA_grid_bound'    # Directory containing spatial grid files

# Initialize empty GeoDataFrame to store all job data
jobs_df = gpd.GeoDataFrame()

# Load all job data from parquet files and merge them
for (path, dirs, files) in os.walk(job_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.parquet'):  # Process only parquet files
            input_file = "%s/%s" % (path, fname)
            temp_jobs = gpd.read_parquet(input_file)
            # Rename column if necessary (handle potential column naming variations)
            try:
                temp_jobs = temp_jobs.rename({'v_CA16_5774..Worked.at.usual.place':'jobs'}, axis=1)    
            except:
                pass
            # Calculate area in square kilometers for each geometry
            temp_jobs['area'] = temp_jobs['geometry'].area / 10**6
            temp_jobs = temp_jobs[['jobs', 'area', 'geometry']]
            # Append to the combined jobs dataframe
            jobs_df = pd.concat([jobs_df, temp_jobs])
         
# Normalize jobs by area to get job density
jobs_df['jobs'] = jobs_df['jobs'] / jobs_df['area']
jobs_df = jobs_df[['jobs', 'geometry']]

# Columns of interest for additional data assignment
col_list = ['retailers', 'office', 'retail']

# Process each city grid file
for (path, dirs, files) in os.walk(grid_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.parquet'):
            input_file = "%s/%s" % (path, fname)
            city = fname.replace('.parquet', '').split('_')[-1]  # Extract city name from filename
            
            # Define output filename for training grid
            out_nm = './training_data/MA_training_data/' + city + '_training_grid.parquet'

            print(city)
            start = datetime.now()
            
            # Load the city grid boundary
            bound_data = gpd.read_parquet(input_file)
            bound_data = bound_data[['id', 'geometry']]
            
            # Check or create spatial index (optional but speeds up spatial operations)
            spatial_index = bound_data.sindex
            bound_data.has_sindex
            
            # Assign retailer, office, and retail building counts to each grid cell
            for col_nm in col_list:
                target_dir = './' + col_nm + '/' + city + '_' + col_nm + '.parquet'
                count_data = gpd.read_parquet(target_dir)
                # Uses assign_variable() from tools.py to spatially join and assign counts
                bound_data = assign_variable(bound_data, count_data, col_nm)
            
            # Combine 'retail' counts into 'retailers' and drop the separate 'retail' column
            bound_data['retailers'] = bound_data['retailers'] + bound_data['retail']
            bound_data = bound_data.drop('retail', axis=1)
                
            # Spatial join between grid cells and job density data
            dfsjoin = gpd.sjoin(bound_data, jobs_df, op="intersects")
            
            # Aggregate total jobs within each grid cell
            dfpivot = pd.pivot_table(dfsjoin, index='id', values='jobs', aggfunc='sum')
            dfpivot = dfpivot.reset_index(drop=False)
            dfpivot = dfpivot[['id', 'jobs']]
                        
            # Merge job data back into the grid dataframe
            bound_data = bound_data.merge(dfpivot, how='left', on='id')
            
            # Fill missing job values with zero
            bound_data = bound_data.fillna(0)
                                        
            # Save the completed training grid with all variables assigned
            bound_data.to_parquet(out_nm, index=False)
            end = datetime.now()
            print('Duration: {}'.format(end - start))