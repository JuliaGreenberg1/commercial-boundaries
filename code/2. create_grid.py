#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import os
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Set working directory to your base directory (update this path)
os.chdir("put your base directory")

# Load the top 50 metropolitan areas file and convert it to EPSG:3857 projection
ma_america = gpd.read_parquet('put the given metropolitan area files (top_50_ma_us_ca.parquet)')
ma_america = ma_america.to_crs('EPSG:3857')

# Loop through each metropolitan area in the dataset
for x in ma_america.index:
    # Extract a single metropolitan area boundary
    temp_ma = ma_america.loc[ma_america.index == x].copy()
    nm = temp_ma['NAME'].tolist()[0]  # Get the metro area name
    
    # Define the output filename for the grid
    out_grid_nm = './MA_grid_bound/base_grid_' + nm + '.parquet'
    
    # If the grid for this metro area already exists, skip it
    if os.path.isfile(out_grid_nm):
        pass
    else:
        print(nm)  # Print city name being processed
        start = datetime.now()  # Start timing the process

        # Set grid cell size (50m x 50m)
        width = 50
        height = 50
    
        # Get bounding box coordinates for the metro area
        xmin, ymin, xmax, ymax = temp_ma.total_bounds

        # Calculate how many rows and columns are needed to cover the bounding box
        rows = int(np.ceil((ymax - ymin) / height))
        cols = int(np.ceil((xmax - xmin) / width))

        # Initialize origin points for grid generation
        XleftOrigin = xmin
        XrightOrigin = xmin + width
        YtopOrigin = ymax
        YbottomOrigin = ymax - height

        polygons = []  # List to hold each grid cell polygon

        # Generate grid cells by looping through columns and rows
        for i in tqdm(range(cols)):  # Iterate over columns
            Ytop = YtopOrigin
            Ybottom = YbottomOrigin
            for j in range(rows):  # Iterate over rows
                # Create a rectangular polygon for each grid cell
                polygons.append(
                    Polygon([
                        (XleftOrigin, Ytop),
                        (XrightOrigin, Ytop),
                        (XrightOrigin, Ybottom),
                        (XleftOrigin, Ybottom)
                    ])
                )
                # Move down to the next row
                Ytop -= height
                Ybottom -= height

            # Move to the next column
            XleftOrigin += width
            XrightOrigin += width
            
        # Create a GeoDataFrame from the generated polygons
        grid = gpd.GeoDataFrame({'geometry': polygons})
        grid = grid.reset_index(drop=False)  # Add index as ID
        grid.columns = ['id', 'geometry']  # Name the columns
        grid.crs = 'EPSG:3857'  # Set projection
        
        # Save the grid to a parquet file
        out_grid_nm = './MA_grid_bound/base_grid_' + nm + '.parquet'
        grid.to_parquet(out_grid_nm)
        
        # Print how long it took to process
        end = datetime.now()
        print('Duration: {}'.format(end - start))