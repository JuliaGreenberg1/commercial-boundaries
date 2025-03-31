#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import geopandas as gpd
from datetime import datetime
from tools import *  # Custom module containing functions like convert_array() and save_gz()

# Path where the training grid data (with assigned variables) is stored
training_data_path = 'put your path'

# List of variables of interest to extract from the grid files
var_list = ['retailers', 'office', 'jobs']

# Loop through each training grid parquet file in the specified directory
for (path, dirs, files) in os.walk(training_data_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.parquet'):  # Process only parquet files
            input_file = "%s/%s" % (path, fname)
            city = fname.split('_')[0]  # Extract the city name from the filename
            print(input_file)
            
            start = datetime.now()  # Start timing for each city

            # Convert the grid data into a structured NumPy array with the selected variables
            fin_out = convert_array(input_file, var_list)  # convert_array() is likely a custom function from tools.py
            
            # Define the output filename for the compressed numpy array
            np_nm = './data/MA_base_training_array/array_' + city + '.npy.gz'
            
            # Save the resulting numpy array in compressed format
            save_gz(np_nm, fin_out)  # save_gz() is a custom function from tools.py
            
            end = datetime.now()
            print('Duration: {}'.format(end - start))  # Print how long processing took for this city