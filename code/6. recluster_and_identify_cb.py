#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import datetime

# Set working directory to the project directory (adjust as needed)
os.chdir('/Users/byeonghwa/Desktop/all/retail_boundary/github_commit/code')
from tools import *  # Import custom utility functions (e.g., reclustering, recluster_and_assign_att, identify_commercial_boundary)

# Number of clusters (labels) used from the prior training result
nlabel = 58

# Paths to the base training arrays and output clustering results
training_data_path = './data/MA_base_training_array'
out_array_path = './result/MA_output_array/' + str(nlabel)

# Perform reclustering to merge and clean up cluster results
# This creates a recluster_dict mapping old cluster labels to new consolidated groups
recluster_dict = reclustering(training_data_path, out_array_path, nlabel)

# Loop through each city's clustering output array
for (path, dirs, files) in os.walk(out_array_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if (ext == '.gz'):  # Process only compressed cluster output files
            input_file = "%s/%s" % (path, fname)
            city = fname.split('_')[2].split('.')[0]  # Extract city name from filename
            start = datetime.now()
            print(city)
            
            # Apply reclustering and assign attributes to each cluster in this city
            recluster_and_assign_att(input_file, city, recluster_dict)
            
            # Identify the final commercial boundary cluster (typically cluster '1' after reclustering)
            commercial_bound_recluster = 1
            out_nm = './result/MA_output_grid/' + str(nlabel) + '/MA_output_attri_' + city + '.parquet'
            
            # Mark and save the identified commercial boundaries for this city
            identify_commercial_boundary(out_nm, city, commercial_bound_recluster)
            
            end = datetime.now()
            print('Duration: {}'.format(end - start))  # Log how long processing took for this city