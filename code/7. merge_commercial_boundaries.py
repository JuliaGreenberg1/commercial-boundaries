#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Path where individual identified commercial boundary (CB) files are stored
cb_path = 'put your path'

# Initialize an empty GeoDataFrame to store combined commercial boundaries
all_cb = gpd.GeoDataFrame()

# Loop through each file in the CB directory and combine them
for (path, dirs, files) in os.walk(cb_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if ext == '.parquet':
            input_file = "%s/%s" % (path, fname)
            print(input_file)
            temp_cb = gpd.read_parquet(input_file)
            # Drop columns not needed for final merged output
            temp_cb = temp_cb.drop(['out_bound', 'out_bound_mean'], axis=1)
            all_cb = pd.concat([all_cb, temp_cb])

# Reset index and calculate area for each polygon
all_cb = all_cb.reset_index(drop=True)
all_cb['Area(m2)'] = all_cb.geometry.area

# Start process of removing smaller overlapping polygons
start = datetime.now()

# Slightly shrink geometries to avoid shared boundary overlap artifacts
all_cb.geometry = all_cb.geometry.buffer(-0.1)

# Find overlaps and mark smaller polygons for removal
to_remove = []
for i, row in tqdm(all_cb.iterrows()):
    for j, other_row in all_cb.iterrows():
        if i != j and row.geometry.intersects(other_row.geometry):
            if row.geometry.area < other_row.geometry.area:
                to_remove.append(i)
                break

end = datetime.now()
print('Duration: {}'.format(end - start))

# Drop the smaller, overlapping polygons
all_cb = all_cb.drop(to_remove)

# Re-expand geometries to their original size
all_cb.geometry = all_cb.geometry.buffer(0.1)

# Reset index and assign unique IDs to each commercial boundary
all_cb = all_cb.reset_index(drop=True)
all_cb = all_cb.reset_index(drop=False)
all_cb = all_cb.rename({'index': 'CB_ID'}, axis=1)

# Save the cleaned, merged commercial boundaries dataset
out_nm = 'put your path'
all_cb.to_parquet(out_nm)