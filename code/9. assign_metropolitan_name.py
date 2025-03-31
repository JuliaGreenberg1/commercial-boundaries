#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd

# Load the top 50 metropolitan areas (MSAs) and commercial boundary (CB) results
msa_df = gpd.read_parquet('./base_data/top_50_ma_us_ca.parquet')
cb_df = gpd.read_parquet('put_your_cb_result')

# Reproject both datasets to EPSG:3857 for spatial operations
msa_df = msa_df.to_crs('EPSG:3857')
cb_df = cb_df.to_crs('EPSG:3857')

# Rename columns in CB data for consistency
cb_df.columns = ['CB_ID', 'geometry', 'Area(m2)', 'Commercial_score']

# Perform spatial intersection to assign each CB polygon to an MSA it overlaps with
inter = gpd.overlay(cb_df, msa_df, how='intersection')

# Calculate the intersection area to resolve overlaps
inter['area'] = inter.geometry.area

# Sort by intersection area so that the largest intersection is retained in case of duplicates
inter.sort_values(by='area', inplace=True)
inter.drop_duplicates(subset='CB_ID', keep='last', inplace=True)
inter.drop(columns=['area'], inplace=True)

# Identify commercial boundaries that did not intersect any MSA
out_bound_id = list(inter['CB_ID'].unique().tolist())
out_df = cb_df.loc[~cb_df['CB_ID'].isin(out_bound_id)]

# For those CBs that didn't intersect, assign them to the nearest MSA using spatial join
out_bound_inter = out_df.sjoin_nearest(msa_df)
out_bound_inter = out_bound_inter.drop('index_right', axis=1)

# Combine intersect-based and nearest-assigned results into one dataframe
assign_name = pd.concat([inter, out_bound_inter])
assign_name = assign_name.reset_index(drop=True)

# Ensure GeoDataFrame structure is preserved
assign_name = gpd.GeoDataFrame(assign_name, geometry='geometry')

# Rename columns for final output consistency
assign_name = assign_name.rename({'NAME': 'MSA_NAME',
                                  'STATE': 'STATE/PROVINCE'}, axis=1)

# Check that all CB polygons were assigned an MSA
if len(cb_df) == len(assign_name):
    print('process worked well!')
else:
    print('something wrong')

# Save the final output (commercial boundaries with assigned MSAs) as a GeoPackage
out_nm = 'put your out name'
assign_name.to_file(out_nm, driver='GPKG')