#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextily as ctx
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from shapely.geometry import box
import os

# Create bounding boxes around each metropolitan area
box_ma = gpd.GeoDataFrame()
ma_base = gpd.read_parquet('./base_data/top_50_ma_us_ca.parquet')
ma_base = ma_base.to_crs('EPSG:3857')

for i in ma_base.index:
    temp_ma = ma_base.loc[ma_base.index == i]
    city = temp_ma['NAME'].values[0]
    bbox_polygon = gpd.GeoDataFrame([city], index=[0], columns=['NAME'], crs='epsg:3857', geometry=[box(*temp_ma.total_bounds)])
    box_ma = pd.concat([box_ma, bbox_polygon])

box_ma = box_ma.reset_index(drop=True)
box_ma = gpd.GeoDataFrame(box_ma, geometry=box_ma.geometry)

# Change working directory
os.chdir('put your path')

# Load validation datasets:
# 1. Downtown boundaries (excluding Honolulu)
base_downtown = gpd.read_parquet('./validation_data/downtown_recovery_bound.parquet')
base_downtown = base_downtown.to_crs('EPSG:3857')
base_downtown = base_downtown.loc[base_downtown['city'] != 'Honolulu']

# 2. Brookings commercial core and CBD boundaries (excluding Honolulu)
brookings = gpd.read_parquet('./validation_data/brookings_cc_cbd.parquet')
brookings = brookings.loc[brookings['CBSA_NAME'] != 'Urban Honolulu, HI']
brookings = brookings.to_crs('EPSG:3857')
brookings_cc = brookings.loc[brookings['DTWN_TYPE'] == 'Commercial Core'].reset_index(drop=True)
brookings_cbd = brookings.loc[brookings['DTWN_TYPE'] == 'Central Business District'].reset_index(drop=True)

# 3. CDRC US Retail Centre dataset
cdrc_bound = gpd.read_file('./validation_data/US_RetailCentres.gpkg')
cdrc_bound = cdrc_bound.to_crs('EPSG:3857')

# 4. Identified commercial boundaries
cb_bound = gpd.read_file('./MA_commercial_boundary/Commercial_boundaries.gpkg')
cb_bound = cb_bound.to_crs('EPSG:3857')

# ---------------------------------------------------------------------------------------
# Validate coverage between downtown boundaries and identified commercial boundaries
overall_overlap = []
for i in base_downtown.index:
    temp_down = base_downtown.loc[base_downtown.index == i]
    city = temp_down['city'].values[0]
    clip_downtown = gpd.clip(cb_bound, temp_down)
    clip_downtown['area'] = clip_downtown.geometry.area
    temp_list = [city, temp_down.geometry.area.values[0], clip_downtown['area'].sum()]
    overall_overlap.append(temp_list)

down_val_df = pd.DataFrame(overall_overlap, columns=['city', 'area_down', 'area_cb'])
down_val_df['ratio'] = down_val_df['area_cb'] / down_val_df['area_down']
canada_dt = ['Calgary', 'Edmonton', 'Halifax', 'Mississauga', 'Montreal',
             'Ottawa', 'Quebec', 'Toronto', 'Vancouver', 'Winnipeg', 'London']
down_val_df['Country'] = down_val_df['city'].apply(lambda x: 'CA' if x in canada_dt else 'US')
print(down_val_df['area_down'].sum() / 1_000_000, down_val_df['area_cb'].sum() / 1_000_000,
      down_val_df['area_cb'].sum() / down_val_df['area_down'].sum())
down_val_df.to_csv('./validation_data/area_ratio_downtwon.csv', index=False)

# ---------------------------------------------------------------------------------------
# Validate coverage between CDRC retail centres and identified commercial boundaries
clip_cb_bound = gpd.clip(cb_bound, cdrc_bound)
clip_cb_bound['area'] = clip_cb_bound.geometry.area
cdrc_bound['area'] = cdrc_bound.geometry.area
cdrc_bound_sel = gpd.sjoin(cdrc_bound, box_ma)
cdrc_ratio = {
    'cdrc_area': [cdrc_bound['area'].sum()],
    'cdrc_sel_area': [cdrc_bound_sel['area'].sum()],
    'clip_cb_area': [clip_cb_bound['area'].sum()],
}
cdrc_ratio = pd.DataFrame(cdrc_ratio)
cdrc_ratio['ratio'] = cdrc_ratio['clip_cb_area'] / cdrc_ratio['cdrc_area']
cdrc_ratio['ratio_sel'] = cdrc_ratio['clip_cb_area'] / cdrc_ratio['cdrc_sel_area']
cdrc_ratio.to_csv('./validation_data/area_ratio_cdrc.csv', index=False)

# ---------------------------------------------------------------------------------------
# Validate coverage per city between CDRC retail centres and commercial boundaries
overall_overlap_cdrc = []
for i in box_ma.index:
    temp_ma = box_ma.loc[box_ma.index == i]
    city = temp_ma['NAME'].values[0]
    if city in canada_dt:
        continue
    sel_cdrc = gpd.sjoin(cdrc_bound, temp_ma)
    sel_cb = gpd.sjoin(cb_bound, temp_ma)
    clip_cb = gpd.clip(sel_cb, sel_cdrc)
    clip_cb['area'] = clip_cb.geometry.area
    sel_cdrc['area'] = sel_cdrc.geometry.area
    temp_list = [city, sel_cdrc['area'].sum(), clip_cb['area'].sum()]
    overall_overlap_cdrc.append(temp_list)

overall_overlap_cdrc_df = pd.DataFrame(overall_overlap_cdrc, columns=['city', 'area_cdrc', 'area_cb'])
overall_overlap_cdrc_df['ratio'] = overall_overlap_cdrc_df['area_cb'] / overall_overlap_cdrc_df['area_cdrc']
overall_overlap_cdrc_df.to_csv('./validation_data/area_ratio_cdrc_by_city.csv', index=False)

# ---------------------------------------------------------------------------------------
# Validate coverage between Brookings commercial core and identified commercial boundaries
overall_overlap_brook_cc = []
for i in brookings_cc.index:
    temp_cc = brookings_cc.loc[brookings_cc.index == i]
    city = temp_cc['DTWN_NAME'].values[0].split(',')[0]
    clip_downtown = gpd.clip(cb_bound, temp_cc)
    clip_downtown['area'] = clip_downtown.geometry.area
    temp_list = [city, temp_cc.geometry.area.values[0], clip_downtown['area'].sum()]
    overall_overlap_brook_cc.append(temp_list)

down_val_boork_cc_df = pd.DataFrame(overall_overlap_brook_cc, columns=['city', 'area_brook', 'area_cb'])
down_val_boork_cc_df['ratio'] = down_val_boork_cc_df['area_cb'] / down_val_boork_cc_df['area_brook']
print(down_val_boork_cc_df['area_brook'].sum() / 1_000_000, down_val_boork_cc_df['area_cb'].sum() / 1_000_000,
      down_val_boork_cc_df['area_cb'].sum() / down_val_boork_cc_df['area_brook'].sum())
down_val_boork_cc_df.to_csv('./validation_data/area_ratio_brookings_cc.csv', index=False)

# ---------------------------------------------------------------------------------------
# Validate coverage between Brookings CBD and identified commercial boundaries
overall_overlap_brook_cbd = []
for i in brookings_cbd.index:
    temp_cbd = brookings_cbd.loc[brookings_cbd.index == i]
    city = temp_cbd['DTWN_NAME'].values[0].split(',')[0]
    clip_downtown = gpd.clip(cb_bound, temp_cbd)
    clip_downtown['area'] = clip_downtown.geometry.area
    temp_list = [city, temp_cbd.geometry.area.values[0], clip_downtown['area'].sum()]
    overall_overlap_brook_cbd.append(temp_list)

down_val_boork_cbd_df = pd.DataFrame(overall_overlap_brook_cbd, columns=['city', 'area_brook', 'area_cb'])
down_val_boork_cbd_df = pd.pivot_table(down_val_boork_cbd_df, index='city', values=['area_brook', 'area_cb'], aggfunc='sum')
down_val_boork_cbd_df = down_val_boork_cbd_df.reset_index(drop=False)
down_val_boork_cbd_df['ratio'] = down_val_boork_cbd_df['area_cb'] / down_val_boork_cbd_df['area_brook']
print(down_val_boork_cbd_df['area_brook'].sum() / 1_000_000, down_val_boork_cbd_df['area_cb'].sum() / 1_000_000,
      down_val_boork_cbd_df['area_cb'].sum() / down_val_boork_cbd_df['area_brook'].sum())
down_val_boork_cbd_df.to_csv('./validation_data/area_ratio_brookings_cbd.csv', index=False)

# ---------------------------------------------------------------------------------------
# Generate visual maps for validation by city
for i in base_downtown.index:
    temp_down = base_downtown.loc[base_downtown.index == i]
    city = temp_down['city'].values[0]
    temp_brok_cc = gpd.sjoin(brookings_cc, temp_down)
    temp_brok_cbd = gpd.sjoin(brookings_cbd, temp_down)

    fig, ax = plt.subplots(figsize=(10, 10))
    extent = temp_brok_cc.total_bounds if len(temp_brok_cc) > 0 else temp_down.total_bounds
    buffer_value = 500
    extent[0] -= buffer_value
    extent[1] -= buffer_value
    extent[2] += buffer_value
    extent[3] += buffer_value

    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])

    temp_down.boundary.plot(ax=ax, color='red', linewidth=1.2, label='Downtown Boundary')
    cdrc_bound.plot(ax=ax, color='none', hatch='\\\\', label='CDRC US Retail Centre Boundaries')
    cb_bound.plot(ax=ax, color='#003F5B', alpha=0.65, label='Commercial Boundaries')

    if len(temp_brok_cc) > 0:
        temp_brok_cc.boundary.plot(ax=ax, color='#FFA602', linewidth=2, label='Brookings Commercial Core Boundary')            
        temp_brok_cbd.boundary.plot(ax=ax, color='#a5fa50', linewidth=2, label='Brookings Central Business District Boundary')
        LegendElement = [
            Line2D([0], [0], color='red', lw=2, label='Downtown Boundary'),
            Line2D([0], [0], color='#FFA602', lw=2, label='Brookings Commercial Core Boundary'),
            Line2D([0], [0], color='#a5fa50', lw=2, label='Brookings Central Business District Boundary'),
            mpatches.Patch(edgecolor='black', facecolor='white', hatch='\\\\', label='CDRC US Retail Centre Boundaries'),
            mpatches.Patch(color='#003F5B', label='Commercial Boundaries', alpha=0.5),
        ]
    else:
        LegendElement = [
            Line2D([0], [0], color='red', lw=2, label='Downtown Boundary'),
            mpatches.Patch(edgecolor='black', facecolor='white', hatch='\\\\', label='CDRC US Retail Centre Boundaries'),
            mpatches.Patch(color='#003F5B', label='Commercial Boundaries', alpha=0.5),
        ]

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.legend(handles=LegendElement, loc='upper right')
    plt.title(city)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()