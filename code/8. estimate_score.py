#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:30:36 2024

@author: byeonghwa
"""

import numpy as np
import geopandas as gpd
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the merged commercial boundary dataset
cb = gpd.read_parquet('put_your_merged_cb_results')

# Keep only necessary columns for scoring
cb = cb[['index', 'cluster', 'geometry', 'retailers', 'office', 'jobs', 'area']]

# Define variables to use for the commercial score
sel_list = ['retailers', 'office', 'jobs']

# Apply log transformation to reduce skew and compress large values
for sel in sel_list:
    cb[sel] = cb[sel].apply(lambda x: np.log1p(x))

# Standardize features to zero mean and unit variance
ss = StandardScaler()
cb[sel_list] = ss.fit_transform(cb[sel_list])

# Perform PCA to reduce the three features into a single commercial score dimension
pca = PCA(n_components=1)
pca.fit(cb[sel_list])
cb['commercial_score'] = pca.transform(cb[sel_list])

# Print minimum and maximum of the raw PCA scores for inspection
print(cb['commercial_score'].min(), cb['commercial_score'].max())

# Rescale the commercial score between 1 and 10 for interpretability
mm = MinMaxScaler(feature_range=(1, 10))
cb['commercial_score'] = mm.fit_transform(cb[['commercial_score']])

# Print PCA explained variance ratio and singular values to understand feature contribution
print('pca_ratio', pca.explained_variance_ratio_)
print('pca_value', pca.singular_values_)

# Plot a histogram of the commercial scores to visualize their distribution
plt.figure(figsize=(10, 6))
plt.hist(cb['commercial_score'], bins=20, color='Orange', alpha=0.5)
plt.xlabel('Commercial Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("put_your_output_figure", dpi=400)
plt.show()

# Keep only essential columns with the computed score
cb = cb[['index', 'geometry', 'area', 'commercial_score']]

# Save the processed commercial boundaries with scores to a parquet file
cb.to_parquet('put_your_processed_cb_results_dir')