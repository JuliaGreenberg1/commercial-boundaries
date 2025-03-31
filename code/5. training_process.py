#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import os
from tools import *  # Custom module with helper functions (e.g., load_gz, save_gz, crop_array)
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader
import model  # Custom PyTorch model definition (MyNet)

# Set computing device: prioritize CUDA, then MPS (Apple), else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else "cpu")
print(device)

# Set random seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Path to the preprocessed training arrays (one per metro area)
training_data_path = './data/MA_base_training_array'

# Parameters for cropping and training
num_example = 1000
window_size = 100
batch_size = 32

###############################################################################
# Create random weighted crops from each city's array
mm = MinMaxScaler(feature_range=(10, 1000))  # Scale file sizes to control crop weighting
size_dict = {}

# Measure file sizes (to weight sampling)
for (path, dirs, files) in os.walk(training_data_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if ext == '.gz':
            input_file = "%s/%s" % (path, fname)
            print(input_file)
            city = fname.split('_')[-1].split('.')[0]
            size_dict[city] = os.path.getsize(input_file)

# Scale file sizes and convert to integer weights
size_df = pd.DataFrame([size_dict]).T
size_df.columns = ['size']
size_df[['size']] = mm.fit_transform(size_df[['size']])
size_df['size'] = size_df['size'].astype(int)
size_dict = size_df.to_dict()['size']
print(size_dict)

# Perform weighted random cropping for each city's training array
i = 0
for (path, dirs, files) in os.walk(training_data_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if ext == '.gz':
            input_file = "%s/%s" % (path, fname)
            print(input_file)
            city = fname.split('_')[-1].split('.')[0]
            fin_out = load_gz(input_file)  # Load compressed numpy array
            fin_out = local_minmax_transform(fin_out)  # Normalize values locally
            print(fin_out.shape)

            # Generate random crops (patches)
            temp_crop = crop_array(fin_out, size_dict[city], window_size)

            # Concatenate crops from all cities
            if i == 0:
                ran_crop = temp_crop.copy()
                i += 1
            else:
                ran_crop = np.vstack([ran_crop, temp_crop])

print(ran_crop.shape)
# Save the combined random crops to file
out_nm = './data/random_crop/local_minmax_random_crop_weight_100.npy.gz'
save_gz(out_nm, ran_crop)

##############################################################################
# Create a PyTorch tensor and DataLoader for training
data = torch.Tensor(ran_crop).to(device)
print(data.shape)

training_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

##############################################################################
# Train the model with hyperparameter grid search

# Hyperparameters setup
input_dim = data.shape[1]
nChannel = 64
lr_list = [0.1, 0.01, 0.001]
stepsize_sim = 1
step_con_list = [1, 5, 10, 50]
epochs_list = [3, 4, 5]
minLabels = 3

results = {}  # To store results from each hyperparameter combination

# Grid search loop
for epochs in epochs_list:
    for learning_rate in lr_list:
        for stepsize_con in step_con_list:
            # Define loss functions
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_hpy = torch.nn.L1Loss(size_average=True)  # Horizontal continuity loss
            loss_hpz = torch.nn.L1Loss(size_average=True)  # Vertical continuity loss
            
            # Create continuity loss target tensors
            HPy_target = torch.zeros(ran_crop.shape[-2] - 1, ran_crop.shape[-1], nChannel, batch_size).to(device)
            HPz_target = torch.zeros(ran_crop.shape[-2], ran_crop.shape[-1] - 1, nChannel, batch_size).to(device)
            
            # Initialize model
            model_net = model.MyNet(input_dim, nChannel).to(device)
            optimizer = torch.optim.SGD(model_net.parameters(), lr=learning_rate, momentum=0.9)
            
            a = 0
            repeat = 0
            losses_train = []

            # Training loop
            for e in range(epochs):
                for batch_idx, data in enumerate(tqdm(training_loader)):
                    optimizer.zero_grad()
                    output = model_net(data).to(device)
                    output = output.permute(2, 3, 1, 0).contiguous().view(-1, nChannel)

                    # Compute continuity losses between adjacent pixels
                    outputHP = output.reshape((data.shape[-2], data.shape[-1], nChannel, batch_size))
                    HPy = outputHP[1:, :, :, :] - outputHP[0:-1, :, :, :]
                    HPz = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
                    lhpy = loss_hpy(HPy, HPy_target)
                    lhpz = loss_hpz(HPz, HPz_target)

                    # Compute similarity loss
                    ignore, target = torch.max(output, 1)
                    im_target = target.data.cpu().numpy()
                    nLabels = len(np.unique(im_target))

                    # Total loss combines similarity and continuity terms
                    loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)

                    loss.backward()
                    optimizer.step()
                    losses_train.append(loss.item())

                    # Early stopping if labels collapse
                    if nLabels <= minLabels:
                        print("nLabels", nLabels, "reached minLabels", minLabels, ".")
                        break

                print(f"{e}/{epochs} | label num: {nLabels} | loss: {loss.item()}")
                if a == 0:
                    repeat_labels = nLabels
                    a += 1
                else:
                    if repeat_labels == nLabels:
                        repeat += 1
                    else:
                        repeat_labels = nLabels
                        repeat = 0

                if repeat == 2:  # Stop if label counts stabilize
                    break

            # Save trained model and training losses
            print('Saving results')
            model_weights_svname = f'./saved/MA_colab_model_weight_{nLabels}_{stepsize_con}.pth'
            torch.save(model_net.state_dict(), model_weights_svname)

            loss_df = pd.DataFrame(losses_train, columns=['loss'])
            loss_df.to_csv(f'./saved/MA_colab_training_los_{nLabels}_{stepsize_con}.csv', index=False, sep=',')

            results[(epochs, learning_rate, stepsize_con, nLabels)] = loss.item()

# Organize grid search results
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Loss'])
results_df = results_df.reset_index(drop=False).rename({'index': 'parameters'}, axis=1)
results_df['Loss_rank'] = results_df['RMSE'].rank(method='min', ascending=True)
results_df = results_df.sort_values(by='Loss_rank', ascending=True).reset_index(drop=True)
final_parameter = results_df.at[0, 'parameters']

##############################################################################
# Prediction on each city's data using the best model

nLabels = final_parameter[-1]
for (path, dirs, files) in os.walk(training_data_path):
    for fname in files:
        ext = os.path.splitext(fname)[-1]
        if ext == '.gz':
            input_file = "%s/%s" % (path, fname)
            city = fname.split('_')[1].split('.')[0]

            out_nm = f'./result/MA_output_array/{nLabels}/MA_output_{city}_{nLabels}_{stepsize_con}.gz'

            print(input_file)
            model_net = model.MyNet(input_dim, nChannel).to(device)
            model_net.load_state_dict(torch.load(f'./saved/MA_colab_model_weight_{nLabels}_{stepsize_con}.pth'))

            fin_out = load_gz(input_file)
            fin_out = local_minmax_transform(fin_out)
            print(fin_out.shape)

            all_data = torch.Tensor(np.expand_dims(fin_out, axis=0)).to(device)
            with torch.no_grad():
                model_net.eval()
                output = model_net(all_data).to(device)
                output = output.permute(2, 3, 1, 0).contiguous().view(-1, nChannel)
                ignore, target = torch.max(output, 1)
                im_target = target.data.cpu().numpy()
                im_target = im_target.reshape(fin_out.shape[-2], fin_out.shape[-1])

            save_gz(out_nm, im_target)

            # Save cluster map visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(im_target)
            plt.tight_layout()
            plt.savefig(f'./result/MA_cluster_figure/MA_cluster_map_{city}_{nLabels}_{stepsize_con}.png', dpi=400)
            plt.close()