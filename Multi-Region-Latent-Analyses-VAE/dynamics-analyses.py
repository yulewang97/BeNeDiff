import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

def zscore_3d(arr):
    means = np.mean(arr, axis=(0, 1), keepdims=True)
    stds = np.std(arr, axis=(0, 1), keepdims=True)
    zscored = (arr - means) / stds
    return zscored

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

hdf5_file = '../datasets/mSM36/05-Dec-2017-wpaw/data.hdf5'
numpy_dir = os.path.join('..', 'numpy_arrays')

region_dict = {}
with h5py.File(hdf5_file, 'r', libver='latest', swmr=True) as file:
    regions = list(file['regions']['indxs_consolidate_lr'].keys())
    for region in regions:
        region_idxs = file['regions']['indxs_consolidate_lr'][region][()]
        print('%s: %i dims' % (region, region_idxs.shape[1]))
        region_dict[region] = region_idxs

region_name = "VIS_R"
neural_file = '../datasets/WFCI/neural_whole_array-05-Dec.npy'
latents_label_file = '../datasets/WFCI/behavioral_labels_psvae_7.npy'
behavior_indices = [0, 1, 3, 4, 5, 6]
seed = 2027
neural_region_array = np.load(neural_file)[:, :, region_dict[region_name][0]]
label_whole_array = np.load(latents_label_file)[:, :, behavior_indices]

print(region_name + " neural_region_array shape:", neural_region_array.shape)
print("label_whole_array shape:", label_whole_array.shape)

neural_whole_array = zscore_3d(neural_region_array)
label_whole_array = zscore_3d(label_whole_array)

import numpy as np

neural_data = np.random.rand(100, 50, 200)

mean = np.mean(neural_data, axis=(0, 1), keepdims=True)
std = np.std(neural_data, axis=(0, 1), keepdims=True)
zscored_neural_data = (neural_data - mean) / std
print(mean.shape)

trial_number, trial_length, neuron_number = neural_whole_array.shape
trial_number, trial_length, label_number = label_whole_array.shape

neural_trial_to2d_array = neural_whole_array.reshape(trial_number * trial_length, neuron_number)
label_trial_to2d_array = label_whole_array.reshape(trial_number * trial_length, label_number)

from scipy.ndimage import gaussian_filter

numpy_dir = os.path.join('..', 'numpy_arrays')

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

fetched_trial_id = 145

paw_ids = np.array([2, 3])
sigma = 0.25

ssp_l_latents = np.load(os.path.join(numpy_dir, 'zscore1_pure_neural_6_yw_50.0_region_SSp_L_seed_2025_model.pth_latents.npy')).reshape(trial_number, trial_length, -1)
ssp_r_latents = np.load(os.path.join(numpy_dir, 'zscore1_pure_neural_6_yw_100.0_region_SSp_R_seed_2024_model.pth_latents.npy')).reshape(trial_number, trial_length, -1)
behavior_labels_paw = label_whole_array[:, :, paw_ids]

fetched_behavior_labels = label_whole_array[fetched_trial_id, :, paw_ids]
fetched_ssp_l_latents = ssp_l_latents[fetched_trial_id, :, paw_ids]
fetched_ssp_r_latents = ssp_r_latents[fetched_trial_id, :, paw_ids]

def affine_transform(X, Y):
    n, d = X.shape

    X_h = np.hstack([X, np.ones((n, 1))])

    def residuals(params):
        A = params[:d*d].reshape((d, d))
        t = params[d*d:].reshape((d, 1))
        Y_pred = (A @ X.T + t).T
        return (Y - Y_pred).flatten()

    initial_params = np.hstack([np.eye(d).flatten(), np.zeros(d)])

    result = least_squares(residuals, initial_params)

    A = result.x[:d*d].reshape((d, d))
    t = result.x[d*d:].reshape((d, 1))
    return A, t

X = fetched_ssp_l_latents
Y = fetched_ssp_r_latents

A, t = affine_transform(X, Y)

print("Affine Transformation Matrix (A):")
print(A)

print("Translation Vector (t):")
print(t)

X_transformed = (A @ X.T + t).T

print("Transformed Source Trajectory (X_transformed):")
print(X_transformed)
