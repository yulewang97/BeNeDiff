from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

import math_functions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

def zscore_3d(arr, axis=1):
    means = np.mean(arr, axis=(0, 1), keepdims=True)
    stds = np.std(arr, axis=(0, 1), keepdims=True)
    zscored = (arr - means) / stds
    return zscored

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

hdf5_file = '../datasets/mSM36/05-Dec-2017-wpaw/data.hdf5'

region_dict = {}
with h5py.File(hdf5_file, 'r', libver='latest', swmr=True) as file:
    regions = list(file['regions']['indxs_consolidate_lr'].keys())
    for region in regions:
        region_idxs = file['regions']['indxs_consolidate_lr'][region][()]
        print('%s: %i dims' % (region, region_idxs.shape[1]))
        region_dict[region] = region_idxs
region_name = "VIS_L"

supervised_y_weight = 10.0
vae_model_beta_value = 1.0

region_weight_dict = {'VIS_L': supervised_y_weight, "SSp_L": supervised_y_weight, "MOs_L": supervised_y_weight,
                      'VIS_R': supervised_y_weight, "SSp_R": supervised_y_weight, "MOs_R": supervised_y_weight}

neural_file = '../datasets/WFCI/neural_whole_array-05-Dec.npy'
latents_label_file = '../datasets/WFCI/behavioral_labels_psvae_7.npy'

behavior_indices = [0, 1, 3, 4, 5, 6]
seed = 2027
neural_region_array = np.load(neural_file)[:, :, region_dict[region_name][0]]
label_whole_array = np.load(latents_label_file)[:, :, behavior_indices]

print("neural_region_array shape:", neural_region_array.shape)
print("label_whole_array shape:", label_whole_array.shape)

neural_whole_array = zscore_3d(neural_region_array, axis=1)
label_whole_array = zscore_3d(label_whole_array, axis=2)

import os
import datetime
import logging

log_dir = os.path.join('..', 'logger')
ckpts_dir = os.path.join('..', 'checkpoints-rw')
numpy_dir = os.path.join('..', 'numpy_arrays')

X_trial = neural_whole_array
Y_trial = label_whole_array
X_trial_flatten = X_trial.reshape(-1, X_trial.shape[-1])
Y_trial_flatten = Y_trial.reshape(-1, Y_trial.shape[-1])

random_seeds = np.arange(2024, 2025)

x_size = X_trial.shape[-1]
hidden_size = 6 + 2
behave_size = Y_trial.shape[-1]
vae_model_weight_y = region_weight_dict[region_name]
rnn_encoder_lay_num, rnn_decoder_lay_num = 2, 1

def log_results(seed):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    exp_name = 'tc_vae_neural_' + str(hidden_size - 6) + '_yw_' + str(vae_model_weight_y) + '_betav_' + str(vae_model_beta_value) + '_region_' + str(region_name) + '_layer_' + str(rnn_encoder_lay_num) + '_' + str(rnn_decoder_lay_num) + '_seed_' + str(seed)

    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'{exp_name}_{formatted_date}.log'
    full_path = os.path.join(log_dir, filename)

    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')

    file_handler = logging.FileHandler(full_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, ckpts_dir, exp_name

def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, is_mss=False):
    batch_size, hidden_dim = latent_sample.shape

    log_q_zCx = math_functions.log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    zeros = torch.zeros_like(latent_sample)
    log_pz = math_functions.log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = math_functions.matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        log_iw_mat = math_functions.log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

mse_loss = nn.MSELoss(reduction='sum')

class RNNNeuralDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNNeuralDecoder, self).__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            self.rnn,
            self.fc
        )

    def forward(self, input, hidden=None):
        output = self.layers[0](input)
        output = self.layers[1](output)

        output, hidden = self.rnn(output, hidden)

        output = self.fc(output)
        return output

class RNN_VAE_Model(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, vae_model_beta_value, rnn_encoder_lay_num, rnn_decoder_lay_num):
        super(RNN_VAE_Model, self).__init__()
        
        self.x_dim = x_dim
        self.rnn_encoder_lay_num = rnn_encoder_lay_num
        self.rnn_decoder_lay_num = rnn_decoder_lay_num

        layers = []
        prev_size = x_dim
        hidden_sizes = [x_dim]
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.GRU(x_dim, x_dim, self.rnn_encoder_lay_num, batch_first=True))
        
        self.encoder = nn.Sequential(*layers)

        self.z_mean_WT = nn.Linear(x_dim, z_dim, bias=True)
        self.z_log_var_WT = nn.Linear(x_dim, z_dim, bias=True)

        self.BT = nn.Linear(z_dim, x_dim, bias=False)

        self.label_dim_n = Y_trial.shape[-1]
        self.WT = nn.Parameter(torch.Tensor(self.label_dim_n))

        self.beta_kl = vae_model_beta_value
        self.weight_x = 2. * 20 / x_dim
        self.weight_y = vae_model_weight_y

        self.neural_decoder = RNNNeuralDecoder(z_dim, x_dim, x_dim, num_layers=self.rnn_decoder_lay_num)

        self.behavior_decoder = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, y_dim)
        )

    def encode(self, x):
        _x, _ = self.encoder(x)
        z_mean = self.z_mean_WT(_x)
        z_log_var = self.z_log_var_WT(_x)

        return z_mean, z_log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_hat = self.neural_decoder(z)
        y_hat = z[:, :, :6].mul(self.WT)
        return x_hat, y_hat

    def forward(self, x, phase):
        mu, logvar = self.encode(x)
        z = mu if phase != 'Train' else self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar
    
    def tc_vae_loss_func(self, x_hat, y_hat, x, y, z, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = self.beta_kl * kl_div

        mse_x = self.weight_x * mse_loss(x_hat, x)
        mse_y = self.weight_y * mse_loss(y_hat, y)

        log_pz, log_qz, log_prod_qzi, log_q_zCx
