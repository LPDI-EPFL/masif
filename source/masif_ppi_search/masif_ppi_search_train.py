# Header variables and parameters.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import numpy.matlib as matlib
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)

"""
masif_ppi_search_train.py: Entry function to train the MaSIF-search neural network.
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

params = masif_opts["ppi_search"]

binder_rho_wrt_center = np.load(params["cache_dir"] + "/binder_rho_wrt_center.npy")
binder_theta_wrt_center = np.load(params["cache_dir"] + "/binder_theta_wrt_center.npy")
binder_input_feat = np.load(params["cache_dir"] + "/binder_input_feat.npy")
binder_mask = np.load(params["cache_dir"] + "/binder_mask.npy")
binder_input_feat = mask_input_feat(binder_input_feat, params["feat_mask"])

pos_training_idx = (np.load(params["cache_dir"] + "/pos_training_idx.npy")).astype(int)
pos_val_idx = (np.load(params["cache_dir"] + "/pos_val_idx.npy")).astype(int)
pos_test_idx = (np.load(params["cache_dir"] + "/pos_test_idx.npy")).astype(int)
pos_rho_wrt_center = np.load(params["cache_dir"] + "/pos_rho_wrt_center.npy")
pos_theta_wrt_center = np.load(params["cache_dir"] + "/pos_theta_wrt_center.npy")
pos_input_feat = np.load(params["cache_dir"] + "/pos_input_feat.npy")
pos_mask = np.load(params["cache_dir"] + "/pos_mask.npy")
pos_input_feat = mask_input_feat(pos_input_feat, params["feat_mask"])
pos_names = np.load(params["cache_dir"] + "/pos_names.npy")

neg_training_idx = (np.load(params["cache_dir"] + "/neg_training_idx.npy")).astype(int)
neg_val_idx = (np.load(params["cache_dir"] + "/neg_val_idx.npy")).astype(int)
neg_test_idx = (np.load(params["cache_dir"] + "/neg_test_idx.npy")).astype(int)
neg_rho_wrt_center = np.load(params["cache_dir"] + "/neg_rho_wrt_center.npy")
neg_theta_wrt_center = np.load(params["cache_dir"] + "/neg_theta_wrt_center.npy")
neg_input_feat = np.load(params["cache_dir"] + "/neg_input_feat.npy")
neg_mask = np.load(params["cache_dir"] + "/neg_mask.npy")
neg_input_feat = mask_input_feat(neg_input_feat, params["feat_mask"])

if "pids" not in params:
    params["pids"] = ["p1", "p2"]

# Build the neural network model -- Multi channel, slow.
from masif_modules.MaSIF_ppi_search import MaSIF_ppi_search

learning_obj = MaSIF_ppi_search(
    params["max_distance"],
    n_thetas=16,
    n_rhos=5,
    n_rotations=16,
    idx_gpu="/gpu:0",
    feat_mask=params["feat_mask"],
)

# Compute the list of binders.


print(params["feat_mask"])

if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])

# train
from masif_modules.train_ppi_search import train_ppi_search

train_ppi_search(
    learning_obj,
    params,
    binder_rho_wrt_center,
    binder_theta_wrt_center,
    binder_input_feat,
    binder_mask,
    pos_training_idx,
    pos_val_idx,
    pos_test_idx,
    pos_rho_wrt_center,
    pos_theta_wrt_center,
    pos_input_feat,
    pos_mask,
    neg_training_idx,
    neg_val_idx,
    neg_test_idx,
    neg_rho_wrt_center,
    neg_theta_wrt_center,
    neg_input_feat,
    neg_mask,
)

