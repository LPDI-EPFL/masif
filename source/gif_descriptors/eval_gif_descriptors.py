# all_test_to_all_train.sh: Evaluate Geometric Invariant Fingerprint descriptors 
#  (descriptors by Yin. et al. PNAS 2019) for comparison to MaSIF. The input used is the training data.
# Pablo Gainza - LPDI STI EPFL 2019
# Released under an Apache License 2.0

import os
import numpy as np
import sys
import importlib
import ipdb
from sklearn import metrics
from default_config.masif_opts import masif_opts


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)


# Evaluate the yin pnas descriptors using the same dataset as MaSIF descriptors.
# Compute the distance dependent histogram of normals.
def compute_dfss_histogram(rho, ddc, mask):
    # Bin it
    histogram = np.zeros((4, 15))
    alive = np.where(mask > 0)[0]
    kij = ddc[alive]
    D = rho[alive]

    for j in range(len(kij)):
        if D[j] < 1 or D[j] >= 9:
            continue
        if kij[j] < -0.7 or kij[j] > 0.7:
            continue
        radial_bin = np.int((D[j] - 1) // 2)
        curv_bin = np.int(np.floor(10 * (kij[j] + 0.7)))
        histogram[radial_bin][curv_bin] += 1
    # Reshape the histogram to 1 dimension:
    histogram = np.reshape(histogram, -1)
    # Normalize the histogram.
    norm = np.linalg.norm(histogram)
    histogram = histogram / norm

    return histogram


params = masif_opts["ppi_search"]

custom_params_file = sys.argv[1]
custom_params = importlib.import_module(custom_params_file, package=None)
custom_params = custom_params.custom_params

for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]

out_dir = params["gif_eval_out"]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

test_idx = np.load(params["cache_dir"] + "/pos_test_idx.npy")
pos_feat = np.load(params["cache_dir"] + "/pos_input_feat.npy")[:, :, 1]
pos_mask = np.load(params["cache_dir"] + "/pos_mask.npy")
pos_rho = np.load(params["cache_dir"] + "/pos_rho_wrt_center.npy")

binder_feat = np.load(params["cache_dir"] + "/binder_input_feat.npy")[:, :, 1]
binder_mask = np.load(params["cache_dir"] + "/binder_mask.npy")
binder_rho = np.load(params["cache_dir"] + "/binder_rho_wrt_center.npy")

neg_feat = np.load(params["cache_dir"] + "/neg_input_feat.npy")[:, :, 1]
neg_mask = np.load(params["cache_dir"] + "/neg_mask.npy")
neg_rho = np.load(params["cache_dir"] + "/neg_rho_wrt_center.npy")

# Compute histograms for positives.

pos_dists = []
for idx in test_idx:
    desc1 = compute_dfss_histogram(pos_rho[idx], pos_feat[idx], pos_mask[idx])
    desc2 = compute_dfss_histogram(binder_rho[idx], -binder_feat[idx], binder_mask[idx])
    dist = np.linalg.norm(desc1 - desc2)
    pos_dists.append(dist)

neg_dists = []
test_idx2 = test_idx.copy()
np.random.shuffle(test_idx2)
for ix, idx in enumerate(test_idx):
    idx2 = test_idx2[ix]
    desc1 = compute_dfss_histogram(neg_rho[idx], neg_feat[idx], neg_mask[idx])
    # For second escriptor is the binder with a negated input, as described in the paper. 
    desc2 = compute_dfss_histogram(binder_rho[idx], -binder_feat[idx], binder_mask[idx])
    dist = np.linalg.norm(desc1 - desc2)
    neg_dists.append(dist)

auc = compute_roc_auc(pos_dists, neg_dists)
print("GIF descriptors ROC AUC: {}".format(1 - auc))
np.save(out_dir + "/pos_dists.npy", pos_dists)
np.save(out_dir + "/neg_dists.npy", neg_dists)
