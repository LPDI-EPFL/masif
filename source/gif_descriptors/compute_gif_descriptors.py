import os
import numpy as np
from IPython.core.debugger import set_trace
from default_config.masif_opts import masif_opts
import sys

# This code computes Geometric Invariant Fingerprint descriptors for full proteins as originally proposed in:
# S. Yin, et al. PNAS September 29, 2009 106 (39) 16622-16626
# Implemented by Pablo Gainza LPDI - EPFL 2016-2019
# Released under an Apache License 2.0

# Compute the histogram of distance-dependent curvature.
def compute_dfss_histogram(rho, ddc, mask):
    # Bin it
    histogram = np.zeros((4, 15))
    alive = np.where(mask > 0)[0]
    kij = ddc[alive]
    D = rho[alive]

    # GIF descriptors were proposed for patches of radius 9A.
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


out_dir = masif_opts["ppi_search"]["gif_descriptors_out"]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

list_fn = sys.argv[1]
list_file = open(list_fn).readlines()

precomputation_dir = masif_opts["ppi_search"]["masif_precomputation_dir"]
for ppi_pair_id in list_file:
    ppi_pair_id = ppi_pair_id.rstrip()
    mydir = os.path.join(precomputation_dir, ppi_pair_id)
    for pid in ["p1", "p2"]:
        # The distance dependent curvature is stored in the second dimension of input features.
        # All other features are ignored.
        try:
            feat = np.load(mydir + "/" + pid + "_input_feat.npy")[:, :, 1]
            mask = np.load(mydir + "/" + pid + "_mask.npy")
            rho = np.load(mydir + "/" + pid + "_rho_wrt_center.npy")
        except:
            print("Error opening {}".format(ppi_pair_id))
            continue
        myoutdir = os.path.join(out_dir, ppi_pair_id)
        if not os.path.exists(myoutdir):
            os.mkdir(myoutdir)

        # Compute histograms for all descriptors.
        all_desc_straight = []
        all_desc_flipped = []
        for i in range(len(mask)):
            desc_straight = compute_dfss_histogram(rho[i], feat[i], mask[i])
            desc_flipped = compute_dfss_histogram(rho[i], -feat[i], mask[i])
            all_desc_straight.append(desc_straight)
            all_desc_flipped.append(desc_flipped)
        np.save(myoutdir + "/" + pid + "_desc_straight", all_desc_straight)
        np.save(myoutdir + "/" + pid + "_desc_flipped", all_desc_flipped)
