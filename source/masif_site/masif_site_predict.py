# Header variables and parameters.
import time
import os
import numpy as np
from IPython.core.debugger import set_trace
import sys
import importlib
from masif_modules.train_masif_site import run_masif_site
from default_config.masif_opts import masif_opts

"""
masif_site_predict.py: Evaluate one or multiple proteins on MaSIF-site. 
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


params = masif_opts["site"]
custom_params_file = sys.argv[1]
custom_params = importlib.import_module(custom_params_file, package=None)
custom_params = custom_params.custom_params

for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]


# Shape precomputation dir.
parent_in_dir = params["masif_precomputation_dir"]
eval_list = []

if len(sys.argv) == 3:
    ppi_pair_ids = [sys.argv[2]]
# Read a list of pdb_chain entries to evaluate.
elif len(sys.argv) == 4 and sys.argv[2] == "-l":
    listfile = open(sys.argv[3])
    ppi_pair_ids = []
    for line in listfile:
        eval_list.append(line.rstrip())
    for mydir in os.listdir(parent_in_dir):
        ppi_pair_ids.append(mydir)
else:
    sys.exit(1)

# Build the neural network model
from masif_modules.MaSIF_site import MaSIF_site

learning_obj = MaSIF_site(
    params["max_distance"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4,
    idx_gpu="/gpu:0",
    feat_mask=params["feat_mask"],
    n_conv_layers=params["n_conv_layers"],
)
print("Restoring model from: " + params["model_dir"] + "model")
learning_obj.saver.restore(learning_obj.session, params["model_dir"] + "model")

if not os.path.exists(params["out_pred_dir"]):
    os.makedirs(params["out_pred_dir"])

idx_count = 0
for ppi_pair_id in ppi_pair_ids:
    print(ppi_pair_id)
    in_dir = parent_in_dir + ppi_pair_id + "/"

    fields = ppi_pair_id.split('_')
    if len(fields) < 2:
        continue
    pdbid = ppi_pair_id.split("_")[0]
    chain1 = ppi_pair_id.split("_")[1]
    pids = ["p1"]
    chains = [chain1]
    if len(fields) == 3 and fields[2] != "":
        chain2 = fields[2]
        pids = ["p1", "p2"]
        chains = [chain1, chain2]

    for ix, pid in enumerate(pids):
        pdb_chain_id = pdbid + "_" + chains[ix]
        if (
            len(eval_list) > 0
            and pdb_chain_id not in eval_list
            and pdb_chain_id + "_" not in eval_list
        ):
            continue

        print("Evaluating {}".format(pdb_chain_id))

        try:
            rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
        except:
            print("File not found: {}".format(in_dir + pid + "_rho_wrt_center.npy"))
            continue
        theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
        input_feat = np.load(in_dir + pid + "_input_feat.npy")
        input_feat = mask_input_feat(input_feat, params["feat_mask"])
        mask = np.load(in_dir + pid + "_mask.npy")
        indices = np.load(in_dir + pid + "_list_indices.npy", encoding="latin1", allow_pickle=True)
        labels = np.zeros((len(mask)))

        print("Total number of patches:{} \n".format(len(mask)))

        tic = time.time()
        scores = run_masif_site(
            params,
            learning_obj,
            rho_wrt_center,
            theta_wrt_center,
            input_feat,
            mask,
            indices,
        )
        toc = time.time()
        print(
            "Total number of patches for which scores were computed: {}\n".format(
                len(scores[0])
            )
        )
        print("GPU time (real time, not actual GPU time): {:.3f}s".format(toc-tic))
        np.save(
            params["out_pred_dir"] + "/pred_" + pdbid + "_" + chains[ix] + ".npy",
            scores,
        )

