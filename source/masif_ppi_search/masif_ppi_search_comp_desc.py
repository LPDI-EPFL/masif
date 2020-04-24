# Header variables and parameters.
import pymesh
import sys
import os
import time
import numpy as np
from IPython.core.debugger import set_trace
from sklearn import metrics
import importlib
from default_config.masif_opts import masif_opts

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)


params = masif_opts["ppi_search"]

custom_params_file = sys.argv[1]
custom_params = importlib.import_module(custom_params_file, package=None)
custom_params = custom_params.custom_params

for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]

# Read the positive first
parent_in_dir = params["masif_precomputation_dir"]

np.random.seed(0)


#   Load existing network.
print("Reading pre-trained network")
from masif_modules.MaSIF_ppi_search import MaSIF_ppi_search

learning_obj = MaSIF_ppi_search(
    params["max_distance"],
    n_thetas=16,
    n_rhos=5,
    n_rotations=16,
    idx_gpu="/gpu:0",
    feat_mask=params["feat_mask"],
)
learning_obj.saver.restore(learning_obj.session, params["model_dir"] + "model")

from masif_modules.train_ppi_search import compute_val_test_desc

idx_count = 0
all_pos_dists = []
all_neg_dists = []
all_pos_dists_pos_neg = []
all_neg_dists_pos_neg = []
if not os.path.exists(params["desc_dir"]):
    os.makedirs(params["desc_dir"])

eval_list = []
if len(sys.argv) == 3:
    ppi_list = [sys.argv[2]]
# Read a list of pdb_chain entries to evaluate.
elif len(sys.argv) == 4 and sys.argv[2] == "-l":
    listfile = open(sys.argv[3])
    ppi_list = []
    for line in listfile:
        eval_list.append(line.rstrip())
    for mydir in os.listdir(parent_in_dir):
        ppi_list.append(mydir)
else:
    sys.exit(1)

logfile = open(os.path.join(params["desc_dir"], "log.txt"), "w+")
for count, ppi_pair_id in enumerate(ppi_list):

    if len(eval_list) > 0 and ppi_pair_id not in eval_list:
        continue

    in_dir = parent_in_dir + ppi_pair_id + "/"
    print(ppi_pair_id)

    out_desc_dir = os.path.join(params["desc_dir"], ppi_pair_id)
    if not os.path.exists(os.path.join(out_desc_dir, 'p1_desc_straight.npy')):
        os.mkdir(out_desc_dir)
#    else:
#        # Ignore this one as it was already computed.
#        print('Ignoring descriptor computation for {} as it was already computed'.format(ppi_pair_id))
#        continue

    pdbid = ppi_pair_id.split("_")[0]
    chain1 = ppi_pair_id.split("_")[1]
    if len(ppi_pair_id.split("_")) > 2: 
        chain2 = ppi_pair_id.split("_")[2]
    else:
        chain2 = ''


    # Read shape complementarity labels if chain2 != ''
    if chain2 != '':
        try:
            labels = np.load(in_dir + "p1" + "_sc_labels.npy")
            mylabels = labels[0]
            labels = np.median(mylabels, axis=1)
        except:# Exception, e:
            print('Could not open '+in_dir+'p1'+'_sc_labels.npy: '+str(e))
            continue
        print("Number of vertices: {}".format(len(labels)))

        # pos_labels: points that pass the sc_filt.
        pos_labels = np.where(
            (labels > params["min_sc_filt"]) & (labels < params["max_sc_filt"])
        )[0]
        l = pos_labels
    else:
        l = []



    if len(l) > 0 and chain2 != "":
        ply_fn1 = masif_opts['ply_file_template'].format(pdbid, chain1)
        v1 = pymesh.load_mesh(ply_fn1).vertices[l]
        from sklearn.neighbors import NearestNeighbors

        ply_fn2 = masif_opts['ply_file_template'].format(pdbid, chain2 )
        v2 = pymesh.load_mesh(ply_fn2).vertices

        # For each point in v1, find the closest point in v2.
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(v2)
        d, r = nbrs.kneighbors(v1)
        d = np.squeeze(d, axis=1)
        r = np.squeeze(r, axis=1)

        # Contact points: those within a cutoff distance.
        contact_points = np.where(d < params["pos_interface_cutoff"])[0]
        if len(contact_points) > 0:
            k1 = l[contact_points]  # contact points protein 1
            k2 = r[contact_points]  # contact points protein 2
            assert len(k1) == len(k2)
        else:
            l = []

    tic = time.time()
    pid = "p1"
    try:
        p1_rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
    except:
        continue
    p1_theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
    p1_input_feat = np.load(in_dir + pid + "_input_feat.npy")
    p1_input_feat = mask_input_feat(p1_input_feat, params["feat_mask"])
    p1_mask = np.load(in_dir + pid + "_mask.npy")
    idx1 = np.array(range(len(p1_rho_wrt_center)))
    print("Data loading time: {:.2f}s".format(time.time() - tic))
    tic = time.time()
    desc1_str = compute_val_test_desc(
        learning_obj,
        idx1,
        p1_rho_wrt_center,
        p1_theta_wrt_center,
        p1_input_feat,
        p1_mask,
        batch_size=1000,
        flip=False,
    )
    desc1_flip = compute_val_test_desc(
        learning_obj,
        idx1,
        p1_rho_wrt_center,
        p1_theta_wrt_center,
        p1_input_feat,
        p1_mask,
        batch_size=1000,
        flip=True,
    )
    print("Running time: {:.2f}s".format(time.time() - tic))

    if chain2 != "":
        pid = "p2"
        p2_rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
        p2_theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
        p2_input_feat = np.load(in_dir + pid + "_input_feat.npy")
        p2_input_feat = mask_input_feat(p2_input_feat, params["feat_mask"])
        p2_mask = np.load(in_dir + pid + "_mask.npy")
        idx2 = np.array(range(len(p2_rho_wrt_center)))
        desc2_str = compute_val_test_desc(
            learning_obj,
            idx2,
            p2_rho_wrt_center,
            p2_theta_wrt_center,
            p2_input_feat,
            p2_mask,
            batch_size=1000,
            flip=False,
        )
        desc2_flip = compute_val_test_desc(
            learning_obj,
            idx2,
            p2_rho_wrt_center,
            p2_theta_wrt_center,
            p2_input_feat,
            p2_mask,
            batch_size=1000,
            flip=True,
        )

        max_label = np.max(labels)
        logfile.write("{}: max label: {} \n".format(ppi_pair_id, max_label))

    # Save descriptors
    np.save(os.path.join(out_desc_dir, "p1_desc_straight.npy"), desc1_str)
    np.save(os.path.join(out_desc_dir, "p1_desc_flipped.npy"), desc1_flip)

    if chain2 != "":
        np.save(os.path.join(out_desc_dir, "p2_desc_straight.npy"), desc2_str)
        np.save(os.path.join(out_desc_dir, "p2_desc_flipped.npy"), desc2_flip)

    # For sanity and statistics: Compute ROC AUC between points that pass the filter and a randomly chosen set.
    if chain2 != "" and len(l) > 0:
        np.random.shuffle(idx1)
        kneg1 = idx1[: len(k1)]
        np.random.shuffle(idx2)
        kneg2 = idx2[: len(k2)]
        # Compute pos_dists
        pos_dists = np.sqrt(np.sum(np.square(desc1_str[k1] - desc2_flip[k2]), axis=1))
        neg_dists = np.sqrt(
            np.sum(np.square(desc1_str[kneg1] - desc2_flip[kneg2]), axis=1)
        )
        roc_auc = 1.0 - compute_roc_auc(pos_dists, neg_dists)
        all_pos_dists.append(pos_dists)
        all_neg_dists.append(neg_dists)
        logfile.write(
            "{}: ROC AUC: {:.6f}; num pos: {}; mean_pos: {} ; mean_neg: {} \n".format(
                ppi_pair_id, roc_auc, len(k1), np.mean(pos_dists), np.mean(neg_dists)
            )
        )
        logfile.flush()

        np.random.shuffle(idx2)
        kneg2 = idx2[: len(k2)]
        # Compute pos_dists
        pos_dists = np.sqrt(np.sum(np.square(desc1_str[k1] - desc2_flip[k2]), axis=1))
        neg_dists = np.sqrt(
            np.sum(np.square(desc1_str[k1] - desc2_flip[kneg2]), axis=1)
        )
        roc_auc = 1.0 - compute_roc_auc(pos_dists, neg_dists)
        all_pos_dists_pos_neg.append(pos_dists)
        all_neg_dists_pos_neg.append(neg_dists)
        logfile.write(
            "{}: Pos_neg ROC AUC: {:.6f}; num pos: {}; mean_pos: {} ; mean_neg: {} \n".format(
                ppi_pair_id, roc_auc, len(k1), np.mean(pos_dists), np.mean(neg_dists)
            )
        )
        logfile.flush()


if len(all_pos_dists) > 0:
    all_pos_dists = np.concatenate(all_pos_dists, axis=0)
    all_neg_dists = np.concatenate(all_neg_dists, axis=0)

    roc_auc = 1.0 - compute_roc_auc(all_pos_dists, all_neg_dists)
    logfile.write(
        "Global ROC AUC: {:.6f}; num pos: {}\n".format(roc_auc, len(all_pos_dists))
    )
    np.save(params["desc_dir"] + "/all_pos_dists.npy", all_pos_dists)
    np.save(params["desc_dir"] + "/all_neg_dists.npy", all_neg_dists)

    all_pos_dists_pos_neg = np.concatenate(all_pos_dists_pos_neg, axis=0)
    all_neg_dists_pos_neg = np.concatenate(all_neg_dists_pos_neg, axis=0)
    roc_auc = 1.0 - compute_roc_auc(all_pos_dists_pos_neg, all_neg_dists_pos_neg)
    logfile.write(
        "Global ROC AUC: {:.6f}; num pos: {}\n".format(
            roc_auc, len(all_pos_dists_pos_neg)
        )
    )
    np.save(params["desc_dir"] + "/all_pos_dists_pos_neg.npy", all_pos_dists_pos_neg)
    np.save(params["desc_dir"] + "/all_neg_dists_pos_neg.npy", all_neg_dists_pos_neg)

