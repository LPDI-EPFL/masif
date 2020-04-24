#!/usr/bin/env python
# coding: utf-8
import sys
import time
import sklearn.metrics
from geometry.open3d_import import *
import numpy as np
import os
from alignment_utils_masif_search import compute_nn_score, rand_rotation_matrix, \
        get_center_and_random_rotate, get_patch_geo, multidock, test_alignments, \
       subsample_patch_coords 
from transformation_training_data.score_nn import ScoreNN
from scipy.spatial import cKDTree
from Bio.PDB import *
import copy
import scipy.sparse as spio
from default_config.masif_opts import masif_opts
import sys

"""
second_stage_alignment_nn.py: Second stage alignment code for benchmarking MaSIF-search.
                            This code benchmarks MaSIF-search by generating 3D alignments
                            of the protein.
                            The method consists of two stages: 
                            (1) Read a database of MaSIF-search fingerprint descriptors for each overlapping patch, and find the top K decoys that are the most similar to the 
                            target 
                            (2) Align and score these patches:
                                (2a) Use the RANSAC algorithm + the iterative closest point algorithm to align each patch
                                (2b) Use a pre trained neural network to score the alignment.
                            
Pablo Gainza and Freyr Sverrisson - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""


# Start measuring the cpu clock time here. 
# We will not count the time required to align the structures and verify the ground truth. 
#               This time will be subtracted at the end.
global_start_time = time.clock()
global_ground_truth_time = 0.0

# Read the pre-trained neural network.
nn_model = ScoreNN()
print(sys.argv)

if len(sys.argv) != 6 or (sys.argv[5] != "masif" and sys.argv[5] != "gif"):
    print("Usage: {} data_dir K ransac_iter num_success gif|masif".format(sys.argv[0]))
    print("data_dir: Location of data directory.")
    print("K: Number of decoy descriptors per target")
    print("ransac_iter: number of ransac iterations.")
    print("num_success: true alignment within short list of size num_success")
    sys.exit(1)

data_dir = sys.argv[1]
K = int(sys.argv[2])
ransac_iter = int(sys.argv[3])
num_success = int(sys.argv[4])
method = sys.argv[5]

# Location of surface (ply) files. 
surf_dir = os.path.join(data_dir, masif_opts["ply_chain_dir"])

if method == "gif":
    desc_dir = os.path.join(data_dir, masif_opts["ppi_search"]["gif_descriptors_out"])
else:  # MaSIF
    desc_dir = os.path.join(data_dir, masif_opts["ppi_search"]["desc_dir"])

# Directory of pdb files (used to compute the ground truth).
pdb_dir = os.path.join(data_dir, masif_opts["pdb_chain_dir"])
precomp_dir = os.path.join(
    data_dir, masif_opts["ppi_search"]["masif_precomputation_dir"]
)
precomp_dir_9A = os.path.join(
    data_dir, masif_opts["site"]["masif_precomputation_dir"]
)

# List of PDBID_CHAIN1_CHAIN2 ids that will be used in this benchmark. 
benchmark_list = "../benchmark_list.txt"


pdb_list = open(benchmark_list).readlines()[0:100]
pdb_list = [x.rstrip() for x in pdb_list]

"""
This is where the actual protocol starts. 
"""

# Read all surfaces.
all_pc = []
all_desc = []

rand_list = np.copy(pdb_list)
#np.random.seed(0)
np.random.shuffle(rand_list)
rand_list = rand_list[0:100]

p2_descriptors_straight = []
p2_point_clouds = []
p2_patch_coords = []
p2_names = []

# First we read in all the decoy 'binder' shapes. 
# Read all of p2. p2 will have straight descriptors.
for i, pdb in enumerate(rand_list):
    print("Loading patch coordinates for {}".format(pdb))
    pdb_id = pdb.split("_")[0]
    chains = pdb.split("_")[1:]
    # Descriptors for global matching.
    p2_descriptors_straight.append(
        np.load(os.path.join(desc_dir, pdb, "p2_desc_straight.npy"))
    )

    p2_point_clouds.append(
        read_point_cloud(
            os.path.join(surf_dir, "{}.ply".format(pdb_id + "_" + chains[1]))
        )
    )

    # Read patch coordinates. 

    pc = subsample_patch_coords(pdb, "p2", precomp_dir_9A)
    p2_patch_coords.append(pc)

    p2_names.append(pdb)


import time
import scipy.spatial


all_positive_scores = []
all_positive_rmsd = []
all_negative_scores = []
# Match all descriptors.
count_found = 0
all_rankings_desc = []

# Now go through each target (p1 in every case) and dock each 'decoy' binder to it. 
# The target will have flipped (inverted) descriptors.
for target_ix, target_pdb in enumerate(rand_list):
    cycle_start_time = time.clock()
    print('Docking all binders on target: {} '.format(target_pdb))
    target_pdb_id = target_pdb.split("_")[0]
    chains = target_pdb.split("_")[1:]

    # Load target descriptors for global matching.
    target_desc = np.load(os.path.join(desc_dir, target_pdb, "p1_desc_flipped.npy"))

    # Load target point cloud
    target_pc = os.path.join(surf_dir, "{}.ply".format(target_pdb_id + "_" + chains[0]))
    source_pc_gt = os.path.join(
        surf_dir, "{}.ply".format(target_pdb_id + "_" + chains[1])
    )
    target_pcd = read_point_cloud(target_pc)

    # Read the point with the highest shape compl.
    sc_labels = np.load(os.path.join(precomp_dir, target_pdb, "p1_sc_labels.npy"))
    center_point = np.argmax(np.median(np.nan_to_num(sc_labels[0]), axis=1))

    # Go through each source descriptor, find the top descriptors, store id+pdb
    num_negs = 0
    all_desc_dists = []
    all_pdb_id = []
    all_vix = []
    gt_dists = []

    # This is where the desriptors are actually compared (stage 1 of the MaSIF-search protocol)
    for source_ix, source_pdb in enumerate(rand_list):

        source_desc = p2_descriptors_straight[source_ix]

        desc_dists = np.linalg.norm(source_desc - target_desc[center_point], axis=1)
        all_desc_dists.append(desc_dists)
        all_pdb_id.append([source_pdb] * len(desc_dists))
        all_vix.append(np.arange(len(desc_dists)))

        if source_pdb == target_pdb:
            source_pcd = p2_point_clouds[source_ix]
            eucl_dists = np.linalg.norm(
                np.asarray(source_pcd.points)
                - np.asarray(target_pcd.points)[center_point, :],
                axis=1,
            )
            eucl_closest = np.argsort(eucl_dists)
            gt_dists = desc_dists[eucl_closest[0:50]]
            gt_count = len(source_desc)

    all_desc_dists = np.concatenate(all_desc_dists, axis=0)
    all_pdb_id = np.concatenate(all_pdb_id, axis=0)
    all_vix = np.concatenate(all_vix, axis=0)

    ranking = np.argsort(all_desc_dists)

    # Load target geodesic distances.
    target_coord = subsample_patch_coords(target_pdb, "p1", precomp_dir_9A, [center_point])

    # Get the geodesic patch and descriptor patch for the target.
    target_patch, target_patch_descs = get_patch_geo(
        target_pcd, target_coord, center_point, target_desc, flip=True
    )

    # Make a ckdtree with the target.
    target_ckdtree = cKDTree(target_patch.points)

    ## Load the structures of the target and the source (to get the ground truth).
    parser = PDBParser()
    target_struct = parser.get_structure(
        "{}_{}".format(target_pdb_id, chains[0]),
        os.path.join(pdb_dir, "{}_{}.pdb".format(target_pdb_id, chains[0])),
    )
    gt_source_struct = parser.get_structure(
        "{}_{}".format(target_pdb_id, chains[1]),
        os.path.join(pdb_dir, "{}_{}.pdb".format(target_pdb_id, chains[1])),
    )
    # Get coordinates of atoms for the ground truth and target.
    target_atom_coords = [atom.get_coord() for atom in target_struct.get_atoms()]
    target_ca_coords = [
        atom.get_coord() for atom in target_struct.get_atoms() if atom.get_id() == "CA"
    ]
    target_atom_coord_pcd = PointCloud()
    target_ca_coord_pcd = PointCloud()
    target_atom_coord_pcd.points = Vector3dVector(np.array(target_atom_coords))
    target_ca_coord_pcd.points = Vector3dVector(np.array(target_ca_coords))
    target_atom_pcd_tree = KDTreeFlann(target_atom_coord_pcd)
    target_ca_pcd_tree = KDTreeFlann(target_ca_coord_pcd)

    found = False
    myrank_desc = float("inf")

    chosen_top = ranking[0:K]

    pos_scores = []
    pos_rmsd = []
    neg_scores = []

    # This is where the matched descriptors are actually aligned.
    for source_ix, source_pdb in enumerate(rand_list):
        viii = chosen_top[np.where(all_pdb_id[chosen_top] == source_pdb)[0]]
        source_vix = all_vix[viii]

        if len(source_vix) == 0:
            continue

        source_desc = p2_descriptors_straight[source_ix]
        source_pcd = copy.deepcopy(p2_point_clouds[source_ix])

        source_coords = p2_patch_coords[source_ix]

        # Randomly rotate and translate.
        random_transformation = get_center_and_random_rotate(source_pcd)
        source_pcd.transform(random_transformation)
        # Dock and score each matched patch. 
        all_results, all_source_patch, all_source_scores = multidock(
            source_pcd,
            source_coords,
            source_desc,
            source_vix,
            target_patch,
            target_patch_descs,
            target_ckdtree,
            nn_model, 
            ransac_iter=ransac_iter
        )
        num_negs = num_negs

        # If this is the source_pdb, get the ground truth. The ground truth evaluation time is ignored for this and all other methods. 
        gt_start_time = time.clock()
        if source_pdb == target_pdb:

            for j, res in enumerate(all_results):
                rmsd, clashing, structure_coord_pcd, structure_coord_pcd_notTransformed = test_alignments(
                    res.transformation,
                    random_transformation,
                    gt_source_struct,
                    target_ca_pcd_tree,
                    target_atom_pcd_tree,
                    radius=0.5,
                )
                score = all_source_scores[j]
                if rmsd < 5.0 and res.fitness > 0:
                    rank_val = np.where(chosen_top == viii[j])[0][0]
                    pos_rmsd.append(rmsd)
                    found = True
                    myrank_desc = min(rank_val, myrank_desc)
                    pos_scores.append(score)
                else:
                    neg_scores.append(score)
        else:
            for j in range(len(all_source_scores)):
                score = all_source_scores[j]
                neg_scores.append(score)
    if found:
        count_found += 1
        all_rankings_desc.append(myrank_desc)
        print('Descriptor rank: {}'.format(myrank_desc))
        print('Mean positive score: {}, mean negative score: {}'.format(np.mean(pos_scores), np.mean(neg_scores)))
        max_pos_score = np.max(pos_scores)
        rank = np.sum(neg_scores > max_pos_score)+1
        print('Neural network rank: {}'.format(rank))
        y_true = np.concatenate([np.zeros_like(pos_scores), np.ones_like(neg_scores)])
        y_pred = np.concatenate([pos_scores, neg_scores])
        auc = 1.0 - sklearn.metrics.roc_auc_score(y_true, y_pred)
        print('ROC AUC (protein): {:.3f}'.format(auc))
    else:
        print("N/D")
    gt_end_time = time.clock()
    global_ground_truth_time += (gt_end_time - gt_start_time)

    all_positive_rmsd.append(pos_rmsd)
    all_positive_scores.append(pos_scores)
    all_negative_scores.append(neg_scores)
    cycle_end_time = time.clock()
    cycle_time = cycle_end_time-cycle_start_time - (gt_end_time - gt_start_time)
    print("Cycle took {:.2f} cpu seconds (excluding ground truth time) ".format(cycle_time))
    # Go through every top descriptor.

# We stop measuring the time at this point. 
global_end_time = time.clock()

# CPU time in minutes.
global_cpu_time = global_end_time - global_start_time - global_ground_truth_time
# Convert to minutes. 
global_cpu_time = global_cpu_time/60

print("All alignments took {} min".format(global_cpu_time))


all_pos = []
all_neg = []
ranks = []
unranked = 0

pos_inliers = []
neg_inliers = []

pos_inlier_patch_ratio = []
neg_inlier_patch_ratio = []

pos_inlier_outlier_ratio = []
neg_inlier_outlier_ratio = []
rmsds = []
# Print and write the output information from MaSIF-search
for pdb_ix in range(len(all_positive_scores)):
    if len(all_positive_scores[pdb_ix]) == 0:
        print("N/D")
        unranked += 1
    else:
        pos_scores = all_positive_scores[pdb_ix]
        neg_scores = [x for x in all_negative_scores[pdb_ix]]
        best_pos_score = np.max(pos_scores)
        best_pos_score_ix = np.argmax(pos_scores)
        best_rmsd = all_positive_rmsd[pdb_ix][best_pos_score_ix]

        number_better_than_best_pos = np.sum(neg_scores > best_pos_score) + 1
        if number_better_than_best_pos > num_success:
            print("{} N/D".format(rand_list[pdb_ix]))
            unranked += 1
        else:
            rmsds.append(best_rmsd)
            print(
                "{} {} out of {} -- pos scores: {}".format(
                    rand_list[pdb_ix],
                    number_better_than_best_pos,
                    len(neg_scores) + len(pos_scores),
                    len(pos_scores),
                )
            )
            ranks.append(number_better_than_best_pos)
ranks = np.array(ranks)
print("Median rank for correctly ranked ones: {}".format(np.median(ranks)))
print("Mean rank for correctly ranked ones: {}".format(np.mean(ranks)))
print(
    "Number in top 100 {} out of {}".format(np.sum(ranks <= 100), len(all_positive_scores))
)
print(
    "Number in top 10 {} out of {}".format(np.sum(ranks <= 10), len(all_positive_scores))
)
print(
    "Number in top 1 {} out of {}".format(np.sum(ranks <= 1), len(all_positive_scores))
)

outfile = open("results_{}.txt".format(method), "a+")
outfile.write("K,Total,Top2000,Top1000,Top100,Top10,Top5,Top1,MeanRMSD,Time(min)\n")
top2000= np.sum(ranks<=2000)
top1000= np.sum(ranks<=1000)
top100= np.sum(ranks<=100)
top10= np.sum(ranks<=10)
top5= np.sum(ranks<=5)
top1= np.sum(ranks<=1)
meanrmsd = np.mean(rmsds)


outline = "{},{},{},{},{},{},{},{},{},{}\n".format(
    K, len(all_positive_scores), top2000, top1000, top100, top10, top5, top1, meanrmsd, global_cpu_time
)
outfile.write(outline)

outfile.close()

