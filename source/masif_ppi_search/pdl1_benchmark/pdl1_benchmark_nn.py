#!/usr/bin/env python
# coding: utf-8
import pymesh
from IPython.core.debugger import set_trace
from scipy.spatial import cKDTree
import time
import os
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from Bio.PDB import *
import copy
import scipy.sparse as spio
import sys

# import the right version of open3d
from geometry.open3d_import import PointCloud, read_point_cloud, \
        Vector3dVector, Feature, registration_ransac_based_on_feature_matching, \
       TransformationEstimationPointToPoint, CorrespondenceCheckerBasedOnEdgeLength, \
      CorrespondenceCheckerBasedOnDistance, CorrespondenceCheckerBasedOnNormal, \
     RANSACConvergenceCriteria 

# Local imports
from default_config.masif_opts import masif_opts
from alignment_utils_masif_search import get_patch_geo, multidock, \
        subsample_patch_coords, compute_nn_score, get_target_vix
from transformation_training_data.score_nn import ScoreNN

"""
Hard-coded configuration, change accordingly!"

Target name - hard coded and tested with PD-L1 (PDB id: 4ZQK). You can change you your own target and test. 
In general this will work well with targets where MaSIF-site labels the site well and where there is a high
amount of shape complementarity
"""
target_name = "4ZQK_A"
target_ppi_pair_id = "4ZQK_A_B"

"""
Descriptor cutoff: This is the key parameter for the speed of the method. The lower the value, 
the faster the method, but also the higher the number of false negatives. Values ABOVE
this cutoff are discareded. Recommended values: 1.7-2.2. 
"""
DESC_DIST_CUTOFF=1.7

"""
Iface cutoff: Patches are also filtered by their MaSIF-site score. Patches whose center
point has a value BELOW this score are discarded. 
The higher the value faster the method, but also the higher the number of false negatives. 
Recommended values: 0.8
"""
IFACE_CUTOFF=0.8

def blockPrint():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


nn_model = ScoreNN()
start_time = time.time()

"""
pdl1_benchmark.py: Scan a large database of proteins for binders of PD-L1. The ground truth is PD-L1 in the bound state (chain A of PDB id: 4ZQK)
Pablo Gainza and Freyr Sverrisson - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""


masif_root = os.environ["masif_root"]
top_dir = os.path.join(masif_root, "data/masif_pdl1_benchmark/")
surf_dir = os.path.join(top_dir, masif_opts["ply_chain_dir"])
iface_dir = os.path.join(
    top_dir, masif_opts["site"]["out_pred_dir"]
)
ply_iface_dir = os.path.join(
    top_dir, masif_opts["site"]["out_surf_dir"]
)

desc_dir = os.path.join(masif_opts["ppi_search"]["desc_dir"])

pdb_dir = os.path.join(top_dir, masif_opts["pdb_chain_dir"])
precomp_dir = os.path.join(
    top_dir, masif_opts["site"]["masif_precomputation_dir"]
)






# Go through every 9A patch in top_dir -- get the one with the highest iface mean 12A around it.
target_ply_fn = os.path.join(ply_iface_dir, target_name + ".ply")

mesh = pymesh.load_mesh(target_ply_fn)

iface = mesh.get_attribute("vertex_iface")

target_coord = subsample_patch_coords(target_ppi_pair_id, "p1", precomp_dir)
target_vix = get_target_vix(target_coord, iface)

target_pcd = read_point_cloud(target_ply_fn)
target_desc = np.load(os.path.join(desc_dir, target_ppi_pair_id, "p1_desc_flipped.npy"))

# Get the geodesic patch and descriptor patch for the target.
target_patch, target_patch_descs = get_patch_geo(
    target_pcd, target_coord, target_vix, target_desc, flip=True, outward_shift=0.25
)

out_patch = open("target.vert", "w+")
for point in target_patch.points:
    out_patch.write("{}, {}, {}\n".format(point[0], point[1], point[2]))
out_patch.close()


# Match descriptors that have a descriptor distance less than K
def match_descriptors(
    in_desc_dir, in_iface_dir, pids, target_desc, desc_dist_cutoff=2.2, iface_cutoff=0.8
):

    all_matched_names = []
    all_matched_vix = []
    all_matched_desc_dist = []
    count_proteins = 0
    for ppi_pair_id in os.listdir(in_desc_dir):
        if ".npy" in ppi_pair_id or ".txt" in ppi_pair_id:
            continue
        mydescdir = os.path.join(in_desc_dir, ppi_pair_id)
        for pid in pids:
            try:
                fields = ppi_pair_id.split("_")
                if pid == "p1":
                    pdb_chain_id = fields[0] + "_" + fields[1]
                elif pid == "p2":
                    pdb_chain_id = fields[0] + "_" + fields[2]
                iface = np.load(in_iface_dir + "/pred_" + pdb_chain_id + ".npy")[0]
                descs = np.load(mydescdir + "/" + pid + "_desc_straight.npy")
            except:
                continue
            print(pdb_chain_id)
            name = (ppi_pair_id, pid)
            count_proteins += 1

            diff = np.sqrt(np.sum(np.square(descs - target_desc), axis=1))

            true_iface = np.where(iface > iface_cutoff)[0]
            near_points = np.where(diff < desc_dist_cutoff)[0]

            selected = np.intersect1d(true_iface, near_points)
            if len(selected > 0):
                all_matched_names.append([name] * len(selected))
                all_matched_vix.append(selected)
                all_matched_desc_dist.append(diff[selected])
                print("Matched {}".format(ppi_pair_id))
                print("Scores: {} {}".format(iface[selected], diff[selected]))

    print("Iterated over {} proteins.".format(count_proteins))
    return all_matched_names, all_matched_vix, all_matched_desc_dist, count_proteins

def align_and_save(
    out_filename_base,
    patch,
    transformation,
    source_structure,
):
    structure_atoms = [atom for atom in source_structure.get_atoms()]
    structure_coords = [x.get_coord() for x in structure_atoms]

    structure_coord_pcd = PointCloud()
    structure_coord_pcd.points = Vector3dVector(structure_coords)
    structure_coord_pcd.transform(transformation)

    for ix, v in enumerate(structure_coord_pcd.points):
        structure_atoms[ix].set_coord(v)

    io = PDBIO()
    io.set_structure(source_structure)
    io.save(out_filename_base + ".pdb")
    # Save patch
    out_patch = open(out_filename_base + ".vert", "w+")
    for point in patch.points:
        out_patch.write("{}, {}, {}\n".format(point[0], point[1], point[2]))
    out_patch.close()

    return 0

## Load the structures of the target
target_pdb_id = "4ZQK"
target_chain = "A"
target_pdb_dir = pdb_dir
parser = PDBParser()
target_struct = parser.get_structure(
    "{}_{}".format(target_pdb_id, target_chain),
    os.path.join(target_pdb_dir, "{}_{}.pdb".format(target_pdb_id, target_chain)),
)

# Make a ckdtree with the target.
target_ckdtree = cKDTree(target_patch.points)

desc_scores = []
desc_pos = []
inlier_scores = []
inlier_pos = []

(matched_names, matched_vix, matched_desc_dist, count_proteins) = match_descriptors(
    desc_dir, iface_dir, ["p1", "p2"], target_desc[target_vix],
    desc_dist_cutoff=DESC_DIST_CUTOFF, iface_cutoff=IFACE_CUTOFF
)

matched_names = np.concatenate(matched_names, axis=0)
matched_vix = np.concatenate(matched_vix, axis=0)
matched_desc_dist = np.concatenate(matched_desc_dist, axis=0)

matched_dict = {}
out_log = open("log.txt", "w+")
out_log.write("Total number of proteins {}\n".format(count_proteins))
for name_ix, name in enumerate(matched_names):
    name = (name[0], name[1])
    if name not in matched_dict:
        matched_dict[name] = []
    matched_dict[name].append(matched_vix[name_ix])

desc_scores = []
inlier_scores = []

for name in matched_dict.keys():
    ppi_pair_id = name[0]
    pid = name[1]
    pdb = ppi_pair_id.split("_")[0]

    if pid == "p1":
        chain = ppi_pair_id.split("_")[1]
    else:
        chain = ppi_pair_id.split("_")[2]

    # Load source ply file, coords, and descriptors.
    tic = time.time()

    print("{}".format(pdb + "_" + chain))
    blockPrint()
    source_pcd = read_point_cloud(
        os.path.join(surf_dir, "{}.ply".format(pdb + "_" + chain))
    )
    enablePrint()
    #    print('Reading ply {}'.format(time.time()- tic))
    enablePrint()

    tic = time.time()
    source_vix = matched_dict[name]
#    try:
    source_coords = subsample_patch_coords(
            ppi_pair_id, pid,precomp_dir, cv=source_vix, 
        )
#    except:
#    print("Coordinates not found. continuing.")
#    continue
    source_desc = np.load(
        os.path.join(desc_dir, ppi_pair_id, pid + "_desc_straight.npy")
    )

    # Perform all alignments to target.
    tic = time.time()
    all_results, all_source_patch, all_source_scores = multidock(
        source_pcd,
        source_coords,
        source_desc,
        source_vix,
        target_patch,
        target_patch_descs,
        target_ckdtree,
        nn_model,
        use_icp = True 
    )
    scores = np.asarray(all_source_scores)
    desc_scores.append(scores)

    # Filter anything above 9 .
    top_scorers = np.where(scores > 0.9)[0]

    if len(top_scorers) > 0:

        # Load source structure
        # Perform the transformation on the atoms
        for j in top_scorers:
            print("{} {} {}".format(ppi_pair_id, scores[j], pid))
            out_log.write("{} {} {}\n".format(ppi_pair_id, scores[j], pid))
            source_struct = parser.get_structure(
                "{}_{}".format(pdb, chain),
                os.path.join(pdb_dir, "{}_{}.pdb".format(pdb, chain)),
            )
            res = all_results[j]
            if not os.path.exists("out/" + pdb):
                os.makedirs("out/" + pdb)

            out_fn = "out/" + pdb + "/{}_{}_{}".format(pdb, chain, j)

            # Align and save the pdb + patch
            align_and_save(
                out_fn,
                all_source_patch[j],
                res.transformation,
                source_struct,
            )


end_time = time.time()
out_log.write("Took {}s\n".format(end_time - start_time))

