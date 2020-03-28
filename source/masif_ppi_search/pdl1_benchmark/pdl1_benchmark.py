#!/usr/bin/env python
# coding: utf-8
import pymesh
from IPython.core.debugger import set_trace
import time
import os
from default_config.masif_opts import masif_opts
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from Bio.PDB import *
import copy
import scipy.sparse as spio
import sys

start_time = time.time()

"""
pdl1_benchmark.py: Scan a large database of proteins for binders of PD-L1. The ground truth is PD-L1 in the bound state (chain A of PDB id: 4ZQK)
Pablo Gainza and Freyr Sverrisson - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""


def blockPrint():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


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

# Extract a geodesic patch.
def get_patch_geo(
    pcd, patch_coords, center, descriptors, outward_shift=0.25, flip=False
):
    """
        Get a patch based on geodesic distances. 
        pcd: the point cloud.
        patch_coords: the geodesic distances.
        center: the index of the center of the patch
        descriptors: the descriptors for every point in the original surface.
        outward_shift: expand the surface by a float value (for better alignment)
        flip: invert the surface?
    """

    idx = patch_coords[center]
    try:
        pts = np.asarray(pcd.points)[idx, :]
    except:
        set_trace()
    nrmls = np.asarray(pcd.normals)[idx, :]
    # Expand the surface in the direction of the normals. 
    pts = pts + outward_shift * nrmls
    if flip:
        nrmls = -np.asarray(pcd.normals)[idx, :]

    patch = PointCloud()
    patch.points = Vector3dVector(pts)
    patch.normals = Vector3dVector(nrmls)
    patch_descs = Feature()
    patch_descs.data = descriptors[idx, :].T
    return patch, patch_descs

def subsample_patch_coords(top_dir, pdb, pid, cv=None, frac=1.0, radius=12.0):
    """
        subsample_patch_coords: Read the geodesic coordinates in an easy to access format.
        pdb: the id of the protein pair in PDBID_CHAIN1_CHAIN2 format.
        pid: 'p1' if you want to read CHAIN1, 'p2' if you want to read CHAIN2
        cv: central vertex (list of patches to select; if None, select all)
    """
    if cv is None:
        pc = np.load(os.path.join(top_dir, pdb, pid+'_list_indices.npy'))
    else:
        temp = np.load(os.path.join(top_dir, pdb, pid+'_list_indices.npy'))[cv]
        pc = {}
        for ix, key in enumerate(cv):
            pc[key] = temp[ix]

    return pc


def get_target_vix(pc, iface):
    iface_patch_vals = []
    # Go through each patch.
    for ii in range(len(pc)):

        neigh = pc[ii]
        val = np.mean(iface[neigh])

        iface_patch_vals.append(val)

    target_vix = np.argmax(iface_patch_vals)

    return target_vix


# # Load target patches.


target_name = "4ZQK_A"
target_ppi_pair_id = "4ZQK_A_B"

# Go through every 12A patch in top_dir -- get the one with the highest iface mean 12A around it.
target_ply_fn = os.path.join(ply_iface_dir, target_name + ".ply")

mesh = pymesh.load_mesh(target_ply_fn)

iface = mesh.get_attribute("vertex_iface")

target_coord = subsample_patch_coords(precomp_dir, target_ppi_pair_id, "p1")
target_vix = get_target_vix(target_coord, iface)

target_pcd = read_point_cloud(target_ply_fn)
target_desc = np.load(os.path.join(desc_dir, target_ppi_pair_id, "p1_desc_flipped.npy"))

# Get the geodesic patch and descriptor patch for the target.
target_patch, target_patch_descs = get_patch_geo(
    target_pcd, target_coord, target_vix, target_desc, flip=True, outward_shift=1.0
)

out_patch = open("target.vert", "w+")
for point in target_patch.points:
    out_patch.write("{}, {}, {}\n".format(point[0], point[1], point[2]))
out_patch.close()


# Match descriptors that have a descriptor distance less than K


def match_descriptors(
    in_desc_dir, in_iface_dir, pids, target_desc, desc_dist_cutoff=1.7, iface_cutoff=0.8
):

    all_matched_names = []
    all_matched_vix = []
    all_matched_desc_dist = []
    count_proteins = 0
    for ppi_pair_id in os.listdir(in_desc_dir):
        if ".npy" in ppi_pair_id or ".txt" in ppi_pair_id:
            continue
        if count_proteins > 300 and ('4ZQK' not in ppi_pair_id and '3BIK' not in ppi_pair_id):
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


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def get_center_and_random_rotate(pcd):
    pts = pcd.points
    mean_pt = np.mean(pts, axis=0)
    rand_mat = rand_rotation_matrix()
    transform = np.vstack([rand_mat.T, -mean_pt]).T
    transform = np.vstack([transform, [0, 0, 0, 1]])
    return transform


def multidock(
    source_pcd,
    source_patch_coords,
    source_descs,
    cand_pts,
    target_pcd,
    target_descs,
    ransac_radius=1.0,
    ransac_iter=2000,
):
    all_results = []
    all_source_patch = []
    all_source_scores = []
    patch_time = 0.0
    ransac_time = 0.0
    transform_time = 0.0
    score_time = 0.0
    for pt in cand_pts:
        tic = time.time()
        source_patch, source_patch_descs = get_patch_geo(
            source_pcd, source_patch_coords, pt, source_descs
        )
        patch_time = patch_time + (time.time() - tic)
        tic = time.time()

        result = registration_ransac_based_on_feature_matching(
            source_patch,
            target_pcd,
            source_patch_descs,
            target_descs,
            ransac_radius,
            TransformationEstimationPointToPoint(False),
            3,
            [
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(1.0),
                CorrespondenceCheckerBasedOnNormal(np.pi / 2),
            ],
            RANSACConvergenceCriteria(ransac_iter, 500)
        )
        ransac_time = ransac_time + (time.time() - tic)
        # result = registration_icp(source_patch, target_pcd, 1.5,
        # result.transformation,
        # TransformationEstimationPointToPoint())

        tic = time.time()
        source_patch.transform(result.transformation)
        all_results.append(result)
        all_source_patch.append(source_patch)
        transform_time = transform_time + (time.time() - tic)

        tic = time.time()
        source_scores = compute_desc_dist_score(
            target_pcd,
            source_patch,
            np.asarray(result.correspondence_set),
            target_descs,
            source_patch_descs,
        )
        all_source_scores.append(source_scores)
        score_time = score_time + (time.time() - tic)
    # print('Ransac time = {:.2f}'.format(ransac_time))
    # print('Extraction time = {:.2f}'.format(patch_time))
    # print('Score time = {:.2f}'.format(score_time))

    return all_results, all_source_patch, all_source_scores


def align_and_save(
    out_filename_base,
    patch,
    transformation,
    source_structure,
    target_ca_pcd_tree,
    target_pcd_tree,
    clashing_cutoff=10.0,
    clashing_radius=2.0,
):
    structure_atoms = [atom for atom in source_structure.get_atoms()]
    structure_coords = [x.get_coord() for x in structure_atoms]

    structure_coord_pcd = PointCloud()
    structure_coord_pcd.points = Vector3dVector(structure_coords)
    structure_coord_pcd.transform(transformation)

    clashing = 0
    for point in structure_coord_pcd.points:
        [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, clashing_radius)
        if k > 0:
            clashing += 1

    #
    if clashing < float("inf"):  # clashing_cutoff:
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

    return clashing


# Compute different types of scores:
# -- Inverted sum of the minimum descriptor distances squared cutoff.


def compute_desc_dist_score(
    target_pcd, source_pcd, corr, target_desc, source_desc, cutoff=2.0
):

    # Compute scores based on correspondences.
    if len(corr) < 1:
        dists_cutoff = np.array([1000.0])
        inliers = 0
    else:
        target_p = corr[:, 1]
        source_p = corr[:, 0]
        try:
            dists_cutoff = target_desc.data[:, target_p] - source_desc.data[:, source_p]
        except:
            set_trace()
        dists_cutoff = np.sqrt(np.sum(np.square(dists_cutoff.T), axis=1))
        inliers = len(corr)

    scores_corr = np.sum(np.square(1.0 / dists_cutoff))
    scores_corr_cube = np.sum(np.power(1.0 / dists_cutoff, 3))
    scores_corr_mean = np.mean(np.square(1.0 / dists_cutoff))

    return np.array([scores_corr, inliers, scores_corr_mean, scores_corr_cube]).T


## Load the structures of the target
target_pdb_id = "4ZQK"
target_chain = "A"
target_pdb_dir = pdb_dir
parser = PDBParser()
target_struct = parser.get_structure(
    "{}_{}".format(target_pdb_id, target_chain),
    os.path.join(target_pdb_dir, "{}_{}.pdb".format(target_pdb_id, target_chain)),
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
target_pcd_tree = KDTreeFlann(target_atom_coord_pcd)

desc_scores = []
desc_pos = []
inlier_scores = []
inlier_pos = []

(matched_names, matched_vix, matched_desc_dist, count_proteins) = match_descriptors(
    desc_dir, iface_dir, ["p1", "p2"], target_desc[target_vix]
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
            precomp_dir, ppi_pair_id, pid, cv=source_vix, 
        )
#    except:
#        print("Coordinates not found. continuing.")
#        continue
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
    )
    scores = np.asarray(all_source_scores)
    desc_scores.append(scores[:, 0])
    inlier_scores.append(scores[:, 1])

    # Filter anything above 5.
    top_scorers = np.where(scores[:, 0] > 15)[0]

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
                target_ca_pcd_tree,
                target_pcd_tree,
            )


end_time = time.time()
out_log.write("Took {}s\n".format(end_time - start_time))

