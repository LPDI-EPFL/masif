# coding: utf-8
# ## Imports and helper functions
from IPython.core.debugger import set_trace
import sys
import os, sys, inspect
import os
import numpy as np
import h5py
import scipy.sparse.linalg as la
import scipy.sparse as sp
import scipy
import time
import pyflann

import re

import math

import itertools as it
from sklearn import metrics


def load_matlab_file(path_file, name_field, struct=False):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    out = None
    db = h5py.File(path_file, "r")
    if type(name_field) is tuple:
        if name_field[1] not in db[name_field[0]]:
            return None
        ds = db[name_field[0]][name_field[1]]
    else:
        ds = db[name_field]
    try:
        if "ir" in ds.keys():
            data = np.asarray(ds["data"])
            ir = np.asarray(ds["ir"])
            jc = np.asarray(ds["jc"])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
        if struct:
            out = dict()
            for c_k in ds.keys():
                # THis is a horrible way to manage the exception when shape_comp_25 is not defined
                if c_k.startswith("shape_comp"):
                    try:
                        out[c_k] = np.asarray(ds[c_k])
                    except:
                        continue
                else:
                    out[c_k] = np.asarray(ds[c_k])
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


# From a full shape in a full protein, extract a patch around a vertex.
# If patch_indices = True, then store the indices of all neighbors.
def extract_patch_and_coord(
    vix, shape, coord, max_distance, max_vertices, patch_indices=False
):
    # Member vertices are nonzero elements
    i, j = coord[np.int(vix), : coord.shape[1] // 2].nonzero()

    # D = np.squeeze(np.asarray(coord[np.int(vix),j].todense()))
    D = np.squeeze(np.asarray(coord[np.int(vix), : coord.shape[1] // 2].todense()))
    j = np.where((D < max_distance) & (D > 0))[0]
    max_dist_tmp = max_distance
    old_j = len(j)
    while len(j) > max_vertices:
        max_dist_tmp = max_dist_tmp * 0.95
        j = np.where((D < max_dist_tmp) & (D > 0))[0]
    #    print('j = {} {}'.format(len(j), old_j))
    D = D[j]
    patch = {}
    patch["X"] = shape["X"][0][j]
    patch["Y"] = shape["Y"][0][j]
    patch["Z"] = shape["Z"][0][j]
    patch["charge"] = shape["charge"][0][j]
    patch["hbond"] = shape["hbond"][0][j]
    patch["normal"] = shape["normal"][:, j]
    patch["shape_index"] = shape["shape_index"][0][j]
    if "hphob" in shape:
        patch["hphob"] = shape["hphob"][0][j]

    patch["center"] = np.argmin(D)

    j_theta = j + coord.shape[1] // 2
    theta = np.squeeze(np.asarray(coord[np.int(vix), j_theta].todense()))
    coord = np.concatenate([D, theta], axis=0)

    if patch_indices:
        return patch, coord, j
    else:
        return patch, coord


# FOR DEBUGGING only.... too slow, use precomputed values instead.
from scipy.spatial import KDTree


def compute_shape_complementarity(shape1, coord1, shape2, coord2):
    w = 0.5
    radius = 8.0

    D1 = coord1[: coord1.shape[0] // 2]
    v1 = np.stack([shape1["X"], shape1["Y"], shape1["Z"]], 1)
    v1 = v1[np.where(D1 < radius)]
    n1 = shape1["normal"].T[np.where(D1 < radius)]

    D2 = coord2[: coord2.shape[0] // 2]
    v2 = np.stack([shape2["X"], shape2["Y"], shape2["Z"]], 1)
    v2 = v2[np.where(D2 < radius)]
    n2 = shape2["normal"].T[np.where(D2 < radius)]

    # First v2 -> v1
    kdt = KDTree(v1)
    d, i = kdt.query(v2)
    comp2 = [np.dot(n2[x], -n1[i[x]]) for x in range(len(n2))]
    comp2 = np.multiply(comp2, np.exp(-w * np.square(d)))
    comp2 = np.percentile(comp2, 50)

    # Now v1 -> v2
    kdt = KDTree(v2)
    d, i = kdt.query(v1)
    comp1 = [np.dot(n1[x], -n2[i[x]]) for x in range(len(n1))]
    comp1 = np.multiply(comp1, np.exp(-w * np.square(d)))
    comp1 = np.percentile(comp1, 50)

    return np.mean([comp1, comp2])


def memory():
    import os
    import psutil

    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB...I think
    print("memory use:", memoryUse)


# ## Read input dataset and patch coords (and filter input if desired)

# if do_shape_comp_pairs is false, then use the random coords.
def read_data_from_matfile(
    coord_file, shape_file, seed_pair, params, do_shape_comp_pairs, reshuffle=True
):

    training_idx = []
    val_idx = []
    test_idx = []
    non_sc_idx = []
    # Ignore any shapes with more than X vertices.
    max_shape_size = params["max_shape_size"]

    list_desc = []
    list_coords = []
    list_shape_idx = []
    list_names = []
    positive_pairs_idx = set()
    idx_positives = []
    discarded_sc = 0.0
    discarded_large = 0.0
    ppi_accept_probability = params["ppi_accept_probability"]

    all_patch_coord = {}
    print(coord_file)
    all_patch_coord["p1"] = load_matlab_file(coord_file, ("all_patch_coord", "p1"))
    all_patch_coord["p2"] = load_matlab_file(coord_file, ("all_patch_coord", "p2"))

    if all_patch_coord["p1"] is None:
        return
    if all_patch_coord["p2"] is None:
        return
    p_s = {}
    p_s["p1"] = load_matlab_file(shape_file, "p1", True)
    p_s["p2"] = load_matlab_file(shape_file, "p2", True)

    if True:
        if params["sc_filt"] > 0:
            shape_comp_25_p1 = load_matlab_file(shape_file, ("p1", "shape_comp_25"))
            shape_comp_25_p2 = load_matlab_file(shape_file, ("p2", "shape_comp_25"))
        sc_pairs = load_matlab_file(shape_file, "sc_pairs", True)
        data = np.asarray(sc_pairs["data"])
        ir = np.asarray(sc_pairs["ir"])
        jc = np.asarray(sc_pairs["jc"])
        sc_pairs = sp.csc_matrix((data, ir, jc)).astype(np.float32).nonzero()
        # Go through a subset of all shape complementary pairs.
        num_accepted = 0.0
        order = np.arange(len(sc_pairs[0]))
        # Randomly reshuffle the dataset.
        if reshuffle:
            np.random.shuffle(order)
        pairs_subset = len(order) * ppi_accept_probability
        # Always accept at least one pair.
        pairs_subset = np.ceil(pairs_subset)
        for pair_ix in order[0 : int(pairs_subset)]:
            pix1 = sc_pairs[0][pair_ix]
            pix2 = sc_pairs[1][pair_ix]

            # Filter on SC if desired.
            if params["sc_filt"] > 0:
                sc_filter_val_1 = np.asarray(shape_comp_25_p1[pix1, :].todense())[0]
                sc_filter_val_1 = np.percentile(sc_filter_val_1, 50)
                sc_filter_val_2 = np.asarray(shape_comp_25_p2[pix2, :].todense())[0]
                sc_filter_val_2 = np.percentile(sc_filter_val_2, 50)

                if np.mean([sc_filter_val_1, sc_filter_val_2]) < params["sc_filt"]:
                    discarded_sc += 1
                    continue
            # Extract the vertices within the threshold
            s1, coord1 = extract_patch_and_coord(
                pix1,
                p_s["p1"],
                all_patch_coord["p1"],
                params["max_distance"],
                params["max_shape_size"],
            )
            s2, coord2 = extract_patch_and_coord(
                pix2,
                p_s["p2"],
                all_patch_coord["p2"],
                params["max_distance"],
                params["max_shape_size"],
            )

            if s1["X"].shape[0] > max_shape_size or s2["X"].shape[0] > max_shape_size:
                discarded_large += 1
                continue

            num_accepted += 1

            pair_name = "{}_{}_{}".format(seed_pair, pix1, pix2)
            ids_p1_p2 = (len(list_shape_idx), len(list_shape_idx) + 1)
            positive_pairs_idx.add(ids_p1_p2)
            idx_positives.append(ids_p1_p2)
            list_names.append(pair_name)
            list_names.append(pair_name)

            list_desc.append(s1)
            list_desc.append(s2)
            list_coords.append(coord1)
            list_coords.append(coord2)
            list_shape_idx.append(pair_name + "_p1")
            list_shape_idx.append(pair_name + "_p2")
    print("Num accepted this pair {}".format(num_accepted))
    print("Number of pairs of shapes:  {}".format(len(idx_positives)))
    print("Discarded pairs for size (total):  {}".format(discarded_large))
    print("Discarded pairs for sc (total):  {}".format(discarded_sc))

    return list_desc, list_coords, list_shape_idx, list_names


# if do_shape_comp_pairs is false, then use the random coords.
# if label iface, return a list of points with X of the partner protein.
def read_data_from_matfile_full_protein(
    coord_file,
    shape_file,
    seed_pair,
    params,
    protein_id,
    label_iface=False,
    label_sc=False,
):

    # Ignore any shapes with more than X vertices.
    print("Reading full protein")
    max_shape_size = params["max_shape_size"]

    list_desc = []
    list_coords = []
    list_shape_idx = []
    list_names = []
    discarded_large = 0.0

    if protein_id == "p2":
        other_pid = "p1"
    else:
        other_pid = "p2"

    all_patch_coord = {}
    print(coord_file)
    all_patch_coord[protein_id] = load_matlab_file(
        coord_file, ("all_patch_coord", protein_id)
    )

    if all_patch_coord[protein_id] is None:
        return
    p_s = {}
    p_s[protein_id] = load_matlab_file(shape_file, protein_id, True)
    if label_sc:
        try:
            shape_comp_25 = load_matlab_file(shape_file, (protein_id, "shape_comp_25"))
            shape_comp_50 = load_matlab_file(shape_file, (protein_id, "shape_comp_50"))
            sc_filter_val = np.array([shape_comp_25.todense(), shape_comp_50.todense()])
            # Pad with zeros the last few lines, since they are lost when reading from a sparse matrix.
            pad_val = len(p_s[protein_id]["X"][0]) - sc_filter_val.shape[1]
            if pad_val > 0:
                padding = np.zeros((2, pad_val, 10))
                sc_filter_val = np.concatenate([sc_filter_val, padding], axis=1)
        # Not found: set -1 to all fields
        except:
            sc_filter_val = -np.ones((2, len(p_s[protein_id]["X"][0]), 10))

    [rows, cols] = all_patch_coord[protein_id].nonzero()
    assert len(np.unique(rows)) == len(p_s[protein_id]["X"][0])

    X = p_s[protein_id]["X"][0]
    Y = p_s[protein_id]["Y"][0]
    Z = p_s[protein_id]["Z"][0]

    if label_iface:
        try:
            # Read the interface information from the ply file itself.
            import pymesh

            if protein_id == "p1":
                plyfile = "_".join(seed_pair.split("_")[0:2])
            elif protein_id == "p2":
                plyfile = "{}_{}".format(
                    seed_pair.split("_")[0], seed_pair.split("_")[2]
                )
            plyfile = params["ply_chain_dir"] + plyfile + ".ply"
            iface_mesh = pymesh.load_mesh(plyfile)
            iface_labels = iface_mesh.get_attribute("vertex_iface")
        except:
            print("Unable to label interface as other protein not present. ")
            iface_labels = np.zeros((len(p_s[protein_id]["X"][0]), 1))

    list_indices = []
    for pix in range(len(p_s[protein_id]["X"][0])):
        shape, coord, neigh_indices = extract_patch_and_coord(
            pix,
            p_s[protein_id],
            all_patch_coord[protein_id],
            params["max_distance"],
            params["max_shape_size"],
            patch_indices=True,
        )
        if len(shape["X"]) > max_shape_size:
            discarded_large += 1
            continue
        list_desc.append(shape)
        list_coords.append(coord)
        shape_name = "{}_{}_rand_{}".format(seed_pair, pix, protein_id)
        list_names.append(shape_name)
        list_shape_idx.append(shape_name)
        list_indices.append(neigh_indices)
        if pix % 500 == 0:
            print("pix: {}\n".format(pix))

    print("Number of pairs of shapes:  {}".format(len(list_desc)))
    print("Discarded pairs for size (total):  {}".format(discarded_large))

    if label_iface:
        return (
            list_desc,
            list_coords,
            list_shape_idx,
            list_names,
            X,
            Y,
            Z,
            iface_labels,
            list_indices,
        )
    elif label_sc:
        return (
            list_desc,
            list_coords,
            list_shape_idx,
            list_names,
            X,
            Y,
            Z,
            sc_filter_val,
            list_indices,
        )
    else:
        return list_desc, list_coords, list_shape_idx, list_names, X, Y, Z


# Get the closest X vertices.
def compute_closest_vertices(coords, vix, X=200):
    n = coords.shape[0]
    neigh = coords[vix, :n].nonzero()[1]
    dists = np.asarray(coords[vix, neigh].todense())[0]
    neigh_dists = zip(dists, neigh)
    neigh_dists = sorted(neigh_dists)
    # Only take into account the closest X vertices.
    neigh_dists = neigh_dists[:X]
    max_dist = neigh_dists[-1]
    return neigh_dists, max_dist


def read_data_from_matfile_binding_pair(coord_file, shape_file, seed_pair, params, pid):
    # Ignore any shapes with more than X vertices.
    max_shape_size = params["max_shape_size"]

    list_desc = []
    list_coords = []
    list_shape_idx = []
    list_names = []

    other_pid = "p1"
    if pid == "p1":
        other_pid = "p2"

    # Read coordinates.

    all_patch_coord = {}
    all_patch_coord[pid] = load_matlab_file(coord_file, ("all_patch_coord", pid))
    all_patch_coord[other_pid] = load_matlab_file(
        coord_file, ("all_patch_coord", other_pid)
    )

    if all_patch_coord[pid] is None:
        return
    if all_patch_coord[other_pid] is None:
        return

    # Read shapes.
    p_s = {}
    p_s[pid] = load_matlab_file(shape_file, pid, True)
    p_s[other_pid] = load_matlab_file(shape_file, other_pid, True)

    # Compute all points in pid that are within 1.0A of a point in the other protein.
    n_pid = range(len(p_s[pid]))
    v1 = np.vstack([p_s[pid]["X"][0], p_s[pid]["Y"][0], p_s[pid]["Z"][0]]).T
    v2 = np.vstack(
        [p_s[other_pid]["X"][0], p_s[other_pid]["Y"][0], p_s[other_pid]["Z"][0]]
    ).T

    flann = pyflann.FLANN()
    closest_vertex_in_v2, dist = flann.nn(v2, v1)
    dist = np.sqrt(dist)
    iface1 = np.where(dist <= params["pos_interface_cutoff"])[0]

    # Find the corresponding point in iface2.
    iface2 = closest_vertex_in_v2[iface1]

    # Now to iface1 add all negatives.
    iface1_neg = np.where(dist > params["neg_interface_cutoff"])[0]
    K = params["neg_surf_accept_probability"] * len(iface1_neg)
    k = np.random.choice(len(iface1_neg), int(K))
    iface1_neg = iface1_neg[k]

    labels1 = np.concatenate([np.ones_like(iface1), np.zeros_like(iface1_neg)], axis=0)
    iface1 = np.concatenate([iface1, iface1_neg], axis=0)

    # Compute flann from iface2 to iface1
    flann = pyflann.FLANN()
    closest_vertex_in_v1, dist = flann.nn(v1, v2)
    dist = np.sqrt(dist)
    iface2_neg = np.where(dist > params["neg_interface_cutoff"])[0]
    # Randomly sample iface2_neg
    K = params["neg_surf_accept_probability"] * len(iface2_neg)
    k = np.random.choice(len(iface2_neg), int(K))
    iface2_neg = iface2_neg[k]

    labels2 = np.concatenate([np.ones_like(iface2), np.zeros_like(iface2_neg)], axis=0)
    iface2 = np.concatenate([iface2, iface2_neg], axis=0)

    list_desc_binder = []
    list_coord_binder = []
    list_names_binder = []
    list_vert_binder = []
    for vix in iface1:
        s, coord = extract_patch_and_coord(
            vix,
            p_s[pid],
            all_patch_coord[pid],
            params["max_distance"],
            params["max_shape_size"],
        )
        name = "{}_{}_{}".format(seed_pair, pid, vix)
        list_desc_binder.append(s)
        list_coord_binder.append(coord)
        list_names_binder.append(name)
        list_vert_binder.append(v1[vix])

    list_desc_pos = []
    list_coord_pos = []
    list_names_pos = []
    list_vert_pos = []
    for vix in iface2:
        s, coord = extract_patch_and_coord(
            vix,
            p_s[other_pid],
            all_patch_coord[other_pid],
            params["max_distance"],
            params["max_shape_size"],
        )
        name = "{}_{}_{}".format(seed_pair, other_pid, vix)
        list_desc_pos.append(s)
        list_coord_pos.append(coord)
        list_names_pos.append(name)
        list_vert_pos.append(v2[vix])

    return (
        list_desc_binder,
        list_coord_binder,
        list_names_binder,
        list_vert_binder,
        labels1,
        list_desc_pos,
        list_coord_pos,
        list_names_pos,
        list_vert_pos,
        labels2,
    )


# Select points that are farther than X from the other protein.
def read_data_from_matfile_negative_patches(
    coord_file, shape_file, ppi_pair_id, params, pid, other_pid
):
    all_patch_coord = {}
    all_patch_coord[pid] = load_matlab_file(coord_file, ("all_patch_coord", pid))
    if all_patch_coord[pid] is None:
        return

    # Read shapes.
    p_s = {}
    p_s[pid] = load_matlab_file(shape_file, pid, True)
    p_s[other_pid] = load_matlab_file(shape_file, other_pid, True)

    n = len(p_s[pid]["X"][0])
    vert = np.vstack([p_s[pid]["X"][0], p_s[pid]["Y"][0], p_s[pid]["Z"][0]]).T
    other_vert = np.vstack(
        [p_s[other_pid]["X"][0], p_s[other_pid]["Y"][0], p_s[other_pid]["Z"][0]]
    ).T

    flann = pyflann.FLANN()
    closest_vertex_in_v2, dist = flann.nn(v2, v1)
    dist = np.sqrt(dist)
    iface1 = np.where(dist > 1.5)[0]

    # Find the corresponding point in iface2.
    iface2 = closest_vertex_in_v2[iface1]

    neigh_dists, _ = compute_closest_vertices(all_patch_coord[pid], rvix, X=25)
    neigh = [x[1] for x in neigh_dists]

    list_desc_neg = []
    list_coord_neg = []
    list_names_neg = []
    list_vert_neg = []
    for vix in neigh:
        s, coord = extract_patch_and_coord(
            vix,
            p_s[pid],
            all_patch_coord[pid],
            params["max_distance"],
            params["max_shape_size"],
        )
        name = "{}_{}_{}".format(ppi_pair_id, pid, vix)
        list_desc_neg.append(s)
        list_coord_neg.append(coord)
        list_names_neg.append(name)
        list_vert_neg.append(rvert[vix])

    return list_desc_neg, list_coord_neg, list_names_neg, list_vert_neg

