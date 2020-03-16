#!/usr/bin/env python
# coding: utf-8
import sys
from open3d import *
import numpy as np
import os
from sklearn.manifold import TSNE
from Bio.PDB import *
import copy
import scipy.sparse as spio
from default_config.masif_opts import masif_opts
import sys

"""
second_stage_alignment.py: Second stage alignment code for benchmarking MaSIF-search WITHOUT neural network scoring. 
                            It is recommended to use masif_second_stage_nn.py instead.
                            This code benchmarks MaSIF-search by generating 3D alignments
                            of the protein.
                            The method consists of two stages: 
                            (1) Read a database of MaSIF-search fingerprint descriptors for each overlapping patch, and find the top K decoys that are the most similar to the 
                            target 
                            (2) Align and score these patches:
                                (2a) Use the RANSAC algorithm + the iterative closest point algorithm to align each patch
                                (2b) Use a simple function based on the fingerprints of all aligned points to score the alignment
                            
Pablo Gainza  and Freyr Sverrisson- LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

print(sys.argv)
if len(sys.argv) != 7 or (sys.argv[5] != "masif" and sys.argv[5] != "gif"):
    print("Usage: {} data_dir K ransac_iter num_success gif|masif".format(sys.argv[0]))
    print("data_dir: Location of data directory.")
    print("K: Number of descriptors to run")
    print("ransac_iter: number of ransac iterations.")
    print("num_success: true alignment within short list of size num_success")
    sys.exit(1)

data_dir = sys.argv[1]
K = int(sys.argv[2])
ransac_iter = int(sys.argv[3])
num_success = int(sys.argv[4])
method = sys.argv[5]
random_seed = int(sys.argv[6])

surf_dir = os.path.join(data_dir, masif_opts["ply_chain_dir"])

if method == "gif":
    desc_dir = os.path.join(data_dir, masif_opts["ppi_search"]["gif_descriptors_out"])
else:  # MaSIF
    desc_dir = os.path.join(data_dir, masif_opts["ppi_search"]["desc_dir"])

pdb_dir = os.path.join(data_dir, masif_opts["pdb_chain_dir"])
precomp_dir = os.path.join(
    data_dir, masif_opts["ppi_search"]["masif_precomputation_dir"]
)
precomp_dir_9A = os.path.join(
    data_dir, masif_opts["site"]["masif_precomputation_dir"]
)

benchmark_list = "../benchmark_list.txt"


pdb_list = open(benchmark_list).readlines()[0:100]
pdb_list = [x.rstrip() for x in pdb_list]


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
    """
        Get the center of a point cloud and randomly rotate it.
        pcd: the point cloud.
    """
    pts = pcd.points
    mean_pt = np.mean(pts, axis=0)
    # pts = pts - mean_pt
    rand_mat = rand_rotation_matrix()
    # pts = Vector3dVector(np.dot(pts,rand_mat))
    transform = np.vstack([rand_mat.T, -mean_pt]).T
    # transform = np.vstack([np.diag([1,1,1]),-mean_pt]).T
    transform = np.vstack([transform, [0, 0, 0, 1]])
    return transform


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
    pts = np.asarray(pcd.points)[idx, :]
    nrmls = np.asarray(pcd.normals)[idx, :]
    pts = pts + outward_shift * nrmls
    if flip:
        nrmls = -np.asarray(pcd.normals)[idx, :]

    patch = PointCloud()
    patch.points = Vector3dVector(pts)
    patch.normals = Vector3dVector(nrmls)
    patch_descs = Feature()
    patch_descs.data = descriptors[idx, :].T
    return patch, patch_descs


def multidock(
    source_pcd,
    source_patch_coords,
    source_descs,
    cand_pts,
    target_pcd,
    target_descs,
    ransac_radius=1.0,
):
    """
    Multi-docking protocol: Here is where the alignment is actually made. 
    This method aligns each of the K prematched decoy patches to the target using tehe
    RANSAC algorithm followed by icp
    """
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
                CorrespondenceCheckerBasedOnDistance(2.0),
                CorrespondenceCheckerBasedOnNormal(np.pi / 2),
            ],
            RANSACConvergenceCriteria(ransac_iter, 500), random_seed
        )
        result = registration_icp(source_patch, target_pcd, 
            1.0, result.transformation, TransformationEstimationPointToPlane())
        ransac_time = ransac_time + (time.time() - tic)

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

    return all_results, all_source_patch, all_source_scores


def test_alignments(
    transformation,
    random_transformation,
    source_structure,
    target_ca_pcd_tree,
    target_pcd_tree,
    radius=2.0,
    interface_dist=10.0,
):
    """
    Verify the alignment against the ground truth. 
    """
    structure_coords = np.array(
        [
            atom.get_coord()
            for atom in source_structure.get_atoms()
            if atom.get_id() == "CA"
        ]
    )
    structure_coord_pcd = PointCloud()
    structure_coord_pcd.points = Vector3dVector(structure_coords)
    structure_coord_pcd_notTransformed = copy.deepcopy(structure_coord_pcd)
    structure_coord_pcd.transform(random_transformation)
    structure_coord_pcd.transform(transformation)

    clashing = 0
    for point in structure_coord_pcd.points:
        [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, radius)
        if k > 0:
            clashing += 1

    interface_atoms = []
    for i, point in enumerate(structure_coords):
        [k, idx, _] = target_ca_pcd_tree.search_radius_vector_3d(point, interface_dist)
        if k > 0:
            interface_atoms.append(i)
    rmsd = np.sqrt(
        np.mean(
            np.square(
                np.linalg.norm(
                    structure_coords[interface_atoms, :]
                    - np.asarray(structure_coord_pcd.points)[interface_atoms, :],
                    axis=1,
                )
            )
        )
    )
    return (
        rmsd,
        clashing,
        structure_coord_pcd,
        structure_coord_pcd_notTransformed,
    )  # , structure, structure_coord_pcd


# Compute different types of scores:
# -- Inverted sum of the minimum descriptor distances squared cutoff.
def compute_desc_dist_score(
    target_pcd, source_pcd, corr, target_desc, source_desc, cutoff=2.0
):
    """
        compute_desc_dist_score: a simple scoring based on fingerprints 
    """

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


from IPython.core.debugger import set_trace


def subsample_patch_coords(pdb, pid, cv=None):
    """
        subsample_patch_coords: Read the geodesic coordinates in an easy to access format.
        pdb: the id of the protein pair in PDBID_CHAIN1_CHAIN2 format.
        pid: 'p1' if you want to read CHAIN1, 'p2' if you want to read CHAIN2
        cv: central vertex 
    """

    if cv is None:
        pc = np.load(os.path.join(precomp_dir_9A, pdb, pid+'_list_indices.npy'))
    else:
        pc = {}
        pc[cv] = np.load(os.path.join(precomp_dir_9A, pdb, pid+'_list_indices.npy'))[cv]


    return pc


# Read all surfaces.
all_pc = []
all_desc = []

rand_list = np.copy(pdb_list)
np.random.seed(0)
np.random.shuffle(rand_list)
rand_list = rand_list[0:100]

p2_descriptors_straight = []
p2_point_clouds = []
p2_patch_coords = []
p2_names = []

# Read all of p2. p2 will have straight descriptors.
for i, pdb in enumerate(rand_list):
    print("Running on {}".format(pdb))
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

    # Read patch coordinates. Subsample

    pc = subsample_patch_coords(pdb, "p2")
    p2_patch_coords.append(pc)

    p2_names.append(pdb)


import time
import scipy.spatial

# Read all of p1, the target. p1 will have flipped descriptors.

all_positive_scores = []
all_positive_rmsd = []
all_negative_scores = []
# Match all descriptors.
count_found = 0
all_rankings_desc = []
all_time_global = []
for target_ix, target_pdb in enumerate(rand_list):
    tic = time.time()
    print(target_pdb)
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
    tic_global = time.time()
    num_negs = 0
    all_desc_dists = []
    all_pdb_id = []
    all_vix = []
    gt_dists = []

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
    target_coord = subsample_patch_coords(target_pdb, "p1", center_point)

    # Get the geodesic patch and descriptor patch for the target.
    target_patch, target_patch_descs = get_patch_geo(
        target_pcd, target_coord, center_point, target_desc, flip=True
    )

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
    time_global = time.time() - tic

    # Go through every source pdb.
    for source_ix, source_pdb in enumerate(rand_list):
        tic = time.time()
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
        all_results, all_source_patch, all_source_scores = multidock(
            source_pcd,
            source_coords,
            source_desc,
            source_vix,
            target_patch,
            target_patch_descs,
        )
        toc = time.time()
        time_global += toc - tic
        num_negs = num_negs

        # If this is the source_pdb, get the ground truth.
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
                score = list(all_source_scores[j])
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
                score = list(all_source_scores[j])
                neg_scores.append(score)
    if found:
        count_found += 1
        all_rankings_desc.append(myrank_desc)
        print(myrank_desc)
    else:
        print("N/D")

    all_positive_rmsd.append(pos_rmsd)
    all_positive_scores.append(pos_scores)
    all_negative_scores.append(neg_scores)
    print("Took {:.2f}s".format(time_global))
    all_time_global.append(time_global)
    # Go through every top descriptor.


print("All alignments took {}".format(np.sum(all_time_global)))


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

for pdb_ix in range(len(all_positive_scores)):
    if len(all_positive_scores[pdb_ix]) == 0:
        print("N/D")
        unranked += 1
    else:
        pos_scores = [x[0] for x in all_positive_scores[pdb_ix]]
        neg_scores = [x[0] for x in all_negative_scores[pdb_ix]]
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
print(
    "Number in top 100 {} out of {}".format(np.sum(ranks <= 100), len(all_positive_scores))
)
print(
    "Number in top 10 {} out of {}".format(np.sum(ranks <= 10), len(all_positive_scores))
)
print(
    "Number in top 1 {} out of {}".format(np.sum(ranks <= 1), len(all_positive_scores))
)
print("Median rank for correctly ranked ones: {}".format(np.median(ranks)))
print("Mean rank for correctly ranked ones: {}".format(np.mean(ranks)))
print(
    "Number failed {} out of {}".format(np.median(unranked), len(all_positive_scores))
)

outfile = open("results_{}.txt".format(method), "a+")
outfile.write("K,Total,Top2000,Top1000,Top100,Top10,Top5,Top1,MeanRMSD,Time\n")
top2000= np.sum(ranks<=2000)
top1000= np.sum(ranks<=1000)
top100= np.sum(ranks<=100)
top10= np.sum(ranks<=10)
top5= np.sum(ranks<=5)
top1= np.sum(ranks<=1)
meanrmsd = np.mean(rmsds)
runtime = np.sum(all_time_global)


outline = "{},{},{},{},{},{},{},{},{},{}\n".format(
    K, len(all_positive_scores), top2000, top1000, top100, top10, top5, top1, meanrmsd, runtime
)
outfile.write(outline)

outfile.close()

sys.exit(0)

