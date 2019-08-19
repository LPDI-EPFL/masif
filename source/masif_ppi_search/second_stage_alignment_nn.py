#!/usr/bin/env python
# coding: utf-8
import sys
import time
import sklearn.metrics
from open3d import *
import numpy as np
import os
from transformation_training_data.score_nn import ScoreNN
from scipy.spatial import cKDTree
from Bio.PDB import *
import copy
import scipy.sparse as spio
from default_config.masif_opts import masif_opts
import sys

# Start measuring the cpu clock time here. 
# We will not count the time required to align the structures and verify the ground truth. 
#               This time will be subtracted at the end.
global_start_time = time.clock()
global_ground_truth_time = 0.0

nn_model = ScoreNN()
print(sys.argv)
if len(sys.argv) != 7 or (sys.argv[5] != "masif" and sys.argv[5] != "gif"):
    print("Usage: {} data_dir K ransac_iter num_success gif|masif".format(sys.argv[0]))
    print("data_dir: Location of data directory.")
    print("K: Number of descriptors to run")
    print("ransac_iter: number of ransac iterations.")
    print("num_success: true alignment within short list of size num_success")
    print("random seed: to randomize RANSAC")
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

coord_dir = os.path.join(data_dir, masif_opts["coord_dir_npy"])

pdb_dir = os.path.join(data_dir, masif_opts["pdb_chain_dir"])
precomp_dir = os.path.join(
    data_dir, masif_opts["ppi_search"]["masif_precomputation_dir"]
)

benchmark_list = "../benchmark_list.txt"


pdb_list = open(benchmark_list).readlines()[0:100]
pdb_list = [x.rstrip() for x in pdb_list]

def compute_nn_score(
    target_ckdtree,
    target_patch,
    source_patch,
    target_descs,
    source_patch_descs):

    # Neural network max size is 200. Those bigger must be trimmed.
    npoints = np.asarray(source_patch.points).shape[0]
    n_features = 3
    max_npoints = 200
    # Compute nn scores
    # Compute all points correspondences and distances for nn
    nn_dists, nn_inds = target_ckdtree.query(source_patch.points)
    desc_dists = np.linalg.norm(source_patch_descs.data.T-target_descs.data.T[nn_inds],axis=1)
    normal_dp = np.diag(np.dot(np.asarray(source_patch.normals),np.asarray(target_patch.normals)[nn_inds].T))
    features = np.zeros((npoints,n_features))
    nn_dists[nn_dists<0.5] = 0.5
    features[:npoints,0] = 1.0/nn_dists
    features[:npoints,1] = 1.0/desc_dists
    features[:npoints,2] = normal_dp

    features_trimmed = np.zeros((1,max_npoints,n_features))
    if npoints>max_npoints:
        selected_rows = np.random.choice(features.shape[0],max_npoints,replace=False)
        features_trimmed[0,:,:] = features[selected_rows]
    else:
        features_trimmed[0,:features.shape[0],:] = features

    pred = nn_model.eval(features_trimmed)
    return pred[0][1]

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
    target_ckdtree,
    ransac_radius=1.0,
):
    all_results = []
    all_source_patch = []
    all_source_scores = []
    patch_time = 0.0
    ransac_time = 0.0
    transform_time = 0.0
    score_time = 0.0
    for pt in cand_pts:
        source_patch, source_patch_descs = get_patch_geo(
            source_pcd, source_patch_coords, pt, source_descs
        )

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
                CorrespondenceCheckerBasedOnDistance(1.5),
                CorrespondenceCheckerBasedOnNormal(np.pi / 2),
            ],
            RANSACConvergenceCriteria(ransac_iter, 500), random_seed
        )
        result = registration_icp(source_patch, target_pcd, 
            1.0, result.transformation, TransformationEstimationPointToPlane(),
            )

        source_patch.transform(result.transformation)
        all_results.append(result)
        all_source_patch.append(source_patch)

        source_scores = compute_nn_score(
            target_ckdtree,
            target_pcd, 
            source_patch,
            target_descs,
            source_patch_descs
        )
        all_source_scores.append(source_scores)

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


def subsample_patch_coords(pdb, pid, cv=None, frac=1.0, radius=9.0):
    patch_coords = spio.load_npz(os.path.join(coord_dir, pdb, pid + ".npz"))

    if cv is None:
        D = np.squeeze(
            np.asarray(patch_coords[:, : patch_coords.shape[1] // 2].todense())
        )
    else:
        D = np.squeeze(
            np.asarray(patch_coords[cv, : patch_coords.shape[1] // 2].todense())
        )
    # Get nonzero fields, points under radius.
    idx = np.where(np.logical_and(D > 0.0, D < radius))

    # Convert to dictionary;

    pc = {}
    for ii in range(len(idx[0])):
        # With probability frac, ignore this entry point - always include the center poitn.
        if cv is None:
            cvix = idx[0][ii]
            val = idx[1][ii]
        else:
            cvix = cv
            val = idx[0][ii]
        if np.random.random() < frac or cvix == val:
            if cvix not in pc:
                pc[cvix] = []
            pc[cvix].append(val)

    return pc


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

    # Read patch coordinates. Subsample

    pc = subsample_patch_coords(pdb, "p2", frac=1.0)
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
    target_coord = subsample_patch_coords(target_pdb, "p1", center_point, frac=1.0)

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

    # Go thorugh every source pdb.
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
        all_results, all_source_patch, all_source_scores = multidock(
            source_pcd,
            source_coords,
            source_desc,
            source_vix,
            target_patch,
            target_patch_descs,
            target_ckdtree
        )
        num_negs = num_negs

        # If this is the source_pdb, get the ground truth.
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
        auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
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

