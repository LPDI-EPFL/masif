import numpy as np 
from IPython.core.debugger import set_trace
import copy
from Bio.PDB import *
import os
from geometry.open3d_import import *

def compute_nn_score(
    target_ckdtree,
    target_patch,
    source_patch,
    target_descs,
    source_patch_descs,
    nn_model):
    # Compute the score of the neural network. 
    # target_ckdtree: KD-tree of the target patch point cloud. 
    # target_patch: Patch of the target. 
    # source_patch: Patch of the source.
    # target_descs: Fingerprint descriptors of the target
    # source_patch_descs:  Fingerprint descriptors of the source.
    # Returns: the score of the alignment. 

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

    # Evaluate patch with neural network. 
    pred = nn_model.eval(features_trimmed)
    return pred[0][1]

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix. used to randomize initial pose. 

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


def multidock(
    source_pcd, # the candidate 'decoy' patches
    source_patch_coords, # The geodesic coordinates of the decoy patches
    source_descs, # The source descriptors
    cand_pts, # The central vertex of the candidate patches.
    target_pcd, # The target patch's point cloud
    target_descs, # The descriptors for the target patch
    target_ckdtree, # A kd-tree for the target patch for fast searches. 
    nn_model, # The neural network model
    ransac_radius=1.0, # The radius fro RANSAC inliers.
    ransac_iter=2000,
    use_icp=True
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
        source_patch, source_patch_descs = get_patch_geo(
            source_pcd, source_patch_coords, pt, source_descs
        )

        # Align the two patches
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
            RANSACConvergenceCriteria(ransac_iter, 500)
        )
        # Optimize the alignment using RANSAC.
        if use_icp:
            result = registration_icp(source_patch, target_pcd, 
            1.0, result.transformation, TransformationEstimationPointToPlane(),
            )

        source_patch.transform(result.transformation)
        all_results.append(result)
        all_source_patch.append(source_patch)

        # Compute the neural network score for each alignment.
        source_scores = compute_nn_score(
            target_ckdtree,
            target_pcd, 
            source_patch,
            target_descs,
            source_patch_descs,
            nn_model
        )
        all_source_scores.append(source_scores)

    return all_results, all_source_patch, all_source_scores


def test_alignments(
    transformation, # the 4D transformation matrix 
    random_transformation, # The random transformation matrix used before.
    source_structure, # The source (decoy) binder structure in a Biopython object 
    target_ca_pcd_tree, # The target c-alphas in an Open3D point cloud tree format.
    target_pcd_tree, # The target atoms in an Open 3D point cloud tree format.
    radius=2.0, # The radius for clashes (unused) 
    interface_dist=10.0, # The interface cutoff to define the interface.
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
    # Compute clashes (unused now) 
    for point in structure_coord_pcd.points:
        [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, radius)
        if k > 0:
            clashing += 1

    interface_atoms = []
    # Compute structures.
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
        compute_desc_dist_score: a simple scoring based on fingerprints (unused currently in this protocol)
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




def subsample_patch_coords(pdb, pid, precomp_dir, cv=None):
    """
        subsample_patch_coords: Read the geodesic coordinates in an easy to access format.
        pdb: the id of the protein pair in PDBID_CHAIN1_CHAIN2 format.
        pid: 'p1' if you want to read CHAIN1, 'p2' if you want to read CHAIN2
        cv: central vertex (list of patches to select; if None, select all)
    """

    if cv is None:
        pc = np.load(os.path.join(precomp_dir, pdb, pid+'_list_indices.npy'))
    else:
        pc = {}
        coords = np.load(os.path.join(precomp_dir, pdb, pid+'_list_indices.npy'))[cv]
        for iii, v in enumerate(cv):
            pc[v] = coords[iii]


    return pc

def get_target_vix(pc, iface):
    """ 
        Get center of the patch on the target surface that has the highest mean iface score.
        pc: patch coords
        iface: a vector with interface scores for the target.
    """
    iface_patch_vals = []
    # Go through each patch.
    for ii in range(len(pc)):

        neigh = pc[ii]
        val = np.mean(iface[neigh])

        iface_patch_vals.append(val)

    target_vix = np.argmax(iface_patch_vals)

    return target_vix
