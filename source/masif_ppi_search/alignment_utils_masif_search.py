import numpy as np 
import scipy
from IPython.core.debugger import set_trace
import copy
from Bio.PDB import *
import os
from geometry.open3d_import import *
from masif_ppi_search.differentiable_alignment.rand_rotation import center_patch

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
    pred = nn_model['scoring_nn'].eval(features_trimmed)
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

# Align source patch to target patch with SVD. 
def align_source_to_target_svd(source_patch, source_patch_descs, target_pcd, target_patch_descs, target_ckd_desc, nn_models):
    # Compute the nearest neighborhood from target to source. 
    sdesc = source_patch_descs.data.T
    desc_d_patch, desc_r_patch = target_ckd_desc.query(sdesc)

    patch1 = np.asarray(target_pcd.points)
    patch1_n = np.asarray(target_pcd.normals)

    patch2 = np.asarray(source_patch.points)
    patch2_copy = np.copy(patch2)
    patch2_n = np.asarray(source_patch.normals)

    dist_0 = np.sqrt(np.sum(np.square(patch1[0] - patch2[0])))

    # We must center the two patches for the neural network. 
    patch1, translation_value1 = center_patch(patch1)
    patch2, translation_value2 = center_patch(patch2)

    # Feat1: the descriptor distance between points. 
    feat1 = desc_d_patch
    feat1 = np.expand_dims(feat1, axis=1)

    # Feat2: xyz of target, according to correspondences
    feat2 = patch1[desc_r_patch]

    # Feat3: xyz of source, 
    feat3 = patch2

    # Feat4: normal of target, according to correspondences
    feat4= patch1_n[desc_r_patch]

    # Feat5: noraml of source
    feat5 = patch2_n

    # features: nx13 matrix, where n is the number of points. 
    features = np.concatenate([feat1, feat2, feat3, feat4, feat5], axis=1)

    # Pad with repeat ? 
    if len(feat1) < 100: # Set to actual params!.
        repeat_corrs = np.arange(len(feat1))
        repeat_corrs = np.concatenate([repeat_corrs, repeat_corrs, repeat_corrs], axis=0)
        np.random.shuffle(repeat_corrs)
        repeat_corrs = repeat_corrs[:100-len(feat1)]
        features_pad = np.zeros((100, 13))
        features_pad[0:len(feat1),:] = features
        features_pad[len(feat1):,:] = features[repeat_corrs,:]
        features = features_pad

    features = np.expand_dims(features, axis=0)
    
    # Get the predicted correspondences.
    ypred = nn_models['corr_nn'].eval(features)
    ypred = np.squeeze(ypred)

    # Replace the descriptor distance feature with the correspondences.
    features[:,:,0] = ypred

    # Get the predicted alignment
    # The rotation and translation are stored in fields 6:15 and 15:18 
    #           respectively; they are the same for the entire patch, obviously 
    new_coords_norm2_R = nn_models['align_nn'].eval(features)
    # Get the rotation matrix
    new_coords2 = new_coords_norm2_R[:,:,0:3]
    R = new_coords_norm2_R[0,0,6:15]
    trans_vec = new_coords_norm2_R[0,0,15:18]
    R = np.reshape(R, [3,3])
    eval_patch = np.dot(R, patch2.T).T + trans_vec
    # Ensure nothing went wrong with R and the eval_patch
    diff = np.mean(np.sqrt(np.sum(np.square(new_coords2[0,:len(feat1),:] - eval_patch), axis=1)))
    assert(np.mean(diff) < 1e4)

    # Make a transformation matrix 
    trans_matrix = np.identity(4)
    trans_matrix[0:3,0:3] = R
    trans_matrix[0:3,3] = trans_vec.T 

    translation_mat1 = np.identity(4)
    translation_mat1[0:3,3] = translation_value1

    translation_mat2 = np.identity(4)
    translation_mat2[0:3,3] = -translation_value2

    if (dist_0 < 1.5):
        test_pcd = PointCloud()
        in_patch = patch2_copy - translation_value2
        test_pcd.points = Vector3dVector(in_patch)
        test_pcd.normals = Vector3dVector(patch2_n)
        test_pcd.transform(trans_matrix)
        test_points = np.asarray(test_pcd.points)
        test_points = test_points + translation_value1
        rmsd = np.sqrt(np.mean(np.sum(np.square(patch2_copy - test_points), axis=1)))
    
    return trans_matrix, translation_mat1, translation_mat2

def multidock_cn_svd(
    source_pcd, # the candidate 'decoy' patches
    source_patch_coords, # The geodesic coordinates of the decoy patches
    source_descs, # The source descriptors
    cand_pts, # The central vertex of the candidate patches.
    target_pcd, # The target patch's point cloud
    target_descs, # The descriptors for the target patch
    target_ckdtree, # A kd-tree for the target patch for fast searches. 
    nn_models, # The neural network models for correspondences, alignment, and scoring.
    use_icp=True
):
    """
    Multi-docking protocol using context normalization and SVD (i.e. avoids RANSAC).
    Here is where the alignment is actually made. 
    This method aligns each of the K prematched decoy patches to the target by first using
    a neural network to establish correspondences and then using a neural network to align using SVD.
    """

    all_results = []
    all_source_patch = []
    all_source_scores = []
    all_translations_target = []
    all_translations_source = []
    all_rotations = []
    patch_time = 0.0
    ransac_time = 0.0
    transform_time = 0.0
    score_time = 0.0

    tdesc = np.asarray(target_descs.data).T
    desc_kdt_patch = scipy.spatial.cKDTree(tdesc)

    for pt in cand_pts:
        source_patch, source_patch_descs = get_patch_geo(
            source_pcd, source_patch_coords, pt, source_descs
        )

        patch1 = np.asarray(target_pcd.points)
        patch2 = np.asarray(source_patch.points)

        dist_0 = np.sqrt(np.sum(np.square(patch1[0] - patch2[0])))
        svd_transformation, translation_mat1, translation_mat2 = align_source_to_target_svd(source_patch, source_patch_descs, target_pcd, target_descs, desc_kdt_patch, nn_models)

        # Transform the patch:
        orig_points = np.copy(np.asarray(source_patch.points))
        source_patch.transform(translation_mat2)
        source_patch.transform(svd_transformation)
        source_patch.transform(translation_mat1)

        #if dist_0 < 1.5:
        #    set_trace()
        # Optimize the alignment using ICP.
        if use_icp:
            result = registration_icp(source_patch, target_pcd, 
                1.0, np.identity(4), TransformationEstimationPointToPlane(),
                )
            result = result.transformation
        else:
            result = np.identity(4)

        source_patch.transform(result)
        all_results.append(result)
        all_translations_target.append(translation_mat1)
        all_translations_source.append(translation_mat2)
        all_rotations.append(svd_transformation)
        all_source_patch.append(source_patch)

        # Compute the score 
        source_scores = compute_nn_score(
            target_ckdtree,
            target_pcd, 
            source_patch,
            target_descs,
            source_patch_descs,
            nn_models
        )
        all_source_scores.append(source_scores)

    return all_results, all_source_patch, all_source_scores, all_rotations, all_translations_target, all_translations_source

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
    svd_rotation, \
    svd_translation_target, \
    svd_translation_source, \
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
    # Apply a translation: the source patch translation to the center
    structure_coord_pcd.transform(svd_translation_source)
    structure_coord_pcd.transform(svd_rotation)
    structure_coord_pcd.transform(svd_translation_target)
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
