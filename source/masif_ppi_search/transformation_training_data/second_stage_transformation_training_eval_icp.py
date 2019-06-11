#!/usr/bin/env python
# coding: utf-8
import sys
from open3d import *
#import ipdb
import numpy as np
import os
from sklearn.manifold import TSNE
from Bio.PDB import *
import copy
import scipy.sparse as spio
from default_config.masif_opts import masif_opts
import sys
from scipy.spatial import cKDTree

print(sys.argv)
if len(sys.argv) != 6:
    print('Usage: {} data_dir K ransac_iter patch_radius output_dir'.format(sys.argv[0]))
    print('data_dir: Location of data directory.')
    print('K: Number of descriptors to run')
    sys.exit(1)

data_dir = sys.argv[1]
K=int(sys.argv[2])
ransac_iter = int(sys.argv[3])
PATCH_RADIUS = float(sys.argv[4])
out_base = sys.argv[5]


#scratch_dir = '/
surf_dir = os.path.join(data_dir,masif_opts['ply_chain_dir'])
  
desc_dir = os.path.join(data_dir,masif_opts['ppi_search']['desc_dir'])

# chemical and all features descriptors, iface.
iface_dir = os.path.join(data_dir, '/home/gainza/lpdi_fs/masif/data/masif_site/output/all_feat_3l/pred_data/')
desc_dir_no_scfilt_chem = os.path.join(data_dir,'descriptors/sc_nofilt/chem/')
desc_dir_no_scfilt_all_feat = os.path.join(data_dir,'descriptors/sc_nofilt/all_feat/')

coord_dir = os.path.join(data_dir,masif_opts['coord_dir_npy'])

pdb_dir = os.path.join(data_dir, masif_opts['pdb_chain_dir'])
precomp_dir = os.path.join(data_dir, masif_opts['ppi_search']['masif_precomputation_dir'])
precomp_dir_9A = os.path.join(data_dir, masif_opts['site']['masif_precomputation_dir'])

benchmark_list = 'training_list.txt'

# In[3]:

pdb_list = open(benchmark_list).readlines()
pdb_list = [x.rstrip() for x in pdb_list]

 

# In[4]:


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

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def get_center_and_random_rotate(pcd):
    pts = pcd.points
    mean_pt = np.mean(pts, axis=0)
    #pts = pts - mean_pt
    rand_mat = rand_rotation_matrix()
    #pts = Vector3dVector(np.dot(pts,rand_mat))
    transform = np.vstack([rand_mat.T,-mean_pt]).T
    #transform = np.vstack([np.diag([1,1,1]),-mean_pt]).T
    transform = np.vstack([transform, [0,0,0,1]])
    return transform

def get_patch_mesh_geo(triangle_mesh,patch_coords,center,descriptors,outward_shift=0.30, flip=False):
    idx = patch_coords[center]
    n = len(triangle_mesh.vertices)
    pts = np.asarray(triangle_mesh.vertices)[idx,:]
    nrmls = np.asarray(triangle_mesh.vertex_normals)[idx,:]
    pts = pts + outward_shift*nrmls
    if flip:
        nrmls = -np.asarray(triangle_mesh.vertex_normals)[idx,:]

    patch = TriangleMesh()
    patch.vertices= Vector3dVector(pts)
    patch.vertex_normals = Vector3dVector(nrmls)

    # Extract triangulation.
    m = np.zeros(n,dtype=int)

    for i in range(len(idx)):
        m[idx[i]] = i
    f = triangle_mesh.triangles
    nf = len(f)

    idx = set(idx)
    subf = [[m[f[i][0]], m[f[i][1]], m[f[i][2]]] for i in range(nf) \
             if f[i][0] in idx and f[i][1] in idx and f[i][2] in idx ]

    patch.triangles = Vector3iVector(np.asarray(subf))
    return patch

# Descriptors is a vector of descriptors of different types of different types.
def get_patch_geo(pcd,patch_coords,geo_dists, center,descriptors, iface_scores, outward_shift=0.30, flip=False):
    idx = patch_coords[center]
    patch_geo_dists = geo_dists[center]
    pts = np.asarray(pcd.points)[idx,:]
    nrmls = np.asarray(pcd.normals)[idx,:]
    pts = pts + outward_shift*nrmls
    if flip:
        nrmls = -np.asarray(pcd.normals)[idx,:]

    patch = PointCloud()
    patch.points = Vector3dVector(pts)
    patch.normals = Vector3dVector(nrmls)
    patch_descs = []
    for desc_type in range(len(descriptors)):
        pd = Feature()
        pd.data = descriptors[desc_type][idx,:].T
        patch_descs.append(pd)
    patch_iface_scores = iface_scores[idx]
    return patch, patch_descs, patch_iface_scores, patch_geo_dists

def multidock(source_pcd,source_patch_coords,source_descs,cand_pts,target_pcd,target_descs,\
            source_iface,source_geo_dists,\
            target_patch_mesh, target_patch_mesh_centroids,ransac_radius=1.0):
    all_results = []
    all_source_patch = []
    all_source_patch_descs = []
    all_source_scores = []
    all_source_geo_dists = []
    all_source_inlier_onehot = []
    all_source_iface_scores = []
    patch_time = 0.0
    ransac_time = 0.0
    transform_time = 0.0
    score_time = 0.0
    for pt in cand_pts:
        tic = time.time()
        source_patch, source_patch_descs, source_patch_iface_scores, source_patch_geo_dists = \
            get_patch_geo(source_pcd,source_patch_coords,source_geo_dists, pt,source_descs, source_iface)
        patch_time = patch_time + (time.time() - tic)
        tic = time.time()
        if True:
            result = registration_ransac_based_on_shape_complementarity(
                source_patch, target_pcd, target_patch_mesh, target_patch_mesh_centroids, source_patch_descs[0], target_descs[0],
                1.5,
                TransformationEstimationPointToPoint(False), 3,
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(2.0),
                CorrespondenceCheckerBasedOnNormal(np.pi/2)],
                RANSACConvergenceCriteria(10000, 500),0,3)
#            result = registration_icp_based_on_shape_complementarity(source_patch, target_patch, 
#                    target_patch_mesh, target_patch_mesh_centroids, source_patch_descs[0], target_descs[0],
#                    2.0, result.transformation, TransformationEstimationPointToPoint(), ICPConvergenceCriteria(max_iteration = 100), 0)
            result_icp = registration_icp(source_patch, target_patch, 
                    1.0, result.transformation, TransformationEstimationPointToPlane())
#                    1.5, result.transformation, TransformationEstimationPointToPoint())
        else:
            result = registration_ransac_based_on_feature_matching(
                source_patch, target_pcd, source_patch_descs, target_descs,
                ransac_radius,
                TransformationEstimationPointToPoint(False), 3,
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(1.0),
                CorrespondenceCheckerBasedOnNormal(np.pi/2)],
                RANSACConvergenceCriteria(ransac_iter, 500), 0.0, 2)
        ransac_time = ransac_time + (time.time() - tic)
        tic = time.time()
        source_patch.transform(result_icp.transformation)
        all_results.append(result_icp)
        all_source_patch.append(source_patch) 
        all_source_patch_descs.append(source_patch_descs) 
        all_source_iface_scores.append(source_patch_iface_scores)
        all_source_geo_dists.append(source_patch_geo_dists)
        inliers = np.zeros((len(source_patch.points)))
        vals_w_corr = np.asarray(result.correspondence_set)[:,0]
        inliers[vals_w_corr] = 1.0
        all_source_inlier_onehot.append(inliers)
        transform_time = transform_time + (time.time() - tic)
        
        tic = time.time()
        source_scores = compute_desc_dist_score(target_pcd, source_patch, np.asarray(result.correspondence_set), target_descs[0], source_patch_descs[0])
        all_source_scores.append(source_scores)
        score_time = score_time + (time.time() - tic)
    
    return all_results, all_source_patch, all_source_scores, all_source_patch_descs, all_source_iface_scores, all_source_geo_dists, all_source_inlier_onehot

def test_alignments(transformation, random_transformation,source_structure, target_ca_pcd_tree, target_pcd_tree, radius=2.0, interface_dist=10.0):
    structure_ca_coords = np.array([atom.get_coord() for atom in source_structure.get_atoms() if atom.get_id() == 'CA'])
    structure_ca_coord_pcd = PointCloud()
    structure_ca_coord_pcd.points = Vector3dVector(structure_ca_coords)
    structure_ca_coord_pcd_notTransformed = copy.deepcopy(structure_ca_coord_pcd)
    structure_ca_coord_pcd.transform(random_transformation)
    structure_ca_coord_pcd.transform(transformation)
        
    structure_coords = np.array([atom.get_coord() for atom in source_structure.get_atoms()])
    structure_coord_pcd = PointCloud()
    structure_coord_pcd.points = Vector3dVector(structure_coords)
    structure_coord_pcd_notTransformed = copy.deepcopy(structure_coord_pcd)
    structure_coord_pcd.transform(random_transformation)
    structure_coord_pcd.transform(transformation)
    
    #clashing_ca = 0
    #total_ca_atoms = 0
    #for point in structure_ca_coord_pcd.points:
    #    total_ca_atoms += 1
    #    [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, radius)
    #    if k>0:
    #        clashing_ca += 1
    
    #clashing = 0
    #total_atoms = 0
    #for point in structure_coord_pcd.points:
    #    total_atoms += 1
    #    [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, radius)
    #    if k>0:
    #        clashing += 1
    d_nn_ca, i_nn_ca = target_pcd_tree.query(np.asarray(structure_ca_coord_pcd.points),k=1,distance_upper_bound=radius)
    d_nn, i_nn = target_pcd_tree.query(np.asarray(structure_coord_pcd.points),k=1,distance_upper_bound=radius)
    clashing_ca = np.sum(d_nn_ca<=radius)
    clashing = np.sum(d_nn<=radius)
    total_ca_atoms = np.asarray(structure_ca_coord_pcd.points).shape[0]
    total_atoms = np.asarray(structure_coord_pcd.points).shape[0]
    
    #interface_atoms = []
    #for i,point in enumerate(structure_coords):
    #    [k, idx, _] = target_ca_pcd_tree.search_radius_vector_3d(point, interface_dist)
    #    if k>0:
    #        interface_atoms.append(i)
    d_nn_interface, i_nn_interface = target_pcd_tree.query(np.asarray(structure_coord_pcd.points),k=1,distance_upper_bound=interface_dist)
    interface_atoms = np.where(d_nn_interface<=interface_dist)[0]
    rmsd = np.sqrt(np.mean(np.square(np.linalg.norm(structure_coords[interface_atoms,:]-np.asarray(structure_coord_pcd.points)[interface_atoms,:],axis=1))))
    return rmsd,clashing_ca,clashing,total_ca_atoms,total_atoms,structure_ca_coord_pcd,structure_ca_coord_pcd_notTransformed#, structure, structure_coord_pcd


def count_clashes(transformation, random_transformation,source_structure, target_ca_pcd_tree, target_pcd_tree, radius=2.0, interface_dist=10.0):
    structure_ca_coords = np.array([atom.get_coord() for atom in source_structure.get_atoms() if atom.get_id() == 'CA'])
    structure_ca_coord_pcd = PointCloud()
    structure_ca_coord_pcd.points = Vector3dVector(structure_ca_coords)
    structure_ca_coord_pcd_notTransformed = copy.deepcopy(structure_ca_coord_pcd)
    structure_ca_coord_pcd.transform(random_transformation)
    structure_ca_coord_pcd.transform(transformation)
        
    structure_coords = np.array([atom.get_coord() for atom in source_structure.get_atoms()])
    structure_coord_pcd = PointCloud()
    structure_coord_pcd.points = Vector3dVector(structure_coords)
    structure_coord_pcd_notTransformed = copy.deepcopy(structure_coord_pcd)
    structure_coord_pcd.transform(random_transformation)
    structure_coord_pcd.transform(transformation)
    
    # Invoke cKDTree with (structure_coord_pcd)

    # Count how many are <radius
    
    #clashing_ca = 0
    #total_ca_atoms = 0
    #for point in structure_ca_coord_pcd.points:
    #    total_ca_atoms += 1
    #    [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, radius)
    #    if k>0:
    #        clashing_ca += 1
    
    #clashing = 0
    #total_atoms = 0
    #for point in structure_coord_pcd.points:
    #    total_atoms += 1
    #    [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, radius)
    #    if k>0:
    #        clashing += 1
    d_nn_ca, i_nn_ca = target_pcd_tree.query(np.asarray(structure_ca_coord_pcd.points),k=1,distance_upper_bound=radius)
    d_nn, i_nn = target_pcd_tree.query(np.asarray(structure_coord_pcd.points),k=1,distance_upper_bound=radius)
    clashing_ca = np.sum(d_nn_ca<=radius)
    clashing = np.sum(d_nn<=radius)
    total_ca_atoms = np.asarray(structure_ca_coord_pcd.points).shape[0]
    total_atoms = np.asarray(structure_coord_pcd.points).shape[0]
    
    return clashing_ca,clashing,total_ca_atoms,total_atoms#, structure, structure_coord_pcd
# In[5]:


# Compute different types of scores: 
# -- Inverted sum of the minimum descriptor distances squared cutoff.
def compute_desc_dist_score(target_pcd, source_pcd, corr, target_desc, source_desc, cutoff=2.0):

    # Compute scores based on correspondences.
    if len(corr) < 1:
        dists_cutoff= np.array([1000.0])
        inliers = 0
    else:
        target_p = corr[:,1]
        source_p = corr[:,0]    
        try:
            dists_cutoff = target_desc.data[:,target_p] - source_desc.data[:,source_p]
        except:
            set_trace()
        dists_cutoff = np.sqrt(np.sum(np.square(dists_cutoff.T), axis=1))
        inliers = len(corr)
    
    scores_corr = np.sum(np.square(1.0/dists_cutoff))
    scores_corr_cube = np.sum(np.power(1.0/dists_cutoff, 3))
    scores_corr_mean = np.mean(np.square(1.0/dists_cutoff))

    return np.array([scores_corr, inliers, scores_corr_mean, scores_corr_cube]).T


def subsample_patch_coords_fast(top_dir, pdb, pid, cv=None, frac = 1.0, radius=12.0):
    patch_coords = np.load(os.path.join(top_dir,pdb,pid+'_list_indices.npy'))[cv]
    pc = {}
    for ii, key in enumerate(cv):
        pc[key] = patch_coords[ii]
    return pc

from IPython.core.debugger import set_trace
def subsample_patch_coords(pdb, pid, cv=None, radius=9.0):
    patch_coords = spio.load_npz(os.path.join(coord_dir,pdb,pid+'.npz'))
    
    D = np.squeeze(np.asarray(patch_coords[:,:patch_coords.shape[1]//2].todense()))

    # Convert to dictionary; 
    pc= {}
    geodists = {}
    if cv is None: 
        for ii in range(len(D)): 
            idx = np.where(np.logical_and(D[ii] <= radius, D[ii] > 0.0))[0]
            pc[ii] = idx
            geodists[ii] = D[ii][idx]
    else: 
        for ii in cv:
            idx = np.where(np.logical_and(D[ii] <= radius, D[ii] > 0.0))[0]
            pc[ii] = idx
            geodists[ii] = D[ii][idx]
    
    return pc, geodists
    
# Read all surfaces. 
all_pc = []
all_desc = []

np.random.seed(0)
rand_list = np.copy(pdb_list)
np.random.shuffle(rand_list)
rand_list = rand_list[0:300]

p2_descriptors_straight = []
p2_point_clouds = []
p2_patch_coords = []
p2_names = []

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
for target_ix,target_pdb in enumerate(rand_list):
    outdir = out_base+'/'+target_pdb+'/'
    if os.path.exists(outdir):
        continue
    tic = time.time()
    print(target_pdb)
    target_pdb_id = target_pdb.split('_')[0]
    chains = target_pdb.split('_')[1:]
        
    # Load target descriptors for global matching. 
    try:
        target_desc_sc05 = np.load(os.path.join(desc_dir,target_pdb,'p1_desc_flipped.npy'))
        target_desc_no_scfilt_chem = np.load(os.path.join(desc_dir_no_scfilt_chem,target_pdb,'p1_desc_flipped.npy'))
        target_desc_no_scfilt_all_feat = np.load(os.path.join(desc_dir_no_scfilt_all_feat,target_pdb,'p1_desc_flipped.npy'))
    except:
        print("error")
        continue
    target_desc = [target_desc_sc05, target_desc_no_scfilt_chem, target_desc_no_scfilt_all_feat]

    # Load target iface prediction
    pdb_chain = '_'.join([target_pdb.split('_')[0], target_pdb.split('_')[1]])
    try:
        target_iface = np.load(os.path.join(iface_dir, 'pred_'+pdb_chain+'.npy'))[0]
    except:
        print("error loading target iface: {}".format(os.path.join(iface_dir, 'pred_'+pdb_chain+'.npy')))
        continue


    # Load target point cloud
    target_pc = os.path.join(surf_dir,'{}.ply'.format(target_pdb_id+'_'+chains[0]))
    source_pc_gt = os.path.join(surf_dir,'{}.ply'.format(target_pdb_id+'_'+chains[1]))
    target_pcd = read_point_cloud(target_pc)

    target_mesh = read_triangle_mesh(target_pc)
    
    # Read the point with the highest shape compl.
    sc_labels = np.load(os.path.join(precomp_dir,target_pdb,'p1_sc_labels.npy'))
    center_point = np.argmax(np.median(np.nan_to_num(sc_labels[0]),axis=1))

    # Go through each source descriptor, find the top descriptors, store id+pdb
    tic_global = time.time()
    num_negs = 0
    all_desc_dists = []
    all_pdb_id = []
    all_vix = []
    gt_dists = []

    for source_ix, source_pdb in enumerate(rand_list):

        if source_pdb != target_pdb:
            continue
        
        source_desc = np.load(os.path.join(desc_dir,source_pdb,'p2_desc_straight.npy'))

        
        desc_dists = np.linalg.norm(source_desc - target_desc[0][center_point],axis=1)
        all_desc_dists.append(desc_dists) 
        all_pdb_id.append([source_pdb]*len(desc_dists))
        all_vix.append(np.arange(len(desc_dists)))
        
            
    all_desc_dists = np.concatenate(all_desc_dists, axis =0)
    all_pdb_id = np.concatenate(all_pdb_id, axis = 0)
    all_vix = np.concatenate(all_vix, axis = 0)
    
    ranking = np.argsort(all_desc_dists)

    # Load target geodesic distances.
    target_coord, target_geo_dists = subsample_patch_coords(target_pdb, 'p1',[center_point], radius=PATCH_RADIUS)

    # Get the geodesic patch and descriptor patch for the target.
    target_patch, target_patch_descs, target_patch_iface_scores, target_patch_geo_dists = \
             get_patch_geo(target_pcd,target_coord,target_geo_dists,center_point,\
                     target_desc, target_iface,flip=True)

    target_patch_mesh = get_patch_mesh_geo(target_mesh,target_coord,center_point,target_desc,flip=False)

    # Make a pointcloud where the centroids of the mesh have a point
    centroids = []
    for f in target_patch_mesh.triangles:
        v0 = target_patch_mesh.vertices[f[0]]
        v1 = target_patch_mesh.vertices[f[1]]
        v2 = target_patch_mesh.vertices[f[2]]
        centroids.append((v0+v1+v2)/3)
    triangle_centroids = PointCloud()
    triangle_centroids.points = Vector3dVector(np.asarray(centroids))
    
    ## Load the structures of the target and the source (to get the ground truth).    
    parser = PDBParser()
    target_struct = parser.get_structure('{}_{}'.format(target_pdb_id,chains[0]), os.path.join(pdb_dir,'{}_{}.pdb'.format(target_pdb_id,chains[0])))
    gt_source_struct = parser.get_structure('{}_{}'.format(target_pdb_id,chains[1]), os.path.join(pdb_dir,'{}_{}.pdb'.format(target_pdb_id,chains[1])))
    # Get coordinates of atoms for the ground truth and target. 
    target_atom_coords = [atom.get_coord() for atom in target_struct.get_atoms()]
    target_ca_coords = [atom.get_coord() for atom in target_struct.get_atoms() if atom.get_id() == 'CA']
    target_atom_coord_pcd = PointCloud()
    target_ca_coord_pcd = PointCloud()
    target_atom_coord_pcd.points = Vector3dVector(np.array(target_atom_coords))
    target_ca_coord_pcd.points = Vector3dVector(np.array(target_ca_coords))
    # Create with cKDTree 
    target_atom_pcd_tree = cKDTree(np.array(target_atom_coords))#KDTreeFlann(target_atom_coord_pcd)
    target_ca_pcd_tree = cKDTree(np.array(target_ca_coords))#KDTreeFlann(target_ca_coord_pcd)
    
    found = False
    myrank_desc = float('inf')
    
    chosen_top = ranking[0:K]
    
    pos_scores = []
    pos_rmsd = []
    neg_scores = []
    time_global = time.time() - tic

    # For each source pdb, for each matched patch store: 
    # Random, and alignment transformation. 
    random_transformations = []
    alignment_transformations = []
    # The aligned source patches themselves. 
    aligned_source_patches = []
    aligned_source_patches_normals = []
    aligned_source_patches_descs_0 = []
    aligned_source_patches_descs_1 = []
    aligned_source_patches_descs_2 = []
    aligned_source_patches_iface = []
    aligned_source_patches_geo_dists = []
    clashing_info = []
    inlier_data = []

    # The geodesic distance from the center 

    # Chemical descriptors, no sc filter
    # all feature descriptors, no sc filter 
    # The index of the center. 
    center_index_sources = []
    # The scores of each transformation.
    scores_transforms = []
    # The RMSDs, inf if not from same complex
    source_patch_rmsds = []
    # The pdb names 
    source_patch_names = []

    # The interface score values. 
    target_iface_scores = []
    
    # Go thorugh every source pdb. 
    for source_ix, source_pdb in enumerate(rand_list):
        tic = time.time()
        viii = chosen_top[np.where(all_pdb_id[chosen_top] == source_pdb)[0]]

        source_vix = all_vix[viii]
        
        if len(source_vix) == 0:
            continue

        # Continue with this pdb.    
        pdb_id = source_pdb.split('_')[0]
        chain = source_pdb.split('_')[2]
        print('Reading: {}'.format(source_pdb))
        source_pcd = read_point_cloud(os.path.join(surf_dir,'{}.ply'.format(pdb_id+'_'+chain)))

        source_desc_sc05 = np.load(os.path.join(desc_dir,source_pdb,'p2_desc_straight.npy'))
        try:
            source_coords, source_geo_dists = subsample_patch_coords(source_pdb, 'p2', cv=source_vix, radius=PATCH_RADIUS)

            source_desc_no_scfilt_chem = np.load(os.path.join(desc_dir_no_scfilt_chem,source_pdb,'p2_desc_straight.npy'))
            source_desc_no_scfilt_all_feat = np.load(os.path.join(desc_dir_no_scfilt_all_feat,source_pdb,'p2_desc_straight.npy'))

            source_desc = [source_desc_sc05, source_desc_no_scfilt_chem, source_desc_no_scfilt_all_feat] 

            # Load source iface prediction
            pdb_chain = pdb_id+'_'+chain
            source_iface = np.load(os.path.join(iface_dir, 'pred_'+pdb_chain+'.npy'))[0]
        except:
            continue
        
        # Randomly rotate and translate.  
        random_transformation = get_center_and_random_rotate(source_pcd)#np.diag([1,1,1,1])#get_center_and_random_rotate(source_pcd)
        source_pcd.transform(random_transformation)
        tic = time.time()
        all_results, all_source_patch, all_source_scores, \
            all_source_descs, all_source_iface, all_source_geo_dists, all_source_inlier_onehot \
                = multidock(source_pcd, source_coords, source_desc,source_vix, target_patch, \
                        target_patch_descs, source_iface, source_geo_dists, target_patch_mesh, triangle_centroids) 
        toc = time.time()
        print('Ransac time {}'.format((toc - tic)))
        num_negs = num_negs

        for j,res in enumerate(all_results):
            if res.fitness == 0:
                continue
            score = list(all_source_scores[j])
            random_transformations.append(random_transformation)
            alignment_transformations.append(res.transformation)
            aligned_source_patches.append(np.asarray(all_source_patch[j].points))
            aligned_source_patches_normals.append(np.asarray(all_source_patch[j].normals))
            aligned_source_patches_descs_0.append(np.asarray(all_source_descs[j][0].data).T)
            aligned_source_patches_descs_1.append(np.asarray(all_source_descs[j][1].data).T)
            aligned_source_patches_descs_2.append(np.asarray(all_source_descs[j][2].data).T)
            aligned_source_patches_iface.append(all_source_iface[j])
            aligned_source_patches_geo_dists.append(all_source_geo_dists[j])
            inlier_data.append(all_source_inlier_onehot)
            scores_transforms.append(score)
            center_index_sources.append(source_vix[j])
            source_patch_names.append(source_pdb)

        # If this is the source_pdb, get the ground truth. 
        if source_pdb == target_pdb: 
            
            tic_cc = time.time()
            for j,res in enumerate(all_results):
                rmsd, clashing_ca,clashing,total_ca_atoms,total_atoms, structure_coord_pcd, structure_coord_pcd_notTransformed = test_alignments(res.transformation, random_transformation, gt_source_struct,                     target_ca_pcd_tree, target_atom_pcd_tree, radius=0.5)
                score = list(all_source_scores[j])
                if rmsd < 2.0 and res.fitness > 0: 
                    #print('{} {}'.format(rank_val, rmsd))
                    rank_val = np.where(chosen_top == viii[j])[0][0]
                    pos_rmsd.append(rmsd)
                    found = True
                    myrank_desc = min(rank_val, myrank_desc)
                    pos_scores.append(score)
                    source_patch_rmsds.append(rmsd)
                    clashing_info.append([clashing_ca,clashing,total_ca_atoms,total_atoms])
                elif res.fitness > 0:
                    neg_scores.append(score)
                    source_patch_rmsds.append(rmsd)
                    clashing_info.append([clashing_ca,clashing,total_ca_atoms,total_atoms])
            toc_cc = time.time()
            print('count clashes time',toc_cc-tic_cc)
        else:
            tic_cc = time.time()
            tic_load_struct = time.time()
            source_struct = parser.get_structure('{}_{}'.format(pdb_id,chain), os.path.join(pdb_dir,'{}_{}.pdb'.format(pdb_id,chain)))
            print('Load structure time: {}'.format(time.time()- tic_load_struct))
            for j in range(len(all_source_scores)):
                if all_results[j].fitness > 0:
                    res = all_results[j]
                    clashing_ca,clashing,total_ca_atoms,total_atoms =  count_clashes(res.transformation, random_transformation, source_struct, target_ca_pcd_tree, target_atom_pcd_tree, radius=0.5)
                    score = list(all_source_scores[j])
                    neg_scores.append(score)
                    source_patch_rmsds.append(float('inf'))
                    clashing_info.append([clashing_ca,clashing,total_ca_atoms,total_atoms])
            toc_cc = time.time()
            print('count clashes time',toc_cc-tic_cc)
    if found: 
        count_found += 1
        all_rankings_desc.append(myrank_desc)
        print(myrank_desc)
    else:
        print('N/D')
        
    all_positive_rmsd.append(pos_rmsd)
    all_positive_scores.append(pos_scores)
    all_negative_scores.append(neg_scores)
    print('Took {:.2f}s'.format(time_global))
    all_time_global.append(time_global)
    # Make out directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Save data for this source patch. 
    np.save(os.path.join(outdir,'source_patch_rmsds'), source_patch_rmsds)
    # Save : 
    # target->source corr
    # target desc_no_scfilt_chem
    # target desc_no_scfilt_all_feat
    # target geodesic radius 
    # target iface pred

    
            
print('All alignments took {}'.format(np.sum(all_time_global)))


# In[18]:


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
        print('N/D')
        unranked+=1
    else: 
        pos_scores = [x[1] for x in all_positive_scores[pdb_ix]]
        neg_scores = [x[1] for x in all_negative_scores[pdb_ix]]
        best_pos_score = np.max(pos_scores)
        best_pos_score_ix = np.argmax(pos_scores)
        best_rmsd = all_positive_rmsd[pdb_ix][best_pos_score_ix]

        number_better_than_best_pos = np.sum(neg_scores > best_pos_score )+1
        if number_better_than_best_pos > num_success:
            print('{} N/D'.format(rand_list[pdb_ix]))
            unranked += 1
        else:
            rmsds.append(best_rmsd)
            print('{} {} out of {} -- pos scores: {}'.format(rand_list[pdb_ix], number_better_than_best_pos, len(neg_scores)+len(pos_scores), len(pos_scores)))
            ranks.append(number_better_than_best_pos)
        
print('Median rank for correctly ranked ones: {}'.format(np.median(ranks)))
print('Mean rank for correctly ranked ones: {}'.format(np.mean(ranks)))
print('Number failed {} out of {}'.format(np.median(unranked), len(all_positive_scores)))

outfile = open('results_{}.txt'.format(method), 'a+')
outfile.write('K Total NumCorrect MeanRank MedianRank MeanRMSD Time\n')
numcorrect = len(all_positive_scores) - unranked
meanrank = np.mean(ranks)
medianrank = np.median(ranks)
meanrmsd = np.mean(rmsds)
runtime = np.sum(all_time_global)


outline = '{} {} {} {} {} {} {}\n'.format(K, len(all_positive_scores), numcorrect, meanrank, medianrank, meanrmsd, runtime)
outfile.write(outline)

outfile.close()

sys.exit(0)
