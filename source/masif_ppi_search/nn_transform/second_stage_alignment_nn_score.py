#!/usr/bin/env python
# coding: utf-8
import sys
from scipy.spatial import cKDTree
from IPython.core.debugger import set_trace
from open3d import *
import numpy as np
import os
from sklearn.manifold import TSNE
from Bio.PDB import *
import copy
import scipy.sparse as spio
from default_config.masif_opts import masif_opts
import sys
import tensorflow as tf
from  masif_ppi_search.nn_transform.score_with_nn import Masif_search_score

print(sys.argv)
if len(sys.argv) != 6 or (sys.argv[5] != 'masif' and sys.argv[5] != 'gif'):
    print('Usage: {} data_dir K ransac_iter num_success gif|masif'.format(sys.argv[0]))
    print('data_dir: Location of data directory.')
    print('K: Number of descriptors to run')
    print('ransac_iter: number of ransac iterations.')
    print('num_success: true alignment within short list of size num_success')
    sys.exit(1)

nn_score = Masif_search_score()

data_dir = sys.argv[1]
K=int(sys.argv[2])
ransac_iter = int(sys.argv[3])
num_success = int(sys.argv[4])
method = sys.argv[5]
PATCH_RADIUS=9.0
RANSAC_DESCRIPTOR_INDEX = 3

#scratch_dir = '/
surf_dir = os.path.join(data_dir,masif_opts['ply_chain_dir'])

desc_dir_old_model = '/home/gainza/lpdi_fs/masif_paper/masif/data/masif_ppi_search/descriptors/sc05/all_feat'
#desc_dir_old_model = '/home/gainza/lpdi_fs/masif_paper/masif/data/masif_ppi_search_ub/descriptors/sc05/all_feat'

desc_dir = os.path.join(data_dir,masif_opts['ppi_search']['desc_dir'])

# chemical and all features descriptors, iface.
#iface_dir = os.path.join(data_dir, '/home/gainza/lpdi_fs/masif/data/masif_site/output/all_feat_3l/pred_data/')
desc_dir_no_scfilt_chem = os.path.join(data_dir,'descriptors/sc_nofilt/chem/')
desc_dir_no_scfilt_all_feat = os.path.join(data_dir,'descriptors/sc_nofilt/all_feat/')

coord_dir = os.path.join(data_dir,masif_opts['coord_dir_npy'])

pdb_dir = os.path.join(data_dir, masif_opts['pdb_chain_dir'])
precomp_dir = os.path.join(data_dir, masif_opts['ppi_search']['masif_precomputation_dir'])

benchmark_list = '../benchmark_list.txt'

# In[3]:

pdb_list = open(benchmark_list).readlines()[0:100]
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

# Descriptors is a vector of descriptors of different types of different types.
def get_patch_geo(pcd,patch_coords,geo_dists, center,descriptors, outward_shift=0.0, flip=False):
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
    return patch, patch_descs, patch_geo_dists

def multidock(source_pcd,source_patch_coords,source_descs,cand_pts,target_pcd,
            target_descs, source_geo_dists, target_patch_geo_dists, \
            target_ckdtree, ransac_radius=1.0):
    all_results = []
    all_source_patch = []
    all_source_patch_descs = []
    all_source_scores = []
    all_source_geo_dists = []
    patch_time = 0.0
    ransac_time = 0.0
    transform_time = 0.0
    score_time = 0.0
    for pt in cand_pts:
        tic = time.time()
        source_patch, source_patch_descs, source_patch_geo_dists = \
            get_patch_geo(source_pcd,source_patch_coords,source_geo_dists, pt,source_descs)
        patch_time = patch_time + (time.time() - tic)
        tic = time.time()
        result = registration_ransac_based_on_feature_matching(
            source_patch, target_pcd, source_patch_descs[RANSAC_DESCRIPTOR_INDEX], target_descs[RANSAC_DESCRIPTOR_INDEX],
            ransac_radius,
            TransformationEstimationPointToPoint(False), 3,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(2.0),
            CorrespondenceCheckerBasedOnNormal(np.pi/2)],
            RANSACConvergenceCriteria(ransac_iter, 500))
        ransac_time = ransac_time + (time.time() - tic)
        
        tic = time.time()
        source_patch.transform(result.transformation)
        all_results.append(result)
        all_source_patch.append(source_patch) 
        all_source_patch_descs.append(source_patch_descs) 
        all_source_geo_dists.append(source_patch_geo_dists)
        transform_time = transform_time + (time.time() - tic)
        
        tic = time.time()
        source_scores = compute_desc_dist_score(target_pcd, source_patch, np.asarray(result.correspondence_set),
                source_patch_geo_dists, target_patch_geo_dists,
                target_descs, source_patch_descs, \
                target_ckdtree )
        all_source_scores.append(source_scores)
        score_time = score_time + (time.time() - tic)
    
    return all_results, all_source_patch, all_source_scores, all_source_patch_descs, all_source_geo_dists

def test_alignments(transformation, random_transformation,source_structure, target_ca_pcd_tree, target_pcd_tree, radius=2.0, interface_dist=10.0):
    structure_coords = np.array([atom.get_coord() for atom in source_structure.get_atoms() if atom.get_id() == 'CA'])
    structure_coord_pcd = PointCloud()
    structure_coord_pcd.points = Vector3dVector(structure_coords)
    structure_coord_pcd_notTransformed = copy.deepcopy(structure_coord_pcd)
    structure_coord_pcd.transform(random_transformation)
    structure_coord_pcd.transform(transformation)
        
    clashing = 0
    for point in structure_coord_pcd.points:
        [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, radius)
        if k>0:
            clashing += 1
    
    interface_atoms = []
    for i,point in enumerate(structure_coords):
        [k, idx, _] = target_ca_pcd_tree.search_radius_vector_3d(point, interface_dist)
        if k>0:
            interface_atoms.append(i)
    rmsd = np.sqrt(np.mean(np.square(np.linalg.norm(structure_coords[interface_atoms,:]-np.asarray(structure_coord_pcd.points)[interface_atoms,:],axis=1))))
    return rmsd,clashing,structure_coord_pcd,structure_coord_pcd_notTransformed#, structure, structure_coord_pcd


# In[5]:


# Compute different types of scores: 
# -- Inverted sum of the minimum descriptor distances squared cutoff.
def compute_desc_dist_score(target_pcd, source_pcd, corr, 
        source_patch_geo_dists, target_patch_geo_dists,
        target_desc, source_desc, 
        target_ckdtree ):
        

    # Compute scores based on correspondences.
    if len(corr) < 1:
        dists_cutoff_0= np.array([1000.0])
        dists_cutoff_1= np.array([1000.0])
        dists_cutoff_2= np.array([1000.0])
        inliers = 0
    else:
        target_p = corr[:,1]
        source_p = corr[:,0]    
        try:
            dists_cutoff_0 = target_desc[0].data[:,target_p] - source_desc[0].data[:,source_p]
            dists_cutoff_1 = target_desc[1].data[:,target_p] - source_desc[1].data[:,source_p]
            dists_cutoff_2 = target_desc[2].data[:,target_p] - source_desc[2].data[:,source_p]
        except:
            set_trace()
        dists_cutoff_0 = np.sqrt(np.sum(np.square(dists_cutoff_0.T), axis=1))
        dists_cutoff_1 = np.sqrt(np.sum(np.square(dists_cutoff_1.T), axis=1))
        dists_cutoff_2 = np.sqrt(np.sum(np.square(dists_cutoff_2.T), axis=1))
        inliers = len(corr)
    
    scores_corr_0 = np.sum(np.square(1.0/dists_cutoff_0))
    scores_corr_1 = np.sum(np.square(1.0/dists_cutoff_1))
    scores_corr_2 = np.sum(np.square(1.0/dists_cutoff_2))

    # Compute nn scores 
    # Compute all points correspondences and distances for nn
    d, r = target_ckdtree.query(source_pcd.points)

    # r: for every point in source, what is its correspondence in target

    feat0 = d # distances between points 
    # feat1-feat3: descriptors
    feat1 = target_desc[0].data[:,r] - source_desc[0].data[:,:]
    feat1 = np.sqrt(np.sum(np.square(feat1.T), axis=1))
    feat2 = target_desc[1].data[:,r] - source_desc[1].data[:,:]
    feat2 = np.sqrt(np.sum(np.square(feat2.T), axis=1))
    feat3 = target_desc[2].data[:,r] - source_desc[2].data[:,:]
    feat3 = np.sqrt(np.sum(np.square(feat3.T), axis=1))
    # feat4 = source_geo_dists
    feat4 = source_patch_geo_dists
    # feat5 = target_geo_dists
    feat5 = target_patch_geo_dists[r]
    # feat6-7 = source iface, target iface
#    feat6 = source_patch_iface_scores
#    feat7 = target_patch_iface_scores[r]
    # feat8: normal dot product
    n1 = np.asarray(source_pcd.normals)
    n2 = np.asarray(target_pcd.normals)[r]
    feat8 = np.multiply(n1, n2).sum(1)


#    feat8 = np.diag(np.dot(np.asarray(source_pcd.normals), np.asarray(target_pcd.normals)[r].T))

    nn_score_pred = nn_score.eval_model(feat0, feat1, feat2, feat3, feat4, feat5, feat8)

    return np.array([scores_corr_0, inliers, scores_corr_1, scores_corr_2, nn_score_pred[0][0]]).T


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

rand_list = np.copy(pdb_list)
np.random.seed(0)
np.random.shuffle(rand_list)
rand_list = rand_list[0:100]

p2_descriptors_straight = []
p2_point_clouds = []
p2_patch_coords = []
p2_geo_dists = []
p2_names = []

# Read all of p2. p2 will have straight descriptors. 
for i,pdb in enumerate(rand_list):
    print("Running on {}".format(pdb))
#    print(i/len(pdb_list))
    pdb_id = pdb.split('_')[0]
    chains = pdb.split('_')[1:]
    # Descriptors for global matching. 
    p2_descriptors_straight.append(np.load(os.path.join(desc_dir_old_model,pdb,'p2_desc_straight.npy')))

    p2_point_clouds.append(read_point_cloud(os.path.join(surf_dir,'{}.ply'.format(pdb_id+'_'+chains[1]))))
    
    # Read patch coordinates. Subsample 
    
    #p2_patch_coords.append(spio.load_npz(os.path.join(data_dir,'03-coords_npy',pdb,'p2.npz')))
    pc,geo_dists = subsample_patch_coords(pdb, 'p2', radius=PATCH_RADIUS)
    p2_patch_coords.append(pc)
    p2_geo_dists.append(geo_dists)
    
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
for target_ix,target_pdb in enumerate(rand_list):
    tic = time.time()
    print(target_pdb)
    target_pdb_id = target_pdb.split('_')[0]
    chains = target_pdb.split('_')[1:]
        
    # Load target descriptors for global matching. 
    #try:
    target_desc_sc05 = np.load(os.path.join(desc_dir,target_pdb,'p1_desc_flipped.npy'))
    target_desc_no_scfilt_chem = np.load(os.path.join(desc_dir_no_scfilt_chem,target_pdb,'p1_desc_flipped.npy'))
    target_desc_no_scfilt_all_feat = np.load(os.path.join(desc_dir_no_scfilt_all_feat,target_pdb,'p1_desc_flipped.npy'))
    target_desc_old_model = np.load(os.path.join(desc_dir_old_model,target_pdb,'p1_desc_flipped.npy'))
    #except:
    #    continue
    target_desc = [target_desc_sc05, target_desc_no_scfilt_chem, target_desc_no_scfilt_all_feat,target_desc_old_model]

    #try:
    #except:
    #    continue


    # Load target point cloud
    target_pc = os.path.join(surf_dir,'{}.ply'.format(target_pdb_id+'_'+chains[0]))
    source_pc_gt = os.path.join(surf_dir,'{}.ply'.format(target_pdb_id+'_'+chains[1]))
    target_pcd = read_point_cloud(target_pc)
    
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

        source_desc = p2_descriptors_straight[source_ix]
        
        try:
            desc_dists = np.linalg.norm(source_desc - target_desc[RANSAC_DESCRIPTOR_INDEX][center_point],axis=1)
        except:
            set_trace()
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
    target_patch, target_patch_descs, target_patch_geo_dists = \
             get_patch_geo(target_pcd,target_coord,target_geo_dists,center_point,\
                     target_desc, flip=True)

    # Make a ckdtree with the target.
    target_ckdtree = cKDTree(target_patch.points)
    
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
    target_atom_pcd_tree = KDTreeFlann(target_atom_coord_pcd)
    target_ca_pcd_tree = KDTreeFlann(target_ca_coord_pcd)
    
    found = False
    myrank_desc = float('inf')
    
    chosen_top = ranking[0:K]
    
    pos_scores = []
    pos_rmsd = []
    neg_scores = []
    time_global = time.time() - tic
    
    # Go thorugh every source pdb. 
    data_loading_time = 0.0
    for source_ix, source_pdb in enumerate(rand_list):
        tic = time.time()
        viii = chosen_top[np.where(all_pdb_id[chosen_top] == source_pdb)[0]]
        source_vix = all_vix[viii]
        
        if len(source_vix) == 0:
            continue

        # Continue with this pdb.    
        pdb_id = source_pdb.split('_')[0]
        chain = source_pdb.split('_')[2]
        source_pcd = copy.deepcopy(p2_point_clouds[source_ix])
        
        source_coords = p2_patch_coords[source_ix]
        source_geo_dists = p2_geo_dists[source_ix]

        source_desc_old_model = p2_descriptors_straight[source_ix]
        tic = time.time()
        #try:

        source_desc_sc05 = np.load(os.path.join(desc_dir,source_pdb,'p2_desc_straight.npy'))
        source_desc_no_scfilt_chem = np.load(os.path.join(desc_dir_no_scfilt_chem,source_pdb,'p2_desc_straight.npy'))
        source_desc_no_scfilt_all_feat = np.load(os.path.join(desc_dir_no_scfilt_all_feat,source_pdb,'p2_desc_straight.npy'))

        source_desc = [source_desc_sc05, source_desc_no_scfilt_chem, source_desc_no_scfilt_all_feat,source_desc_old_model] 

        pdb_chain = pdb_id+'_'+chain
        #except:
        #    continue
        toc = time.time()
        data_loading_time += toc - tic
        
        # Randomly rotate and translate.  
        random_transformation = get_center_and_random_rotate(source_pcd)#np.diag([1,1,1,1])#get_center_and_random_rotate(source_pcd)
        source_pcd.transform(random_transformation)
        all_results, all_source_patch, all_source_scores, \
            all_source_descs, all_source_geo_dists \
                = multidock(source_pcd, source_coords, source_desc,source_vix, target_patch, \
                        target_patch_descs, \
                        source_geo_dists, target_patch_geo_dists, target_ckdtree) 
        toc = time.time()
        time_global += (toc - tic)
        num_negs = num_negs

        # If this is the source_pdb, get the ground truth. 
        if source_pdb == target_pdb: 
            
            for j,res in enumerate(all_results):
                rmsd, clashing, structure_coord_pcd, structure_coord_pcd_notTransformed = \
                    test_alignments(res.transformation, random_transformation, gt_source_struct,\
                            target_ca_pcd_tree, target_atom_pcd_tree, radius=0.5)
                score = list(all_source_scores[j])
                if rmsd < 5.0 and res.fitness > 0: 
                    #print('{} {}'.format(rank_val, rmsd))
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
        pos_scores_np = np.array(pos_scores)
        neg_scores_np = np.array(neg_scores)
        my_nn_score_pos = np.max(pos_scores_np[:,4])
        ranking_score_nn = np.sum(neg_scores_np[:,4]>=my_nn_score_pos)
        my_d2_score_pos = np.max(pos_scores_np[:,0])
        ranking_score_d2 = np.sum(neg_scores_np[:,0]>=my_d2_score_pos)
        print('1/d2 rank: {}'.format(ranking_score_d2+1)) 
        print('Neural net rank: {}'.format(ranking_score_nn+1)) 
    else:
        print('N/D')
    print('Data loading time: {}'.format(data_loading_time))
        
    all_positive_rmsd.append(pos_rmsd)
    all_positive_scores.append(pos_scores)
    all_negative_scores.append(neg_scores)
    print('Took {:.2f}s'.format(time_global))
    all_time_global.append(time_global)
    # Go through every top descriptor. 
    
            
print('All alignments took {}'.format(np.sum(all_time_global)))


# In[18]:

score_type_names = ['sc05', 'points', 'chem_allp','allp', 'nn']

for score_type in range(0,5):
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
    num_solvable = 0

    for pdb_ix in range(len(all_positive_scores)):
        if len(all_positive_scores[pdb_ix]) == 0:
            print('N/D')
            unranked+=1
        else: 
            num_solvable += 1
            pos_scores = [x[score_type] for x in all_positive_scores[pdb_ix]]
            neg_scores = [x[score_type] for x in all_negative_scores[pdb_ix]]
            best_pos_score = np.max(pos_scores)
            best_pos_score_ix = np.argmax(pos_scores)
            best_rmsd = all_positive_rmsd[pdb_ix][best_pos_score_ix]

            number_better_than_best_pos = np.sum(neg_scores >= best_pos_score )+1
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
    outfile.write('K Total NumCorrect MeanRank MedianRank MeanRMSD Time NumSolvable score_type desc_ix\n')
    numcorrect = len(all_positive_scores) - unranked
    meanrank = np.mean(ranks)
    medianrank = np.median(ranks)
    meanrmsd = np.mean(rmsds)
    runtime = np.sum(all_time_global)


    outline = '{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {} {} {}\n'.format(K, len(all_positive_scores), numcorrect, meanrank, medianrank, meanrmsd, runtime, num_solvable, score_type_names[score_type], RANSAC_DESCRIPTOR_INDEX)
    outfile.write(outline)

    outfile.close()

sys.exit(0)
