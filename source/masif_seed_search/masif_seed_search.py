#!/usr/bin/env python
# coding: utf-8

# In[1]:
from IPython.core.debugger import set_trace
import time
import os 
from default_config.masif_opts import masif_opts
from input_output.save_ply import save_ply
from open3d import *
import pymesh
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from Bio.PDB import *
import copy
import scipy.sparse as spio
import sys
import shutil
import pymesh

start_time = time.time()


# In[133]:
custom_params_fn = sys.argv[1]
custom_params_obj = importlib.import_module(custom_params_fn, package=None)
params = custom_params_obj.params

def get_patch_mesh_geo(triangle_mesh,patch_coords,center,descriptors,outward_shift=0.0, flip=False):
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


# Extract a geodesic patch. 
def get_patch_geo(pcd,patch_coords,center,descriptors,outward_shift=0.0, flip=False):
    idx = patch_coords[center]
    pts = np.asarray(pcd.points)[idx,:]
    nrmls = np.asarray(pcd.normals)[idx,:]
    pts = pts + outward_shift*nrmls
    if flip:
        nrmls = -np.asarray(pcd.normals)[idx,:]

    patch = PointCloud()
    patch.points = Vector3dVector(pts)
    patch.normals = Vector3dVector(nrmls)
    patch_descs = Feature()
    patch_descs.data = descriptors[idx,:].T
    return patch, patch_descs
    
def subsample_patch_coords_fast(top_dir, pdb, pid, cv=None, frac = 1.0, radius=12.0):
    if cv is None:
        patch_coords = np.load(os.path.join(top_dir,pdb,pid+'_list_indices.npy'))
        cv = np.arange(0,patch_coords.shape[0])
    else:
        patch_coords = np.load(os.path.join(top_dir,pdb,pid+'_list_indices.npy'))[cv]
    pc = {}
    for ii, key in enumerate(cv):
        pc[key] = patch_coords[ii]
    return pc

def subsample_patch_coords(top_dir, pdb, pid, cv=None, frac = 1.0, radius=12.0):
    patch_coords = spio.load_npz(os.path.join(top_dir,pdb,pid+'.npz'))
    
    if cv is None:
        D = np.squeeze(np.asarray(patch_coords[:,:patch_coords.shape[1]//2].todense()))
    else:
        D = np.squeeze(np.asarray(patch_coords[cv,:patch_coords.shape[1]//2].todense()))
    # Get nonzero fields, points under radius. 
    idx = np.where(np.logical_and(D>0.0,D<radius))
    
    # Convert to dictionary; 
    
    pc = {}
    for ii in range(len(idx[0])):
        # With probability frac, ignore this entry point - always include the center point.
        if cv is None:
            cvix = idx[0][ii]
            val = idx[1][ii]
        else:
            cvix = cv[ii]
            val = idx[0][ii]
        if np.random.random() < frac or cvix == val:
            if cvix not in pc:
                pc[cvix] = []
            pc[cvix].append(val)
    
    return pc

# Get the vertex indices of target sites. 
def get_target_vix(pc, iface, num_sites=1):
    target_vertices = []
    for site_ix in range(num_sites):
        iface_patch_vals = []
        # Go through each patch.
        best_ix = -1
        best_val = float("-inf")
        best_neigh = []
        for ii in pc.keys():
            neigh = pc[ii]
            val = np.mean(iface[neigh])
            if val > best_val:
                best_ix = ii
                best_val = val
                best_neigh = neigh
            
        # Now that a site has been identified, clear the iface values so that the same site is not picked again.
        iface[best_neigh] = 0
        target_vertices.append(best_ix)
    
    return target_vertices


# # Load target patches.

# In[164]:

# Go through each target
# Go through each 

target_name = sys.argv[2]
target_ppi_pair_id = target_name+'_'

# Go through every 12A patch in top_dir -- get the one with the highest iface mean 12A around it.
target_ply_fn = os.path.join(params['target_ply_iface_dir'], target_name+'.ply')

mesh = pymesh.load_mesh(target_ply_fn)

iface = mesh.get_attribute('vertex_iface')

target_coord = subsample_patch_coords_fast(params['target_precomp_dir'], target_ppi_pair_id, 'p1')
target_vertices= get_target_vix(target_coord, iface,num_sites=params['num_sites'])

target_pcd = read_point_cloud(target_ply_fn)
target_mesh = read_triangle_mesh(target_ply_fn)
target_desc = np.load(os.path.join(params['target_desc_dir'], target_ppi_pair_id, 'p1_desc_flipped.npy'))


# Match descriptors that have a descriptor distance less than K 

def match_descriptors(in_desc_dir, in_iface_dir, pids, target_desc, desc_dist_cutoff=params['desc_dist_cutoff'], iface_cutoff=params['iface_cutoff'] ):

    all_matched_names = []
    all_matched_vix = []
    all_matched_desc_dist = []
    count_proteins = 0
    for ppi_pair_id in os.listdir(in_desc_dir):
        if 'seed_pdb_list' in params and ppi_pair_id not in params['seed_pdb_list']:
            continue
#        if 'CAF1' not in ppi_pair_id:
#            continue
        if '.npy' in ppi_pair_id or '.txt' in ppi_pair_id:
            continue
        mydescdir = os.path.join(in_desc_dir, ppi_pair_id)
        for pid in pids: 
            try:
                fields = ppi_pair_id.split('_')
                if pid == 'p1': 
                    pdb_chain_id = fields[0]+'_'+fields[1]
                elif pid == 'p2': 
                    pdb_chain_id = fields[0]+'_'+fields[2]
                iface = np.load(in_iface_dir+'/pred_'+pdb_chain_id+'.npy')[0]
                descs = np.load(mydescdir+'/'+pid+'_desc_straight.npy')
            except:
                continue
            print(pdb_chain_id)
            name =  (ppi_pair_id, pid)
            count_proteins+= 1
            
            diff = np.sqrt(np.sum(np.square(descs - target_desc), axis=1))
            
            true_iface = np.where(iface > iface_cutoff)[0]
            near_points = np.where(diff < desc_dist_cutoff)[0]
            
            selected = np.intersect1d(true_iface, near_points)
            if len(selected > 0):            
                all_matched_names.append([name]*len(selected))
                all_matched_vix.append(selected)
                all_matched_desc_dist.append(diff[selected])
                print('Matched {}'.format(ppi_pair_id))
                print('Scores: {} {}'.format(iface[selected], diff[selected]))

    
    print('Iterated over {} proteins.'.format(count_proteins))
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

def multidock(source_pcd,source_patch_coords,source_descs,cand_pts,target_pcd,target_descs,
       target_patch_mesh, target_patch_mesh_centroids, 
        ransac_radius=params['ransac_radius'], ransac_iter=params['ransac_iter']):
    all_results = []
    all_source_patch = []
    all_source_scores = []
    patch_time = 0.0
    ransac_time = 0.0
    transform_time = 0.0
    score_time = 0.0
    for pt in cand_pts:
        tic = time.time()
        source_patch,source_patch_descs= get_patch_geo(source_pcd,source_patch_coords,pt,source_descs)
        patch_time = patch_time + (time.time() - tic)
        tic = time.time()
        if params['ransac_type'] == 'shape_comp':
            result = registration_ransac_based_on_shape_complementarity(
                source_patch, target_pcd, target_patch_mesh, target_patch_mesh_centroids, source_patch_descs, target_descs,
                ransac_radius,
                TransformationEstimationPointToPoint(False), 3,
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(2.0),
                CorrespondenceCheckerBasedOnNormal(np.pi/2)],
                RANSACConvergenceCriteria(ransac_iter, 500),0,3)
        else: 
            result = registration_ransac_based_on_feature_matching(
                source_patch, target_pcd, source_patch_descs, target_descs,
                ransac_radius,
                TransformationEstimationPointToPoint(False), 3,
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(1.0),
                CorrespondenceCheckerBasedOnNormal(np.pi/2)],
                RANSACConvergenceCriteria(ransac_iter, 500), 0.0, 2
            )
        #print('{} {}'.format(len(np.asarray(result.correspondence_set)), result.fitness))
        ransac_time = ransac_time + (time.time() - tic)
        #result = registration_icp(source_patch, target_pcd, 1.5,
        #result.transformation,
        #TransformationEstimationPointToPoint())
        
        tic = time.time()
        source_patch.transform(result.transformation)
        all_results.append(result)
        all_source_patch.append(source_patch) 
        transform_time = transform_time + (time.time() - tic)
        
        tic = time.time()
        source_scores = compute_desc_dist_score(target_pcd, source_patch, np.asarray(result.correspondence_set), target_descs, source_patch_descs)
        all_source_scores.append(source_scores)
        score_time = score_time + (time.time() - tic)
    #print('Ransac time = {:.2f}'.format(ransac_time))
    #print('Extraction time = {:.2f}'.format(patch_time))
    #print('Score time = {:.2f}'.format(score_time))
    
    return all_results,all_source_patch, all_source_scores

def align_and_save(out_filename_base, patch, transformation, source_structure, target_ca_pcd_tree,
        target_pcd_tree, clashing_cutoff = 10.0,
        clashing_radius=2.0):
    structure_atoms = [atom for atom in source_structure.get_atoms()]
    structure_coords = [x.get_coord() for x in structure_atoms]
    
    structure_coord_pcd = PointCloud()
    structure_coord_pcd.points = Vector3dVector(structure_coords)
    structure_coord_pcd.transform(transformation)
    
    #all_new_coords = np.asarray(structure_coord_pcd.points)
    #for ix, atom in enumerate(structure.get_atoms()):
    #    atom.set_coord(all_new_coords[ix])
        
    clashing = 0
    for point in structure_coord_pcd.points:
        [k, idx, _] = target_pcd_tree.search_radius_vector_3d(point, clashing_radius)
        if k>0:
            clashing += 1
            
    #
    if clashing < clashing_cutoff:
        for ix, v in enumerate(structure_coord_pcd.points):
            structure_atoms[ix].set_coord(v)
        

        io=PDBIO()
        io.set_structure(source_structure)
        io.save(out_filename_base+'.pdb')
        # Save patch 
        out_patch = open(out_filename_base+'.vert', 'w+')
        for point in patch.points: 
            out_patch.write('{}, {}, {}\n'.format(point[0], point[1], point[2]))
        out_patch.close()
        return True
    else: 
        return False
            




# In[13]:


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


# Open the pdb structure of the target
parser = PDBParser()
target_pdb_path = os.path.join(params['target_pdb_dir'],'{}.pdb'.format(target_name))
target_ply_path = os.path.join(params['target_ply_iface_dir'],'{}.ply'.format(target_name))
target_struct = parser.get_structure(target_pdb_path, target_pdb_path)
target_atom_coords = [atom.get_coord() for atom in target_struct.get_atoms()]
target_ca_coords = [atom.get_coord() for atom in target_struct.get_atoms() if atom.get_id() == 'CA']
target_atom_coord_pcd = PointCloud()
target_ca_coord_pcd = PointCloud()
target_atom_coord_pcd.points = Vector3dVector(np.array(target_atom_coords))
target_ca_coord_pcd.points = Vector3dVector(np.array(target_ca_coords))
target_ca_pcd_tree = KDTreeFlann(target_ca_coord_pcd)
target_pcd_tree = KDTreeFlann(target_atom_coord_pcd)

outdir = params['out_dir_template'].format(target_name)
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Copy the pdb structure and the ply file of the target 
shutil.copy(target_pdb_path, outdir)
shutil.copy(target_ply_path, outdir)

# Go through every target site in the target
out_log = open('log.txt', 'w+')
for site_ix, site_vix in enumerate(target_vertices):

    out_log.write('Starting site {}\n'.format(site_vix))
    site_outdir = os.path.join(outdir, 'site_{}'.format(site_ix))
    if not os.path.exists(site_outdir):
        os.makedirs(site_outdir)
    # Write out the patch itself. 
    out_patch = open(site_outdir+'/target.vert', 'w+')
    # Get the geodesic patch and descriptor patch for each target patch
    target_patch, target_patch_descs = get_patch_geo(target_pcd,target_coord,site_vix,target_desc,flip=True, outward_shift=0.0)

    # Get the geodesic patch for the target.
    target_patch_mesh = get_patch_mesh_geo(target_mesh,target_coord,site_vix,target_desc,flip=False)

    # Make a pointcloud where the centroids of the mesh have a point
    centroids = []
    for f in target_patch_mesh.triangles:
        v0 = target_patch_mesh.vertices[f[0]]
        v1 = target_patch_mesh.vertices[f[1]]
        v2 = target_patch_mesh.vertices[f[2]]
        centroids.append((v0+v1+v2)/3)
    triangle_centroids = PointCloud()
    triangle_centroids.points = Vector3dVector(np.asarray(centroids))

    for point in target_patch.points: 
        out_patch.write('{}, {}, {}\n'.format(point[0], point[1], point[2]))
    out_patch.close()

    desc_scores = []
    desc_pos = []
    inlier_scores = []
    inlier_pos = []

    (matched_names, matched_vix, matched_desc_dist, count_proteins) = match_descriptors(params['seed_desc_dir'], params['seed_iface_dir'], ['p1', 'p2'], target_desc[site_vix])        
    
    matched_names = np.concatenate(matched_names, axis=0)
    matched_vix = np.concatenate(matched_vix, axis=0)
    matched_desc_dist = np.concatenate(matched_desc_dist, axis=0)

    matched_dict = {}
    out_log.write('Total number of proteins {}\n'.format(count_proteins))
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
        pdb = ppi_pair_id.split('_')[0]

        if pid == 'p1':
            chain = ppi_pair_id.split('_')[1]
        else: 
            chain = ppi_pair_id.split('_')[2]
            
        # Load source ply file, coords, and descriptors.
        tic = time.time()
        
        print('{}'.format(pdb+'_'+chain))
        source_pcd = read_point_cloud(os.path.join(params['seed_surf_dir'],'{}.ply'.format(pdb+'_'+chain)))
    #    print('Reading ply {}'.format(time.time()- tic))

        tic = time.time()
        source_vix = matched_dict[name]
        try:
            source_coords = subsample_patch_coords_fast(params['seed_precomp_dir'], ppi_pair_id, pid, cv=source_vix, frac=1.0)
        except:
            print('Coordinates not found. continuing.')
            continue
    #    source_coords = subsample_patch_coords(coords_dir, ppi_pair_id, pid, frac=1.0)
    #    print('Reading coords {}'.format(time.time()- tic))
        source_desc = np.load(os.path.join(params['seed_desc_dir'],ppi_pair_id,pid+'_desc_straight.npy'))
        
        # Perform all alignments to target. 
        tic = time.time()
        all_results, all_source_patch, all_source_scores = multidock(source_pcd,source_coords,source_desc,source_vix,target_patch,target_patch_descs, target_patch_mesh, triangle_centroids) 
    #    print('Multidock took {}'.format(time.time()- tic))
        scores = np.asarray(all_source_scores)
        desc_scores.append(scores[:,0])
        inlier_scores.append(scores[:,1])
        
        # Filter anything above cutoff
        top_scorers = np.where(scores[:,0] > params['post_alignment_score_cutoff'])[0]
        
        if len(top_scorers) > 0:
            source_outdir = os.path.join(site_outdir, '{}'.format(ppi_pair_id))
            if not os.path.exists(source_outdir):
                os.makedirs(source_outdir)

            # Load source structure 
            # Perform the transformation on the atoms
            for j in top_scorers:
                print('{} {} {}'.format(ppi_pair_id, scores[j], pid))
                parser = PDBParser()
                source_struct = parser.get_structure('{}_{}'.format(pdb,chain), os.path.join(params['seed_pdb_dir'],'{}_{}.pdb'.format(pdb,chain)))
                res = all_results[j]
                    
                out_fn = source_outdir+'/{}_{}_{}'.format(pdb, chain, j)
                
                # Align and save the pdb + patch 
                is_not_clashing = align_and_save(out_fn, all_source_patch[j], res.transformation, source_struct, target_ca_pcd_tree,target_pcd_tree,\
                                                params['clashing_cutoff'])
                if is_not_clashing:
                    out_log.write('{} {} {}\n'.format(ppi_pair_id, scores[j], pid))

                    # Align and save the ply file for convenience.     
                    mesh = pymesh.load_mesh(os.path.join(params['seed_surf_dir'],'{}.ply'.format(pdb+'_'+chain)))
                    
                    # Output the score for convenience. 
                    out_score = open(out_fn+'.score', 'w+')
                    out_score.write('{} {} {}\n'.format(ppi_pair_id, scores[j], pid))
                    out_score.close()


                    source_pcd_copy = copy.deepcopy(source_pcd)
                    source_pcd_copy.transform(res.transformation)
                    out_vertices = np.asarray(source_pcd_copy.points)
                    out_normals = np.asarray(source_pcd_copy.normals)
                    save_ply(out_fn+'.ply', out_vertices, mesh.faces, out_normals, charges=mesh.get_attribute('vertex_charge'))
        

# In[ ]:

end_time = time.time()
out_log.write('Took {}s\n'.format(end_time-start_time))



