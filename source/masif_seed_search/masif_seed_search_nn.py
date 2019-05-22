#!/usr/bin/env python
import sys
from scipy.spatial import cKDTree
from IPython.core.debugger import set_trace
import time
import os 
from default_config.masif_opts import masif_opts
from open3d import *
from input_output.simple_mesh import Simple_mesh
from  score_with_nn import Masif_search_score

import numpy as np
import os
from Bio.PDB import *
import copy
import shutil

from alignment_utils import *

start_time = time.time()

# In[133]:
custom_params_fn = sys.argv[1]
custom_params_obj = importlib.import_module(custom_params_fn, package=None)
params = custom_params_obj.params

# # Load target patches.

if len(sys.argv) == 3: 
    target_name = sys.argv[2]
    target_ppi_pair_id = target_name+'_'
    target_pid = 'p1'
    target_chain_ix = 1
elif len(sys.argv) == 4: 
    target_pid = sys.argv[3]
    target_ppi_pair_id = sys.argv[2]
    fields = target_ppi_pair_id.split('_')
    if target_pid == 'p1': 
        target_name = '_'.join([fields[0], fields[1]])
        target_chain_ix = 1
    if target_pid == 'p2': 
        target_name = '_'.join([fields[0], fields[2]])
        target_chain_ix = 2


# Go through every 12A patch in top_dir -- get the one with the highest iface mean 12A around it.
target_ply_fn = os.path.join(params['target_ply_iface_dir'], target_name+'.ply')

mesh = Simple_mesh()
mesh.load_mesh(target_ply_fn)

iface = mesh.get_attribute('vertex_iface')

# Initialize neural network 
nn_score = Masif_search_score(params['nn_score_weights'], max_npoints=params['max_npoints'], nn_score_cutoff=params['nn_score_cutoff'])
target_coord, target_geodists = get_geodists_and_patch_coords(params['target_precomp_dir'], target_ppi_pair_id, target_pid)

target_vertices= get_target_vix(target_coord, iface,num_sites=params['num_sites'])

target_paths = {}
target_paths['surf_dir'] = params['target_surf_dir'] 
target_paths['iface_dir'] = params['target_iface_dir'] 
target_paths['desc_dir'] = params['target_desc_dir'] 
target_paths['desc_dir_sc_nofilt_all_feat'] = params['target_desc_dir_sc_nofilt_all_feat']
target_paths['desc_dir_sc_nofilt_chem'] = params['target_desc_dir_sc_nofilt_chem']

source_paths = {}
source_paths['surf_dir'] = params['seed_surf_dir'] 
source_paths['iface_dir'] = params['seed_iface_dir'] 
source_paths['desc_dir'] = params['seed_desc_dir'] 
source_paths['desc_dir_sc_nofilt_all_feat'] = params['seed_desc_dir_sc_nofilt_all_feat']
source_paths['desc_dir_sc_nofilt_chem'] = params['seed_desc_dir_sc_nofilt_chem']
target_pcd, target_desc, target_iface, target_mesh = load_protein_pcd(target_ppi_pair_id, target_chain_ix, target_paths, flipped_features=True, read_mesh=True)

# Match descriptors that have a descriptor distance less than K 

# Compute different types of scores: 
# -- Inverted sum of the minimum descriptor distances squared cutoff.

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
    target_patch, target_patch_descs, target_patch_iface = \
                get_patch_geo(target_pcd,target_coord,site_vix,\
                        target_desc, target_iface, flip_normals=True)

    # Make a ckdtree with the target.
    target_ckdtree = cKDTree(target_patch.points)

    # Compute the geodesic coordinates of the target.
    target_patch_geodists = target_geodists[site_vix]
     
    # Get the geodesic patch for the target, including the face information. This one is not flipped.
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

    desc_pos = []
    inlier_pos = []

    (matched_names, matched_vix, matched_desc_dist, count_proteins) = match_descriptors(params['seed_desc_dir'], params['seed_iface_dir'], ['p1', 'p2'], target_desc[0][site_vix], params)        
    
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
    
    inlier_scores = []

    for name in matched_dict.keys():
        ppi_pair_id = name[0]
        pid = name[1]
        pdb = ppi_pair_id.split('_')[0]

        if pid == 'p1':
            chain = ppi_pair_id.split('_')[1]
            chain_number = 1
        else: 
            chain = ppi_pair_id.split('_')[2]
            chain_number = 2
            
        # Load source ply file, coords, and descriptors.
        tic = time.time()
        
        print('{}'.format(pdb+'_'+chain))
        try:
            source_pcd, source_desc, source_iface = load_protein_pcd(ppi_pair_id, chain_number, source_paths, flipped_features=False, read_mesh=False)
        except:
            continue
    #    print('Reading ply {}'.format(time.time()- tic))

        tic = time.time()
        source_vix = matched_dict[name]
        try:
            source_coord, source_geodists =  get_geodists_and_patch_coords(params['seed_precomp_dir'], ppi_pair_id, pid, cv=source_vix)
        except:
            print('Coordinates not found. continuing.')
            continue
        
        # Perform all alignments to target. 
        tic = time.time()
        all_results, all_source_patch, all_source_scores = multidock(
                source_pcd, source_coord, source_desc,
                source_vix, target_patch, target_patch_descs, 
                target_patch_mesh, triangle_centroids, 
                source_geodists, target_patch_geodists, target_ckdtree,
                source_iface, target_patch_iface, nn_score, params
                ) 
    #    print('Multidock took {}'.format(time.time()- tic))
        all_point_importance = [x[1] for x in all_source_scores]
        all_source_scores = [x[0] for x in all_source_scores]
        scores = np.asarray(all_source_scores)
        
        # Filter anything above cutoff
        top_scorers = np.where(scores[:,4] > params['nn_score_cutoff'])[0]
        
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
                                                clashing_cutoff=params['clashing_cutoff'], point_importance=all_point_importance[j])
                if is_not_clashing:
                    out_log.write('{} {} {}\n'.format(ppi_pair_id, scores[j], pid))

                    # Align and save the ply file for convenience.     
                    mesh = Simple_mesh()
                    mesh.load_mesh(os.path.join(params['seed_surf_dir'],'{}.ply'.format(pdb+'_'+chain)))
                    
                    # Output the score for convenience. 
                    out_score = open(out_fn+'.score', 'w+')
                    out_score.write('{} {} {} {} {}\n'.format(j , ppi_pair_id, scores[j][0], scores[j][1], scores[j][4], pid))
                    out_score.close()

                    source_pcd_copy = copy.deepcopy(source_pcd)
                    source_pcd_copy.transform(res.transformation)
                    out_vertices = np.asarray(source_pcd_copy.points)
                    out_normals = np.asarray(source_pcd_copy.normals)
                    mesh.set_attribute('vertex_x', out_vertices[:,0])
                    mesh.set_attribute('vertex_y', out_vertices[:,1])
                    mesh.set_attribute('vertex_z', out_vertices[:,2])
                    mesh.set_attribute('vertex_nx', out_normals[:,0])
                    mesh.set_attribute('vertex_ny', out_normals[:,1])
                    mesh.set_attribute('vertex_nz', out_normals[:,2])
                    mesh.vertices = out_vertices
                    mesh.set_attribute('vertex_iface', source_iface) 
                    mesh.save_mesh(out_fn+'.ply')
                    #save_ply(out_fn+'.ply', out_vertices, mesh.faces, out_normals, charges=mesh.get_attribute('vertex_charge'))
        

# In[ ]:

end_time = time.time()
out_log.write('Took {}s\n'.format(end_time-start_time))



