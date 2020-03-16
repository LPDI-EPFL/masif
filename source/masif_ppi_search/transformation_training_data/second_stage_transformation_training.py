#!/usr/bin/env python
from IPython.core.debugger import set_trace
#from transformation_training_data.second_stage_transformation_training_helper import * 
from second_stage_transformation_training_helper import * 
# coding: utf-8
import sys
from open3d import *
#import ipdb
import numpy as np
import os
from Bio.PDB import *
import copy
import scipy.sparse as spio
from default_config.masif_opts import masif_opts
import sys
from scipy.spatial import cKDTree

"""
second_stage_transformation_training.py: Generate real and 'decoy' alignments to train a neural network to discriminate real docking poses vs. false ones.
    Pablo Gainza - LPDI STI EPFL 2019
    Released under an Apache License 2.0
"""

print(sys.argv)
if len(sys.argv) != 7:
    print('Usage: {} data_dir K ransac_iter patch_radius output_dir pdb_list_index'.format(sys.argv[0]))
    print('data_dir: Location of data directory.')
    print('K: Number of descriptors to run')
    sys.exit(1)

data_dir = sys.argv[1]
K=int(sys.argv[2])
ransac_iter = int(sys.argv[3])
# set patch radius fixed at 9A
#PATCH_RADIUS = float(sys.argv[4])
PATCH_RADIUS = 9.0
out_base = sys.argv[5]
pdb_list_index = int(sys.argv[6])

surf_dir = os.path.join(data_dir,masif_opts['ply_chain_dir'])
  
desc_dir = os.path.join(data_dir,masif_opts['ppi_search']['desc_dir'])

pdb_dir = os.path.join(data_dir, masif_opts['pdb_chain_dir'])
precomp_dir = os.path.join(data_dir, masif_opts['ppi_search']['masif_precomputation_dir'])
precomp_dir_9A = os.path.join(data_dir, masif_opts['site']['masif_precomputation_dir'])

training_list = 'training_transformations.txt'

pdb_list = open(training_list).readlines()
pdb_list = [x.rstrip() for ix, x in enumerate(pdb_list) if ix % 1000 == pdb_list_index ]

# Read all surfaces. 
all_pc = []
all_desc = []

rand_list = np.copy(pdb_list)
np.random.seed(0)

p2_descriptors_straight = []
p2_point_clouds = []
p2_patch_coords = []
p2_names = []

import scipy.spatial 

# Read all of p1, the target. p1 will have flipped descriptors.
all_positive_scores = []
all_negative_scores = []
# Match all descriptors. 
count_found = 0
all_rankings_desc = []
for target_ix,target_pdb in enumerate(rand_list):
    outdir = out_base+'/'+target_pdb+'/'
    if os.path.exists(outdir):
        continue
    print(target_pdb)
    target_pdb_id = target_pdb.split('_')[0]
    chains = target_pdb.split('_')[1:]
        
    # Load target descriptors for global matching. 
    try:
        target_desc = np.load(os.path.join(desc_dir,target_pdb,'p1_desc_flipped.npy'))
    except:
        print('Error opening {}'.format(os.path.join(desc_dir,target_pdb,'p1_desc_flipped.npy')))
        continue

    # Load target iface prediction
    pdb_chain = '_'.join([target_pdb.split('_')[0], target_pdb.split('_')[1]])

    # Load target point cloud
    target_pc = os.path.join(surf_dir,'{}.ply'.format(target_pdb_id+'_'+chains[0]))
    source_pc_gt = os.path.join(surf_dir,'{}.ply'.format(target_pdb_id+'_'+chains[1]))
    target_pcd = read_point_cloud(target_pc)

    target_mesh = read_triangle_mesh(target_pc)
    
    # Read the patch center with the highest shape compl (i.e. the center of the interface)
    sc_labels = np.load(os.path.join(precomp_dir,target_pdb,'p1_sc_labels.npy'))
    center_point = np.argmax(np.median(np.nan_to_num(sc_labels[0]),axis=1))

    # Go through each source descriptor, find the top descriptors, store id+pdb
    all_desc_dists = []
    all_pdb_id = []
    all_vix = []
    gt_dists = []

    for source_ix, source_pdb in enumerate(rand_list):
        source_desc = np.load(os.path.join(desc_dir,source_pdb,'p2_desc_straight.npy'))
        desc_dists = np.linalg.norm(source_desc - target_desc[center_point],axis=1)
        all_desc_dists.append(desc_dists) 
        all_pdb_id.append([source_pdb]*len(desc_dists))
        all_vix.append(np.arange(len(desc_dists)))
            
    all_desc_dists = np.concatenate(all_desc_dists, axis =0)
    all_pdb_id = np.concatenate(all_pdb_id, axis = 0)
    all_vix = np.concatenate(all_vix, axis = 0)
    
    ranking = np.argsort(all_desc_dists)

    # Load target geodesic distances.
    # Assume 9A patches. 
    target_coord = np.load(os.path.join(precomp_dir_9A, target_pdb,'p1_list_indices.npy'))

    # Get the geodesic patch and descriptor patch for the target.
    target_patch, target_patch_descs, = \
             get_patch_geo(target_pcd,target_coord,center_point, target_desc, flip=True)

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
    # Create a search tree to quickly check RMSDs.
    target_atom_pcd_tree = cKDTree(np.array(target_atom_coords))
    target_ca_pcd_tree = cKDTree(np.array(target_ca_coords))
    
    found = False
    myrank_desc = float('inf')
    
    chosen_top = ranking[0:K]

    # The aligned source patches themselves. 
    aligned_source_patches = []
    aligned_source_normals = []
    aligned_source_patches_descs = []

    # The RMSDs, inf if not from same complex
    source_patch_rmsds = []
    # The pdb names 
    source_patch_names = []

    # The interface score values. 
    target_iface_scores = []

    # Go through every source pdb. 
    for source_ix, source_pdb in enumerate(rand_list):
        viii = chosen_top[np.where(all_pdb_id[chosen_top] == source_pdb)[0]]

        source_vix = all_vix[viii]
        
        if len(source_vix) == 0:
            continue

        print('Source pdb: {}'.format(source_pdb))

        # Continue with this pdb.    
        pdb_id = source_pdb.split('_')[0]
        chain = source_pdb.split('_')[2]
        source_pcd = read_point_cloud(os.path.join(surf_dir,'{}.ply'.format(pdb_id+'_'+chain)))

        source_desc = np.load(os.path.join(desc_dir,source_pdb,'p2_desc_straight.npy'))
        source_coords = np.load(os.path.join(precomp_dir_9A, source_pdb,'p2_list_indices.npy'))
        
        # Randomly rotate and translate.  
        random_transformation = get_center_and_random_rotate(source_pcd)

        source_pcd.transform(random_transformation)
        all_results, all_source_patch, all_source_descs= \
                multidock(source_pcd, source_coords, source_desc,source_vix, target_patch, target_patch_descs) 

        for j,res in enumerate(all_results):
            if res.fitness == 0:
                continue
            aligned_source_patches.append(np.asarray(all_source_patch[j].points))
            aligned_source_normals.append(np.asarray(all_source_patch[j].normals))
            aligned_source_patches_descs.append(np.asarray(all_source_descs[j].data).T)
            source_patch_names.append(source_pdb)

        # If this is the ground truth, source_pdb, check if the alignment is correct. 
        if source_pdb == target_pdb: 
            print(source_pdb)
            for j,res in enumerate(all_results):
                rmsd = test_alignments(res.transformation, random_transformation, gt_source_struct,\
                        target_ca_pcd_tree, target_atom_pcd_tree)
                
                if rmsd < 5.0 and res.fitness > 0: 
                    rank_val = np.where(chosen_top == viii[j])[0][0]
                    found = True
                    myrank_desc = min(rank_val, myrank_desc)
                    source_patch_rmsds.append(rmsd)
                elif res.fitness > 0 :
                    source_patch_rmsds.append(rmsd)
        else:
            for j, res in enumerate(all_results):
                if res.fitness > 0:
                    source_patch_rmsds.append(float('inf'))
            # Make sure the data has the same size.
            assert(len(aligned_source_patches) == len(source_patch_rmsds))
    if found: 
        count_found += 1
        all_rankings_desc.append(myrank_desc)
        print(myrank_desc)
    else:
        print('N/D')
        

    # Make out directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Save training data for this source patch. 
    np.save(os.path.join(outdir,'source_patch_names'), source_patch_names)
    np.save(os.path.join(outdir,'aligned_source_patches'), np.asarray(aligned_source_patches))
    np.save(os.path.join(outdir,'aligned_source_normals'), np.asarray(aligned_source_normals))
    np.save(os.path.join(outdir,'aligned_source_patches_descs'), np.asarray(aligned_source_patches_descs))
    np.save(os.path.join(outdir,'source_patch_rmsds'), source_patch_rmsds)
    np.save(os.path.join(outdir,'target_patch'), np.asarray(target_patch.points))
    np.save(os.path.join(outdir,'target_patch_normals'), np.asarray(target_patch.normals))
    np.save(os.path.join(outdir,'target_patch_descs'), np.asarray(target_patch_descs.data).T)

