#!/usr/bin/env python
from IPython.core.debugger import set_trace
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

"""
second_stage_transformation_training_helper.py: Helper functions for the second strage transformation data generation.
    Pablo Gainza - LPDI STI EPFL 2019
    Released under an Apache License 2.0
"""

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
    rand_mat = rand_rotation_matrix()
    transform = np.vstack([rand_mat.T,-mean_pt]).T
    transform = np.vstack([transform, [0,0,0,1]])
    return transform

# Descriptors is a vector of descriptors of different types of different types.
def get_patch_geo(pcd,patch_coords,center,descriptors, outward_shift=0.25, flip=False):
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

def multidock(source_pcd,source_patch_coords,source_descs,cand_pts,target_pcd,target_descs,\
            ransac_radius=1.0, ransac_iter=2000):
    all_results = []
    all_source_patch = []
    all_source_patch_descs = []
    for pt in cand_pts:
        try:
            source_patch, source_patch_descs= \
            get_patch_geo(source_pcd,source_patch_coords,pt,source_descs)
        except:
            set_trace()
        result = registration_ransac_based_on_feature_matching(
            source_patch, target_pcd, source_patch_descs, target_descs,
            ransac_radius,
            TransformationEstimationPointToPoint(False), 3,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(1.0),
            CorrespondenceCheckerBasedOnNormal(np.pi/2)],
            RANSACConvergenceCriteria(ransac_iter, 500))
        result = registration_icp(source_patch, target_pcd, 
                    1.0, result.transformation, TransformationEstimationPointToPlane())

        source_patch.transform(result.transformation)
        all_results.append(result)
        all_source_patch.append(source_patch) 
        all_source_patch_descs.append(source_patch_descs) 
   
    return all_results, all_source_patch, all_source_patch_descs 

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
    
    d_nn_ca, i_nn_ca = target_pcd_tree.query(np.asarray(structure_ca_coord_pcd.points),k=1,distance_upper_bound=radius)
    d_nn, i_nn = target_pcd_tree.query(np.asarray(structure_coord_pcd.points),k=1,distance_upper_bound=radius)
    clashing_ca = np.sum(d_nn_ca<=radius)
    clashing = np.sum(d_nn<=radius)
    total_ca_atoms = np.asarray(structure_ca_coord_pcd.points).shape[0]
    total_atoms = np.asarray(structure_coord_pcd.points).shape[0]
    
    d_nn_interface, i_nn_interface = target_pcd_tree.query(np.asarray(structure_coord_pcd.points),k=1,distance_upper_bound=interface_dist)
    interface_atoms = np.where(d_nn_interface<=interface_dist)[0]
    rmsd = np.sqrt(np.mean(np.square(np.linalg.norm(structure_coords[interface_atoms,:]-np.asarray(structure_coord_pcd.points)[interface_atoms,:],axis=1))))
    return rmsd

