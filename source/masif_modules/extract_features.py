# coding: utf-8
# ## Imports and helper functions
from IPython.core.debugger import set_trace
import sys
import os, sys, inspect
import os
import numpy as np
import h5py
import scipy.sparse.linalg as la
import scipy.sparse as sp
import scipy
import time

import re

import math

import itertools as it
from sklearn import metrics


# In[2]:

# Function to compute the mean normal of vertices within r radius of the center of the patch.
def mean_normal_center_patch(D, n, r):
    c_normal = [n[i] for i in range(len(D)) if D[i] <= r]
    mean_normal = np.mean(c_normal, axis=0, keepdims=True).T
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    return np.squeeze(mean_normal)


# Compute the distant dependent distribution of features, Yin et al PNAS 2009
# Returns a vector with the dddc for each point.
def compute_dddc(X, Y, Z, n, D, c_p):
    # Mean normal 2.5A around the center point
    i = c_p
    ni = mean_normal_center_patch(D, n, 2.5)
    r = np.stack([X, Y, Z], 1)
    dij = np.linalg.norm(r - r[i], axis=1)
    # Compute the step function sf:
    sf = r + n
    sf = sf - (ni + r[i])
    sf = np.linalg.norm(sf, axis=1)
    sf = sf - dij
    sf[sf > 0] = 1
    sf[sf < 0] = -1
    sf[sf == 0] = 0
    # Compute the curvature between i and j
    dij[dij == 0] = 1e-8
    kij = np.divide(np.linalg.norm(n - ni, axis=1), dij)
    kij = np.multiply(sf, kij)
    # Ignore any values greater than 0.7 and any values smaller than 0.7
    kij[kij > 0.7] = 0
    kij[kij < -0.7] = 0

    return kij


# In[6]:
def extract_features(list_desc, list_coords):
    list_rho_wrt_center = []
    list_theta_wrt_center = []
    list_isc = []
    list_normals_proj = []
    list_electrostatics = []
    list_hbond = []
    list_hphob = []
    for k in range(len(list_desc)):
        # Rho and theta coordinates for convolution
        list_rho_wrt_center.append(list_coords[k][: list_coords[k].shape[0] // 2])
        list_theta_wrt_center.append(list_coords[k][list_coords[k].shape[0] // 2 :])
        c_isc = list_desc[k]["shape_index"]
        # Feature 1: list_isc, Shape index(called isc here)
        list_isc.append(c_isc)
        # Feature 2: list_normals_proj: dddc , the distance dependent distribution of curvatures.
        n = list_desc[k]["normal"]
        X = np.squeeze(np.asarray(list_desc[k]["X"]))
        Y = np.squeeze(np.asarray(list_desc[k]["Y"]))
        Z = np.squeeze(np.asarray(list_desc[k]["Z"]))
        D = list_coords[k][: list_coords[k].shape[0] // 2]
        c_p = np.int(list_desc[k]["center"])
        dddc = compute_dddc(X, Y, Z, n.T, D, c_p)
        list_normals_proj.append(dddc)
        # Feature 3: Electrostatics.
        c_charge = list_desc[k]["charge"]
        list_electrostatics.append(np.asarray(c_charge))
        # Feature 4: Hbonds
        hbond = list_desc[k]["hbond"]
        list_hbond.append(np.asarray(hbond))
        # Feature 5: hydrophobicity
        if "hphob" in list_desc[k]:
            hphob = list_desc[k]["hphob"]
            # set from -1 to 1.
            hphob = hphob / 4.5
            list_hphob.append(hphob)

    norm_list_electrostatics = normalize_electrostatics(list_electrostatics)
    if len(list_hphob) > 0:
        return (
            list_rho_wrt_center,
            list_theta_wrt_center,
            list_isc,
            list_normals_proj,
            norm_list_electrostatics,
            list_hbond,
            list_hphob,
        )
    else:
        return (
            list_rho_wrt_center,
            list_theta_wrt_center,
            list_isc,
            list_normals_proj,
            norm_list_electrostatics,
            list_hbond,
        )


def normalize_electrostatics(list_electrostatics):
    # Normalize electrostatics.
    upper_threshold = 3
    lower_threshold = -3
    norm_list_electrostatics = []
    for k in range(len(list_electrostatics)):
        elec = np.copy(list_electrostatics[k])
        elec[elec > upper_threshold] = upper_threshold
        elec[elec < lower_threshold] = lower_threshold
        elec = elec - lower_threshold
        elec = elec / (upper_threshold - lower_threshold)
        elec = 2 * elec - 1
        norm_list_electrostatics.append(elec)
    return norm_list_electrostatics


# In[13]
def compute_max_vertices(list_rho_wrt_center):
    max_distance = 0
    max_num_vertices = 0
    index_of_max_num_vertices = -1
    max_num_vertices_found = -1
    all_max_dists = []
    for k in range(len(list_rho_wrt_center)):
        all_max_dists.append(np.max(list_rho_wrt_center[k]))
        max_distance = np.maximum(np.max(list_rho_wrt_center[k]), max_distance)
        max_num_vertices = np.maximum(list_rho_wrt_center[k].shape[0], max_num_vertices)
        if max_num_vertices > max_num_vertices_found:
            max_num_vertices_found = max_num_vertices
            index_of_max_num_vertices = k

