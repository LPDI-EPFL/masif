# coding: utf-8
# ## Imports and helper functions
import sys
import os, sys, inspect
import os
import numpy as np
import h5py
import scipy.sparse.linalg as la
import scipy.sparse as sp
import scipy
import time
from IPython.core.debugger import set_trace
import re

import math

import itertools as it
from sklearn import metrics


def compute_input_feat(
    list_rho_wrt_center,
    list_theta_wrt_center,
    list_isc,
    list_normals_proj,
    list_hbond,
    norm_list_electrostatics,
    max_num_vertices,
    list_hphob=None,
    feat_mask=[1.0, 1.0, 1.0, 1.0, 1.0],
):

    # place all the coordinates in a unique matrix for simplifying further processing of such data
    rho_wrt_center = np.zeros((len(list_rho_wrt_center), max_num_vertices))
    theta_wrt_center = np.zeros((len(list_theta_wrt_center), max_num_vertices))
    mask = np.zeros((len(list_theta_wrt_center), max_num_vertices, 1))
    num_feat = int(sum(feat_mask))
    input_feat = np.zeros((len(list_theta_wrt_center), max_num_vertices, num_feat))
    print(feat_mask)
    for k in range(len(list_rho_wrt_center)):
        rho_wrt_center[k, : list_rho_wrt_center[k].shape[0]] = np.squeeze(
            np.asarray(list_rho_wrt_center[k])
        )
        theta_wrt_center[k, : list_theta_wrt_center[k].shape[0]] = np.squeeze(
            np.asarray(list_theta_wrt_center[k])
        )
        c = 0
        if feat_mask[0] == 1.0:
            input_feat[k, : list_theta_wrt_center[k].shape[0], c] = feat_mask[
                0
            ] * np.squeeze(np.asarray(list_isc[k]))
            c = c + 1
        if feat_mask[1] == 1.0:
            input_feat[k, : list_theta_wrt_center[k].shape[0], c] = feat_mask[
                1
            ] * np.squeeze(np.asarray(list_normals_proj[k]))
            c = c + 1
        if feat_mask[2] == 1.0:
            input_feat[k, : list_theta_wrt_center[k].shape[0], c] = feat_mask[
                2
            ] * np.squeeze(np.asarray(list_hbond[k]))
            c = c + 1
        if feat_mask[3] == 1.0:
            input_feat[k, : list_theta_wrt_center[k].shape[0], c] = feat_mask[
                3
            ] * np.squeeze(np.asarray(norm_list_electrostatics[k]))
            c = c + 1
        if list_hphob is not None and feat_mask[4] == 1.0:
            input_feat[k, : list_theta_wrt_center[k].shape[0], c] = feat_mask[
                4
            ] * np.squeeze(np.asarray(list_hphob[k]))
            c = c + 1

        mask[k, : list_theta_wrt_center[k].shape[0], 0] = 1
    theta_wrt_center[theta_wrt_center < 0] += 2 * np.pi
    return rho_wrt_center, theta_wrt_center, input_feat, mask

