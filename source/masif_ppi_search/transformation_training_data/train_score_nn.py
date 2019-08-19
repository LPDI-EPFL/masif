import tensorflow as tf
import numpy.matlib 
import os
import numpy as np
from IPython.core.debugger import set_trace
from scipy.spatial import cKDTree
from score_nn import ScoreNN
from sklearn.metrics import roc_auc_score
from tensorflow import keras
import time
#import pandas as pd
import pickle
import sys

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

np.random.seed(42)
tf.random.set_random_seed(42)

data_dir = 'transformation_data/'

with open('../lists/training.txt') as f:
    training_list = f.read().splitlines()

n_positives = 1
n_negatives = 100
max_rmsd = 5.0
max_npoints = 200
n_features = 3

data_list = os.listdir(data_dir)
data_list = [os.path.join(data_dir,d) for d in data_list \
            if (os.path.exists(os.path.join(data_dir,d,'target_patch.npy'))) \
            and str(d).split('/')[-1] in training_list]

all_features = np.empty((len(data_list)*(n_positives+n_negatives),max_npoints,n_features))
all_labels = np.empty((len(data_list)*(n_positives+n_negatives),1))
all_scores = np.empty((len(data_list)*(n_positives+n_negatives),1))
all_npoints = []
all_idxs = []
all_nsources = []
n_samples = 0

if not os.path.exists('transformation_data/all_labels.npy'):
    for i,d in enumerate(data_list):
        
        if i > 500:
            break
        source_patches = np.load(os.path.join(d,'aligned_source_patches.npy'), allow_pickle=True)
        source_normals = np.load(os.path.join(d,'aligned_source_normals.npy'), allow_pickle=True)
        target_patch = np.load(os.path.join(d,'target_patch.npy'))
        source_descs = np.load(os.path.join(d,'aligned_source_patches_descs.npy'), allow_pickle=True)
        target_desc = np.load(os.path.join(d,'target_patch_descs.npy'), allow_pickle=True)
        target_normals = np.load(os.path.join(d,'target_patch_normals.npy'), allow_pickle=True)

        
        # If target patch has more than 200 points, then trim it down.
        if len(target_patch) > max_npoints: 
            subsample = np.random.choice(np.arange(len(target_patch)), size=max_npoints)
            target_patch= target_patch[subsample]
            target_desc = target_desc[subsample]

        # Find correspondences between source and target.
        # Make a target_patch cKDTree
        ckd = cKDTree(target_patch)
        features = np.zeros((len(source_descs), max_npoints, n_features))
        inlier_scores = []
        for iii, source_patch in enumerate(source_patches): 
            source_desc = source_descs[iii]
            sn = source_normals[iii]
            # If source_patch is larger than 200, trim it down. 
            if len(source_patch) > max_npoints: 
                subsample = np.random.choice(np.arange(len(source_patch)), max_npoints)
                source_patch= source_patch[subsample]
                source_desc = source_desc[subsample]
                sn = sn[subsample]

            dist, corr = ckd.query(source_patch)
            dist[dist<0.5] = 0.5
            dist = 1.0/dist
            # Compute the descriptor distance. 
            desc_dist = 1.0/np.sqrt(np.sum(np.square(source_desc - target_desc[corr]), axis=1))
            # Compute the normal product.
            norm_prd = np.multiply(sn, target_normals[corr]).sum(1)

            try:
                features[iii, 0:len(desc_dist), :] = np.vstack([dist, desc_dist, norm_prd]).T
            except: 
                set_trace()

        source_patch_rmsds = np.load(os.path.join(d,'source_patch_rmsds.npy'), allow_pickle=True)
        assert(len(source_patch_rmsds)== len(source_patches))
        positive_alignments = np.where(source_patch_rmsds<max_rmsd)[0]
        if len(positive_alignments)==0:#<n_positives:
            continue

        chosen_positives = np.random.choice(positive_alignments,n_positives,replace=False)

        negative_alignments = np.where(source_patch_rmsds>=max_rmsd)[0]
        if len(negative_alignments) < n_negatives:
            continue

        print(d)

        chosen_negatives = np.random.choice(negative_alignments,n_negatives,replace=False)
        chosen_alignments = np.concatenate([chosen_positives, chosen_negatives])

        n_sources = len(features)
        features = features[chosen_alignments]
            
        labels = np.expand_dims(np.concatenate([np.ones_like(chosen_positives),np.zeros_like(chosen_negatives)]),1)
        
        all_features[n_samples:n_samples+len(chosen_alignments),:,:] = features
        all_labels[n_samples:n_samples+len(chosen_alignments)] = labels
        n_samples += len(chosen_alignments)
    
    all_features = all_features[:n_samples]
    all_labels = all_labels[:n_samples]

    all_idxs = np.concatenate([(n_positives+n_negatives)*[i] for i in range(int(all_features.shape[0]/(n_positives+n_negatives)))])

    np.save('transformation_data/all_features.npy', all_features)
    np.save('transformation_data/all_labels.npy', all_labels)
    np.save('transformation_data/all_idxs.npy', all_idxs)

else:
    all_features = np.load('transformation_data/all_features.npy')
    all_labels = np.load('transformation_data/all_labels.npy')
    all_idxs = np.load('transformation_data/all_idxs.npy')

nn_model = ScoreNN()
nn_model.train_model(all_features, all_labels, n_negatives, n_positives)

#history = model.fit(all_features,all_labels,batch_size=128,epochs=50,validation_split=0.1,shuffle=True,class_weight={0:1.0/n_negatives,1:1.0/n_positives}, callbacks=callbacks)

