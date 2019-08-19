import tensorflow as tf
import numpy.matlib 
import os
import numpy as np
from IPython.core.debugger import set_trace
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from score_nn import ScoreNN
import time
import pickle
import sys
import sklearn.metrics

# Load neural network
np.random.seed(42)
tf.random.set_random_seed(42)
print('Loading neural network')
nn_model = ScoreNN()

data_dir = 'transformation_data/'

with open('../lists/testing.txt') as f:
    testing_list = f.read().splitlines()

n_positives = 100
n_negatives = 100
max_rmsd = 5.0
max_npoints = 200
inlier_distance = 1.0
n_features = 3

data_list = os.listdir(data_dir)
data_list = [os.path.join(data_dir,d) for d in data_list \
            if (os.path.exists(os.path.join(data_dir,d,'target_patch.npy'))) \
            and str(d).split('/')[-1] in testing_list]

all_features = []
all_labels = []
all_npoints = []
all_idxs = []
all_nsources = []
n_samples = 0

if not os.path.exists('transformation_data/all_labels_test.npy'):
    for i,d in enumerate(data_list):
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
            # Compute the descriptor distance. 
            desc_dist = 1.0/np.sqrt(np.sum(np.square(source_desc - target_desc[corr]), axis=1))
#            norm_prd = np.multiply(sn, target_normals[corr]).sum(1)
            norm_prd = np.diag(np.dot(sn,target_normals[corr].T))
            dist[dist< 0.5] = 0.5
            dist = 1.0/dist
            features[iii, 0:len(desc_dist), :] = np.vstack([dist, desc_dist, norm_prd]).T

        source_patch_rmsds = np.load(os.path.join(d,'source_patch_rmsds.npy'), allow_pickle=True)
        assert(len(source_patch_rmsds)== len(source_patches))
        positive_alignments = np.where(source_patch_rmsds<max_rmsd)[0]
        if len(positive_alignments)==0:#<n_positives:
            continue

        # Choose only one positive.
        chosen_positives = np.random.choice(positive_alignments, 1, replace=False)

        negative_alignments = np.where(source_patch_rmsds>=max_rmsd)[0]
        if len(negative_alignments) < n_negatives:
            continue

        print(d)

        chosen_negatives = np.random.choice(negative_alignments, 100, replace=False)
        chosen_alignments = np.concatenate([chosen_positives,chosen_negatives])

        n_sources = len(features)
        features = features[chosen_alignments]
            
        labels = np.expand_dims(np.concatenate([np.ones_like(chosen_positives),np.zeros_like(chosen_negatives)]),1)
        
        all_features.append(features)
        all_labels.append(labels)
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.save('transformation_data/all_features_test.npy', all_features)
    np.save('transformation_data/all_labels_test.npy', all_labels)

else:
    print('Loading prexisting data')
    all_features = np.load('transformation_data/all_features_test.npy')
    all_labels = np.load('transformation_data/all_labels_test.npy')

# Compare with 1/d^2 and num inliers.
## 
d2_scores = []
inlier_num= []
nn_feat = []
for patch_pair in all_features:
    neigh = np.where(patch_pair[:,0] >= 1.0)[0]
    inlier_num.append(len(neigh))
    d2_score = patch_pair[neigh, 1]
    d2_score = np.sum(np.square(d2_score))
    d2_scores.append(d2_score)
labels = np.squeeze(all_labels)
roc_auc = sklearn.metrics.roc_auc_score(labels, d2_scores)
print('Roc AUC for 1/d2= {:.3f}'.format(roc_auc))

roc_auc = sklearn.metrics.roc_auc_score(labels, inlier_num)
print('Roc AUC for inlier number= {:.3f}'.format(roc_auc))
    
# Evaluate  nn.
print('Evaluating features.')
test = nn_model.eval(all_features)
# Run on the neural network. 
y_pred = test[:,1]
labels = np.squeeze(all_labels)
roc_auc = sklearn.metrics.roc_auc_score(labels, y_pred)
print('roc auc: {:.3f}'.format(roc_auc))

