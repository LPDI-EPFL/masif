import tensorflow as tf
import numpy.matlib 
import os
import numpy as np
from IPython.core.debugger import set_trace
from scipy.spatial import cKDTree
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

with open('../lists/testing.txt') as f:
    training_list = f.read().splitlines()

n_positives = 100
n_negatives = 100
max_rmsd = 5.0
max_npoints = 200
inlier_distance = 1.0
n_features = 2

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
        
        source_patches = np.load(os.path.join(d,'aligned_source_patches.npy'), allow_pickle=True)
        target_patch = np.load(os.path.join(d,'target_patch.npy'))
        source_descs = np.load(os.path.join(d,'aligned_source_patches_descs.npy'), allow_pickle=True)
        target_desc = np.load(os.path.join(d,'target_patch_descs.npy'), allow_pickle=True)

        
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
            # If source_patch is larger than 200, trim it down. 
            if len(source_patch) > max_npoints: 
                subsample = np.random.choice(np.arange(len(source_patch)), max_npoints)
                source_patch= source_patch[subsample]
                source_desc = source_desc[subsample]

            dist, corr = ckd.query(source_patch)
            # Compute the descriptor distance. 
            desc_dist = np.sqrt(np.sum(np.square(source_desc - target_desc[corr]), axis=1))
            try:
                features[iii, 0:len(desc_dist), :] = np.vstack([dist, desc_dist]).T
            except: 
                set_trace()
            # Quickly compute an inlier rate.
            inliers = np.sum(dist < 1.5)/float(len(dist))
            inlier_scores.append(inliers)

        source_patch_rmsds = np.load(os.path.join(d,'source_patch_rmsds.npy'), allow_pickle=True)
        assert(len(source_patch_rmsds)== len(source_patches))
        positive_alignments = np.where(source_patch_rmsds<max_rmsd)[0]
        if len(positive_alignments)==0:#<n_positives:
            continue

        if len(positive_alignments) > n_positives:
            chosen_positives = np.random.choice(positive_alignments,n_positives,replace=False)
        else:
            factor = n_positives/len(positive_alignments)
            positive_alignments = np.repeat(positive_alignments, factor+1)
            chosen_positives = positive_alignments[0:n_positives]

        negative_alignments = np.where(source_patch_rmsds>=max_rmsd)[0]
        if len(negative_alignments) < n_negatives:
            continue

        print(d)

        # Always include include half of the best inliers. 
        negative_alignments_top = np.argsort(inlier_scores)[::-1][:n_negatives//2]
        negative_alignments_top = np.intersect1d(negative_alignments_top, negative_alignments)

        chosen_negatives = np.random.choice(negative_alignments,n_negatives-len(negative_alignments_top),replace=False)
        chosen_alignments = np.concatenate([chosen_positives,negative_alignments_top, chosen_negatives])

        n_sources = len(features)
        features = features[chosen_alignments]
            
        labels = np.expand_dims(np.concatenate([np.ones_like(chosen_positives),np.zeros_like(negative_alignments_top),np.zeros_like(chosen_negatives)]),1)
        
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

reg = keras.regularizers.l2(l=0.0)
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=16,kernel_size=1,strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=32,kernel_size=1,strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=64,kernel_size=1,strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=128,kernel_size=1,strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=256,kernel_size=1,strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(128,activation=tf.nn.relu,kernel_regularizer=reg))
model.add(keras.layers.Dense(64,activation=tf.nn.relu,kernel_regularizer=reg))
model.add(keras.layers.Dense(32,activation=tf.nn.relu,kernel_regularizer=reg))
model.add(keras.layers.Dense(16,activation=tf.nn.relu,kernel_regularizer=reg))
model.add(keras.layers.Dense(8,activation=tf.nn.relu,kernel_regularizer=reg))
model.add(keras.layers.Dense(4,activation=tf.nn.relu,kernel_regularizer=reg))
model.add(keras.layers.Dense(2, activation='softmax'))

opt = keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(model)
#model.load_weights('models/nn_score/trained_model.hdf5')
#pred_auc,pred_solved,max_solved,computed_score_auc,computed_npoints_auc,scoring_df = get_auc(f) 
#print('Model',m,pred_auc,pred_solved,computed_score_auc,computed_npoints_auc)

#y_test_pred = model.predict(features)

