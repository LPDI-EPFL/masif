import tensorflow as tf
import numpy as np
from IPython.core.debugger import set_trace
from scipy.spatial import cKDTree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow import keras
import os
import time
import glob

class Masif_search_score:

    def load_model(self, weights_file):
        reg = keras.regularizers.l2(l=0.0)
        model = keras.models.Sequential()
        #model.add(keras.layers.Conv1D(filters=4,kernel_size=1,strides=1,activation=tf.nn.relu,kernel_regularizer=reg))
        #model.add(keras.layers.Conv1D(filters=8,kernel_size=1,strides=1,activation=tf.nn.relu,kernel_regularizer=reg))
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
        #model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dense(256,activation=tf.nn.relu,kernel_regularizer=reg))
        model.add(keras.layers.Dense(128,activation=tf.nn.relu,kernel_regularizer=reg))
        model.add(keras.layers.Dense(64,activation=tf.nn.relu,kernel_regularizer=reg))
        model.add(keras.layers.Dense(32,activation=tf.nn.relu,kernel_regularizer=reg))
        model.add(keras.layers.Dense(16,activation=tf.nn.relu,kernel_regularizer=reg))
        model.add(keras.layers.Dense(8,activation=tf.nn.relu,kernel_regularizer=reg))
        model.add(keras.layers.Dense(4,activation=tf.nn.relu,kernel_regularizer=reg))
        model.add(keras.layers.Dense(2, activation='softmax'))
        opt = keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.load_weights(weights_file)
        self.model = model

    def __init__(self, weights_file, max_npoints=300, nn_score_cutoff=0.85):
        self.max_npoints = max_npoints
        self.nn_score_cutoff = nn_score_cutoff
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        np.random.seed(42)
        tf.random.set_random_seed(42)
        self.load_model(weights_file)
    
        # Model order: 
        # [Distance, desc_0, desc_1, desc_2, source_geo_dists, target_geo_dists, source_iface, target_iface, normal_dp]
    def eval_model(self, distance, desc_0, desc_1, desc_2, source_geo_dists, target_geo_dists, source_iface, target_iface, normal_dp):

            distance[distance < 0.5] = 0.5
            distance = 1.0/distance
            desc_0 = 1.0/desc_0
            desc_1 = 1.0/desc_1
            desc_2 = 1.0/desc_2
            source_geo_dists[source_geo_dists<1.0] = 1.0
            target_geo_dists[target_geo_dists<1.0] = 1.0
            source_geo_dists = 1.0/source_geo_dists
            target_geo_dists = 1.0/target_geo_dists

            features = np.vstack([distance, desc_0, desc_1, desc_2, source_geo_dists, target_geo_dists, source_iface, target_iface, normal_dp]).T
            max_npoints = self.max_npoints 
            features = np.expand_dims(features, 0)
            n_features = features.shape[2]
    
            features_trimmed = np.zeros((1,max_npoints,n_features))
            for j,f in enumerate(features):
                if f.shape[0]<=max_npoints:
                    features_trimmed[j,:f.shape[0],:] = f
                else:
                    selected_rows = np.random.choice(f.shape[0],max_npoints,replace=False)
                    features_trimmed[j,:,:] = f[selected_rows]
        
            y_test_pred = self.model.predict(features_trimmed)
            y_test_pred = y_test_pred[:,1].reshape((-1,1))

            point_importance = np.zeros(len(distance))
# 
            if len(distance) < max_npoints and y_test_pred[0,0] > self.nn_score_cutoff:
                # Evaluate point by point. 
                for i in range(len(distance)):
                    feat_copy = np.copy(features_trimmed)
                    feat_copy[0,i,:] = 0.0
                    point_val = self.model.predict(feat_copy)
                    point_val = point_val[:,1].reshape((-1,1))
                    point_importance[i] = point_val[0,0] - y_test_pred[0,0]
                # Normalize
                d = point_importance
                const = np.max(np.abs(d))/np.std(d)
                d_std = d/(const*np.std(d))
                point_importance = d_std


            return y_test_pred, point_importance



