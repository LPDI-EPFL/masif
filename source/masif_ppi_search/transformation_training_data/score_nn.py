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

"""
score_nn.py: Class to score protein complex alignments based on a pre-trained neural network (used for MaSIF-search's second stage protocol).
Freyr Sverrisson and Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

class ScoreNN:

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        np.random.seed(42)
        tf.random.set_random_seed(42)

        reg = keras.regularizers.l2(l=0.0)
        model = keras.models.Sequential()

        model.add(keras.layers.Conv1D(filters=8,kernel_size=1,strides=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())
        model.add(keras.layers.Conv1D(filters=16,kernel_size=1,strides=1, input_shape=(200,3)))
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

        self.model = model
        self.restore_model()

    def restore_model(self):
        self.model.load_weights('models/nn_score/trained_model.hdf5')

    def train_model(self, features, labels, n_negatives, n_positives):
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath='models/nn_score/{}.hdf5'.format('trained_model'),save_best_only=True,monitor='val_loss',save_weights_only=True),\
            keras.callbacks.TensorBoard(log_dir='./logs/nn_score',write_graph=False,write_images=True)\
        ]
        history = self.model.fit(features,labels,batch_size=32,epochs=50,validation_split=0.1,shuffle=True, class_weight={0:1.0/n_negatives,1:1.0/n_positives}, callbacks=callbacks)


    def eval(self, features):
        #set_trace()
        y_test_pred = self.model.predict(features)
        return y_test_pred


