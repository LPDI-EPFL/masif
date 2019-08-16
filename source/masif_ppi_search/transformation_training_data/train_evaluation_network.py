import tensorflow as tf
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow import keras
import os
import time
import pickle
import sys

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

np.random.seed(42)
tf.random.set_random_seed(42)

data_dir = "transformation_data/"

with open(
    "../lists/training.txt"
) as f:
    training_list = f.read().splitlines()

with open(
    "../lists/testing.txt"
) as f:
    testing_list = f.read().splitlines()

n_positives = 1
n_negatives = 200
max_rmsd = 5.0
max_npoints = 200
inlier_distance = 1.0
n_features = 3
data_list = glob.glob(data_dir+'*')
data_list = [
    d
    for d in data_list
    if (os.path.exists(d + "/" + "features.npy")) and d.split("/")[-1] in training_list
]
all_features = np.empty(
    (len(data_list) * (n_positives + n_negatives), max_npoints, n_features)
)
all_labels = np.empty((len(data_list) * (n_positives + n_negatives), 1))
all_scores = np.empty((len(data_list) * (n_positives + n_negatives), 1))
all_npoints = []
all_idxs = []
all_nsources = []
n_samples = 0
for i, d in enumerate(data_list):
    if (i % 100 == 0) and (i == 0):
        print(i, "Feature array size (MB)", all_features.nbytes * 1e-6)
        start = time.time()
    elif i % 100 == 0:
        end = time.time()
        print(
            i,
            "Feature array size (MB)",
            all_features.nbytes * 1e-6,
            "Time",
            end - start,
        )
        start = time.time()

    source_patch_rmsds = np.load(d + "/" + "source_patch_rmsds.npy")

    positive_alignments = np.where(source_patch_rmsds < max_rmsd)[0]
    negative_alignments = np.where(source_patch_rmsds >= max_rmsd)[0]

    if len(positive_alignments) == 0:  # <n_positives:
        continue
    if len(negative_alignments) < n_negatives:
        continue
    chosen_positives = np.random.choice(positive_alignments, n_positives, replace=False)
    chosen_negatives = np.random.choice(negative_alignments, n_negatives, replace=False)
    chosen_alignments = np.concatenate([chosen_positives, chosen_negatives])
    try:
        features = np.load(d + "/" + "features.npy", encoding="latin1", allow_pickle=True)
    except:
        continue
    n_sources = len(features)
    features = features[chosen_alignments]
    features_trimmed = np.zeros((len(chosen_alignments), max_npoints, n_features))
    for j, f in enumerate(features):
        if f.shape[0] <= max_npoints:
            features_trimmed[j, : f.shape[0], : f.shape[1]] = f
        else:
            selected_rows = np.random.choice(f.shape[0], max_npoints, replace=False)
            features_trimmed[j, :, : f.shape[1]] = f[selected_rows]

    labels = np.array(
        (source_patch_rmsds[chosen_alignments] < max_rmsd).astype(int)
    ).reshape(-1, 1)

    all_features[
        n_samples : n_samples + len(chosen_alignments), :, :
    ] = features_trimmed
    all_labels[n_samples : n_samples + len(chosen_alignments)] = labels
    n_samples += len(chosen_alignments)

all_features = all_features[:n_samples]
all_labels = all_labels[:n_samples]

all_idxs = np.concatenate(
    [
        (n_positives + n_negatives) * [i]
        for i in range(int(all_features.shape[0] / (n_positives + n_negatives)))
    ]
)

reg = keras.regularizers.l2(l=0.0)
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=8, kernel_size=1, strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=16, kernel_size=1, strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=32, kernel_size=1, strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=64, kernel_size=1, strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=128, kernel_size=1, strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv1D(filters=256, kernel_size=1, strides=1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=reg))
model.add(keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=reg))
model.add(keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=reg))
model.add(keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=reg))
model.add(keras.layers.Dense(8, activation=tf.nn.relu, kernel_regularizer=reg))
model.add(keras.layers.Dense(4, activation=tf.nn.relu, kernel_regularizer=reg))
model.add(keras.layers.Dense(2, activation="softmax"))

opt = keras.optimizers.Adam(lr=1e-4)
model.compile(
    optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/nn_score/trained_model.hdf5",
        save_best_only=True,
        monitor="val_loss",
        save_weights_only=True,
    ),
    keras.callbacks.TensorBoard(
        log_dir="./logs/nn_score", write_graph=False, write_images=True
    ),
]
history = model.fit(
    all_features,
    all_labels,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    shuffle=True,
    class_weight={0: 1.0 / n_negatives, 1: 1.0 / n_positives},
    callbacks=callbacks,
)

#with open("histories/history_dict_3feat_new.pkl", "wb") as f:
#    pickle.dump(history.history, f)
