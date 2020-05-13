import os
import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand import MaSIF_ligand
from masif_modules.read_ligand_tfrecords import _parse_function
from sklearn.metrics import confusion_matrix
import tensorflow as tf

"""
masif_ligand_evaluate_test: Evaluate and test MaSIF-ligand.
Freyr Sverrisson - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

params = masif_opts["ligand"]
# Load testing data
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "testing_data_sequenceSplit_30.tfrecord")
)
testing_data = testing_data.map(_parse_function)

model_dir = params["model_dir"]
output_model = model_dir + "model"

test_set_out_dir = params["test_set_out_dir"]
if not os.path.exists(test_set_out_dir):
    os.makedirs(test_set_out_dir)


with tf.Session() as sess:
    # Build network
    learning_obj = MaSIF_ligand(
        sess,
        params["max_distance"],
        params["n_classes"],
        idx_gpu="/gpu:0",
        feat_mask=params["feat_mask"],
        costfun=params["costfun"],
    )
    # Load pretrained network
    learning_obj.saver.restore(learning_obj.session, output_model)

    num_test_samples = 290
    testing_iterator = testing_data.make_one_shot_iterator()
    testing_next_element = testing_iterator.get_next()

    all_logits_softmax = []
    all_labels = []
    all_pdbs = []
    all_data_loss = []
    for num_test_sample in range(num_test_samples):
        try:
            data_element = sess.run(testing_next_element)
        except:
            continue

        print(num_test_sample)

        labels = data_element[4]
        n_ligands = labels.shape[1]
        pdb_logits_softmax = []
        pdb_labels = []
        for ligand in range(n_ligands):
            # Rows indicate point number and columns ligand type
            pocket_points = np.where(labels[:, ligand] != 0.0)[0]
            label = np.max(labels[:, ligand]) - 1
            pocket_labels = np.zeros(7, dtype=np.float32)
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < 32:
                continue
            pdb_labels.append(label)
            pdb = data_element[5]
            # all_pdbs.append(pdb)

            samples_logits_softmax = []
            samples_data_loss = []
            # Make 100 predictions
            for i in range(100):
                # Sample pocket randomly
                sample = np.random.choice(pocket_points, 32, replace=False)
                feed_dict = {
                    learning_obj.input_feat: data_element[0][sample, :, :],
                    learning_obj.rho_coords: np.expand_dims(data_element[1], -1)[
                        sample, :, :
                    ],
                    learning_obj.theta_coords: np.expand_dims(data_element[2], -1)[
                        sample, :, :
                    ],
                    learning_obj.mask: data_element[3][sample, :, :],
                    learning_obj.labels: pocket_labels,
                    learning_obj.keep_prob: 1.0,
                }

                logits_softmax, data_loss = learning_obj.session.run(
                    [learning_obj.logits_softmax, learning_obj.data_loss],
                    feed_dict=feed_dict,
                )
                samples_logits_softmax.append(logits_softmax)
                samples_data_loss.append(data_loss)

            pdb_logits_softmax.append(samples_logits_softmax)
        np.save(test_set_out_dir + "{}_labels.npy".format(pdb), pdb_labels)
        np.save(test_set_out_dir + "{}_logits.npy".format(pdb), pdb_logits_softmax)

