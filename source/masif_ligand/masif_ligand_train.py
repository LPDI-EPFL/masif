# Header variables and parameters.
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
masif_ligand_train.py: Train MaSIF-ligand. 
Freyr Sverrisson - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

params = masif_opts["ligand"]

# Load dataset
training_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "training_data_sequenceSplit_30.tfrecord")
)
validation_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "validation_data_sequenceSplit_30.tfrecord")
)
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "testing_data_sequenceSplit_30.tfrecord")
)
training_data = training_data.map(_parse_function)
validation_data = validation_data.map(_parse_function)
testing_data = testing_data.map(_parse_function)
out_dir = params["model_dir"]
output_model = out_dir + "model"
if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])

with tf.Session() as sess:
    # Build the neural network model
    learning_obj = MaSIF_ligand(
        sess,
        params["max_distance"],
        params["n_classes"],
        idx_gpu="/gpu:0",
        feat_mask=params["feat_mask"],
        costfun=params["costfun"],
    )
    # learning_obj.saver.restore(learning_obj.session, 'monet_models/model')
    best_validation_loss = 1000
    best_validation_accuracy = 0.0
    total_iterations = 0
    num_epochs = 100
    for num_epoch in range(num_epochs):
        num_training_samples = 1030
        num_validation_samples = 120
        num_testing_samples = 290
        training_iterator = training_data.make_one_shot_iterator()
        training_next_element = training_iterator.get_next()
        validation_iterator = validation_data.make_one_shot_iterator()
        validation_next_element = validation_iterator.get_next()
        testing_iterator = testing_data.make_one_shot_iterator()
        testing_next_element = testing_iterator.get_next()

        training_losses = []
        training_ytrue = []
        training_ypred = []
        print("Total iterations", total_iterations)
        print("Calulating training loss")
        # Compute accuracy on a subset of the training set
        for num_train_sample in range(int(num_training_samples / 10)):
            try:
                data_element = sess.run(training_next_element)
            except:
                continue
            labels = data_element[4]
            n_ligands = labels.shape[1]
            # Choose a random ligand from the structure
            random_ligand = np.random.choice(n_ligands, 1)
            pocket_points = np.where(labels[:, random_ligand] != 0.0)[0]
            label = np.max(labels[:, random_ligand]) - 1
            pocket_labels = np.zeros(7, dtype=np.float32)
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < 32:
                continue
            # For evaluating take the first 32 points of the pocket
            feed_dict = {
                learning_obj.input_feat: data_element[0][pocket_points[:32], :, :],
                learning_obj.rho_coords: np.expand_dims(data_element[1], -1)[
                    pocket_points[:32], :, :
                ],
                learning_obj.theta_coords: np.expand_dims(data_element[2], -1)[
                    pocket_points[:32], :, :
                ],
                learning_obj.mask: data_element[3][pocket_points[:32], :, :],
                learning_obj.labels: pocket_labels,
                learning_obj.keep_prob: 1.0,
            }

            training_loss, training_logits = learning_obj.session.run(
                [learning_obj.data_loss, learning_obj.logits_softmax],
                feed_dict=feed_dict,
            )
            training_losses.append(training_loss)
            training_ytrue.append(label)
            training_ypred.append(np.argmax(training_logits))

        print(
            "Epoch {}, mean training loss {}, median training loss {}".format(
                num_epoch, np.mean(training_losses), np.median(training_losses)
            )
        )
        # Generate confusion matrix
        training_conf_mat = confusion_matrix(training_ytrue, training_ypred)
        # Compute accuracy
        training_accuracy = float(np.sum(np.diag(training_conf_mat))) / np.sum(
            training_conf_mat
        )
        print(training_conf_mat)
        print("Training accuracy:", training_accuracy)

        validation_losses = []
        validation_ytrue = []
        validation_ypred = []
        print("Calulating validation loss")
        # Compute accuracy on the validation set
        for num_val_sample in range(num_validation_samples):
            try:
                data_element = sess.run(validation_next_element)
            except:
                continue
            labels = data_element[4]
            n_ligands = labels.shape[1]
            random_ligand = np.random.choice(n_ligands, 1)
            pocket_points = np.where(labels[:, random_ligand] != 0.0)[0]
            label = np.max(labels[:, random_ligand]) - 1
            pocket_labels = np.zeros(7, dtype=np.float32)
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < 32:
                continue
            feed_dict = {
                learning_obj.input_feat: data_element[0][pocket_points[:32], :, :],
                learning_obj.rho_coords: np.expand_dims(data_element[1], -1)[
                    pocket_points[:32], :, :
                ],
                learning_obj.theta_coords: np.expand_dims(data_element[2], -1)[
                    pocket_points[:32], :, :
                ],
                learning_obj.mask: data_element[3][pocket_points[:32], :, :],
                learning_obj.labels: pocket_labels,
                learning_obj.keep_prob: 1.0,
            }

            validation_loss, validation_logits = learning_obj.session.run(
                [learning_obj.data_loss, learning_obj.logits_softmax],
                feed_dict=feed_dict,
            )
            validation_losses.append(validation_loss)
            validation_ytrue.append(label)
            validation_ypred.append(np.argmax(validation_logits))

        print(
            "Epoch {}, mean validation loss {}, median validation loss {}".format(
                num_epoch, np.mean(validation_losses), np.median(validation_losses)
            )
        )
        validation_conf_mat = confusion_matrix(validation_ytrue, validation_ypred)
        validation_accuracy = float(np.sum(np.diag(validation_conf_mat))) / np.sum(
            validation_conf_mat
        )
        print(validation_conf_mat)
        print("Validation accuracy:", validation_accuracy)
        if validation_accuracy > best_validation_accuracy:
            print("Saving model")
            learning_obj.saver.save(learning_obj.session, output_model)
            best_validation_accuracy = validation_accuracy

        testing_losses = []
        testing_ytrue = []
        testing_ypred = []
        print("Calulating testing loss")
        for num_test_sample in range(num_testing_samples):
            try:
                data_element = sess.run(testing_next_element)
            except:
                continue
            labels = data_element[4]
            n_ligands = labels.shape[1]
            random_ligand = np.random.choice(n_ligands, 1)
            pocket_points = np.where(labels[:, random_ligand] != 0.0)[0]
            label = np.max(labels[:, random_ligand]) - 1
            pocket_labels = np.zeros(7, dtype=np.float32)
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < 32:
                continue
            feed_dict = {
                learning_obj.input_feat: data_element[0][pocket_points[:32], :, :],
                learning_obj.rho_coords: np.expand_dims(data_element[1], -1)[
                    pocket_points[:32], :, :
                ],
                learning_obj.theta_coords: np.expand_dims(data_element[2], -1)[
                    pocket_points[:32], :, :
                ],
                learning_obj.mask: data_element[3][pocket_points[:32], :, :],
                learning_obj.labels: pocket_labels,
                learning_obj.keep_prob: 1.0,
            }

            testing_loss, testing_logits = learning_obj.session.run(
                [learning_obj.data_loss, learning_obj.logits_softmax],
                feed_dict=feed_dict,
            )
            testing_losses.append(testing_loss)
            testing_ytrue.append(label)
            testing_ypred.append(np.argmax(testing_logits))

        print(
            "Epoch {}, mean testing loss {}, median testing loss {}".format(
                num_epoch, np.mean(testing_losses), np.median(testing_losses)
            )
        )
        testing_conf_mat = confusion_matrix(testing_ytrue, testing_ypred)
        testing_accuracy = float(np.sum(np.diag(testing_conf_mat))) / np.sum(
            testing_conf_mat
        )
        print(testing_conf_mat)
        print("Testing accuracy:", testing_accuracy)
        # Stop training if number of iterations has reached 40000
        if total_iterations == 40000:
            break

        # Train the network
        training_losses = []
        training_ytrue = []
        training_ypred = []
        training_iterator = training_data.make_one_shot_iterator()
        training_next_element = training_iterator.get_next()
        for num_sample in range(num_training_samples):
            try:
                data_element = sess.run(training_next_element)
            except:
                continue
            labels = data_element[4]
            n_ligands = labels.shape[1]
            random_ligand = np.random.choice(n_ligands, 1)
            pocket_points = np.where(labels[:, random_ligand] != 0.0)[0]
            label = np.max(labels[:, random_ligand]) - 1
            pocket_labels = np.zeros(7, dtype=np.float32)
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < 32:
                continue
            # Sample 32 points randomly
            sample = np.random.choice(pocket_points, 32, replace=False)
            feed_dict = {
                learning_obj.input_feat: data_element[0][sample, :, :],
                learning_obj.rho_coords: np.expand_dims(data_element[1], -1)[
                    sample, :, :
                ],
                learning_obj.theta_coords: np.expand_dims(data_element[2], -1)[
                    sample, :, :
                ],
                learning_obj.mask: data_element[3][pocket_points[:32], :, :],
                learning_obj.labels: pocket_labels,
                learning_obj.keep_prob: 1.0,
            }

            _, training_loss, norm_grad, logits, logits_softmax, computed_loss = learning_obj.session.run(
                [
                    learning_obj.optimizer,
                    learning_obj.data_loss,
                    learning_obj.norm_grad,
                    learning_obj.logits,
                    learning_obj.logits_softmax,
                    learning_obj.computed_loss,
                ],
                feed_dict=feed_dict,
            )
            training_losses.append(training_loss)
            # training_ytrue.append(label)
            # training_ypred.append(np.argmax(logits_softmax))
            print(
                "Num sample {}\tTraining loss {}\nLabels {}\tSoftmax logits {}\tComputed loss {}\n".format(
                    num_sample,
                    training_loss,
                    pocket_labels,
                    logits_softmax,
                    computed_loss,
                )
            )
            if num_sample % 50 == 0:
                print(
                    "Mean training loss {}, median training loss {}".format(
                        np.mean(training_losses), np.median(training_losses)
                    )
                )
            total_iterations += 1
            if total_iterations == 40000:
                break

