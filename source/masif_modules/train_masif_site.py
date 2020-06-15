import time
import os
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
from sklearn.metrics import accuracy_score, roc_auc_score

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def pad_indices(indices, max_verts):
    padded_ix = np.zeros((len(indices), max_verts), dtype=int)
    for patch_ix in range(len(indices)):
        padded_ix[patch_ix] = np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))]
        )
    return padded_ix


# Run masif site on a protein, on a previously trained network.
def run_masif_site(
    params, learning_obj, rho_wrt_center, theta_wrt_center, input_feat, mask, indices
):
    indices = pad_indices(indices, mask.shape[1])
    mask = np.expand_dims(mask, 2)
    feed_dict = {
        learning_obj.rho_coords: rho_wrt_center,
        learning_obj.theta_coords: theta_wrt_center,
        learning_obj.input_feat: input_feat,
        learning_obj.mask: mask,
        learning_obj.indices_tensor: indices,
    }

    score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
    return score


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)


def train_masif_site(
    learning_obj,
    params,
    batch_size=100,
    num_iterations=100,
    num_iter_test=1000,
    batch_size_val_test=50,
):

    # Open training list.

    list_training_loss = []
    list_training_auc = []
    list_validation_auc = []
    iter_time = []
    best_val_auc = 0

    out_dir = params["model_dir"]
    logfile = open(out_dir + "log.txt", "w")
    for key in params:
        logfile.write("{}: {}\n".format(key, params[key]))

    training_list = open(params["training_list"]).readlines()
    training_list = [x.rstrip() for x in training_list]

    testing_list = open(params["testing_list"]).readlines()
    testing_list = [x.rstrip() for x in testing_list]

    data_dirs = os.listdir(params["masif_precomputation_dir"])
    np.random.shuffle(data_dirs)
    data_dirs = data_dirs
    n_val = len(data_dirs) // 10
    val_dirs = set(data_dirs[(len(data_dirs) - n_val) :])

    for num_iter in range(num_iterations):
        # Start training epoch:
        list_training_loss = []
        list_training_auc = []
        list_val_auc = []
        list_val_pos_labels = []
        list_val_neg_labels = []
        list_val_names = []
        list_training_acc = []
        list_val_acc = []
        logfile.write("Starting epoch {}".format(num_iter))
        print("Starting epoch {}".format(num_iter))
        tic = time.time()
        all_training_labels = []
        all_training_scores = []
        all_val_labels = []
        all_val_scores = []
        all_test_labels = []
        all_test_scores = []
        count_proteins = 0

        list_test_auc = []
        list_test_names = []
        list_test_acc = []
        all_test_labels = []
        all_test_scores = []

        for ppi_pair_id in data_dirs:
            mydir = params["masif_precomputation_dir"] + ppi_pair_id + "/"
            pdbid = ppi_pair_id.split("_")[0]
            chains1 = ppi_pair_id.split("_")[1]
            if len(ppi_pair_id.split("_")) > 2:
                chains2 = ppi_pair_id.split("_")[2]
            else: 
                chains2 = ''
            pids = []
            if pdbid + "_" + chains1 in training_list:
                pids.append("p1")
            if pdbid + "_" + chains2 in training_list:
                pids.append("p2")
            for pid in pids:
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except:
                    continue
                if len(iface_labels) > 8000:
                    continue
                if (
                    np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
                ):
                    continue
                count_proteins += 1

                rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
                theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
                input_feat = np.load(mydir + pid + "_input_feat.npy")
                if np.sum(params["feat_mask"]) < 5:
                    input_feat = mask_input_feat(input_feat, params["feat_mask"])
                mask = np.load(mydir + pid + "_mask.npy")
                mask = np.expand_dims(mask, 2)
                indices = np.load(mydir + pid + "_list_indices.npy", encoding="latin1")
                # indices is (n_verts x <30), it should be
                indices = pad_indices(indices, mask.shape[1])
                tmp = np.zeros((len(iface_labels), 2))
                for i in range(len(iface_labels)):
                    if iface_labels[i] == 1:
                        tmp[i, 0] = 1
                    else:
                        tmp[i, 1] = 1
                iface_labels_dc = tmp
                logfile.flush()
                pos_labels = np.where(iface_labels == 1)[0]
                neg_labels = np.where(iface_labels == 0)[0]
                np.random.shuffle(neg_labels)
                np.random.shuffle(pos_labels)
                # Scramble neg idx, and only get as many as pos_labels to balance the training.
                if params["n_conv_layers"] == 1:
                    n = min(len(pos_labels), len(neg_labels))
                    n = min(n, batch_size // 2)
                    subset = np.concatenate([neg_labels[:n], pos_labels[:n]])

                    rho_wrt_center = rho_wrt_center[subset]
                    theta_wrt_center = theta_wrt_center[subset]
                    input_feat = input_feat[subset]
                    mask = mask[subset]
                    iface_labels_dc = iface_labels_dc[subset]
                    indices = indices[subset]
                    pos_labels = range(0, n)
                    neg_labels = range(n, n * 2)
                else:
                    n = min(len(pos_labels), len(neg_labels))
                    neg_labels = neg_labels[:n]
                    pos_labels = pos_labels[:n]

                feed_dict = {
                    learning_obj.rho_coords: rho_wrt_center,
                    learning_obj.theta_coords: theta_wrt_center,
                    learning_obj.input_feat: input_feat,
                    learning_obj.mask: mask,
                    learning_obj.labels: iface_labels_dc,
                    learning_obj.pos_idx: pos_labels,
                    learning_obj.neg_idx: neg_labels,
                    learning_obj.indices_tensor: indices,
                }

                if ppi_pair_id in val_dirs:
                    logfile.write("Validating on {} {}\n".format(ppi_pair_id, pid))
                    feed_dict[learning_obj.keep_prob] = 1.0
                    training_loss, score, eval_labels = learning_obj.session.run(
                        [
                            learning_obj.data_loss,
                            learning_obj.eval_score,
                            learning_obj.eval_labels,
                        ],
                        feed_dict=feed_dict,
                    )
                    auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                    list_val_pos_labels.append(np.sum(iface_labels))
                    list_val_neg_labels.append(len(iface_labels) - np.sum(iface_labels))
                    list_val_auc.append(auc)
                    list_val_names.append(ppi_pair_id)
                    all_val_labels = np.concatenate([all_val_labels, eval_labels[:, 0]])
                    all_val_scores = np.concatenate([all_val_scores, score])
                else:
                    logfile.write("Training on {} {}\n".format(ppi_pair_id, pid))
                    feed_dict[learning_obj.keep_prob] = 1.0
                    _, training_loss, norm_grad, score, eval_labels = learning_obj.session.run(
                        [
                            learning_obj.optimizer,
                            learning_obj.data_loss,
                            learning_obj.norm_grad,
                            learning_obj.eval_score,
                            learning_obj.eval_labels,
                        ],
                        feed_dict=feed_dict,
                    )
                    all_training_labels = np.concatenate(
                        [all_training_labels, eval_labels[:, 0]]
                    )
                    all_training_scores = np.concatenate([all_training_scores, score])
                    auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                    list_training_auc.append(auc)
                    list_training_loss.append(np.mean(training_loss))
                logfile.flush()

        # Run testing cycle.
        for ppi_pair_id in data_dirs:
            mydir = params["masif_precomputation_dir"] + ppi_pair_id + "/"
            pdbid = ppi_pair_id.split("_")[0]
            chains1 = ppi_pair_id.split("_")[1]
            if len(ppi_pair_id.split("_")) > 2:
                chains2 = ppi_pair_id.split("_")[2]
            else: 
                chains2 = ''
            pids = []
            if pdbid + "_" + chains1 in testing_list:
                pids.append("p1")
            if pdbid + "_" + chains2 in testing_list:
                pids.append("p2")
            for pid in pids:
                logfile.write("Testing on {} {}\n".format(ppi_pair_id, pid))
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except:
                    continue
                if len(iface_labels) > 20000:
                    continue
                if (
                    np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
                ):
                    continue
                count_proteins += 1

                rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
                theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
                input_feat = np.load(mydir + pid + "_input_feat.npy")
                if np.sum(params["feat_mask"]) < 5:
                    input_feat = mask_input_feat(input_feat, params["feat_mask"])
                mask = np.load(mydir + pid + "_mask.npy")
                mask = np.expand_dims(mask, 2)
                indices = np.load(mydir + pid + "_list_indices.npy", encoding="latin1")
                # indices is (n_verts x <30), it should be
                indices = pad_indices(indices, mask.shape[1])
                tmp = np.zeros((len(iface_labels), 2))
                for i in range(len(iface_labels)):
                    if iface_labels[i] == 1:
                        tmp[i, 0] = 1
                    else:
                        tmp[i, 1] = 1
                iface_labels_dc = tmp
                logfile.flush()
                pos_labels = np.where(iface_labels == 1)[0]
                neg_labels = np.where(iface_labels == 0)[0]

                feed_dict = {
                    learning_obj.rho_coords: rho_wrt_center,
                    learning_obj.theta_coords: theta_wrt_center,
                    learning_obj.input_feat: input_feat,
                    learning_obj.mask: mask,
                    learning_obj.labels: iface_labels_dc,
                    learning_obj.pos_idx: pos_labels,
                    learning_obj.neg_idx: neg_labels,
                    learning_obj.indices_tensor: indices,
                }

                feed_dict[learning_obj.keep_prob] = 1.0
                score = learning_obj.session.run(
                    [learning_obj.full_score], feed_dict=feed_dict
                )
                score = score[0]
                auc = metrics.roc_auc_score(iface_labels, score)
                list_test_auc.append(auc)
                list_test_names.append((ppi_pair_id, pid))
                all_test_labels.append(iface_labels)
                all_test_scores.append(score)

        outstr = "Epoch ran on {} proteins\n".format(count_proteins)
        outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_training_auc), np.median(list_training_auc), num_iter
        )
        outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_val_auc), np.median(list_val_auc), num_iter
        )
        outstr += "Per protein AUC mean (test): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_test_auc), np.median(list_test_auc), num_iter
        )
        flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
        flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
        outstr += "Testing auc (all points): {:.2f}".format(
            metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
        )
        outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
        logfile.write(outstr + "\n")
        print(outstr)

        if np.mean(list_val_auc) > best_val_auc:
            logfile.write(">>> Saving model.\n")
            print(">>> Saving model.\n")
            best_val_auc = np.mean(list_val_auc)
            output_model = out_dir + "model"
            learning_obj.saver.save(learning_obj.session, output_model)
            # Save the scores for test.
            np.save(out_dir + "test_labels.npy", all_test_labels)
            np.save(out_dir + "test_scores.npy", all_test_scores)
            np.save(out_dir + "test_names.npy", list_test_names)

    logfile.close()
