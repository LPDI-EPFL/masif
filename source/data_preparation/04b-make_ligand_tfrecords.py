import numpy as np
import tensorflow as tf
from random import shuffle
import os
import glob
from scipy import spatial
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
# List all structures that have been preprocessed
precomputed_pdbs = glob.glob(
    os.path.join(params["masif_precomputation_dir"], "*", "p1_X.npy")
)
precomputed_pdbs = [p.split("/")[-2] for p in precomputed_pdbs]

# Only use the ones selected based on sequence homology
selected_pdbs = np.load(os.path.join("lists", "selected_pdb_ids_30.npy"))
selected_pdbs = selected_pdbs.astype(str)
all_pdbs = [p for p in precomputed_pdbs if p.split("_")[0] in selected_pdbs]

labels_dict = {"ADP": 1, "COA": 2, "FAD": 3, "HEM": 4, "NAD": 5, "NAP": 6, "SAM": 7}

# Structures are randomly assigned to train, validation and test sets
shuffle(all_pdbs)
train = int(len(all_pdbs) * params["train_fract"])
val = int(len(all_pdbs) * params["val_fract"])
test = int(len(all_pdbs) * params["test_fract"])
print("Train", train)
print("Validation", val)
print("Test", test)
train_pdbs = all_pdbs[:train]
val_pdbs = all_pdbs[train : train + val]
test_pdbs = all_pdbs[train + val : train + val + test]
# np.save('lists/train_pdbs_sequence.npy',train_pdbs)
# np.save('lists/val_pdbs_sequence.npy',val_pdbs)
# np.save('lists/test_pdbs_sequence.npy',test_pdbs)

# For this run use the train, validation and test sets actually used
train_pdbs = np.load("lists/train_pdbs_sequence.npy").astype(str)
val_pdbs = np.load("lists/val_pdbs_sequence.npy").astype(str)
test_pdbs = np.load("lists/test_pdbs_sequence.npy").astype(str)
success = 0
precom_dir = params["masif_precomputation_dir"]
ligand_coord_dir = params["ligand_coords_dir"]
tfrecords_dir = params["tfrecords_dir"]
if not os.path.exists(tfrecords_dir):
    os.mkdir(tfrecords_dir)
with tf.python_io.TFRecordWriter(
    os.path.join(tfrecords_dir, "training_data_sequenceSplit_30.tfrecord")
) as writer:
    for i, pdb in enumerate(train_pdbs):
        print("Working on", pdb)
        try:
            # Load precomputed data
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, pdb + "_", "p1_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, pdb + "_", "p1_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_", "p1_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_", "p1_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
                )
            )
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
                )
            ).astype(str)
        except:
            continue

        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int
        )
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ligand_coords = all_ligand_coords[j]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))
            pocket_labels[pocket_points_flatten, j] = labels_dict[structure_ligand]

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pocket_labels.shape)
        pocket_labels = tf.train.Int64List(value=pocket_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        }

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1 == 0:
            print("Training data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(train_pdbs))


success = 0
with tf.python_io.TFRecordWriter(
    os.path.join(tfrecords_dir, "validation_data_sequenceSplit_30.tfrecord")
) as writer:
    for i, pdb in enumerate(val_pdbs):
        try:
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, pdb + "_", "p1_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, pdb + "_", "p1_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_", "p1_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_", "p1_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
                )
            )
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
                )
            ).astype(str)
        except:
            continue

        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int
        )
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ligand_coords = all_ligand_coords[j]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))
            pocket_labels[pocket_points_flatten, j] = labels_dict[structure_ligand]

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pocket_labels.shape)
        pocket_labels = tf.train.Int64List(value=pocket_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        }

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1 == 0:
            print("Validation data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(val_pdbs))


success = 0
with tf.python_io.TFRecordWriter(
    os.path.join(tfrecords_dir, "testing_data_sequenceSplit_30.tfrecord")
) as writer:
    for i, pdb in enumerate(test_pdbs):
        try:
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, pdb + "_", "p1_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, pdb + "_", "p1_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_", "p1_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_", "p1_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
                )
            )
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
                )
            ).astype(str)
        except:
            continue

        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int
        )
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ligand_coords = all_ligand_coords[j]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))
            pocket_labels[pocket_points_flatten, j] = labels_dict[structure_ligand]

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pocket_labels.shape)
        pocket_labels = tf.train.Int64List(value=pocket_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        }

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1 == 0:
            print("Testing data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(test_pdbs))

