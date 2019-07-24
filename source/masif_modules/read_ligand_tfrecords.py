import tensorflow as tf


def _parse_function(example_proto):
    keys_to_features = {
        "input_feat_shape": tf.FixedLenFeature([3], dtype=tf.int64),
        "input_feat": tf.VarLenFeature(dtype=tf.float32),
        "rho_wrt_center_shape": tf.FixedLenFeature([2], dtype=tf.int64),
        "rho_wrt_center": tf.VarLenFeature(dtype=tf.float32),
        "theta_wrt_center_shape": tf.FixedLenFeature([2], dtype=tf.int64),
        "theta_wrt_center": tf.VarLenFeature(dtype=tf.float32),
        "mask_shape": tf.FixedLenFeature([3], dtype=tf.int64),
        "mask": tf.VarLenFeature(dtype=tf.float32),
        "pdb": tf.FixedLenFeature([], dtype=tf.string),
        "pocket_labels_shape": tf.FixedLenFeature([2], dtype=tf.int64),
        "pocket_labels": tf.VarLenFeature(dtype=tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    input_feat = tf.sparse_tensor_to_dense(parsed_features["input_feat"])
    input_feat = tf.reshape(
        input_feat, tf.cast(parsed_features["input_feat_shape"], tf.int32)
    )
    rho_wrt_center = tf.sparse_tensor_to_dense(parsed_features["rho_wrt_center"])
    rho_wrt_center = tf.reshape(
        rho_wrt_center, tf.cast(parsed_features["rho_wrt_center_shape"], tf.int32)
    )
    theta_wrt_center = tf.sparse_tensor_to_dense(parsed_features["theta_wrt_center"])
    theta_wrt_center = tf.reshape(
        theta_wrt_center, tf.cast(parsed_features["theta_wrt_center_shape"], tf.int32)
    )
    mask = tf.sparse_tensor_to_dense(parsed_features["mask"])
    mask = tf.reshape(mask, tf.cast(parsed_features["mask_shape"], tf.int32))
    labels = tf.sparse_tensor_to_dense(parsed_features["pocket_labels"])
    labels = tf.reshape(
        labels, tf.cast(parsed_features["pocket_labels_shape"], tf.int32)
    )
    return (
        input_feat,
        rho_wrt_center,
        theta_wrt_center,
        mask,
        labels,
        parsed_features["pdb"],
    )
