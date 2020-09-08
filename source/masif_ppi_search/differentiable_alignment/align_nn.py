import tensorflow as tf
import numpy as np
import os
from IPython.core.debugger import set_trace
from tensorflow import keras 
from rand_rotation import batch_rand_rotate_center_patch
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from corr_nn_context import CorrespondenceNN

NUM_NEIGH = 100

# point to point rmsd loss. 
def p2p_rmsd_loss(yTrue, yPred):
    yPred = yPred[:,:,0:3]
    yTrue = yTrue[:,:,0:3]
    mydiff = yTrue - yPred # (batch_size, NUM_NEIGH, 3)
    mysquare = K.square(mydiff) # (batch_size, NUM_NEIGH, 3)
    mysum = K.sum(mysquare, axis=2) # (batch_size, NUM_NEIGH)
    mymean = K.mean(mysum, axis=1) # (batch_size)
    mysqrt = K.sqrt(mymean)
    costfunc = K.mean(mysqrt) # (1)
    return costfunc

# EXPERIMENTAL - a shape similarity loss: use both points and normals
# UNSTABLE - code has a bug. 
def shape_sim_loss(yTrue, yPred):
    yPredxyz = yPred[:,:,0:3]
    yPrednorm = yPred[:,:,3:6]
    yTruexyz = yTrue[:,:,0:3]
    yTruenorm = yTrue[:,:,3:6]

    # Compute the point to point distances between all points.
    mydiff = yTruexyz - yPredxyz # (batch_size, NUM_NEIGH, 3)
    mysquare = K.square(mydiff) # (batch_size, NUM_NEIGH, 3)
    p2pdist2 = K.sum(mysquare, axis=2) # (batch_size, NUM_NEIGH)
#    p2pdist= K.sqrt(mysum)

    # This is actually shape similarity, not shape complementarity.
    print("yPrednorm {}".format(yPrednorm.shape))
    print("yTruenorm {}".format(yTruenorm.shape))
    print("p2pdist2 {}".format(p2pdist2.shape))
    ytn = K.permute_dimensions(yTruenorm,(0,2,1)) # (batch_size, 3, NUM_NEIGH
    sc = K.batch_dot(yPrednorm, ytn , axes=2) # (batch_size, NUM_NEIGH, 1)
#    assert(sc.shape[2] ==1)
    print("midpoint {}".format(sc.shape))
    sc = K.squeeze(sc, axis=2)
    assert(sc.shape[1] == 100)
    assert (p2pdist2.shape[1] == sc.shape[1])
    w = 0.5
    sc = tf.math.multiply(sc, K.exp(-w * p2pdist2)) # (batch_size, NUM_NEIGH)
    print("after {}".format(sc.shape))
    assert (len(sc.shape) == 2)
    assert (sc.shape[1] == 100)

    sc = K.mean(sc) #(batch_size, NUM_NEIGH)

    return sc 


class Align_Generator(keras.utils.Sequence):

    def __init__(self, feature_filenames, label_filenames, pred_filenames, batch_size):
        self.feature_filenames, self.label_filenames, self.pred_filenames = feature_filenames, label_filenames, pred_filenames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.feature_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.feature_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_z = self.pred_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        feat_batch = np.array([np.load(file_name) for file_name in batch_x])
        pred_batch = np.array([np.load(file_name) for file_name in batch_z])
        pred_batch = np.expand_dims(pred_batch, 2)
        # DEBUG: set the true correspondences.


        # Randomly rotate xyz2 only.  
        xyz2 = feat_batch[:,:,4:7]
        gt_xyz2 = np.copy(xyz2)
        norm2 = feat_batch[:,:,10:13]
        gt_norm2 = np.copy(norm2)
        xyz2, norm2 = batch_rand_rotate_center_patch(xyz2, norm2)
        feat_batch[:,:,4:7] = xyz2
        feat_batch[:,:,10:13] = norm2

        # Stack weights and coordinates.
        feat_batch = np.concatenate([pred_batch, feat_batch[:,:,1:13]], axis=2)
        assert(feat_batch.shape[1] == NUM_NEIGH)
        assert(feat_batch.shape[2] == 13)
        
        # Labels are the original position of xyz2 and the original normals.
        label_batch = np.concatenate([gt_xyz2,gt_norm2], axis=2)
        assert(label_batch.shape[1] == NUM_NEIGH)
        assert(label_batch.shape[2] == 6)
        assert(len(label_batch.shape) == 3)
        
        return feat_batch, label_batch

# Layer for SVD
class SVDAlign(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SVDAlign, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for shifting the points outward.
        self.outward_shift_val = self.add_weight(name='outward_shift', 
                                      shape=[1],
                                      initializer=keras.initializers.Constant(value=0),
                                      trainable=False)

        # Create a simple single neuron for the weights to create the linear function ax + b
        self.a = self.add_weight(name='a_factor', 
                                      shape=[1],
                                      initializer='ones',
                                      trainable=True)

        self.b = self.add_weight(name='b_constant', 
                                      shape=[1],
                                      initializer='zeros',
                                      trainable=True)

        super(SVDAlign, self).build(input_shape)  # Be sure to call this at the end

    # Performs SVD.
    # Input:
    # Array of size (batch_size, max_points, feats)
    # feats are: w: corr_score
    #            xyz1: xyz of protein 1
    #            xyz2: xyz of protein 2
    #            norm1: normals of protein1
    #            norm2: normals of protein2.
    # Returns: (batch_size, rotated_xyz+rotated_normals)
    def call(self, network_in):
        w = network_in[:,:,0]
        xyz1 = network_in[:,:,1:4]
        xyz2 = network_in[:,:,4:7]
        orig_xyz2 = network_in[:,:,4:7]
        norm1 = network_in[:,:,7:10]
        norm2 = network_in[:,:,10:13]
        orig_norm2 = network_in[:,:,10:13]
        assert(len(xyz1.shape) == 3)
        # Filter the correspondences by a simple neuron and a relu. 
        w = K.expand_dims(w, 2) #(batch_size, NUM_NEIGH, 1)
        assert(len(w.shape) == 3)
        w = self.a*w + self.b #(batch_size, NUM_NEIGH, 1)
        w = K.relu(w) 
        # Add a small value to w to prevent division by zero. 
        w = w+1e-8
        # Move xyz outward by a learned weight
        v1 = xyz1 + norm1*self.outward_shift_val #(batch_size, NUM_NEIGH, 3)
        v2 = xyz2 + norm2*self.outward_shift_val #(batch_size, NUM_NEIGH, 3)
        assert(len(v1.shape) == 3)

        # Move v1 and v2 to the weighted center
        w_sum = K.sum(w, axis=1) # (batch_size, 1)

        v1_ctr = K.sum(v1*w, axis=1)/w_sum # (batch_size, 3)
        v2_ctr = K.sum(v2*w, axis=1)/w_sum # (batch_size, 3)
        v1_ctr = K.expand_dims(v1_ctr, axis=1) # (batch_size, 1, 3)
        v2_ctr = K.expand_dims(v2_ctr, axis=1) # (batch_size, 1, 3)
        v1 = v1 - v1_ctr # (batch_size, NUM_NEIGH, 3)
        v2 = v2 - v2_ctr # (batch_size, NUM_NEIGH, 3)
        print(v1.shape)
        assert(v1.shape[1] == NUM_NEIGH)

        # Compute rotation and translation.
        # Identity matrix. 
        W = tf.linalg.diag(K.reshape(w,[-1,NUM_NEIGH])) # (batch_size, NUM_NEIGH, NUM_NEIGH)
        cov_mat = K.batch_dot(K.permute_dimensions(v2,(0,2,1)), W) # (batch_size, 3, NUM_NEIGH)
        cov_mat = K.batch_dot(cov_mat, v1) # (batch_size, 3, 3)
        s,u,v = tf.linalg.svd(cov_mat) 
        VUt = K.batch_dot(v, K.permute_dimensions(u, (0,2,1))) # batch_size, 3, 3)
        det_VUt = tf.linalg.det(VUt) # (batch_size)
        det_VUt = K.expand_dims(det_VUt, 1)
        # Turn the determinant into a [1, 1, det_VUt] diagonal matrix.
        col1 = K.ones_like(det_VUt)
        col2 = K.ones_like(det_VUt)
        row = K.stack([col1, col2, det_VUt], axis=1) # (batch_size, 3, 1)
        row = tf.reshape(row, [-1, 3]) # (batch_size, 2)
        
        det_matrix = tf.linalg.diag(row) # (batch_size,3,3)

        # Compute the actual rotation matrix
        R = K.batch_dot(v, det_matrix) # (batch_size, 3, 3)
        R = K.batch_dot(R, K.permute_dimensions(u, (0,2,1))) # (batch_size, 3, 3)
        t = K.permute_dimensions(v1_ctr, (0,2,1)) - K.batch_dot(R, K.permute_dimensions(v2_ctr, (0,2,1))) # (batch_size, 3, 1)
        #t = K.permute_dimensions(t, (0,2,1)) # batch_size, 3

        # Rotate+Translate the coordinates for xyz2 
        new_coords = K.batch_dot(R, K.permute_dimensions(orig_xyz2, (0,2,1))) # (batch_size, 3, NUM_NEIGH)
        new_coords = new_coords+t # (batch_size, 3, NUM_NEIGH)
        # Transpose 
        new_coords = K.permute_dimensions(new_coords, (0,2,1)) # (batch_size, NUM_NEIGH, 3)

        # Rotate+translate the coordinates for norm2
        new_norm = K.batch_dot(R, K.permute_dimensions(orig_norm2, (0,2,1))) # (batch_size, 3, NUM_NEIGH)
        new_norm = K.permute_dimensions(new_norm, (0,2,1)) # (batch_size, NUM_NEIGH, 3)
        
        coord_norm = K.concatenate([new_coords, new_norm], axis =2)
        return coord_norm



    def compute_output_shape(self, input_shape):
        return self.output_dim

class AlignNN:

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        np.random.seed(42)
        tf.random.set_random_seed(42)

        reg = keras.regularizers.l2(l=0.0)
        
        # Input is the weights, xyz, and the normals.
        network_in = keras.layers.Input(shape=(NUM_NEIGH,13))
        
        self.svd_layer = SVDAlign((NUM_NEIGH,3))(network_in)
        network_out = self.svd_layer
        model = keras.models.Model(inputs=network_in, outputs=network_out)
        opt = keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer=opt,loss=p2p_rmsd_loss)
        self.model = model
        self.init_data_dir()


    def train_model(self):
        batch_size = 8
        my_training_batch_generator = Align_Generator(self.train_feat_fn, self.train_label_fn, self.train_pred_fn, batch_size)
        my_val_batch_generator = Align_Generator(self.val_feat_fn, self.val_label_fn, self.val_pred_fn,  batch_size)
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath='models/{}.hdf5'.format('trained_align_model'),save_best_only=True,monitor='val_loss',save_weights_only=True),\
            keras.callbacks.TensorBoard(log_dir='./logs/output',write_graph=False,write_images=True)\
        ]
        history = self.model.fit_generator(my_training_batch_generator,  validation_data=my_val_batch_generator,\
                        use_multiprocessing=True,\
                        workers=4, epochs=100, callbacks=callbacks)
        self.print_status()

    def eval(self, features):
        y_test_pred = self.model.predict(features)
        return y_test_pred

    def restore_model(self):
        self.model.load_weights('models/trained_align_model.hdf5')

    def look_into_model(self):
        output = self.model.layers[0].output
        return output

    def print_status(self):
        for layer in self.model.layers:
            print('Trainable weights: {}'.format(layer.trainable_variables))
            print(layer.get_weights())
    
    def init_data_dir(self):
        all_training_pair_ids = os.listdir('data/training/')
        np.random.shuffle(all_training_pair_ids)
        val_split = int(np.floor(0.9 * len(all_training_pair_ids)))
        val_pair_ids = all_training_pair_ids[val_split:]
        train_pair_ids = all_training_pair_ids[:val_split]
        tmpl = 'data/training/{}/features_0.npy'
        self.train_feat_fn = [tmpl.format(x) for x in train_pair_ids]
        self.val_feat_fn = [tmpl.format(x) for x in val_pair_ids]
        tmpl = 'data/training/{}/labels_0.npy'
        self.train_label_fn = [tmpl.format(x) for x in train_pair_ids]
        self.val_label_fn = [tmpl.format(x) for x in val_pair_ids]
        tmpl = 'data/training/{}/pred_0.npy'
        self.train_pred_fn = [tmpl.format(x) for x in train_pair_ids]
        self.val_pred_fn = [tmpl.format(x) for x in val_pair_ids]

