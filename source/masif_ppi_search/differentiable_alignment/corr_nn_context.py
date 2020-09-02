import tensorflow as tf
import numpy as np
import os
from IPython.core.debugger import set_trace
from tensorflow import keras 
from rand_rotation import batch_rand_rotate_center_patch
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Corr_Generator(keras.utils.Sequence):

    def __init__(self, feature_filenames, label_filenames, batch_size):
        self.feature_filenames, self.label_filenames = feature_filenames, label_filenames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.feature_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.feature_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        feat_batch = np.array([np.load(file_name) for file_name in batch_x])
        label_batch = np.array([np.load(file_name) for file_name in batch_y])

        # Should randomly rotate here. 
        xyz2 = feat_batch[:,:,4:7]
        norm2 = feat_batch[:,:,10:13]
        xyz2, norm2 = batch_rand_rotate_center_patch(xyz2, norm2)
        feat_batch[:,:,4:7] = xyz2
        feat_batch[:,:,10:13] = norm2
        
        label_batch = np.expand_dims(label_batch,2)
        return feat_batch, label_batch

class ContextNormalization(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ContextNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ContextNormalization, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #assert isinstance(x, list)
        assert(len(x.shape) == 3)
        mymean = K.mean(x, axis=1)      
        mymean = K.expand_dims(mymean, 1)
        mystd = K.std(x, axis=1)
        mystd = K.expand_dims(mystd, 1)
        return (x - mymean)/mystd

    def compute_output_shape(self, input_shape):
        return self.output_dim

class CorrespondenceNN:

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        np.random.seed(42)
        tf.random.set_random_seed(42)

        reg = keras.regularizers.l2(l=0.0)
#        model = keras.models.Sequential()
    
        # Initial multilayer perceptron 
        network_in = keras.layers.Input(shape=(200,13))
        dense1 = keras.layers.Dense(128)(network_in)
        resnet1 = self.resnet(dense1)
        resnet2 = self.resnet(resnet1)
        resnet3 = self.resnet(resnet2)
        resnet4 = self.resnet(resnet3)
        resnet5 = self.resnet(resnet4)
        resnet6 = self.resnet(resnet5)
        resnet7 = self.resnet(resnet6)
        resnet8 = self.resnet(resnet7)
        resnet9 = self.resnet(resnet8)
        resnet10 = self.resnet(resnet9)
        resnet11 = self.resnet(resnet10)
        resnet12 = self.resnet(resnet11)
        dense_out1 = keras.layers.Dense(1)(resnet12)
        relu_out = keras.layers.ReLU()(dense_out1)
        dense_out2 = keras.layers.Dense(1, activation='sigmoid')(relu_out)

        # Multiply the xyz coordinates of the points and the output of the 
        model = keras.models.Model(inputs=network_in, outputs=dense_out2)

        opt = keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

        self.model = model
        for layer in model.layers:
            print('Trainable weights: {}'.format(layer.trainable_variables))
            print(layer.output_shape)

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



    def resnet(self, resnet_in):
        p1 = keras.layers.Dense(128)(resnet_in)
        cn1 = ContextNormalization((200,128))(p1)
        bn1 = keras.layers.BatchNormalization()(cn1)
        relu1 = keras.layers.ReLU()(bn1)
        p2 = keras.layers.Dense(128)(relu1)
        cn2 = ContextNormalization((200,128))(p2)
        bn2 = keras.layers.BatchNormalization()(cn2)
        relu2 = keras.layers.ReLU()(bn2)
        resnet_out = keras.layers.Add()([resnet_in, relu2])
        return resnet_out

    def train_model(self):
        batch_size = 8
        my_training_batch_generator = Corr_Generator(self.train_feat_fn, self.train_label_fn, batch_size)
        my_val_batch_generator = Corr_Generator(self.val_feat_fn, self.val_label_fn, batch_size)
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath='models/{}.hdf5'.format('trained_corr_model'),save_best_only=True,monitor='val_loss',save_weights_only=True),\
            keras.callbacks.TensorBoard(log_dir='./logs/output',write_graph=False,write_images=True)\
        ]
        history = self.model.fit_generator(my_training_batch_generator,  validation_data=my_val_batch_generator,\
                        use_multiprocessing=True,\
                        workers=4, epochs=1000, callbacks=callbacks)

    def eval(self, features):
        y_test_pred = self.model.predict(features)
        return y_test_pred

    def restore_model(self):
        self.model.load_weights('models/trained_corr_model.hdf5')

    def look_into_model(self):
        output = self.model.layers[0].output
        return output

