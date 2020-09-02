import tensorflow as tf
import numpy as np
import os
from IPython.core.debugger import set_trace
from tensorflow import keras 
from corr_nn_context import CorrespondenceNN

corr_nn = CorrespondenceNN()
# Load all the training data. 
#features = features[:,:,0]
#features = np.expand_dims(features, 2)
corr_nn.train_model()


