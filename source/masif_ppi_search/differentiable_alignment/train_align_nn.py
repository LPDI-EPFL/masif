import tensorflow as tf
import numpy as np
import os
from IPython.core.debugger import set_trace
from tensorflow import keras 
from align_nn import AlignNN

align_nn = AlignNN()
# Load all the training data. 
#features = features[:,:,0]
#features = np.expand_dims(features, 2)
align_nn.train_model()


