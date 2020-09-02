import tensorflow as tf
import sklearn.metrics
import numpy as np
import sys
import os
from IPython.core.debugger import set_trace
from tensorflow import keras 
from corr_nn_context import CorrespondenceNN

corr_nn = CorrespondenceNN()
corr_nn.restore_model()

test_pair_ids = os.listdir('data/testing/')
tmpl = 'data/testing/{}/features_0.npy'
test_feat_fn = [tmpl.format(x) for x in test_pair_ids]
tmpl = 'data/testing/{}/labels_0.npy'
test_label_fn = [tmpl.format(x) for x in test_pair_ids]

all_roc_nn = []
all_roc_bl = []
all_labels = []
all_scores_nn = []
all_scores_bl = []
for i in range(len(test_pair_ids)):
    feat = np.load(test_feat_fn[i])
    feat = np.expand_dims(feat,0)
    ypred = corr_nn.eval(feat)
    label = np.load(test_label_fn[i])
    ypred = np.squeeze(ypred)
    all_scores_nn.append(ypred)
    all_labels.append(label)
    all_roc_nn.append(sklearn.metrics.roc_auc_score(label, ypred))
    ypred = 1.0/feat[:,:,0]
    ypred = np.squeeze(ypred)
    all_scores_bl.append(ypred)
    all_roc_bl.append(sklearn.metrics.roc_auc_score(label, ypred))


all_scores_nn = np.concatenate(all_scores_nn)
all_scores_bl = np.concatenate(all_scores_bl)
all_labels = np.concatenate(all_labels)
print('NN Median roc auc score (per protein): {}'.format(np.median(all_roc_nn)))
print('NN Mean roc auc score (per protein): {}'.format(np.mean(all_roc_nn)))
print('*** NN roc auc score (all points): {}'.format(sklearn.metrics.roc_auc_score(all_labels, all_scores_nn)))
print('Baseline Median roc auc score (per protein): {}'.format(np.median(all_roc_bl)))
print('Baseline Mean roc auc score (per protein): {}'.format(np.mean(all_roc_bl)))
print('***Baseline roc auc score (all points): {}'.format(sklearn.metrics.roc_auc_score(all_labels, all_scores_bl)))

sys.exit(0)

