import tensorflow as tf
import sklearn.metrics
import numpy as np
from rand_rotation import rand_rotate_center_patch
import sys
import os
from IPython.core.debugger import set_trace
from tensorflow import keras 
from corr_nn_context import CorrespondenceNN

corr_nn = CorrespondenceNN()
corr_nn.restore_model()

eval_pair_ids = os.listdir('data/testing/')
tmpl = 'data/testing/{}/features_0.npy'
eval_feat_fn = [tmpl.format(x) for x in eval_pair_ids]
tmpl = 'data/testing/{}/labels_0.npy'
eval_label_fn = [tmpl.format(x) for x in eval_pair_ids]
tmpl = 'data/testing/{}/pred_0.npy'
eval_pred_fn = [tmpl.format(x) for x in eval_pair_ids]



all_roc_nn = []
all_roc_bl = []
all_labels = []
all_scores_nn = []
all_scores_bl = []
for i in range(len(eval_pair_ids)):
    feat = np.load(eval_feat_fn[i])
    feat = np.expand_dims(feat,0)

    # Randomly rotate features.
    xyz2 = np.squeeze(feat[:,:,4:7])
    norm2 = np.squeeze(feat[:,:,10:13])
    xyz2, norm2 = rand_rotate_center_patch(xyz2, norm2)
    feat[:,:,4:7] = xyz2
    feat[:,:,10:13] = norm2
    
    ypred = corr_nn.eval(feat)
    label = np.load(eval_label_fn[i])
    ypred = np.squeeze(ypred)
    # Save the correspondences predictions.
    np.save(eval_pred_fn[i], ypred)
    all_scores_nn.append(ypred)
    all_labels.append(label)
    all_roc_nn.append(sklearn.metrics.roc_auc_score(label, ypred))
    ypred = 1.0/feat[:,:,1]
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

# Load all the testing data. 
#features = features[:,:,0]
#features = np.expand_dims(features,2)
# Compute the median roc auc using descriptor distances.
all_roc_bl = []
for i in range(len(features)):
    mymask = int(np.sum(mask[i]))
    ypred = 1.0/features[i,:mymask,0]
    ytrue = labels[i,:mymask]
    all_roc_bl.append(sklearn.metrics.roc_auc_score(ytrue, ypred))

print('Baseline Median roc auc score: {}'.format(np.median(all_roc_bl)))
print('Baseline Mean roc auc score: {}'.format(np.mean(all_roc_bl)))

y_pred_all = corr_nn.eval(features)
all_roc_nn = []
for i in range(len(features)):
    mymask = int(np.sum(mask[i]))
    ypred = y_pred_all[i,:mymask] 
    ytrue = labels[i,:mymask]
    all_roc_nn.append(sklearn.metrics.roc_auc_score(ytrue, ypred))
print('NN Median roc auc score: {}'.format(np.median(all_roc_nn)))
print('NN Mean roc auc score: {}'.format(np.mean(all_roc_nn)))
#np.save('y_pred_eval.npy', y_pred_all)
