import tensorflow as tf
import numpy as np
import os
from IPython.core.debugger import set_trace
from tensorflow import keras 
from align_nn import AlignNN
from rand_rotation import batch_rand_rotate_center_patch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

align_nn = AlignNN()
align_nn.restore_model()
# Load all the training data. 
#features = features[:,:,0]
#features = np.expand_dims(features, 2)


test_pair_ids = os.listdir('data/testing/')
tmpl = 'data/testing/{}/features_0.npy'
test_feat_fn = [tmpl.format(x) for x in test_pair_ids]
tmpl = 'data/testing/{}/pred_0.npy'
test_pred_fn = [tmpl.format(x) for x in test_pair_ids]
tmpl = 'data/testing/{}/labels_0.npy'
test_label_fn = [tmpl.format(x) for x in test_pair_ids]
all_rmsd = []

for i in range(len(test_pair_ids)):
    if 'Store' in test_pair_ids[i] or 'PGC' in test_pair_ids[i]:
        continue
    feat = np.load(test_feat_fn[i])
    feat = np.expand_dims(feat,0)
    pred = np.load(test_pred_fn[i])
    pred = np.expand_dims(pred,0)
    pred = np.expand_dims(pred,2)
    label = np.load(test_label_fn[i])
    
    # Randomly rotate xyz2 only.  
    xyz2 = feat[:,:,4:7]
    gt_xyz2 = np.copy(xyz2)
    norm2 = feat[:,:,10:13]
    xyz2, norm2 = batch_rand_rotate_center_patch(xyz2, norm2)
    feat[:,:,4:7] = xyz2
    feat[:,:,10:13] = norm2

    feat = np.concatenate([pred, feat[:,:,1:13]], axis=2)
    new_coords = align_nn.eval(feat)

#    gt_xyz2 = np.reshape(gt_xyz2, [1,600])
#    new_coords = np.reshape(new_coords, [1,200,3])
    rmsd1 = np.sqrt(np.mean(np.sum(np.square(gt_xyz2 - new_coords), axis=2)))

    print('Point to point RMSD: {:.3f}'.format(rmsd1))

    all_rmsd.append(rmsd1)

sns.distplot(all_rmsd)
all_rmsd = np.array(all_rmsd)
plt.savefig('all_rmsd.png')
print("Number below 5A: {} out of {} fraction: {}".format(np.sum(all_rmsd < 5.0), len(all_rmsd), np.sum(all_rmsd < 5.0)/float(len(all_rmsd))))
print("Median RMSD: {}".format(np.median(all_rmsd)))
print("Mean RMSD: {}".format(np.mean(all_rmsd)))

np.save('nn_rmsd_test_set.npy', all_rmsd)
