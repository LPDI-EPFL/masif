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
import open3d as o3d

align_nn = AlignNN()
align_nn.restore_model()
# Load all the training data. 
#features = features[:,:,0]
#features = np.expand_dims(features, 2)

do_icp = True

test_pair_ids = os.listdir('data/testing/')
tmpl = 'data/testing/{}/features_0.npy'
test_feat_fn = [tmpl.format(x) for x in test_pair_ids]

tmpl = 'data/testing/{}/patch1_0.npy'
test_patch1_fn = [tmpl.format(x) for x in test_pair_ids]

tmpl = 'data/testing/{}/patch1_n0.npy'
test_patch1_n_fn = [tmpl.format(x) for x in test_pair_ids]

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
    gt_norm2 = np.copy(norm2)
    xyz2, norm2 = batch_rand_rotate_center_patch(xyz2, norm2)
    feat[:,:,4:7] = xyz2
    feat[:,:,10:13] = norm2

    feat = np.concatenate([pred, feat[:,:,1:13]], axis=2)
    new_coords_norm2 = align_nn.eval(feat)
    new_coords2 = new_coords_norm2[:,:,0:3]
    new_norm2 = new_coords_norm2[:,:,3:6]


    rmsd1 = np.sqrt(np.mean(np.sum(np.square(gt_xyz2 - new_coords2), axis=2)))

    if do_icp:
        # Patch1 is the target
        patch1_pcd = o3d.geometry.PointCloud()
        p1 = np.squeeze(np.load(test_patch1_fn[i]))
        p1n = np.squeeze(np.load(test_patch1_n_fn[i]))
        p1 = p1+0.25*p1n
        patch1_pcd.points = o3d.Vector3dVector(p1)
        patch1_pcd.normals= o3d.Vector3dVector(p1n)

        # Take the new coordinates and new normals 
        patch2_pcd = o3d.geometry.PointCloud()
        p2 = np.squeeze(new_coords2)
        p2n = np.squeeze(new_norm2)
        p2 = p2+0.25*p2n
        patch2_pcd.points = o3d.Vector3dVector(p2)
        patch2_pcd.normals = o3d.Vector3dVector(p2n)

        # Do ICP on the new coordinates for a final refinement.
        result = o3d.registration_icp(patch2_pcd, patch1_pcd,
                1.5, estimation_method=o3d.TransformationEstimationPointToPlane(),
                )

        patch2_pcd.transform(result.transformation)
        patch2_pred = np.asarray(patch2_pcd.points)
        p2p_dist = np.sqrt(np.sum(np.square(np.squeeze(gt_xyz2) - patch2_pred), axis=1))
        rmsd2 = np.sqrt(np.mean(np.sum(np.square(np.squeeze(gt_xyz2) - patch2_pred), axis=1)))

        print(result)
        print('Point to point RMSD: before {:.3f} after: {:.3f}, inliers = {}'.format(rmsd1, rmsd2, np.sum(p2p_dist<1.0)))
        all_rmsd.append(rmsd2)
        
    else:

        print('Point to point RMSD: {:.3f}'.format(rmsd1))
        all_rmsd.append(rmsd1)

sns.distplot(all_rmsd)
all_rmsd = np.array(all_rmsd)
plt.savefig('all_rmsd.png')
print("Number below 5A: {} out of {} fraction: {}".format(np.sum(all_rmsd < 5.0), len(all_rmsd), np.sum(all_rmsd < 5.0)/float(len(all_rmsd))))
print("Median RMSD: {}".format(np.median(all_rmsd)))
print("Mean RMSD: {}".format(np.mean(all_rmsd)))

np.save('nn_rmsd_test_set.npy', all_rmsd)
