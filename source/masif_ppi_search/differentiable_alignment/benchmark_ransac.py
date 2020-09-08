import numpy as np 
from IPython.core.debugger import set_trace
import os
import open3d as o3d

# Go through each test subdirectory
test_pair_ids = os.listdir('data/testing/')
tmpl = 'data/testing/{}/patch1_0.npy'
test_patch1_fn = [tmpl.format(x) for x in test_pair_ids]
tmpl = 'data/testing/{}/patch1_n0.npy'
test_patch1_n_fn = [tmpl.format(x) for x in test_pair_ids]
tmpl = 'data/testing/{}/patch1_d0.npy'
test_patch1_d_fn = [tmpl.format(x) for x in test_pair_ids]
tmpl = 'data/testing/{}/patch2_0.npy'
test_patch2_fn = [tmpl.format(x) for x in test_pair_ids]
tmpl = 'data/testing/{}/patch2_n0.npy'
test_patch2_n_fn = [tmpl.format(x) for x in test_pair_ids]
tmpl = 'data/testing/{}/patch2_d0.npy'
test_patch2_d_fn = [tmpl.format(x) for x in test_pair_ids]

all_rmsds = []

for i in range(len(test_pair_ids)):
    # Load data
    patch1 = np.load(test_patch1_fn[i].format(test_pair_ids[i]))
    patch1_n = np.load(test_patch1_n_fn[i].format(test_pair_ids[i]))
    patch1_d = np.load(test_patch1_d_fn[i].format(test_pair_ids[i]))
    patch2 = np.load(test_patch2_fn[i].format(test_pair_ids[i]))
    patch2_n = np.load(test_patch2_n_fn[i].format(test_pair_ids[i]))
    patch2_d = np.load(test_patch2_d_fn[i].format(test_pair_ids[i]))

    patch2_gt = np.copy(patch2)

    # Push patches out by 0.25 in the direction of the normal
    patch1 = patch1 + 0.25*patch1_n
    patch2 = patch2 + 0.25*patch2_n

    # Make point clouds. 
    patch1_pcd = o3d.geometry.PointCloud()
    patch1_pcd.points = o3d.Vector3dVector(patch1)
    # Flip target normals!
    patch1_pcd.normals = o3d.Vector3dVector(-patch1_n)
    patch1_desc = o3d.Feature()
    patch1_desc.data = patch1_d.T

    patch2_pcd = o3d.geometry.PointCloud()
    patch2_pcd.points = o3d.Vector3dVector(patch2)
    patch2_pcd.normals = o3d.Vector3dVector(patch2_n)
    patch2_desc = o3d.Feature()
    patch2_desc.data = patch2_d.T

    ransac_iter = 2000
    ransac_radius = 1.5
    # Run RANSAC.
    result = o3d.registration_ransac_based_on_feature_matching(
            patch2_pcd, patch1_pcd, patch2_desc, patch1_desc,
            ransac_radius,
            o3d.TransformationEstimationPointToPoint(False), 3,
            [o3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.CorrespondenceCheckerBasedOnDistance(1.5),
            o3d.CorrespondenceCheckerBasedOnNormal(np.pi/2)],
            o3d.RANSACConvergenceCriteria(ransac_iter, 500))

    result = o3d.registration_icp(patch2_pcd, patch1_pcd,
            1.0, result.transformation, o3d.TransformationEstimationPointToPlane(),
            )

    patch2_pcd.transform(result.transformation)
    patch2_pred = np.asarray(patch2_pcd.points)
    
    # Estimate RMSD
    rmsd1 = np.sqrt(np.mean(np.sum(np.square(patch2_gt - patch2_pred), axis=1)))
    print(rmsd1)
    all_rmsds.append(rmsd1)

all_rmsds = np.array(all_rmsds)
print("Number below 5A: {} out of {} fraction: {}".format(np.sum(all_rmsds < 5.0), len(all_rmsds), np.sum(all_rmsds < 5.0)/float(len(all_rmsds))))
print("Median RMSD: {}".format(np.median(all_rmsds)))
print("Mean RMSD: {}".format(np.mean(all_rmsds)))

np.save('ransac_rmsds.npy', all_rmsds)

