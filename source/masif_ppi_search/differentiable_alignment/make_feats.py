import scipy.spatial
import sys
import networkx as nx
import pymesh
import sklearn.metrics
import numpy as np
import os
from IPython.core.debugger import set_trace
from rand_rotation import rand_rotate_center_patch, center_patch

def get_geodesic_neighs(mesh, source):
    # Use Dijkstra to get the distances 
    # Graph
    G=nx.Graph()
    n = len(mesh.vertices)
    G.add_nodes_from(np.arange(n))

    # Get edges
    f = np.array(mesh.faces, dtype = int)
    rowi = np.concatenate([f[:,0], f[:,0], f[:,1], f[:,1], f[:,2], f[:,2]], axis = 0)
    rowj = np.concatenate([f[:,1], f[:,2], f[:,0], f[:,2], f[:,0], f[:,1]], axis = 0)
    edges = np.stack([rowi, rowj]).T
    verts = mesh.vertices

    # Get weights
    edgew = verts[rowi] - verts[rowj]
    edgew = scipy.linalg.norm(edgew, axis=1)
    wedges = np.stack([rowi, rowj, edgew]).T

    G.add_weighted_edges_from(wedges)
    dists = nx.single_source_dijkstra_path_length(G, source)
    # Get the closest 200 neighbors
    dists = [dists[x] for x in range(len(mesh.vertices))]
    dists = np.array(dists)
    neigh = dists.argsort()[0:200]
    return neigh



#from corr_nn import CorrespondenceNN

# precomputation dir
precomp = "../data_preparation/04b-precomputation_12A/precomputation/{}/{}"
plydir = "../data_preparation/01-benchmark_surfaces/{}_{}.ply"
desc = "../descriptors/sc05/all_feat/{}/{}"

training_list = open('../lists/training.txt', 'r').readlines()
training_list = [x.rstrip() for x in training_list]
testing_list = open('../lists/testing.txt', 'r').readlines()
testing_list = [x.rstrip() for x in testing_list]


all_pairs = os.listdir("../descriptors/sc05/all_feat/")


train_labels = []
train_features = []
train_names = []
train_mask = []
train_dists = []
test_labels = []
test_features = []
test_names = []
test_mask = []
test_dists = []

all_pairs = [sys.argv[1]]

for ix, pairid in enumerate(all_pairs):

    print ("\n\n##########")
    print(pairid)
    # Open the surface files. 
    fields = pairid.split('_')
    try:
        mesh1 = pymesh.load_mesh(plydir.format(fields[0], fields[1]))
    except:
        continue
    mesh2 = pymesh.load_mesh(plydir.format(fields[0], fields[2]))

    # Load vertices
    v1 = mesh1.vertices
    v2 = mesh2.vertices

    # Load normals.
    n1x = mesh1.get_attribute('vertex_nx')
    n1y = mesh1.get_attribute('vertex_ny')
    n1z = mesh1.get_attribute('vertex_nz')
    n1 = np.stack([n1x,n1y,n1z]).T

    n2x = mesh2.get_attribute('vertex_nx')
    n2y = mesh2.get_attribute('vertex_ny')
    n2z = mesh2.get_attribute('vertex_nz')
    n2 = np.stack([n2x,n2y,n2z]).T
    
    d1 = np.load(desc.format(pairid, 'p1_desc_flipped.npy'))
    d2 = np.load(desc.format(pairid, 'p2_desc_straight.npy'))

    # Load the shape comp labels and find the center of the interface of p1.
    sc_labels = np.load(precomp.format(pairid, 'p1_sc_labels.npy'))
    center_point = np.argmax(np.median(np.nan_to_num(sc_labels[0]),axis=1))

    # Find neighbors between the two proteins.
    kdt = scipy.spatial.cKDTree(v2)
    euc_d, euc_r = kdt.query(v1)
    assert(len(euc_d) == len(v1))
    cp_2 = euc_r[center_point]

    neigh1 = get_geodesic_neighs(mesh1, center_point)
    neigh2 = get_geodesic_neighs(mesh2, cp_2)
    assert(len(neigh1) == 200)
    assert(len(neigh2) == 200)

    # Get K patch-pairs between these proteins. 
    K = 1
    # Plot descriptor distance for matches according to descriptor nearest neighbors between the two patches.
    desc_kdt_patch = scipy.spatial.cKDTree(d1[neigh1])

    for i in range(K):
        # Find the correspondences from patch2 to patch1 based on descriptor distance. 
        desc_d_patch, desc_r_patch = desc_kdt_patch.query(d2[neigh2])
#        desc_d_patch = desc_d_patch[:,i]
#        desc_r_patch = desc_r_patch[:,i]
        # From those correspondences found based on descriptor distances , find which are really true. 
        pos_dists_nn = []
        neg_dists_nn = []
        feat1 = [] # desc1
        feat2 = [] # xyz1
        feat3 = [] # xyz2
        feat4 = [] # norm1
        feat5 = [] # norm2
        labels = []
        euc_dists = []
        y_pred_bl = []

        # Move both patches to the center of the coordinate system.
        patch1 = v1[neigh1]
        patch2 = v2[neigh2]
        patch1_n = n1[neigh1]
        patch2_n = n2[neigh2]
        patch1, patch2 =  center_patch(patch1, patch2)

        for piii in range(len(neigh2)):
            pix1 = neigh1[desc_r_patch[piii]]
            pix2 = neigh2[piii]
            dist = np.sqrt(np.sum(np.square(v1[pix1] - v2[pix2])))
            desc_dist = desc_d_patch[piii]

            # feat 1 is regular desc distance
            feat1.append([desc_dist])
            # feat 2-5 is the xyz,xyz of the two patches. 
            pvi1 = patch1[desc_r_patch[piii]]
            pvi2 = patch2[piii]
            pn1 = patch1_n[desc_r_patch[piii]]
            pn2 = patch2_n[piii]
            
            # feat 2 is xyz1
            feat2.append(pvi1)
            # feat 3 is xyz2
            feat3.append(pvi2)
            # feat 4 is n1
            feat4.append(pn1)
            # feat 5 is n2
            feat5.append(pn2)
            euc_dists.append(dist)
            if dist < 3.0: 
                pos_dists_nn.append(desc_dist)
                labels.append(1.0)
            else:
                neg_dists_nn.append(desc_dist)
                labels.append(0.0)

        
        pos_dists_nn = np.array(pos_dists_nn)
        neg_dists_nn = np.array(neg_dists_nn)
        print('Number of positives: {}; number of negatives: {}'.format(len(pos_dists_nn), len(neg_dists_nn)))
        if len(pos_dists_nn) < 3:
            continue
        feat1 = np.array(feat1)
        # Print the ROC AUC of this.
        y_true = np.concatenate([np.ones_like(pos_dists_nn), np.zeros_like(neg_dists_nn)], axis=0)

        y_pred = np.concatenate([1.0/pos_dists_nn, 1.0/neg_dists_nn], axis=0)
        roc_nn = sklearn.metrics.roc_auc_score(y_true, y_pred)
        print('ROC AUC score with nearest neighbors : {:.3f}'.format(roc_nn))

        # Store features and coordinates.
        features = np.concatenate([feat1, feat2, feat3, feat4, feat5], axis=1)
        # Add padding. 
        assert(len(features) == 200)

        if pairid in training_list: 
            train_features.append(features)
            train_labels.append(labels)
            train_names.append(pairid)
            train_dists.append(euc_dists)
            outdir = 'data/training/{}'.format(pairid)
        elif pairid in testing_list:
            test_features.append(features)
            test_labels.append(labels)
            test_names.append(pairid)
            test_dists.append(euc_dists)
            outdir = 'data/testing/{}'.format(pairid)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        np.save(os.path.join(outdir,'features_{}.npy'.format(i)), features)
        np.save(os.path.join(outdir,'labels_{}.npy'.format(i)), labels)
        np.save(os.path.join(outdir,'euc_dists_{}.npy'.format(i)), euc_dists)
        # Save both patches. 
        np.save(os.path.join(outdir,'patch1_{}.npy'.format(i)), patch1)
        np.save(os.path.join(outdir,'patch1_d{}.npy'.format(i)), d1[neigh1])
        np.save(os.path.join(outdir,'patch2_d{}.npy'.format(i)), d2[neigh2])
        np.save(os.path.join(outdir,'patch2_{}.npy'.format(i)), patch2)
        np.save(os.path.join(outdir,'patch1_n{}.npy'.format(i)), patch1_n)
        np.save(os.path.join(outdir,'patch2_n{}.npy'.format(i)), patch2_n)


