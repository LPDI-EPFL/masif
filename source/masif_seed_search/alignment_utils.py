import open3d as o3d
import copy 
from IPython.core.debugger import set_trace
import numpy as np
from pathlib import Path
import scipy.sparse as spio
from input_output.simple_mesh import Simple_mesh
from Bio.PDB import PDBParser, PDBIO
import tensorflow as tf
from tensorflow import keras
import os


def load_protein_pcd(full_pdb_id, chain_number, paths, flipped_features=False, read_mesh=True):
    """
    Returns the protein point cloud, list of descriptors, interface predictions
    and patch coordinates given full pdb name (str), chain number (int) and if descriptors
    should be flipped or not (bool).
    """

    full_pdb_id_split = full_pdb_id.split('_')
    pdb_id = full_pdb_id_split[0]
    chain_ids = full_pdb_id_split[1:]
    pdb_chain = chain_ids[chain_number - 1]

    pdb_pcd = o3d.read_point_cloud(
        str(Path(paths['surf_dir']) /
            '{}_{}.ply'.format(
            pdb_id,
            pdb_chain)))

    if read_mesh == True:
        pdb_mesh = o3d.read_triangle_mesh(
            str(Path(paths['surf_dir']) /
            '{}_{}.ply'.format(
            pdb_id,
            pdb_chain))) 

    pdb_iface = np.squeeze(np.load(
        Path(paths['iface_dir']) /
        'pred_{}_{}.npy'.format(
            pdb_id,
            pdb_chain)))

    if flipped_features:
        pdb_desc_sc05 = np.load(
            Path(paths['desc_dir']) /
            full_pdb_id /
            'p{}_desc_flipped.npy'.format(chain_number))
        pdb_desc_sc_nofilt_chem = np.load(
            Path(paths['desc_dir_sc_nofilt_chem']) /
            full_pdb_id /
            'p{}_desc_flipped.npy'.format(chain_number))
        pdb_desc_sc_nofilt_all_feat = np.load(
            Path(paths['desc_dir_sc_nofilt_all_feat']) /
            full_pdb_id /
            'p{}_desc_flipped.npy'.format(chain_number))
    else:
        pdb_desc_sc05 = np.load(
            Path(paths['desc_dir']) /
            full_pdb_id /
            'p{}_desc_straight.npy'.format(chain_number))
        pdb_desc_sc_nofilt_chem = np.load(
            Path(paths['desc_dir_sc_nofilt_chem']) /
            full_pdb_id /
            'p{}_desc_straight.npy'.format(chain_number))
        pdb_desc_sc_nofilt_all_feat = np.load(
            Path(paths['desc_dir_sc_nofilt_all_feat']) /
            full_pdb_id /
            'p{}_desc_straight.npy'.format(chain_number))


    pdb_desc = np.stack([
        pdb_desc_sc05,
        pdb_desc_sc_nofilt_chem,
        pdb_desc_sc_nofilt_all_feat])

    if read_mesh:
        return pdb_pcd, pdb_desc, pdb_iface, pdb_mesh
    else:
        return pdb_pcd, pdb_desc, pdb_iface 
        


def load_structure(full_pdb_id, chain_number):
    full_pdb_id = full_pdb_id.split('_')
    pdb_id = full_pdb_id[0] + '_' + full_pdb_id[chain_number] + '.pdb'
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, paths['pdb_dir'] / pdb_id)
    return structure


def get_patch_geo(
        pcd,
        patch_coords,
        center,
        descriptors,
        iface, 
        outward_shift=0.25,
        flip_normals=False):
    """
    Returns a patch from a point cloud pcd with center point center (int),
    based on geodesic distances from patch coords and corresponding Feature descriptors.
    """
    patch_idxs = patch_coords[center]
    patch_pts = np.asarray(pcd.points)[patch_idxs, :]
    patch_nrmls = np.asarray(pcd.normals)[patch_idxs, :]
    patch_pts = patch_pts + outward_shift * patch_nrmls
    if flip_normals:
        patch_nrmls = -patch_nrmls

    patch = o3d.PointCloud()
    patch.points = o3d.Vector3dVector(patch_pts)
    patch.normals = o3d.Vector3dVector(patch_nrmls)
    patch_descs = [o3d.Feature(), o3d.Feature(), o3d.Feature()]
    patch_descs[0].data = descriptors[0,patch_idxs, :].T
    patch_descs[1].data = descriptors[1,patch_idxs, :].T
    patch_descs[2].data = descriptors[2,patch_idxs, :].T
    patch_iface = iface[patch_idxs]
    return patch, patch_descs, patch_iface


def get_patch_mesh_geo(triangle_mesh, patch_coords, center, descriptors, outward_shift=0.25, flip=False):
    """
    Returns a patch, with the mesh, from a point cloud pcd with center point center (int),
    based on geodesic distances from patch coords and corresponding Feature descriptors.
    """
    idx = patch_coords[center]
    n = len(triangle_mesh.vertices)
    pts = np.asarray(triangle_mesh.vertices)[idx, :]
    nrmls = np.asarray(triangle_mesh.vertex_normals)[idx, :]
    pts = pts + outward_shift*nrmls
    if flip:
        nrmls = -np.asarray(triangle_mesh.vertex_normals)[idx, :]

    patch = o3d.TriangleMesh()
    patch.vertices = o3d.Vector3dVector(pts)
    patch.vertex_normals = o3d.Vector3dVector(nrmls)

    # Extract triangulation.
    m = np.zeros(n, dtype=int)

    for i in range(len(idx)):
        m[idx[i]] = i
    f = triangle_mesh.triangles
    nf = len(f)

    idx = set(idx)
    subf = [[m[f[i][0]], m[f[i][1]], m[f[i][2]]] for i in range(nf)
            if f[i][0] in idx and f[i][1] in idx and f[i][2] in idx]

    patch.triangles = o3d.Vector3iVector(np.asarray(subf))
    return patch


def get_geodists_and_patch_coords(top_dir, pdb, pid, cv=None):
    """ 
    Load precomputed patch coordinates.
    """
    if cv is None:
        patch_coords = np.load(os.path.join(
            top_dir, pdb, pid+'_geodists_indices.npy'))
        geodists = np.load(os.path.join(
            top_dir, pdb, pid+'_geodists.npy'))
        cv = np.arange(0, patch_coords.shape[0])
    else:
        patch_coords = np.load(os.path.join(
            top_dir, pdb, pid+'_geodists_indices.npy'))[cv]
        geodists = np.load(os.path.join(
            top_dir, pdb, pid+'_geodists.npy'))[cv]
    patch_coords = {key: patch_coords[ii] for ii, key in enumerate(cv)}
    geodists = {key: geodists[ii] for ii, key in enumerate(cv)}
    return patch_coords, geodists

# Get the vertex indices of target sites.
def get_target_vix(pc, iface, num_sites=1):
    target_vertices = []
    for site_ix in range(num_sites):
        iface_patch_vals = []
        # Go through each patch.
        best_ix = -1
        best_val = float("-inf")
        best_neigh = []
        for ii in pc.keys():
            neigh = pc[ii]
            val = np.mean(iface[neigh])
            if val > best_val:
                best_ix = ii
                best_val = val
                best_neigh = neigh

        # Now that a site has been identified, clear the iface values so that the same site is not picked again.
        iface[best_neigh] = 0
        target_vertices.append(best_ix)

    return target_vertices


def match_descriptors(in_desc_dir, in_iface_dir, pids, target_desc, params):

    all_matched_names = []
    all_matched_vix = []
    all_matched_desc_dist = []
    count_proteins = 0
    for ppi_pair_id in os.listdir(in_desc_dir):
        if 'seed_pdb_list' in params and ppi_pair_id not in params['seed_pdb_list']:
            continue
#        if 'CAF1' not in ppi_pair_id:
#            continue
        if '.npy' in ppi_pair_id or '.txt' in ppi_pair_id:
            continue
        mydescdir = os.path.join(in_desc_dir, ppi_pair_id)
        for pid in pids:
            try:
                fields = ppi_pair_id.split('_')
                if pid == 'p1':
                    pdb_chain_id = fields[0]+'_'+fields[1]
                elif pid == 'p2':
                    pdb_chain_id = fields[0]+'_'+fields[2]
                iface = np.load(in_iface_dir+'/pred_'+pdb_chain_id+'.npy')[0]
                descs = np.load(mydescdir+'/'+pid+'_desc_straight.npy')
            except:
                continue
            print(pdb_chain_id)
            name = (ppi_pair_id, pid)
            count_proteins += 1

            diff = np.sqrt(np.sum(np.square(descs - target_desc), axis=1))

            true_iface = np.where(iface > params['iface_cutoff'])[0]
            near_points = np.where(diff < params['desc_dist_cutoff'])[0]

            selected = np.intersect1d(true_iface, near_points)
            if len(selected > 0):
                all_matched_names.append([name]*len(selected))
                all_matched_vix.append(selected)
                all_matched_desc_dist.append(diff[selected])
                print('Matched {}'.format(ppi_pair_id))
                print('Scores: {} {}'.format(iface[selected], diff[selected]))

    print('Iterated over {} proteins.'.format(count_proteins))
    return all_matched_names, all_matched_vix, all_matched_desc_dist, count_proteins

def count_clashes(transformation, source_structure, target_ca_pcd_tree,
                   target_pcd_tree, clashing_radius=2.0):
    structure_atoms = [atom for atom in source_structure.get_atoms()]
    structure_coords = [x.get_coord() for x in structure_atoms]
    structure_ca_coords = [x.get_coord() for x in structure_atoms if x.get_name()=='CA']

    structure_coord_pcd = o3d.PointCloud()
    structure_coord_pcd.points = o3d.Vector3dVector(structure_coords)
    structure_coord_pcd.transform(transformation)
    structure_ca_coord_pcd = o3d.PointCloud()
    structure_ca_coord_pcd.points = o3d.Vector3dVector(structure_ca_coords)
    structure_ca_coord_pcd.transform(transformation)

    clashing = 0
    for point in structure_ca_coord_pcd.points:
        [k, idx, _] = target_pcd_tree.search_radius_vector_3d(
            point, clashing_radius)
        if k > 0:
            clashing += 1

    return clashing


def multidock(source_pcd, source_patch_coords, source_descs, 
            cand_pts, target_pcd, target_descs,
              target_patch_mesh, target_patch_mesh_centroids, 
              source_geodists, target_patch_geodists, target_ckdtree,
              source_iface, target_patch_iface,           
              nn_score, params,
              source_struct, target_ca_pcd_tree,target_pcd_tree):
    ransac_radius=params['ransac_radius'] 
    ransac_iter=params['ransac_iter']
    all_results = []
    all_source_patch = []
    all_source_scores = []
    for pt in cand_pts:
        source_patch, source_patch_descs, source_patch_iface = get_patch_geo(
            source_pcd, source_patch_coords, pt, source_descs, source_iface, outward_shift=params['surface_outward_shift'])
        source_patch_geodists = source_geodists[pt]
        if params['ransac_type'] == 'shape_comp':
            result = o3d.registration_ransac_based_on_shape_complementarity(
                source_patch, target_pcd, target_patch_mesh, target_patch_mesh_centroids, source_patch_descs[0], target_descs[0],
                ransac_radius,
                o3d.TransformationEstimationPointToPoint(False), 3,
                [o3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.CorrespondenceCheckerBasedOnDistance(2.0),
                 o3d.CorrespondenceCheckerBasedOnNormal(np.pi/2)],
                o3d.RANSACConvergenceCriteria(ransac_iter, 500), 0, 3)
            result_icp = o3d.registration_icp(source_patch, target_pcd,
                        1.0, result.transformation, o3d.TransformationEstimationPointToPlane())
            
        else:
            result = o3d.registration_ransac_based_on_feature_matching(
                source_patch, target_pcd, source_patch_descs[0], target_descs[0],
                ransac_radius,
                o3d.TransformationEstimationPointToPoint(False), 3,
                [o3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.CorrespondenceCheckerBasedOnDistance(1.0),
                 o3d.CorrespondenceCheckerBasedOnNormal(np.pi/2)],
                o3d.RANSACConvergenceCriteria(ransac_iter, 500)
            )
        #print('{} {}'.format(len(np.asarray(result.correspondence_set)), result.fitness))
        # result = registration_icp(source_patch, target_pcd, 1.5,
        # result.transformation,
        # TransformationEstimationPointToPoint())

        source_patch.transform(result_icp.transformation)
        all_results.append(result_icp)
        all_source_patch.append(source_patch)
        n_clashes = count_clashes(result_icp.transformation, source_struct, target_ca_pcd_tree,target_pcd_tree)

        source_scores = compute_desc_dist_score(target_pcd, source_patch, np.asarray(result.correspondence_set),
                target_patch_geodists, source_patch_geodists,
                target_descs, source_patch_descs, \
                target_patch_iface, source_patch_iface, 
                target_ckdtree, nn_score,n_clashes)
        all_source_scores.append(source_scores)

    return all_results, all_source_patch, all_source_scores


def align_and_save(out_filename_base, patch, transformation, source_structure, target_ca_pcd_tree,
                   target_pcd_tree, point_importance=None, clashing_cutoff=10.0,
                   clashing_radius=2.0):
    source_structure = copy.deepcopy(source_structure)
    structure_atoms = [atom for atom in source_structure.get_atoms()]
    structure_coords = [x.get_coord() for x in structure_atoms]
    structure_ca_coords = [x.get_coord() for x in structure_atoms if x.get_name()=='CA']

    structure_coord_pcd = o3d.PointCloud()
    structure_coord_pcd.points = o3d.Vector3dVector(structure_coords)
    structure_coord_pcd.transform(transformation)
    structure_ca_coord_pcd = o3d.PointCloud()
    structure_ca_coord_pcd.points = o3d.Vector3dVector(structure_ca_coords)
    structure_ca_coord_pcd.transform(transformation)

    clashing = 0
    for point in structure_ca_coord_pcd.points:
        [k, idx, _] = target_pcd_tree.search_radius_vector_3d(
            point, clashing_radius)
        if k > 0:
            clashing += 1

    #
    if clashing < clashing_cutoff:
        for ix, v in enumerate(structure_coord_pcd.points):
            structure_atoms[ix].set_coord(v)

        io = PDBIO()
        io.set_structure(source_structure)
        io.save(out_filename_base+'.pdb')
        # Save patch
        mesh = Simple_mesh(vertices=patch.points)
        mesh.set_attribute('vertex_charge', point_importance)
        mesh.save_mesh(out_filename_base+'_patch.ply')
        
#        out_patch = open(out_filename_base+'.vert', 'w+')
#        for point in patch.points:
#            out_patch.write('{}, {}, {}\n'.format(
#                point[0], point[1], point[2]))
#        out_patch.close()
        return clashing
    else:
        return clashing


# Compute different types of scores: 
# -- Inverted sum of the minimum descriptor distances squared cutoff.
def compute_desc_dist_score(target_pcd, source_pcd, corr, 
        target_patch_geo_dists, source_patch_geo_dists, 
        target_desc, source_desc, 
        target_patch_iface_scores, source_patch_iface_scores,
        target_ckdtree, nn_score,n_clashes):

    # Compute scores based on correspondences.
    if len(corr) < 1:
        dists_cutoff_0= np.array([1000.0])
        dists_cutoff_1= np.array([1000.0])
        dists_cutoff_2= np.array([1000.0])
        inliers = 0
        target_p = []
        source_p = []
    else:
        target_p = corr[:,1]
        source_p = corr[:,0]    
        try:
            dists_cutoff_0 = target_desc[0].data[:,target_p] - source_desc[0].data[:,source_p]
            dists_cutoff_1 = target_desc[1].data[:,target_p] - source_desc[1].data[:,source_p]
            dists_cutoff_2 = target_desc[2].data[:,target_p] - source_desc[2].data[:,source_p]
        except:
            set_trace()
        dists_cutoff_0 = np.sqrt(np.sum(np.square(dists_cutoff_0.T), axis=1))
        dists_cutoff_1 = np.sqrt(np.sum(np.square(dists_cutoff_1.T), axis=1))
        dists_cutoff_2 = np.sqrt(np.sum(np.square(dists_cutoff_2.T), axis=1))
        inliers = len(corr)
    
    scores_corr_0 = np.sum(np.square(1.0/dists_cutoff_0))
    scores_corr_1 = np.sum(np.square(1.0/dists_cutoff_1))
    scores_corr_2 = np.sum(np.square(1.0/dists_cutoff_2))

    # Compute nn scores 
    # Compute all points correspondences and distances for nn
    d, r = target_ckdtree.query(source_pcd.points)

    # r: for every point in source, what is its correspondence in target

    feat0 = d # distances between points 
    # feat1-feat3: descriptors
    feat1 = target_desc[0].data[:,r] - source_desc[0].data[:,:]
    feat1 = np.sqrt(np.sum(np.square(feat1.T), axis=1))
    feat2 = target_desc[1].data[:,r] - source_desc[1].data[:,:]
    feat2 = np.sqrt(np.sum(np.square(feat2.T), axis=1))
    feat3 = target_desc[2].data[:,r] - source_desc[2].data[:,:]
    feat3 = np.sqrt(np.sum(np.square(feat3.T), axis=1))
    # feat4 = source_geo_dists
    feat4 = source_patch_geo_dists
    # feat5 = target_geo_dists
    feat5 = target_patch_geo_dists[r]
    # feat6-7 = source iface, target iface
    feat6 = source_patch_iface_scores
    feat7 = target_patch_iface_scores[r]
    # feat8: normal dot product
    n1 = np.asarray(source_pcd.normals)
    n2 = np.asarray(target_pcd.normals)[r]
    feat8 = np.multiply(n1, n2).sum(1)
    feat9 = np.zeros((len(d)))
    feat9[source_p] = 1.0
    feat10 = np.ones((len(d)))*n_clashes
#    feat8 = np.diag(np.dot(np.asarray(source_pcd.normals), np.asarray(target_pcd.normals)[r].T))
    nn_score_pred, point_importance = nn_score.eval_model(feat0, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10)

    return (np.array([scores_corr_0, inliers, scores_corr_1, scores_corr_2, nn_score_pred[0][0]]).T, point_importance)

