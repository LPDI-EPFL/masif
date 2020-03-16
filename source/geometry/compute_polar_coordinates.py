"""
compute_polar_coordinates.py: Compute the polar coordinates of all patches. 
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

import sys
from sklearn.manifold import MDS
import networkx as nx
import numpy as np
import scipy.linalg
from IPython.core.debugger import set_trace
from  numpy.linalg import norm
import time
from scipy.sparse import csr_matrix, coo_matrix
import pymesh

def compute_polar_coordinates(mesh, do_fast=True, radius=12, max_vertices=200):
    """
    compute_polar_coordinates: compute the polar coordinates for every patch in the mesh. 
    Returns: 
        rho: radial coordinates for each patch. padded to zero.
        theta: angle values for each patch. padded to zero. 
        neigh_indices: indices of members of each patch. 
        mask: the mask for rho and theta
    """

    # Vertices, faces and normals
    vertices = mesh.vertices
    faces = mesh.faces
    norm1 = mesh.get_attribute('vertex_nx')
    norm2 = mesh.get_attribute('vertex_ny')
    norm3 = mesh.get_attribute('vertex_nz')
    normals = np.vstack([norm1, norm2, norm3]).T

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
    start = time.clock()
    if do_fast:
        dists = nx.all_pairs_dijkstra_path_length(G, cutoff=radius)
    else:
        dists = nx.all_pairs_dijkstra_path_length(G, cutoff=radius*2)
    d2 = {}
    for key_tuple in dists:
        d2[key_tuple[0]] = key_tuple[1]
    end = time.clock()
    print('Dijkstra took {:.2f}s'.format((end-start)))
    D = dict_to_sparse(d2)

    # Compute the faces per vertex.
    idx = {}
    for ix, face in enumerate(mesh.faces):
        for i in range(3):
            if face[i] not in idx:
                idx[face[i]] = []
            idx[face[i]].append(ix)


    i = np.arange(D.shape[0])
    # Set diagonal elements to a very small value greater than zero..
    D[i,i] = 1e-8
    # Call MDS for all points.
    mds_start_t = time.clock()

    if do_fast:
        theta = compute_theta_all_fast(D, vertices, faces, normals, idx, radius)
    else:
        theta = compute_theta_all(D, vertices, faces, normals, idx, radius)

    
    # Output a few patches for debugging purposes.
    # extract a patch
    #for i in [0,100,500,1000,1500,2000]:
    #    neigh = D[i].nonzero()
    #    ii = np.where(D[i][neigh] < radius)[1]
    #    neigh_i = neigh[1][ii]
    #    subv, subn, subf = extract_patch(mesh, neigh_i, i)
    #    # Output the patch's rho and theta coords
    #    output_patch_coords(subv, subf, subn, i, neigh_i, theta[i], D[i, :])
    

    mds_end_t = time.clock()
    print('MDS took {:.2f}s'.format((mds_end_t-mds_start_t)))
    
    n = len(d2)
    theta_out = np.zeros((n, max_vertices))
    rho_out= np.zeros((n, max_vertices))
    mask_out = np.zeros((n, max_vertices))
    # neighbors of each key. 
    neigh_indices = []
    
    # Assemble output.
    for i in range(n): 
        dists_i = d2[i]
        sorted_dists_i = sorted(dists_i.items(), key=lambda kv: kv[1])
        neigh = [int(x[0]) for x in sorted_dists_i[0:max_vertices]] 
        neigh_indices.append(neigh)
        rho_out[i,:len(neigh)]= np.squeeze(np.asarray(D[i,neigh].todense()))
        theta_out[i,:len(neigh)]= np.squeeze(theta[i][neigh])
        mask_out[i,:len(neigh)] = 1
    # have the angles between 0 and 2*pi
    theta_out[theta_out < 0] +=2 * np.pi

    return rho_out, theta_out, neigh_indices, mask_out

def compute_thetas(plane, vix, verts, faces, normal, neighbors, idx):
    """
    compute_thetas: compute the angles of each vertex with respect to some
    random direction. Ensure that theta runs clockwise with respect to the
    normals. 
    Args: 
        plane: the 2D plane of the vertices in the patch as computed by multidimensional scaling
        vix: the index of the center in the plane. 
        mesh: The full mesh of the protein.
        neighbors: the indices of the patch vertices 
        idx: a list of faces indexed per vertex.
    Returns:
        thetas: theta values for the patch.
    """
    plane_center_ix = np.where(neighbors == vix)[0][0]
    thetas = np.zeros(len(verts))
    # Center the plane so that the origin is at (0,0).
    plane = plane-plane[plane_center_ix]

    # Choose one of the neighboring triangles, one such that all neighbors are in neighbors. 
    valid = False
    for i in range(len(idx[vix])):
        tt = idx[vix][i]
        tt = faces[tt]
        # Check that all of the members of the triangle are in neighbors.
        check_valid = [x for x in tt if x in neighbors]
        if len(check_valid) == 3:
            valid = True
            break
    try:
        assert(valid)
    except:
        set_trace()
     
    # Compute the normal for tt by averagin over the vertex normals
    normal_tt = np.mean([normal[tt[0]], normal[tt[1]], normal[tt[2]]], axis=0)

    # Find the two vertices (v1ix and v2ix) in tt that are not vix
    neigh_tt = [x for x in tt if x != vix]
    v1ix = neigh_tt[0]
    v2ix = neigh_tt[1]
    # Find the index of the entry for v1ix and v2ix in neighbors
    v1ix_plane = np.where(neighbors == v1ix)[0][0]
    v2ix_plane = np.where(neighbors == v2ix)[0][0]

    # Compute  normalization to make all vectors equal to 1.
    norm_plane = np.sqrt(np.sum(np.square(plane),axis=1))
    # the normalization value at the center should be 1.
    norm_plane[plane_center_ix] = 1.0
    norm_plane = np.stack([norm_plane, norm_plane], axis=1)

    # compute vectors from the center point to each vertex in the plane.  
    vecs = np.divide(plane,norm_plane)
    vecs[plane_center_ix] = [0,0]
    vecs = np.stack([vecs[:,0], vecs[:,1], np.zeros(len(vecs))], axis=1)
    # ref_vec: the vector between the origin and point v1ix, which will be used to compute all angles.
    ref_vec = vecs[v1ix_plane]

    # Compute the actual angles
    term1 = np.sqrt(np.sum(np.square(np.cross(vecs,ref_vec)),axis=1))
    term1 = np.arctan2(term1, np.dot(vecs,ref_vec))
    normal_plane = [0.0,0.0,1.0]
    theta = np.multiply(term1, np.sign(np.dot(vecs,np.cross(normal_plane,ref_vec))))

    # Compute the sign of the angle between v2ix and v1ix in 3D to ensure that the angles are always in the same direction.
    v0 = verts[vix]
    v1 = verts[v1ix]
    v2 = verts[v2ix]
    v1 = v1 - v0
    v1 = v1/np.linalg.norm(v1)
    v2 = v2 - v0
    v2 = v2/np.linalg.norm(v2)
    angle_v1_v2 = np.arctan2(norm(np.cross(v2,v1)),np.dot(v2,v1))*np.sign(np.dot(v2,np.cross(normal_tt,v1)))

    sign_3d = np.sign(angle_v1_v2)
    sign_2d = np.sign(theta[v2ix_plane])
    if sign_3d != sign_2d:
        # Invert it to ensure that the angle is always in the same direction
        theta = -theta
    # Set theta == 0 to epsilon to not confuse it in the sparse matrix.
    theta[theta == 0] = np.finfo(float).eps
    thetas[neighbors] = theta

    return thetas

def dict_to_sparse(mydict):
    """ 
        create a sparse matrix from a dictionary
    """

    # Create the appropriate format for the COO format.
    data = []
    row = []
    col = []
    for r in mydict.keys():
        for c in mydict[r].keys():
            r = int(r)
            c = int(c)
            v = mydict[r][c]
            data.append(v)
            row.append(r)
            col.append(c)
    # Create the COO-matrix
    coo = coo_matrix((data,(row,col)))
    # Let Scipy convert COO to CSR format and return
    return csr_matrix(coo)




def extract_patch(mesh, neigh, cv):
    """ 
    Extract a patch from the mesh.
        neigh: the neighboring vertices.
    """
    n = len(mesh.vertices)
    subverts = mesh.vertices[neigh]

    nx = mesh.get_attribute('vertex_nx')
    ny = mesh.get_attribute('vertex_ny')
    nz = mesh.get_attribute('vertex_nz')
    normals = np.vstack([nx, ny, nz]).T
    subn = normals[neigh]


    # Extract triangulation. 
    
    m = np.zeros(n,dtype=int)

    # -1 if not there.
    m = m - 1 
    for i in range(len(neigh)):
        m[neigh[i]] = i
    f = mesh.faces.astype(int)
    nf = len(f)
    
    neigh = set(neigh) 
    subf = [[m[f[i][0]], m[f[i][1]], m[f[i][2]]] for i in range(nf) \
             if f[i][0] in neigh and f[i][1] in neigh and f[i][2] in neigh]
    
    subfaces = subf
    return np.array(subverts), np.array(subn), np.array(subf) 

def output_patch_coords(subv, subf, subn, i, neigh_i, theta, rho): 
    """ 
        For debugging purposes, save a patch to visualize it.
    """ 
    
    mesh = pymesh.form_mesh(subv, subf)
    n1 = subn[:,0]
    n2 = subn[:,1]
    n3 = subn[:,2]
    mesh.add_attribute('vertex_nx')
    mesh.set_attribute('vertex_nx', n1)
    mesh.add_attribute('vertex_ny')
    mesh.set_attribute('vertex_ny', n2)
    mesh.add_attribute('vertex_nz')
    mesh.set_attribute('vertex_nz', n3)

    rho = np.array([rho[0,ix] for ix in range(rho.shape[1]) if ix in neigh_i])
    mesh.add_attribute('rho')
    mesh.set_attribute('rho', rho)

    theta= np.array([theta[ix] for ix in range((theta.shape[0])) if ix in neigh_i])
    mesh.add_attribute('theta')
    mesh.set_attribute('theta', theta)

    charge = np.zeros(len(neigh_i))
    mesh.add_attribute('charge')
    mesh.set_attribute('charge', charge)

    pymesh.save_mesh('v{}.ply'.format(i), mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)

#@jit
def call_mds(mds_obj, pair_dist):
    return mds_obj.fit_transform(pair_dist)

def compute_theta_all(D, vertices, faces, normals, idx, radius):
    mymds = MDS(n_components=2, n_init=1, max_iter=50, dissimilarity='precomputed', n_jobs=10)
    all_theta = []
    for i in range(D.shape[0]):
        if i % 100 == 0:
            print(i)
        # Get the pairs of geodesic distances.
        neigh = D[i].nonzero()
        ii = np.where(D[i][neigh] < radius)[1]
        neigh_i = neigh[1][ii]
        pair_dist_i = D[neigh_i,:][:,neigh_i]
        pair_dist_i = pair_dist_i.todense()

        # Plane_i: the 2D plane for all neighbors of i
        plane_i = call_mds(mymds, pair_dist_i)
    
        # Compute the angles on the plane.
        theta = compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)
        all_theta.append(theta)
    return all_theta


def compute_theta_all_fast(D, vertices, faces, normals, idx, radius):
    """
        compute_theta_all_fast: compute the theta coordinate using an approximation.
        The approximation consists of taking only the inner radius/2 for the multidimensional
        scaling. Then, for points farther than radius/2, the shortest line to the center is used. 
        This speeds up the method by a factor of about 100.
    """
    mymds = MDS(n_components=2, n_init=1, eps=0.1, max_iter=50, dissimilarity='precomputed', n_jobs=1)
    all_theta = []
    start_loop = time.clock()
    only_mds = 0.0
    for i in range(D.shape[0]):
        # Get the pairs of geodesic distances.
        neigh = D[i].nonzero()
        # We will run MDS on only a subset of the points.
        ii = np.where(D[i][neigh] < radius/2)[1]
        neigh_i = neigh[1][ii]
        pair_dist_i = D[neigh_i,:][:,neigh_i]
        pair_dist_i = pair_dist_i.todense()

        # Plane_i: the 2D plane for all neighbors of i
        tic = time.clock()
        plane_i = call_mds(mymds, pair_dist_i)
        toc = time.clock()
        only_mds += (toc - tic)
    
        # Compute the angles on the plane.
        theta = compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)

        # We now must assign angles to all points kk that are between radius/2 and radius from the center.
        kk = np.where(D[i][neigh] >= radius/2)[1]
        neigh_k = neigh[1][kk]
        dist_kk = D[neigh_k,:][:,neigh_i]
        dist_kk = dist_kk.todense()
        dist_kk[dist_kk == 0] = float('inf')
        closest = np.argmin(dist_kk, axis=1)
        closest = np.squeeze(closest)
        closest = neigh_i[closest]
        theta[neigh_k] = theta[closest]

        
        all_theta.append(theta)
    end_loop = time.clock()
    print('Only MDS time: {:.2f}s'.format(only_mds))
    print('Full loop time: {:.2f}s'.format(end_loop-start_loop))
    return all_theta





