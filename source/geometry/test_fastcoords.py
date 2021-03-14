import pymesh 
from IPython.core.debugger import set_trace
import numpy as np
import time
from scipy.sparse import csr_matrix

#import pyximport; pyximport.install()
from compute_fast_polar_coordinates import compute_fast_polar_coordinates
from compute_polar_coordinates import compute_polar_coordinates

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

    #rho = np.array([rho[0,ix] for ix in range(rho.shape[1]) if ix in neigh_i])
    mesh.add_attribute('rho')
    mesh.set_attribute('rho', rho)

    #theta= np.array([theta[ix] for ix in range((theta.shape[0])) if ix in neigh_i])
    mesh.add_attribute('theta')
    mesh.set_attribute('theta', theta)

    charge = 2*np.copy(theta)/(2*np.pi) -1
    mesh.add_attribute('charge')
    mesh.set_attribute('charge', charge)

    pymesh.save_mesh('v{}.ply'.format(i), mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)



#import helloworld
start = time.perf_counter()
mesh = pymesh.load_mesh('4ZQK_A.ply')

# Vertices, faces and normals
vertices = mesh.vertices
faces = mesh.faces
norm1 = mesh.get_attribute('vertex_nx')
norm2 = mesh.get_attribute('vertex_ny')
norm3 = mesh.get_attribute('vertex_nz')
normals = np.vstack([norm1, norm2, norm3]).T

# Build a csr matrix of distances. 
f = np.array(faces, dtype = int)
rowi = np.concatenate([f[:,0], f[:,0], f[:,1], f[:,1], f[:,2], f[:,2]], axis = 0)
rowj = np.concatenate([f[:,1], f[:,2], f[:,0], f[:,2], f[:,0], f[:,1]], axis = 0)


# Distances divided by two because all edges are counted twice? 
data = np.sqrt(np.sum(np.square(vertices[rowi] - vertices[rowj]), axis=1))/2

distmat = csr_matrix((data, (rowi, rowj)))

patch_indices = np.zeros((len(vertices), 200), dtype=np.int32)
patch_rho = np.zeros((len(vertices), 200))
patch_theta = np.zeros((len(vertices), 200))

end = time.perf_counter()
print('Read data/make matrix Took {:.2f}s'.format(end-start))

start = time.perf_counter()
rho_old, theta_old, neigh_indices_old, mask_old = compute_polar_coordinates(mesh, do_fast=True, radius=12, max_vertices=200)
end = time.perf_counter()
print('Old method took {:.2f}s'.format(end-start))
np.save('rho_old.npy', rho_old)
np.save('theta_old.npy', theta_old)
np.save('neigh_indices_old.npy', neigh_indices_old)

v = vertices.astype(np.float64, order='C')
n = normals.astype(np.float64, order='C')
f = faces.astype(np.int32, order='C')

start = time.perf_counter()
compute_fast_polar_coordinates(distmat, v, n, patch_indices, patch_rho, patch_theta)
end= time.perf_counter()
np.save('rho_new.npy', patch_rho)
np.save('theta_new.npy', patch_theta)
np.save('neigh_indices_new.npy', patch_indices)
print('Fast coords took {:.2f}s'.format(end-start))
vix = 0
subv, subn, subf = extract_patch(mesh, patch_indices[vix], vix)
output_patch_coords(subv, subf, subn, vix, patch_indices[vix], patch_theta[vix], patch_rho[vix]) 
print('{},{},{}'.format(v[vix][0], v[vix][1], v[vix][2]))
set_trace()

#np.random.seed(0)
#for i in range(10):
#    vix = np.random.choice(len(vertices))
#    subv, subn, subf = extract_patch(mesh, patch_indices[vix], vix)
#    output_patch_coords(subv, subf, subn, vix, patch_indices[vix], patch_theta[vix], patch_rho[vix]) 
#    print('{},{},{}'.format(v[vix][0], v[vix][1], v[vix][2]))

