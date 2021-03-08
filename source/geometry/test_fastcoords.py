import pymesh 
from IPython.core.debugger import set_trace
import numpy as np
import time
from scipy.sparse import csr_matrix

#import pyximport; pyximport.install()
#from fast_patches_simple import get_patch_coords_fast_simple
#from fast_patches_cython0 import get_patch_coords_fast_cython0
from fast_patches_cython2 import dijkstra_entry
#import helloworld
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
graph = np.zeros((len(vertices), len(vertices)))

set_trace()
start = time.clock()
pablo = dijkstra_entry(distmat, graph)
end = time.clock()
pablo = dijkstra_entry(distmat, graph)
print('Took {:.2f}s'.format(end-start))
set_trace()
#get_patch_coords_fast_cython1(vertices, faces, normals)
#get_patch_coords_fast_cython1(vertices, faces, normals)
#get_patch_coords_fast_cython0(vertices, faces, normals)
#get_patch_coords_fast_simple(vertices, faces, normals)

