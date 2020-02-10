# coding: utf-8
# ## Imports and helper functions
from IPython.core.debugger import set_trace
import pymesh
import numpy as np
import time
import numpy as np

from geometry.compute_polar_coordinates import compute_polar_coordinates
from input_output.save_ply import save_ply

from sklearn import metrics


def read_data_from_surface(ply_fn, params):
    """
    # Read data from a ply file -- decompose into patches. 
    # Returns: 
    # list_desc: List of features per patch
    # list_coords: list of angular and polar coordinates.
    # list_indices: list of indices of neighbors in the patch.
    # list_sc_labels: list of shape complementarity labels (computed here).
    """
    mesh = pymesh.load_mesh(ply_fn)

    # Normals: 
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1,n2,n3], axis=1)

    # Compute the angular and radial coordinates. 
    rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, radius=params['max_distance'], max_vertices=params['max_shape_size'])

    # Compute the principal curvature components for the shape index. 
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)

    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob")/4.5

    # Iface labels (for ground truth only)     
    iface_labels = mesh.get_attribute("vertex_iface") 

    # n: number of patches, equal to the number of vertices.
    n = len(mesh.vertices)
    
    input_feat = np.zeros((n, params['max_shape_size'], 5))

    # Compute the input features for each patch.
    for vix in range(n):
        # Patch members.
        neigh_vix = np.array(neigh_indices[vix])

        # Compute the distance-dependent curvature for all neighbors of the patch. 
        patch_v = mesh.vertices[neigh_vix]
        patch_n = normals[neigh_vix]
        patch_cp = np.where(neigh_vix == vix)[0][0] # central point
        mask_pos = np.where(mask[vix] == 1.0)[0] # nonzero elements
        patch_rho = rho[vix][mask_pos] # nonzero elements of rho
        ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)        
        
        input_feat[vix, :len(neigh_vix), 0] = si[neigh_vix]
        input_feat[vix, :len(neigh_vix), 1] = ddc
        input_feat[vix, :len(neigh_vix), 2] = hbond[neigh_vix]
        input_feat[vix, :len(neigh_vix), 3] = charge[neigh_vix]
        input_feat[vix, :len(neigh_vix), 4] = hphob[neigh_vix]
        
    return input_feat, rho, theta, mask, neigh_indices, iface_labels

# From a full shape in a full protein, extract a patch around a vertex.
# If patch_indices = True, then store the indices of all neighbors.
def extract_patch_and_coord(
    vix, shape, coord, max_distance, max_vertices, patch_indices=False
):
    # Member vertices are nonzero elements
    i, j = coord[np.int(vix), : coord.shape[1] // 2].nonzero()

    # D = np.squeeze(np.asarray(coord[np.int(vix),j].todense()))
    D = np.squeeze(np.asarray(coord[np.int(vix), : coord.shape[1] // 2].todense()))
    j = np.where((D < max_distance) & (D > 0))[0]
    max_dist_tmp = max_distance
    old_j = len(j)
    while len(j) > max_vertices:
        max_dist_tmp = max_dist_tmp * 0.95
        j = np.where((D < max_dist_tmp) & (D > 0))[0]
    #    print('j = {} {}'.format(len(j), old_j))
    D = D[j]
    patch = {}
    patch["X"] = shape["X"][0][j]
    patch["Y"] = shape["Y"][0][j]
    patch["Z"] = shape["Z"][0][j]
    patch["charge"] = shape["charge"][0][j]
    patch["hbond"] = shape["hbond"][0][j]
    patch["normal"] = shape["normal"][:, j]
    patch["shape_index"] = shape["shape_index"][0][j]
    if "hphob" in shape:
        patch["hphob"] = shape["hphob"][0][j]

    patch["center"] = np.argmin(D)

    j_theta = j + coord.shape[1] // 2
    theta = np.squeeze(np.asarray(coord[np.int(vix), j_theta].todense()))
    coord = np.concatenate([D, theta], axis=0)

    if patch_indices:
        return patch, coord, j
    else:
        return patch, coord


# FOR DEBUGGING only.... too slow, use precomputed values instead.
from scipy.spatial import KDTree
def compute_shape_complementarity(shape1, coord1, shape2, coord2):
    w = 0.5
    radius = 8.0

    D1 = coord1[: coord1.shape[0] // 2]
    v1 = np.stack([shape1["X"], shape1["Y"], shape1["Z"]], 1)
    v1 = v1[np.where(D1 < radius)]
    n1 = shape1["normal"].T[np.where(D1 < radius)]

    D2 = coord2[: coord2.shape[0] // 2]
    v2 = np.stack([shape2["X"], shape2["Y"], shape2["Z"]], 1)
    v2 = v2[np.where(D2 < radius)]
    n2 = shape2["normal"].T[np.where(D2 < radius)]

    # First v2 -> v1
    kdt = KDTree(v1)
    d, i = kdt.query(v2)
    comp2 = [np.dot(n2[x], -n1[i[x]]) for x in range(len(n2))]
    comp2 = np.multiply(comp2, np.exp(-w * np.square(d)))
    comp2 = np.percentile(comp2, 50)

    # Now v1 -> v2
    kdt = KDTree(v2)
    d, i = kdt.query(v1)
    comp1 = [np.dot(n1[x], -n2[i[x]]) for x in range(len(n1))]
    comp1 = np.multiply(comp1, np.exp(-w * np.square(d)))
    comp1 = np.percentile(comp1, 50)

    return np.mean([comp1, comp2])


def normalize_electrostatics(in_elec):
    """
        Normalize electrostatics to a value between -1 and 1
    """
    elec = np.copy(in_elec)
    upper_threshold = 3
    lower_threshold = -3
    elec[elec > upper_threshold] = upper_threshold
    elec[elec > upper_threshold] = upper_threshold
    elec = elec - lower_threshold
    elec = elec / (upper_threshold - lower_threshold)
    elec = 2 * elec - 1
    return elec

def mean_normal_center_patch(D, n, r):
    """
        Function to compute the mean normal of vertices within r radius of the center of the patch.
    """
    c_normal = [n[i] for i in range(len(D)) if D[i] <= r]
    mean_normal = np.mean(c_normal, axis=0, keepdims=True).T
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    return np.squeeze(mean_normal)

def compute_ddc(patch_v, patch_n, patch_cp, patch_rho):
    """
        Compute the distance dependent curvature, Yin et al PNAS 2009
            patch_v: the patch vertices
            patch_n: the patch normals
            patch_cp: the index of the central point of the patch 
            patch_rho: the geodesic distance to all members.
        Returns a vector with the ddc for each point in the patch.
    """
    n = patch_n
    r = patch_v
    i = patch_cp
    # Compute the mean normal 2.5A around the center point
    ni = mean_normal_center_patch(patch_rho, n, 2.5)
    dij = np.linalg.norm(r - r[i], axis=1)
    # Compute the step function sf:
    sf = r + n
    sf = sf - (ni + r[i])
    sf = np.linalg.norm(sf, axis=1)
    sf = sf - dij
    sf[sf > 0] = 1
    sf[sf < 0] = -1
    sf[sf == 0] = 0
    # Compute the curvature between i and j
    dij[dij == 0] = 1e-8
    kij = np.divide(np.linalg.norm(n - ni, axis=1), dij)
    kij = np.multiply(sf, kij)
    # Ignore any values greater than 0.7 and any values smaller than 0.7
    kij[kij > 0.7] = 0
    kij[kij < -0.7] = 0

    return kij
