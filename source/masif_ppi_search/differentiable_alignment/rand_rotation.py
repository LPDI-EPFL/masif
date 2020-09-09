import numpy as np
from IPython.core.debugger import set_trace
def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


# Perform a random rotation to a patch, center in origin.
def rand_rotate_center_patch(patch, patch_normals):
    assert(len(patch.shape) == 2)
    M = rand_rotation_matrix()
    patch = np.copy(patch)
    patch_normals = np.copy(patch_normals)
    center_of_mass = np.mean(patch, axis=0)
    patch = patch-center_of_mass
    patch = np.dot(patch, M)
    patch_normals = np.dot(patch_normals, M)
    return patch, patch_normals

# Perform a random rotation to a set of patches in batch mode, center in origin.
def batch_rand_rotate_center_patch(in_patch, in_patch_normals):
    assert(len(in_patch.shape) == 3)
    assert(in_patch_normals.shape[2] == 3)
    batch_patch = []
    batch_norm = []
    for i in range(len(in_patch)):
        M = rand_rotation_matrix()
        patch = np.copy(in_patch[i])
        patch_normals = np.copy(in_patch_normals[i])
        center_of_mass = np.mean(patch, axis=0)
        patch = patch-center_of_mass
        patch = np.dot(patch, M)
        patch_normals = np.dot(patch_normals, M)
        batch_patch.append(patch)
        batch_norm.append(patch_normals)
    return batch_patch, batch_norm

# center patch in origin.
#def center_patch(patch):
#    patch = np.copy(patch)
#    center_of_mass = np.mean(patch, axis=0)
#    patch = patch-center_of_mass
#    return patch

# center both patches in origin.
def center_patch(patch1, patch2):
    patch1 = np.copy(patch1)
    patch2 = np.copy(patch2)
    center_of_mass1 = np.mean(patch1, axis=0)
    center_of_mass2 = np.mean(patch2, axis=0)
    center_of_mass = (center_of_mass1 + center_of_mass2)/2
    assert(len(center_of_mass) == 3)
    patch1 = patch1-center_of_mass
    patch2 = patch2-center_of_mass
    return patch1, patch2, center_of_mass
