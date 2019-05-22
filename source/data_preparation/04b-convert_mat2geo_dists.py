import numpy as np
import sys
import glob, os
import scipy.sparse as sp
import h5py
from IPython.core.debugger import set_trace
from default_config.masif_opts import masif_opts

def load_matlab_file(path_file, name_field, struct=False):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    if type(name_field) is tuple:
        if name_field[1] not in db[name_field[0]]:
            return None
        ds = db[name_field[0]][name_field[1]]
    else:
        ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
        if struct:
            out = dict()
            for c_k in ds.keys():
                out[c_k] = np.asarray(ds[c_k])
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out

app = sys.argv[1]
ppi_pair_id = sys.argv[2]


out_dir = masif_opts[app]['masif_precomputation_dir']+'/'+ppi_pair_id+'/'
radius = masif_opts[app]['max_distance']
in_file = masif_opts['coord_dir']+'/'+ppi_pair_id+'/'+ppi_pair_id+'.mat'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
pids = ['p1']
if ppi_pair_id.split('_')[2] != '':
    pids.append('p2')

for pid in pids:
    coord = load_matlab_file(in_file, ('all_patch_coord',pid))

    all_geodists = []
    all_indices = []
    for i in range(coord.shape[0]):
        n = coord.shape[0]
        indices = coord[i,:n].nonzero()
        row = np.asarray(coord[i][indices])[0]
        subix = np.where(np.logical_and(row > 0.0, row < radius))[0]
        indices= indices[1][subix]
        geodists = row[subix]
        assert geodists.shape == indices.shape
        all_geodists.append(geodists)
        all_indices.append(indices)

    np.save(os.path.join(out_dir, pid+'_geodists.npy'), all_geodists)
    np.save(os.path.join(out_dir, pid+'_geodists_indices.npy'), all_indices)

