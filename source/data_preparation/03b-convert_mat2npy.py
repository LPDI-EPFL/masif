import numpy as np
import sys
import glob, os
import scipy.sparse as sp
import h5py
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

ppi_pair_id = sys.argv[1]

out_dir= masif_opts['coord_dir_npy']+'/'+ppi_pair_id+'/'
in_file = masif_opts['coord_dir']+'/'+ppi_pair_id+'/'+ppi_pair_id+'.mat'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
coord = load_matlab_file(in_file, ('all_patch_coord','p1'))
sp.save_npz(out_dir+'p1.npz',coord)
if ppi_pair_id.split('_')[2] != '':
    coord = load_matlab_file(in_file, ('all_patch_coord','p2'))
    sp.save_npz(out_dir+'p2.npz',coord)
