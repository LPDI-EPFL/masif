import sys
import os
import numpy as np
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)

# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts

np.random.seed(0)

# Load training data (From many files)
from masif_modules.read_data_from_surface import read_data_from_surface

print(sys.argv[2])

if len(sys.argv) <= 1:
    print("Usage: {config} "+sys.argv[0]+" {masif_ppi_search | masif_site} PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)

masif_app = sys.argv[1]

if masif_app == 'masif_ppi_search': 
    params = masif_opts['ppi_search']
elif masif_app == 'masif_site':
    params = masif_opts['site']
    params['ply_chain_dir'] = masif_opts['ply_chain_dir']
elif masif_app == 'masif_ligand':
    params = masif_opts['ligand']

ppi_pair_list = [sys.argv[2]]

total_shapes = 0
total_ppi_pairs = 0
np.random.seed(0)
print('Reading data from input ply surface files.')
for ppi_pair_id in ppi_pair_list:

    all_list_desc = []
    all_list_coords = []
    all_list_shape_idx = []
    all_list_names = []
    idx_positives = []

    my_precomp_dir = params['masif_precomputation_dir']+ppi_pair_id+'/'
    if not os.path.exists(my_precomp_dir):
        os.makedirs(my_precomp_dir)
    
    # Read directly from the ply file.
    fields = ppi_pair_id.split('_')
    ply_file = {}
    ply_file['p1'] = masif_opts['ply_file_template'].format(fields[0], fields[1])

    if fields[2] == '':
        pids = ['p1']
    else:
        ply_file['p2']  = masif_opts['ply_file_template'].format(fields[0], fields[1])
        pids = ['p1', 'p2']
        
    for pid in pids:
        
        input_feat, rho, theta, mask, neigh_indices, iface_labels = read_data_from_surface(ply_file[pid], params)

        np.save(my_precomp_dir+pid+'_rho_wrt_center', rho)
        np.save(my_precomp_dir+pid+'_theta_wrt_center', theta)
        np.save(my_precomp_dir+pid+'_input_feat', input_feat)
        np.save(my_precomp_dir+pid+'_mask', mask)
        np.save(my_precomp_dir+pid+'_list_indices', neigh_indices)
        np.save(my_precomp_dir+pid+'_iface_labels', iface_labels)

