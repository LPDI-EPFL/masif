import sys
import time
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
from masif_modules.read_data_from_surface import read_data_from_surface, compute_shape_complementarity

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

    if len (fields) == 2 or fields[2] == '':
        pids = ['p1']
    else:
        ply_file['p2']  = masif_opts['ply_file_template'].format(fields[0], fields[2])
        pids = ['p1', 'p2']
        
    # Compute shape complementarity between the two proteins. 
    rho = {}
    neigh_indices = {}
    mask = {}
    input_feat = {}
    theta = {}
    iface_labels = {}
    verts = {}

    for pid in pids:
        input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], iface_labels[pid], verts[pid] = read_data_from_surface(ply_file[pid], params)

    if len(pids) > 1 and masif_app == 'masif_ppi_search':
        start_time = time.time()
        p1_sc_labels, p2_sc_labels = compute_shape_complementarity(ply_file['p1'], ply_file['p2'], neigh_indices['p1'],neigh_indices['p2'], rho['p1'], rho['p2'], mask['p1'], mask['p2'], params)
        np.save(my_precomp_dir+'p1_sc_labels', p1_sc_labels)
        np.save(my_precomp_dir+'p2_sc_labels', p2_sc_labels)
        end_time = time.time()
        print("Computing shape complementarity took {:.2f}".format(end_time-start_time))

    # Save data only if everything went well. 
    for pid in pids: 
        np.save(my_precomp_dir+pid+'_rho_wrt_center', rho[pid])
        np.save(my_precomp_dir+pid+'_theta_wrt_center', theta[pid])
        np.save(my_precomp_dir+pid+'_input_feat', input_feat[pid])
        np.save(my_precomp_dir+pid+'_mask', mask[pid])
        np.save(my_precomp_dir+pid+'_list_indices', neigh_indices[pid])
        np.save(my_precomp_dir+pid+'_iface_labels', iface_labels[pid])
        # Save x, y, z
        np.save(my_precomp_dir+pid+'_X.npy', verts[pid][:,0])
        np.save(my_precomp_dir+pid+'_Y.npy', verts[pid][:,1])
        np.save(my_precomp_dir+pid+'_Z.npy', verts[pid][:,2])
