import sys
import os
import numpy as np
from IPython.core.debugger import set_trace

# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts

np.random.seed(0)

# Load training data (From many files)
from masif_modules.read_data_from_matfile import read_data_from_matfile_full_protein
from masif_modules.compute_input_feat import compute_input_feat

print(sys.argv[1])
# Single ppi pair id set
if len(sys.argv) == 2: 
    ppi_pair_list = [sys.argv[1]]
elif len(sys.argv) == 3 and sys.argv[1] == '-l':
    # Read input from file
    list_of_ppi_file = open(sys.argv[2])
    ppi_pair_list = []
    for line in list_of_ppi_file.readlines():
        ppi_pair_list.append(line.rstrip())
else: 
    ppi_pair_list = os.listdir(path_coords)

# Training data are all directories in path_coords
total_shapes = 0
total_ppi_pairs = 0
np.random.seed(0)
print ('Reading data from matfiles.')

#for ppi_pair_id in os.listdir(path_coords):
for ppi_pair_id in ppi_pair_list:

    all_list_desc = []
    all_list_coords = []
    all_list_shape_idx = []
    all_list_names = []
    idx_positives = []

    my_precomp_dir = masif_opts['masif_precomputation_dir']+ppi_pair_id+'/'
    if not os.path.exists(my_precomp_dir):
        os.mkdir(my_precomp_dir)
    
    coord_file = masif_opts['coord_file_template'].format(ppi_pair_id, ppi_pair_id)
    shape_file = masif_opts['mat_file_template'].format(ppi_pair_id, ppi_pair_id)
        
    for pid in ['p1', 'p2']:
#        try:
        
        list_desc, list_coords, list_shape_idx, list_names, X, Y, Z, list_sc_labels, list_indices = \
                read_data_from_matfile_full_protein(coord_file, shape_file, ppi_pair_id, masif_opts, pid, label_sc=True)
#        except Exception, e: 
#            print('Error reading file'+str(e))
#            continue
        if list_desc is []:
            print('List desc is empty')
            continue

        # Extract features and patches from loaded data. 
        from masif_modules.extract_features import extract_features
        print('Extract features')

        list_rho_wrt_center, list_theta_wrt_center, list_isc, list_normals_proj, norm_list_electrostatics, list_hbond, list_hphob = extract_features(list_desc, list_coords)

        rho_wrt_center, theta_wrt_center, input_feat, mask = compute_input_feat(list_rho_wrt_center, list_theta_wrt_center,\
                    list_isc, list_normals_proj, list_hbond, norm_list_electrostatics, masif_opts['max_shape_size'], list_hphob=list_hphob, feat_mask=masif_opts['feat_mask'])

        np.save(my_precomp_dir+pid+'_rho_wrt_center', rho_wrt_center)
        np.save(my_precomp_dir+pid+'_theta_wrt_center', theta_wrt_center)
        np.save(my_precomp_dir+pid+'_input_feat', input_feat)
        np.save(my_precomp_dir+pid+'_mask', mask)
        np.save(my_precomp_dir+pid+'_names', list_names)
        np.save(my_precomp_dir+pid+'_sc_labels', list_sc_labels)
        np.save(my_precomp_dir+pid+'_list_indices', list_indices)
        np.save(my_precomp_dir+pid+'_X', X)
        np.save(my_precomp_dir+pid+'_Y', Y)
        np.save(my_precomp_dir+pid+'_Z', Z)

