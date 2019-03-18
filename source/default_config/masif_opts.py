masif_opts = {}
# Default directories
masif_opts['raw_pdb_dir'] = '00-raw_pdbs/'
masif_opts['pdb_chain_dir'] = '01-benchmark_pdbs/'
masif_opts['ply_chain_dir'] = '01-benchmark_surfaces/'
masif_opts['mat_dir'] = '02-matfile/'
masif_opts['coord_dir'] = '03-coords/'
masif_opts['tmp_dir'] = '/tmp/'
masif_opts['ply_file_template'] = masif_opts['ply_chain_dir']+'/{}_{}.ply'
masif_opts['mat_dir_template'] = masif_opts['mat_dir']+'/{}'
masif_opts['coord_dir_template'] = masif_opts['coord_dir']+'/{}'
masif_opts['mat_file_template'] = masif_opts['mat_dir']+'/{}/{}.mat'
masif_opts['coord_file_template'] = masif_opts['coord_dir']+'/{}/{}.mat'

# Surface features
masif_opts['use_hbond'] = True
masif_opts['use_hphob'] = True
masif_opts['use_apbs'] = True
masif_opts['compute_iface'] = True
# Mesh resolution. Everything gets very slow if it is lower than 1.0 
masif_opts['mesh_res'] = 1.0
masif_opts['feature_interpolation'] = True

# Parameters for shape complementarity calculations.
masif_opts['sc_radius'] = 9.0 
masif_opts['sc_interaction_cutoff'] = 1.5
masif_opts['sc_w'] = 0.25

# Coords params
masif_opts['radius'] = 12.0

# Neural network patch parameters. 
masif_opts['max_shape_size'] = 200
masif_opts['max_distance'] = 12.0 # Radius for the neural network. 
masif_opts['masif_precomputation_dir'] = '04b-precomputation_12A/precomputation/'
masif_opts['feat_mask'] = [1.0]*5
