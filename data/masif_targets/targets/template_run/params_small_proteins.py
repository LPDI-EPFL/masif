import os
from default_config.masif_opts import masif_opts
params = {}
# Directory locations
params['masif_root'] = os.environ['masif_root']
# Seed locations
params['top_seed_dir'] = os.path.join(params['masif_root'], 'data/masif_small_proteins/')
params['seed_surf_dir'] = os.path.join(params['top_seed_dir'], masif_opts['ply_chain_dir'])
params['seed_iface_dir'] = os.path.join(params['masif_root'],'data/masif_small_proteins/',masif_opts['site']['out_pred_dir'])
params['seed_ply_iface_dir'] = os.path.join(params['masif_root'],'data/masif_site',masif_opts['site']['out_surf_dir'])
params['seed_pdb_dir'] = os.path.join(params['top_seed_dir'],masif_opts['pdb_chain_dir'])
params['seed_desc_dir'] = os.path.join(params['top_seed_dir'],masif_opts['ppi_search']['desc_dir'])
# 12A
params['seed_precomp_dir'] = os.path.join(params['top_seed_dir'],masif_opts['ppi_search']['masif_precomputation_dir'])
# 9A
#params['seed_precomp_dir'] = os.path.join(params['top_seed_dir'],masif_opts['site']['masif_precomputation_dir'])

# Target locations
params['top_target_dir'] = os.path.join(params['masif_root'], 'data/masif_targets/')
params['target_surf_dir'] = os.path.join(params['top_target_dir'], masif_opts['ply_chain_dir'])
params['target_iface_dir'] = os.path.join(params['masif_root'],'data/masif_targets',masif_opts['site']['out_pred_dir'])
params['target_ply_iface_dir'] = os.path.join(params['masif_root'],'data/masif_targets',masif_opts['site']['out_surf_dir'])
params['target_pdb_dir'] = os.path.join(params['top_target_dir'],masif_opts['pdb_chain_dir'])
params['target_desc_dir'] = os.path.join(params['top_target_dir'],masif_opts['ppi_search']['desc_dir'])
params['target_desc_dir_no_scfilt_chem'] = os.path.join(params['top_target_dir'],'descriptors/no_scfilt/chem/')
params['target_desc_dir_no_scfilt_all_feat'] = os.path.join(params['top_target_dir'],'descriptors/no_scfilt/all_feat/')
params['target_desc_dir'] = os.path.join(params['top_target_dir'],masif_opts['ppi_search']['desc_dir'])
# 12 A
params['target_precomp_dir'] = os.path.join(params['top_target_dir'],masif_opts['ppi_search']['masif_precomputation_dir'])
# 9 A
#params['target_precomp_dir'] = os.path.join(params['top_target_dir'],masif_opts['site']['masif_precomputation_dir'])

# Number of sites to target in the protein
params['num_sites'] = 1

# Number of clashes to tolerate.
params['clashing_cutoff'] = float('inf')

# Ransac parameters
params['ransac_iter'] = 4000
# Ransac type: normal or shape_comp
params['ransac_type'] = 'shape_comp'
params['ransac_radius'] = 1.5

###
# Score cutoffs -- these are empirical values, if they are too loose, then you get a lot of results. 
# Descriptor distance cutoff for the patch. All scores below this value are accepted for further processing.
params['desc_dist_cutoff'] = 1.7
# Interface cutoff value, all values below this cutoff are accepted.
params['iface_cutoff'] = 0.75
# Post alignment score cutof
params['post_alignment_score_cutoff'] = 22

# Output directory (target_name, target_site, target_
params['out_dir_template'] = 'out_small_12A_1.5ransac/{}/'
