# Header variables and parameters.
import os
import numpy as np
from IPython.core.debugger import set_trace

from run_args import params, pos_precomputation_dir, neg_precomputation_dir

from monet_modules.compute_input_feat import compute_input_feat

if 'pids' not in params: 
    params['pids'] = ['p1', 'p2']

# Read the positive first 
parent_in_dir = pos_precomputation_dir

binder_rho_wrt_center = []
binder_theta_wrt_center = []
binder_input_feat = []
binder_mask = []

pos_rho_wrt_center = []
pos_theta_wrt_center = []
pos_input_feat = []
pos_mask = []

np.random.seed(0)
pos_training_idx = []
pos_val_idx = []
pos_test_idx = []
pos_names = []

pos_training_list = [x.rstrip() for x in open(params['pos_training_list']).readlines()]
pos_testing_list = [x.rstrip() for x in open(params['pos_testing_list']).readlines()]
neg_training_list = pos_training_list
neg_testing_list = pos_testing_list

idx_count = 0
binder_pos_data = {}
for count, ppi_pair_id in enumerate(os.listdir(parent_in_dir)):
    in_dir = parent_in_dir + ppi_pair_id+'/'
    print(ppi_pair_id)

    # Read binder and pos.
    train_val = np.random.random()
    pid = 'p1'
    try:
        labels = np.load(in_dir+pid+'_labels.npy')
    except Exception, e:
        print('Could not open '+in_dir+pid+'_list_rho_wrt_center.npy: '+str(e))
        continue
    # The pos labels are the same for binder and pos. 
    # Randomly choose a subset of the positive shapes.
    pos_labels = labels[labels == 1]
    K = params['pos_surf_accept_probability']*len(pos_labels)
    k = np.random.choice(len(pos_labels), int(K))

    # Read binder first, which is p1.
    list_rho_wrt_center = np.load(in_dir+pid+'_list_rho_wrt_center.npy')[k]
    list_theta_wrt_center = np.load(in_dir+pid+'_list_theta_wrt_center.npy')[k]
    list_isc = np.load(in_dir+pid+'_list_isc.npy')[k]
    list_normals_proj = np.load(in_dir+pid+'_list_normals_proj.npy')[k]
    norm_list_electrostatics = np.load(in_dir+pid+'_norm_list_electrostatics.npy')[k]
    list_hbond = np.load(in_dir+pid+'_list_hbond.npy')[k]
    list_hphob = np.load(in_dir+pid+'_list_hphob.npy')[k]
    list_names = np.load(in_dir+pid+'_list_names.npy')[k]

    rho_wrt_center, theta_wrt_center, input_feat, mask = compute_input_feat(list_rho_wrt_center, list_theta_wrt_center,\
                list_isc, list_normals_proj, list_hbond, norm_list_electrostatics, params['max_shape_size'], list_hphob=list_hphob, feat_mask=params['feat_mask'])

    binder_rho_wrt_center.append(rho_wrt_center)
    binder_theta_wrt_center.append(theta_wrt_center)
    binder_input_feat.append(input_feat)
    binder_mask.append(mask)

    # Read pos, which is p2.
    pid = 'p2'
    list_rho_wrt_center = np.load(in_dir+pid+'_list_rho_wrt_center.npy')[k]
    list_theta_wrt_center = np.load(in_dir+pid+'_list_theta_wrt_center.npy')[k]
    list_isc = np.load(in_dir+pid+'_list_isc.npy')[k]
    list_normals_proj = np.load(in_dir+pid+'_list_normals_proj.npy')[k]
    norm_list_electrostatics = np.load(in_dir+pid+'_norm_list_electrostatics.npy')[k]
    list_hbond = np.load(in_dir+pid+'_list_hbond.npy')[k]
    list_hphob = np.load(in_dir+pid+'_list_hphob.npy')[k]
    list_names = np.load(in_dir+pid+'_list_names.npy')[k]

    rho_wrt_center, theta_wrt_center, input_feat, mask = compute_input_feat(list_rho_wrt_center, list_theta_wrt_center,\
                list_isc, list_normals_proj, list_hbond, norm_list_electrostatics, params['max_shape_size'], list_hphob=list_hphob, feat_mask=params['feat_mask'])

    pos_rho_wrt_center.append(rho_wrt_center)
    pos_theta_wrt_center.append(theta_wrt_center)
    pos_input_feat.append(input_feat)
    pos_mask.append(mask)

    these_ppi_ids = []
    for name in list_names: 
        these_ppi_ids.append(pid+name)
    pos_names.append(these_ppi_ids)

    # Training, validation or test?
    fields = ppi_pair_id.split('_')
    if ppi_pair_id in pos_training_list:
        if train_val <= params['range_val_samples']:
            pos_training_idx = pos_training_idx + range(idx_count, idx_count+len(mask))
        elif train_val > params['range_val_samples']: 
            pos_val_idx = pos_val_idx + range(idx_count, idx_count+len(mask))
    else: 
        pos_test_idx = pos_test_idx + range(idx_count, idx_count+len(mask))
    idx_count += len(mask)

binder_rho_wrt_center = np.concatenate(binder_rho_wrt_center, axis=0)
binder_theta_wrt_center = np.concatenate(binder_theta_wrt_center, axis=0)
binder_input_feat = np.concatenate(binder_input_feat, axis=0)
binder_mask = np.concatenate(binder_mask, axis=0)

pos_rho_wrt_center = np.concatenate(pos_rho_wrt_center, axis=0)
pos_theta_wrt_center = np.concatenate(pos_theta_wrt_center, axis=0)
pos_input_feat = np.concatenate(pos_input_feat, axis=0)
pos_mask = np.concatenate(pos_mask, axis=0)
pos_names = np.concatenate(pos_names, axis=0)
np.save('cache/pos_names.npy', pos_names)

## Negatives.
neg_rho_wrt_center = []
neg_theta_wrt_center = []
neg_input_feat = []
neg_mask = []
neg_names = []
parent_in_dir = neg_precomputation_dir

# Cache all the data as it takes a long time to load otherwise. 
idx_count = 0
neg_training_idx = []
neg_val_idx = []
neg_test_idx = []
print ('Computing negatives now') 
for count, ppi_pair_id in enumerate(os.listdir(parent_in_dir)):
    print(ppi_pair_id)
    in_dir = parent_in_dir + ppi_pair_id+'/'

    train_val = np.random.random()
    for ix, pid in enumerate(['p1','p2']):
        try:
            labels = np.load(in_dir+pid+'_labels.npy')
        except Exception, e:
            print('Could not open '+in_dir+pid+'_list_rho_wrt_center.npy: '+str(e))
            continue

        neg_labels = np.where(labels == 0)[0]
        # Negatives are sampled in the precompuation.
        k = neg_labels
        list_rho_wrt_center = np.load(in_dir+pid+'_list_rho_wrt_center.npy')[k]
        list_theta_wrt_center = np.load(in_dir+pid+'_list_theta_wrt_center.npy')[k]
        list_isc = np.load(in_dir+pid+'_list_isc.npy')[k]
        list_normals_proj = np.load(in_dir+pid+'_list_normals_proj.npy')[k]
        norm_list_electrostatics = np.load(in_dir+pid+'_norm_list_electrostatics.npy')[k]
        list_hbond = np.load(in_dir+pid+'_list_hbond.npy')[k]
        list_hphob = np.load(in_dir+pid+'_list_hphob.npy')[k]
        list_names = np.load(in_dir+pid+'_list_names.npy')[k]

        rho_wrt_center, theta_wrt_center, input_feat, mask = compute_input_feat(list_rho_wrt_center, list_theta_wrt_center,\
                    list_isc, list_normals_proj, list_hbond, norm_list_electrostatics, params['max_shape_size'], list_hphob=list_hphob, feat_mask=params['feat_mask'])

        neg_rho_wrt_center.append(rho_wrt_center)
        neg_theta_wrt_center.append(theta_wrt_center)
        neg_input_feat.append(input_feat)
        neg_mask.append(mask)
        neg_names.append(list_names)
        # Training, validation or test? 
        if ppi_pair_id in neg_training_list:
            if train_val <= params['range_val_samples']:
                neg_training_idx = neg_training_idx + range(idx_count, idx_count+len(mask))
            elif train_val > params['range_val_samples']: 
                neg_val_idx = neg_val_idx + range(idx_count, idx_count+len(mask))
        else: 
            neg_test_idx = neg_test_idx + range(idx_count, idx_count+len(mask))

        idx_count += len(mask)

neg_rho_wrt_center = np.concatenate(neg_rho_wrt_center, axis=0)
neg_theta_wrt_center = np.concatenate(neg_theta_wrt_center, axis=0)
neg_input_feat = np.concatenate(neg_input_feat, axis=0)
neg_mask = np.concatenate(neg_mask, axis=0)

print('Read {} positive shapes'.format(len(pos_input_feat)))
print('Read {} positive training shapes'.format(len(pos_training_idx)))
print('Read {} positive validation shapes'.format(len(pos_val_idx)))
print('Read {} positive testing shapes'.format(len(pos_test_idx)))

print('Read {} negative shapes'.format(len(neg_input_feat)))
print('Read {} negative training shapes'.format(len(neg_training_idx)))
print('Read {} negative validation shapes'.format(len(neg_val_idx)))
print('Read {} negative testing shapes'.format(len(neg_test_idx)))

np.save('cache/binder_rho_wrt_center.npy', binder_rho_wrt_center)
np.save('cache/binder_theta_wrt_center.npy', binder_theta_wrt_center)
np.save('cache/binder_input_feat.npy', binder_input_feat)
np.save('cache/binder_mask.npy', binder_mask)

np.save('cache/pos_training_idx.npy', pos_training_idx)
np.save('cache/pos_val_idx.npy', pos_val_idx)
np.save('cache/pos_test_idx.npy', pos_test_idx)
np.save('cache/pos_rho_wrt_center.npy', pos_rho_wrt_center)
np.save('cache/pos_theta_wrt_center.npy', pos_theta_wrt_center)
np.save('cache/pos_input_feat.npy', pos_input_feat)
np.save('cache/pos_mask.npy', pos_mask)

np.save('cache/neg_training_idx.npy', neg_training_idx)
np.save('cache/neg_val_idx.npy', neg_val_idx)
np.save('cache/neg_test_idx.npy', neg_test_idx)
np.save('cache/neg_rho_wrt_center.npy', neg_rho_wrt_center)
np.save('cache/neg_theta_wrt_center.npy', neg_theta_wrt_center)
np.save('cache/neg_input_feat.npy', neg_input_feat)
np.save('cache/neg_mask.npy', neg_mask)
np.save('cache/neg_names.npy', neg_names)

