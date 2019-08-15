import warnings
warnings.filterwarnings("ignore")
import numpy as np
import sys
import os
from Bio.PDB import *
from sklearn.neighbors import KDTree
from IPython.core.debugger import set_trace

# Pablo Gainza 2019 - eval zdock results for comparison with MaSIF. 

def test_alignment(target_pdb_fn, source_pdb_fn, aligned_pdb_fn, interface_dist = 10.0):
    parser = PDBParser()
    target_struct = parser.get_structure(target_pdb_fn, target_pdb_fn)
    target_coord = np.asarray([atom.get_coord() for atom in target_struct.get_atoms() if atom.get_id() == 'CA'])
    target_atom = [atom for atom in target_struct.get_atoms() if atom.get_id() == 'CA']

    source_struct = parser.get_structure(source_pdb_fn, source_pdb_fn)
    source_coord = np.asarray([atom.get_coord() for atom in source_struct.get_atoms() if atom.get_id() == 'CA'])
    source_atom = [atom for atom in source_struct.get_atoms() if atom.get_id() == 'CA']

    aligned_struct = parser.get_structure(aligned_pdb_fn, aligned_pdb_fn)

    # The following code was replaced by the sklearn code above. I leave it here for comparison purposes
#    flann = pyflann.FLANN()
#    r, d = flann.nn(target_coord, source_coord)
#    d = np.sqrt(d) 

    # Find interface atoms in source. 
    kdt = KDTree(target_coord)
    # For each element in source_coord, find the closest CA atom in target_coord. If it is within interface_dist, then it is interface.
    d, r = kdt.query(source_coord)
    # d is of size source_coord
    # Those atoms in d with less than interface_dist, are interface.
    int_at_ix = np.where(d < interface_dist)[0]
    dists = []
    for at_ix in int_at_ix: 
        res_id = source_atom[at_ix].get_parent().get_id()
        chain_id = source_atom[at_ix].get_parent().get_parent().get_id()
        d = aligned_struct[0][chain_id][res_id]['CA'] - source_atom[at_ix]
        dists.append(d)

    rmsd_source  = np.sqrt(np.mean(np.square(dists)))

    # ZDock sometimes swaps receptor and ligand. So our target could be the actual moving one. Therefore we compute the rmsd of the target. 
    kdt = KDTree(source_coord)
    # For each element in target_coord, find the closest CA atom in source_coord. If it is within interface_dist, then it is interface.
    d, r = kdt.query(target_coord)
    # d is of size source_coord
    # Those atoms in d with less than interface_dist, are interface.
    int_at_ix = np.where(d < interface_dist)[0]
    dists = []
    for at_ix in int_at_ix: 
        res_id = target_atom[at_ix].get_parent().get_id()
        chain_id = target_atom[at_ix].get_parent().get_parent().get_id()
        d = aligned_struct[0][chain_id][res_id]['CA'] - target_atom[at_ix]
        dists.append(d)

    rmsd_target = np.sqrt(np.mean(np.square(dists)))

    # One of the two should be zero
    assert (min(rmsd_source, rmsd_target) < 1e-8)
    

    return max(rmsd_source, rmsd_target)

# ppi_pair_id: pair of proteins in PDBID_CHAIN1_CHAIN2 format
ppi_pair_id = sys.argv[1]

# receptor= target
target_pdb_fn = sys.argv[2]

# ligand= source
source_pdb_fn = sys.argv[3]

# Align all files that start with 'complex' to the ligand (called source here) and receptor (called target here)
# gt_rank: the rank of the ground truth in the actual complex.
gt_rank = 101
for i in range(1,101):
    myfile = 'complex.{}.pdb'.format(i)
    rmsd = test_alignment(target_pdb_fn, source_pdb_fn, myfile)
    # If any aligns within 5.0 A rmsd, print that file and exit. 
    if rmsd <= 5.0:
        gt_rank = i
        break
print('Rank in wildtype: {}'.format(gt_rank))

if gt_rank < 101:
    # Open the output file of the ground truth complex, and check the score of the ground truth (which will be entry numbered by gt_rank's value)
    gt_pdbid = ppi_pair_id.split('_')[0]
    gt_c1 = ppi_pair_id.split('_')[1]
    gt_c2 = ppi_pair_id.split('_')[2]
    gt_fn = 'zdock_'+gt_pdbid+'_'+gt_c1+'_'+gt_pdbid+'_'+gt_c2+'.out'
    gt_l = open(gt_fn).readlines()
    # conformation info starts at line 5
    gt_score_line = gt_l[4+gt_rank]
    gt_score = float(gt_score_line.split()[-1])

    # Now open every other file 
    for decoy_fn in os.listdir('.'):
        if decoy_fn.startswith('zdock_') and decoy_fn.endswith('.out'):
            if decoy_fn != gt_fn:
                decoy_l = open(decoy_fn).readlines() 
                for line in decoy_l[5:]: 
                    score = float(line.split()[-1])
                    if score > gt_score:
                        gt_rank += 1
    print('Rank overall: {}'.format(gt_rank))
else:
    print('Rank overall: N/D')





