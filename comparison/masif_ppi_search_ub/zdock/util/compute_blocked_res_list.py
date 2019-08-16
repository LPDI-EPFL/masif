import numpy as np
import sys
import time
import shutil
import subprocess
import os
#import pyflann
from IPython.core.debugger import set_trace


# Script to block all atoms that are not in the target site-- used to block everything in ZDOCK outside the target site.
# This is done by adding a 19 to column 55-56 for atoms outside target.
# Input in the form 4QVF_A_B, where 4QVF is the pdbid and A, B are the chains. The first chain ('A') is blocked.
target_pdb_pair = sys.argv[1]
outdir = '02-zdock_marked_blocked_pdbs/'

# Precomputation dir for masif. The location of the target site is extracted from here.
precomp_dir = '/home/gainza/lpdi_fs/masif/data/masif_ppi_search_ub/data_preparation/04b-precomputation_12A/precomputation'


# Open the patch information to find the center of the interface. Print all residues that are within X of this. 
sc_labels = np.load(os.path.join(precomp_dir,target_pdb_pair,'p1_sc_labels.npy'))
center_point = np.argmax(np.median(np.nan_to_num(sc_labels[0]),axis=1))
val = np.max(np.median(np.nan_to_num(sc_labels[0]),axis=1))

# Find the vertex with the highest shape complementarity. 
center_point = np.argmax(np.median(np.nan_to_num(sc_labels[0]),axis=1))
print(center_point)
print(val)

# Read the neighbors for this center point.
neigh = np.load(os.path.join(precomp_dir,target_pdb_pair,'p1_list_indices.npy'), encoding='latin1', allow_pickle=True)[center_point]
print(neigh)

# Read the vertices
X = np.load(os.path.join(precomp_dir,target_pdb_pair,'p1_X.npy'))
Y = np.load(os.path.join(precomp_dir,target_pdb_pair,'p1_Y.npy'))
Z = np.load(os.path.join(precomp_dir,target_pdb_pair,'p1_Z.npy'))
v_t = np.stack([X,Y,Z]).T

# Open the pdb file line by line. 
pdbid = target_pdb_pair.split('_')[0]
t_chain = target_pdb_pair.split('_')[1]

pdb_file_t = os.path.join('01-zdock_marked/','{}_{}_m.pdb'.format(pdbid,t_chain))
all_lines = open(pdb_file_t, 'r').readlines()
# Store line_ix and coord for all lines
interface_line_ix = []
target_atom_coords = []
for ix, line in enumerate(all_lines):
    if line.startswith("ATOM"):
        interface_line_ix.append(ix)
        coordx = float(line[30:38])
        coordy = float(line[38:46])
        coordz = float(line[46:54])

        target_atom_coords.append((coordx, coordy, coordz))

target_atom_coords = np.array(target_atom_coords)

# Find the atom coords closest to each point in the neighborhood of the interface
neigh_atom_ix = set()
for point in neigh:
    dists = np.sqrt(np.sum(np.square(target_atom_coords - v_t[point]),axis=1 ))
    closest_at_iii = np.argmin(dists)
    neigh_atom_ix.add(interface_line_ix[closest_at_iii])

# Block every line that is not in the interface, by adding a 19 to columns 55-56
outfile = open(outdir+'{}_{}_m_bl.pdb'.format(pdbid,t_chain),'w')
for ix, line in enumerate(all_lines):
    if line.startswith("ATOM"):
        # Block every line whose index is not in neigh_atom_ix
        if ix not in neigh_atom_ix: 
            listline = list(line)
            listline[55] = '1'
            listline[56] = '9'
            line = ''.join(listline)
        else:
            print(line)
        outfile.write(line)
        
