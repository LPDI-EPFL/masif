""" 
pocket_to_pocket_align.py: Align a list of cofactor pockets in the training set to a list in the testing set. 
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

import sys
from Bio.PDB import * 
from subprocess import Popen, PIPE
from IPython.core.debugger import set_trace
import numpy as np
import os


def align_pocket_pair(pdbfns, cofids, resids, chainids, out_fn_base, cofactor_distance_cutoff = 3.0):
"""
    align_pocket_pair: align each pair of pockets using the program TMalign. Pockets are first 
    extracted from the PDB using biopython; then they are saved in a new pdb and finally they are
    saved. 
    
    The program TMalign must be in the current directory. 
"""
    
    # Make sure the cofactor is the same one. 
    assert(cofids[0] == cofids[1]) 

    # Extract the pocket using Biopython, taking all residues within 3A of the cofactor. 
    # For the pair of input PDBs :
    for ix, pdbfn in enumerate(pdbfns):
        # Store the cofactor coordinates here. 
        cofactor_coords = []
        parser = PDBParser()
        struct = parser.get_structure(pdbfn, pdbfn)
        # GO through all residues and find the coordinates of the cofactor.
        residue_list = Selection.unfold_entities(struct, 'R')
        for residue in residue_list:
            # If this is the cofactor we are looking for:  
            chain = residue.get_parent()

            if residue.get_resname() == cofids[ix] and \
                chain.get_id() == chainids[ix] and \
                str(residue.get_id()[1]) == resids[ix]:
                for atom in residue: 
                    cofactor_coords.append(atom.get_coord())

        cofactor_coords = np.array(cofactor_coords)
        # Go through all atoms and find those that are within cutoff distance of a cofactor atom. 
        atom_list = Selection.unfold_entities(struct, 'A')
        neigh_res = []
        for myatom in atom_list: 
            dists = np.sqrt(np.sum(np.square(myatom.get_coord() - cofactor_coords), axis=1))
            if np.min(dists) < cofactor_distance_cutoff:
                resid = myatom.get_parent().get_id()
                # Only add amino acids (i.e. no HET atoms
                if resid[0] == ' ' \
                        and myatom.get_parent() not in neigh_res:
                    neigh_res.append(myatom.get_parent())

        # Save the residues using a new numbering. 
        structBuild = StructureBuilder.StructureBuilder()
        structBuild.init_structure("output")
        structBuild.init_seg(" ")
        structBuild.init_model(0)
        outputStruct = structBuild.get_structure()
        structBuild.init_chain('X')
        
        for resix, myres in enumerate(neigh_res):
            structBuild.init_residue(myres.get_resname(), ' ', resix+1, ' ')
            for atom in myres:
                outputStruct[0]['X'][(' ', resix+1, ' ')].add(atom)
            
        # Save the extracted pocket, as a single chain in a temporary directory 
        pdbio = PDBIO()
        pdbio.set_structure(outputStruct)
        pdbio.save('{}_{}.pdb'.format(out_fn_base, ix))

    # Run tmalign between the two pockets. 
    args = ['./TMalign', '{}_{}.pdb'.format(out_fn_base,0), '{}_{}.pdb'.format(out_fn_base,1)] 
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    stdout = stdout.decode('ascii')
    tmout = open('{}.tm'.format(out_fn_base), 'w+')
    tmout.write(stdout)
    tmout.close()
    # Parse the TM align output
    tm_score = []
    for line in stdout.splitlines():
        if 'TM-score=' in line:
            tm_score.append(float(line.split()[1]))

    # Return mean tm_score
    return np.mean(tm_score) 


# Open the testing list with all pockets.
testing_list_lines = open('lists/testing_pockets.txt').readlines()
# Directory with all testing pdb files in the format ({PDBID}_{CHAIN}.pdb)
testing_pdb_dir = '../testing_pdbs/'

# Open the training list 
training_list_lines = open('lists/training_pockets.txt').readlines()
# Directory with all training pdb files. 
training_pdb_dir = '../training_pdbs/'

output_dir = 'test_to_train'

# my_ix is the index to the testing id in the testing_pockets.txt file.
my_ix = int(sys.argv[1])-1

# The testing line corresponds to that of my_ix
testing_full_id = testing_list_lines[my_ix]
test_pdbid = testing_full_id.split('_')[0]
test_chainid = testing_full_id.split('_')[1]
test_cofid = testing_full_id.split('_')[2].split('.')[0]
test_resid = testing_full_id.split('_')[2].split('.')[1]
test_cof_chainid = testing_full_id.split('_')[2].split('.')[2].rstrip()
test_pdbfn = os.path.join(testing_pdb_dir, test_pdbid+'_'+test_chainid+'.pdb')

outtm_fn = 'out_pocket/'+testing_full_id.rstrip()+'.out'
if os.stat(outtm_fn).st_size > 0:
    sys.exit(1)

outtm = open(outtm_fn, 'w+')
# Go through each element of the training list
for training_full_id in training_list_lines:
    train_pdbid = training_full_id.split('_')[0]
    train_chainid = training_full_id.split('_')[1]
    train_cofid = training_full_id.split('_')[2].split('.')[0]
    train_resid = training_full_id.split('_')[2].split('.')[1]
    train_cof_chainid = training_full_id.split('_')[2].split('.')[2].rstrip()
    train_pdbfn = os.path.join(training_pdb_dir, train_pdbid+'_'+train_chainid+'.pdb')

    # Ignore those that have a cofactor different from the test cofactor. 
    if train_cofid != test_cofid: 
        continue

    tmpdir = 'tmp_pockets/'+testing_full_id.rstrip()
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    tmpfn_base = os.path.join(tmpdir, testing_full_id.rstrip()+'_'+training_full_id.rstrip())
    # Perform an alignment using tm alignment. 
    tm_score = align_pocket_pair([test_pdbfn, train_pdbfn], \
            [test_cofid, train_cofid], \
            [test_resid, train_resid], 
            [test_cof_chainid, train_cof_chainid], \
            tmpfn_base)
    print('{},{},{:3f}'.format(training_full_id.rstrip(), testing_full_id.rstrip(), tm_score))
    outtm.write('{},{},{:3f}\n'.format(training_full_id.rstrip(), testing_full_id.rstrip(), tm_score))
outtm.close()



    
