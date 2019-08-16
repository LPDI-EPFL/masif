import numpy as np 
import os
import shutil 
from subprocess import Popen, PIPE
from sklearn import KDTree
from Bio.PDB import * 

# Generate surface file for probis for the MaSIF benchmark. 
# Pablo Gainza 2019. 
probisexe="probis/probis"

cofactors = ['HEM', 'NAD', 'NAP', 'SAM', 'ADP', 'FAD', 'COA']

# Go through all training and testing pdbs.
for mydir in ['training_pdbs', 'testing_pdbs']:
    curdir = os.path.join('data/', mydir)
    for fn in os.listdir(curdir):
        
        # Open the pdb file of the cofactor.
        parser = PDBParser()
        full_fn = os.path.join(curdir, fn)
        chains = fn.split('_')[1].split('.')[0]
        struct = parser.get_structure(full_fn, full_fn)
        cof_keys = []
        for model in struct:
            for chain in model: 
                for residue in chain:
                    if residue.get_resname() in cofactors: 
                        mykey = residue.get_resname()+'.'+str(residue.get_id()[1])+'.'+chain.get_id()
                        cof_keys.append(mykey)

        # Compute the surface file (ProBIS surface file) for each cofactor.         
        if mydir == 'training_pdbs':
            outdir = 'data/training_srfs/'
        elif mydir == 'testing_pdbs':
            outdir = 'data/testing_srfs/'

        # Compute the surface file of each pocket. 
        pdbid_chain = fn.split('.')[0]
        for key in cof_keys:
            outfile = outdir+pdbid_chain+'_'+key+'.srf'
            args = [probisexe, '-extract', '-bsite', key, '-dist', '3.0', '-f1', full_fn, '-c1', chains, '-srffile', outfile]
            p2 = Popen(args, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p2.communicate()
            print(stderr)

