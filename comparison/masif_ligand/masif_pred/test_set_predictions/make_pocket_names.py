import os
from IPython.core.debugger import set_trace
import numpy as np
from Bio.PDB import *
pdb_dir = '../../data/testing_pdbs/'
label_names_order = ['ADP', 'COA', 'FAD', 'HEM', 'NAD', 'NAP', 'SAM']
all_masif_names = []
for label_fn in os.listdir('.'):
    if '_labels' in label_fn: 
        cof_labels = []
        pdbid = label_fn.split('_')[0]
        chain_ids = label_fn.split('_')[1]
        labels = np.load(label_fn)
        # Open pdb with PDB Parser
        parser = PDBParser()
        pdbfn=os.path.join(pdb_dir, pdbid+'_'+chain_ids+'.pdb')
        struct = parser.get_structure(pdbfn, pdbfn)
        
        cof_count = 0
        mynames = []
        Failed = False
        Missing = False
        for model in struct:
            for chain in model: 
                for res in chain: 
                    if res.get_resname() in label_names_order and not Failed and not Missing: 
                        if cof_count >= len(labels):
                            print('Error with structure {}'.format(label_fn))
                            Missing = True
                            continue
                        try:
                            assert(res.get_resname() == label_names_order[labels[cof_count]])
                        except:
                            Failed = True
                           
                        name = '{}_{}_{}.{}.{}'.format(pdbid,chain_ids, res.get_resname(),res.get_id()[1], chain.get_id())
#                        print(name)
                        mynames.append(name)
                        all_masif_names.append(name)
                        cof_count += 1
        if not Failed:
            output_names = label_fn.replace('labels', 'names')
            np.save(output_names, mynames)
alln_file = open('all_masif_names.txt', 'w')
for name in all_masif_names:
    alln_file.write(name+'\n')
alln_file.close()


