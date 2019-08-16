import numpy as np 
import shutil 
import os

pdb_database_dir = "/home/gainza/freyr/patches/masif/data/masif_ligand/data_preparation/00b-pdbs_assembly"
confirmed_computed_dir = "/home/gainza/freyr/patches/masif/data/masif_ligand/data_preparation/04a-precomputation_12A/precomputation/"

training_set = np.load('train_pdbs_sequence.npy')
testing_set = np.load('test_pdbs_sequence.npy')

for mydir in os.listdir(confirmed_computed_dir):
    pdbid = mydir.split('_')[0]
    pdbid_chain = mydir.split('_')[0]+'_'+mydir.split('_')[1]
    src_file = os.path.join(pdb_database_dir, pdbid+'.pdb')

    if pdbid_chain in testing_set:
        dst_file = os.path.join('data/testing_pdbs/', pdbid_chain+'.pdb')
        shutil.copy(src_file, dst_file)
    elif pdbid_chain in training_set:
        dst_file = os.path.join('data/training_pdbs/', pdbid_chain+'.pdb')
        shutil.copy(src_file, dst_file)
        

