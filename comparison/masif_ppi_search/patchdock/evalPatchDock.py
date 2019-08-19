import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from subprocess import Popen, PIPE
from IPython.core.debugger import *
import sys
from Bio.PDB import *
import pyflann

def test_alignment(target_pdb_fn, source_pdb_fn, aligned_pdb_fn, interface_dist = 10.0):
    parser = PDBParser()
    target_struct = parser.get_structure(target_pdb_fn, target_pdb_fn)
    target_coord = np.asarray([atom.get_coord() for atom in target_struct.get_atoms() if atom.get_id() == 'CA'])

    source_struct = parser.get_structure(source_pdb_fn, source_pdb_fn)
    source_coord = np.asarray([atom.get_coord() for atom in source_struct.get_atoms() if atom.get_id() == 'CA'])
    source_atom = [atom for atom in source_struct.get_atoms() if atom.get_id() == 'CA']

    aligned_struct = parser.get_structure(aligned_pdb_fn, aligned_pdb_fn)

    # Find interface atoms in source. 
    flann = pyflann.FLANN()
    r, d = flann.nn(target_coord, source_coord)
    d = np.sqrt(d) 
    int_at_ix = np.where(d < interface_dist)[0]

    dists = []
    for at_ix in int_at_ix: 
        res_id = source_atom[at_ix].get_parent().get_id()
        chain_id = source_atom[at_ix].get_parent().get_parent().get_id()
        try:
            d = aligned_struct[0][chain_id][res_id]['CA'] - source_atom[at_ix]
        except: 
            print('Failed for {} '.format(aligned_pdb_fn))
            sys.exit(1)
        dists.append(d)

    rmsd = np.sqrt(np.mean(np.square(dists)))

    return rmsd

# Go to directory

struct_dir = 'pdbs/'
trans_output_bin = "/home/gainza/lpdi_fs/seeder/data/ppi_benchmark_complexes/10-patchdock/PatchDock/transOutput.pl"

pdbid = sys.argv[1]
# Open all results file and get the top 1000 structures by geometric score. 
run_dir = 'run/'+pdbid+'/'
all_scores = []
all_names = []
ground_truth_present = False
count_files = 0
count_file = open('count_files.txt', 'a')
missing_file = []
all_benchmark_files = [x.rstrip() for x in open('../benchmark_list.txt').readlines()]
missing_files = [x for x in all_benchmark_files if x not in os.listdir(run_dir)]

for outputfile in os.listdir(run_dir):

    if '.' in outputfile:
        continue
    if outputfile == pdbid:
        ground_truth_present = True
    lines = open(run_dir+outputfile).readlines()
    start_line = 0
    if len(lines) > 25:
        count_files += 1 
    else: 
        missing_files.append(outputfile)
    for ix, line in enumerate(lines[25:]):
        fields = line.split('|')
        geoscore = float(fields[1])
        if ix == 0 and geoscore == 0.0: 
            set_trace()
        all_scores.append(geoscore)
        all_names.append(outputfile)
count_file.write('{} {} {}\n'.format(pdbid, count_files, missing_files))
count_file.close()

all_scores, all_names = zip(*sorted(zip(all_scores, all_names), reverse=True))
        
# Get the top 1000 - count how many from ground truth 
gt_ix = []
for ix in range(len(all_scores[:10000])):
    if all_names[ix] == pdbid:
        gt_ix.append(ix)

# Translate output
pdb = pdbid.split('_')[0]
t_chain = pdbid.split('_')[1]
s_chain = pdbid.split('_')[2]
target_pdb_fn = struct_dir+pdb+'_'+t_chain+'.pdb'
source_pdb_fn = struct_dir+pdb+'_'+s_chain+'.pdb'
aligned_pdb_fn = run_dir+pdbid+'.{}.pdb'
outfile = open('data.txt', 'a')
if not ground_truth_present:
    outfile.write('Error: groundtruth not present')
else:
    found = 0
    if len(gt_ix) > 0: 
        transo = Popen([trans_output_bin, pdbid, `1`, `len(gt_ix)`], cwd=run_dir, stdout=PIPE, stdin=PIPE)
        transo.communicate()
        for ix in range(len(gt_ix)):
            # Open the PDB. 
            rmsd = test_alignment(target_pdb_fn, source_pdb_fn, aligned_pdb_fn.format(ix+1))
            if rmsd < 5.0:
                found = 1 
                outfile.write('{} {} {} {}\n'.format(gt_ix[ix], rmsd, all_scores[gt_ix[ix]], all_names[gt_ix[ix]]))
                break
    if found == 0: 
        outfile.write('{} {} {} {}\n'.format('N/D', 1000, 1000, pdbid)) 
        print('Not found')
    outfile.close()
    
