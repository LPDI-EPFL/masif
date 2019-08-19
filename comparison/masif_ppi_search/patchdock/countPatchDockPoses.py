import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from subprocess import Popen, PIPE
from IPython.core.debugger import *
import sys
from Bio.PDB import *
import pyflann

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
    for ix, line in enumerate(lines[25:]):
        fields = line.split('|')
        geoscore = float(fields[1])
        all_scores.append(geoscore)
print('Number of scores: {}'.format(len(all_scores)))
