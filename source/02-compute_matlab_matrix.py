#!/usr/local/bin/python
# Pablo Gainza 2016-2017. LPDI IBI STI EPFL
#Compute matlab matrices. 

# System imports
import importlib
import pymesh
import sys
import os
from subprocess import Popen, PIPE
from Bio.PDB import * 
from IPython.core.debugger import set_trace
from triangulation.fixmesh import fix_mesh
import numpy as np
import shutil
from glob import glob
from scipy.spatial import distance
from input_output.read_ply import read_ply
from triangulation.computeCharges import assignChargesToNewMesh
from default_config.masif_opts import masif_opts
import json

# 1) As input: A BINDER_PDB, a TARGET_PDB, and their corresponding surfaces.
if len(sys.argv) != 2: 
    print "Usage: "+sys.argv[0]+" XXXX_A_XY"
    print "A or AB are the chains of the binder surface."
    print "X or XY are the chains of the target surface. (optional)"
    sys.exit(1)
    
#### MAIN PROGRAM:

in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]
chain_ids2 = in_fields[2]
single_chain = False
masif_opts['single_chain'] = False
if chain_ids2 == '':
    single_chain = True
    masif_opts['single_chain'] = True

binder_ply = masif_opts['ply_file_template'].format(pdb_id, chain_ids1)
target_ply = masif_opts['ply_file_template'].format(pdb_id, chain_ids2)

# Read surface files. 
v1, f1, _, c1, _, hb1, hph1 = read_ply(binder_ply)
if not single_chain:
    v2, f2, _, c2, _, hb2, hph2 = read_ply(target_ply)

# 3) Call the matlab function to export to mtlab..

# INITIALIZE MATLAB
print "Initializing matlab" 
import matlab
from core.initialize_matlab import initialize_matlab
eng = initialize_matlab()

# Convert to matlab format. 
v1_ml = matlab.double(np.ndarray.tolist(v1))
f1_ml = matlab.double(np.ndarray.tolist(f1))
c1_ml = matlab.double(np.ndarray.tolist(c1))
hb1_ml = matlab.double(np.ndarray.tolist(hb1))
hph1_ml = matlab.double(np.ndarray.tolist(hph1))

if not single_chain:
    v2_ml = matlab.double(np.ndarray.tolist(v2))
    f2_ml = matlab.double(np.ndarray.tolist(f2))
    c2_ml = matlab.double(np.ndarray.tolist(c2))
    hb2_ml = matlab.double(np.ndarray.tolist(hb2))
    hph2_ml = matlab.double(np.ndarray.tolist(hph2))
else: 
    v2_ml = 0
    f2_ml = 0
    c2_ml = 0
    hb2_ml = 0
    hph2_ml = 0

out_mat_base = masif_opts['mat_file_template'].format(pdb_id, chain_ids1, chain_ids2)
try:
    os.stat(out_mat_base)
except:
    os.makedirs(out_mat_base)

out_mat_file = out_mat_base+"/"+pdb_id+"_"+chain_ids1+'_'+chain_ids2

print "calling matlab" 
central_vertex_index = eng.compute_matlab_matrix(out_mat_file, v1_ml, \
        f1_ml, c1_ml, hb1_ml, hph1_ml, v2_ml, f2_ml, c2_ml, hb2_ml, hph2_ml, masif_opts)

try:
    os.rmdir(out_mat_base)
    print 'Something went wrong'
except:
    print 'Everything OK'
