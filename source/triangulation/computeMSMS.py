import ipdb
import os
from subprocess import Popen, PIPE

from input_output.read_msms import read_msms
from triangulation.xyzrn import output_pdb_as_xyzrn
from default_config.global_vars import msms_bin, xyzrn_bin
from default_config.masif_opts import masif_opts
import random

# Pablo Gainza LPDI EPFL 2017-2019
# Calls MSMS and returns the vertices.
# Special atoms are atoms with a reduced radius.
def computeMSMS(pdb_file,  protonate=True):
    randnum = random.randint(1,10000000)
    file_base = masif_opts['tmp_dir']+"/msms_"+`randnum`
    out_xyzrn = file_base+".xyzrn"

    if protonate:        
        output_pdb_as_xyzrn(pdb_file, out_xyzrn)
    # If we are not using explicit hydrogens, then I invoke the script provided
    # by MSMS, as this script has many atom definitions that are a pain to
    # translate to python.
    else:
        args = [xyzrn_bin, pdb_file]
        f_out_xyzrn = open(out_xyzrn, 'w')
        p2 = Popen(args, stdout=f_out_xyzrn)
        p2.communicate()
        f_out_xyzrn.close()
    # Now run MSMS on xyzrn file
    FNULL = open(os.devnull, 'w')
    args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe",\
                    "1.5", "-if",out_xyzrn,"-of",file_base, "-af", file_base]
    #print msms_bin+" "+`args`
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base+".area")
    next(ses_file) # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]
    return vertices, faces, normals, names, areas

