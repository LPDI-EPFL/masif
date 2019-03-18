# Protonate a pdb using the reduce program
from subprocess import Popen, PIPE
import os

def protonate(in_pdb_file, out_pdb_file):
    
    args = ["reduce", "-HIS", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate() 
    outfile = open(out_pdb_file, 'w')
    outfile.write(stdout)
    
