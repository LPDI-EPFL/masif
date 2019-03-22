from default_config.masif_opts import masif_opts
import sys
import os
print "Initializing matlab" 
from core.initialize_matlab import initialize_matlab

# 1) As input: A BINDER_PDB, a TARGET_PDB, and their corresponding surfaces.
if len(sys.argv) != 2: 
    print "Usage: "+sys.argv[0]+" XXXX_A_XY"
    print "A or AB are the chains of the binder surface."
    print "X or XY are the chains of the target surface. (optional)"
    sys.exit(1)

eng = initialize_matlab()

in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]
chain_ids2 = in_fields[2]

paths = {}
paths['input'] = masif_opts['mat_dir_template'].format(sys.argv[1])
if not os.path.exists(masif_opts['coord_dir']):
    os.mkdir(masif_opts['coord_dir'])
paths['output'] = masif_opts['coord_dir_template'].format(sys.argv[1])
params = masif_opts
pablo = eng.coords_mds(paths, params)
