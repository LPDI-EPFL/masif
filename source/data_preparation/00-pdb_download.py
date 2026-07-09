#!/usr/bin/python
import Bio
from Bio.PDB import *
import sys
import importlib
import os

from default_config.masif_opts import masif_opts
# Local includes
from input_output.protonate import protonate

if len(sys.argv) <= 1:
    print("Usage: "+sys.argv[0]+" PDBID_A_B")
    print("A or B are the chains to include in this pdb.")
    sys.exit(1)

if not os.path.exists(masif_opts['raw_pdb_dir']):
    os.makedirs(masif_opts['raw_pdb_dir'])

if not os.path.exists(masif_opts['tmp_dir']):
    os.mkdir(masif_opts['tmp_dir'])

in_fields = sys.argv[1].split('_')
pdb_id = in_fields[0]

# Download pdb
# NOTE (revival): Biopython's PDBList.retrieve_pdb_file() relies on legacy wwPDB
# download paths that RCSB has retired. It now fails silently, leaving no file and
# breaking the whole pipeline downstream (see issue #85). Fetch the structure
# directly from the current, stable RCSB endpoint instead.
try:
    from urllib.request import urlretrieve  # Python 3
except ImportError:
    from urllib import urlretrieve          # Python 2
pdb_filename = os.path.join(masif_opts['tmp_dir'], pdb_id + ".pdb")
urlretrieve("https://files.rcsb.org/download/%s.pdb" % pdb_id.upper(), pdb_filename)

##### Protonate with reduce, if hydrogens included.
# - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
protonated_file = masif_opts['raw_pdb_dir']+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file
