import os
import sys
from SBI.structure import PDB
from default_config.masif_opts import masif_opts

ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]

if not os.path.exists(masif_opts["ligand"]["assembly_dir"]):
    os.mkdir(masif_opts["ligand"]["assembly_dir"])


def assemble(pdb_id):
    # Reads and builds the biological assembly of a structure
    struct = PDB(
        os.path.join(masif_opts["raw_pdb_dir"], "{}.pdb".format(pdb_id)), header=True
    )
    try:
        struct_assembly = struct.apply_biomolecule_matrices()[0]
    except:
        return 0
    struct_assembly.write(
        os.path.join(masif_opts["ligand"]["assembly_dir"], "{}.pdb".format(pdb_id))
    )
    return 1


in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]

res = assemble(pdb_id)
if res:
    print("Building assembly was successfull for {}".format(pdb_id))
else:
    print("Building assembly was not successfull for {}".format(pdb_id))
