import os
import numpy as np
import sys

from SBI.structure import PDB
from default_config.masif_opts import masif_opts

in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]

if not os.path.exists(masif_opts["ligand"]["ligand_coords_dir"]):
    os.mkdir(masif_opts["ligand"]["ligand_coords_dir"])

# Ligands of interest
ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]

structure_ligands_type = []
structure_ligands_coords = []
try:
    structure = PDB(
        os.path.join(masif_opts["ligand"]["assembly_dir"], "{}.pdb".format(pdb_id))
    )
except:
    print("Problem with opening structure", pdb)
for chain in structure.chains:
    for het in chain.heteroatoms:
        # Check all ligands in structure and save coordinates if they are of interest
        if het.type in ligands:
            structure_ligands_type.append(het.type)
            structure_ligands_coords.append(het.all_coordinates)

np.save(
    os.path.join(
        masif_opts["ligand"]["ligand_coords_dir"], "{}_ligand_types.npy".format(pdb_id)
    ),
    structure_ligands_type,
)
np.save(
    os.path.join(
        masif_opts["ligand"]["ligand_coords_dir"], "{}_ligand_coords.npy".format(pdb_id)
    ),
    structure_ligands_coords,
)
