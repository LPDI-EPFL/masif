#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from IPython.core.debugger import set_trace

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractHelix import extractHelix
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal

# Compute all helices in the PDB chain.
def computeHelices(pdbid, pdbfilename, chain_id):
    # Minimum number of residues to be considered a helix: 10
    min_helix_res = 10
    p = PDBParser()
    structure = p.get_structure(pdbid, pdbfilename)
    model = structure[0]
    # Compute secondary structure of all residues using DSSP
    dssp = DSSP(model, pdbfilename)
    chain_helices = []
    cur_helix = []
    chain = model[chain_id]
    for res in chain:
        # Ignore het atoms.
        if res.get_id()[0] != ' ':
            continue
        mytuple = (chain.get_id(), res.get_id())
        ss = dssp[mytuple][2]
        # If residue is helix and residue is consecutive with respect to the previous one.
        if ss == 'H' and (prev_res.get_id()[1] - res.get_id()[1]) == -1:
            cur_helix.append(res.get_id())
        else:
            if len(cur_helix) > min_helix_res:
                chain_helices.append(cur_helix)
            cur_helix = []

        prev_res = res
    return chain_helices

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)


# Save the chains as separate files. 
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
tmp_dir= masif_opts['tmp_dir']

protonated_file = tmp_dir+"/"+pdb_id+".pdb"
print("Protonating file {}".format(protonated_file))
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# Extract all helices from chain. 
for chain_id in list(chain_ids1):
    helices = computeHelices(pdb_id, pdb_filename, chain_id)
    if len(helices) == 0:
        print("No helices were found in PDB structure.")
    else: 
        print("Extracting helixces for {} chain {}".format(pdb_filename, chain_id))
    for ix, helix in enumerate(helices): 
        out_filename1 = "{}/{}{:03d}_{}".format(tmp_dir,pdb_id,ix,chain_id)
        extractHelix(helix, pdb_filename, out_filename1+".pdb", chain_id)

        # Compute MSMS of surface w/hydrogens, 
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+".pdb",\
            protonate=True)

        # Compute "charged" vertices
        if masif_opts['use_hbond']:
            vertex_hbond = computeCharges(out_filename1, vertices1, names1)

        # For each surface residue, assign the hydrophobicity of its amino acid. 
        if masif_opts['use_hphob']:
            vertex_hphobicity = computeHydrophobicity(names1)

        # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
        vertices2 = vertices1
        faces2 = faces1

        # Fix the mesh.
        mesh = pymesh.form_mesh(vertices2, faces2)
        regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

        # Compute the normals
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
        # Assign charges on new vertices based on charges of old vertices (nearest
        # neighbor)

        if masif_opts['use_hbond']:
            vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                vertex_hbond, masif_opts)

        if masif_opts['use_hphob']:
            vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                vertex_hphobicity, masif_opts)

        if masif_opts['use_apbs']:
            vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)

        iface = np.zeros(len(regular_mesh.vertices))
        # Convert to ply and save.
        save_ply(out_filename1+".ply", regular_mesh.vertices,\
                                regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                                normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity, iface=iface)
        if not os.path.exists(masif_opts['ply_chain_dir']):
            os.makedirs(masif_opts['ply_chain_dir'])
        if not os.path.exists(masif_opts['pdb_chain_dir']):
            os.makedirs(masif_opts['pdb_chain_dir'])
        shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
        shutil.copy(out_filename1+'.pdb', masif_opts['pdb_chain_dir']) 
