#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
import time

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
import pymesh
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.fixmesh import fix_mesh
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeHbonds, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from scipy.spatial import cKDTree

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)


start = time.perf_counter()
print('Starting to protonate')
# Save the chains as separate files. 
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
tmp_dir= masif_opts['tmp_dir']
protonated_file = tmp_dir+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file
end = time.perf_counter()
print('Protonation took: {:.2f}s'.format(end-start))

# Extract chains of interest.
start = time.perf_counter()
out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)
end = time.perf_counter()
print('Chain extraction took: {:.2f}s'.format(end-start))

# Compute MSMS of surface w/hydrogens, 
start = time.perf_counter()
vertices1, faces1, normals1, names1, _ = computeMSMS(out_filename1+".pdb")
end = time.perf_counter()
print('MSMS took: {:.2f}s'.format(end-start))

# Compute "charged" vertices
if masif_opts['use_hbond']:
    vertex_hbond = computeHbonds(out_filename1, vertices1, names1)

# For each surface residue, assign the hydrophobicity of its amino acid. 
if masif_opts['use_hphob']:
    vertex_hphobicity = computeHydrophobicity(names1)
print('Computing charges took: {:.2f}s'.format(end-start))


end = time.perf_counter()
# Fix the mesh.
start = time.perf_counter()
mesh = pymesh.form_mesh(vertices1, faces1)
mesh.add_attribute('vertex_area')
regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
end = time.perf_counter()
print('Mesh fixing took: {:.2f}s'.format(end-start))
    

start = time.perf_counter()
# Compute the normals
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
# Assign charges on new vertices based on charges of old vertices (nearest
# neighbor)
end = time.perf_counter()
print('Computing normals took: {:.2f}s'.format(end-start))

if masif_opts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hbond, masif_opts)

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hphobicity, masif_opts)

if masif_opts['use_apbs']:
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)

start = time.perf_counter()
# Compute the normals
iface = np.zeros(len(regular_mesh.vertices))
if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    # Compute the surface of the entire complex and from that compute the interface.
    # Use highres for this. 
    v3, f3, _, _, _ = computeMSMS(pdb_filename, res=3.0)
    full_regular_mesh = pymesh.form_mesh(v3, f3)
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = cKDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d) # Square d, because this is how it was in the pyflann version.
    assert(len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    end = time.perf_counter()
    print('Computing iface took: {:.2f}s'.format(end-start))
    # Convert to ply and save.
    start = time.perf_counter()
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                        iface=iface)
    end = time.perf_counter()
    print('Saving ply took: {:.2f}s'.format(end-start))

else:
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])
shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
shutil.copy(out_filename1+'.pdb', masif_opts['pdb_chain_dir']) 
print('Exiting')
