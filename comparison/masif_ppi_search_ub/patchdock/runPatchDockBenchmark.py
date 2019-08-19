import numpy as np
import sys
import time
import shutil
import subprocess
import os
import pyflann
from IPython.core.debugger import set_trace
from Bio.PDB import *

struct_dir = "/home/gainza/lpdi_fs/masif_paper/masif/data/masif_ppi_search_ub/data_preparation/01-benchmark_pdbs/"
benchmark_list_fn = "../benchmark_list.txt"
# Precomputation dir for masif. The location of the target vertex is extracted from here.
precomp_dir = (
    "../../../data/masif_ppi_search_ub/data_preparation/04b-precomputation_12A/precomputation"
)
# Set location of your patchdock binarires here.
#pd_bin = "/your/path/to/PatchDock/patch_dock.Linux"
#trans_output_bin = "/your/path/to/PatchDock/transOutput.pl"
pd_bin = "/home/gainza/lpdi_fs/seeder/data/ppi_benchmark_complexes/10-patchdock/PatchDock/patch_dock.Linux"
trans_output_bin = "/home/gainza/lpdi_fs/seeder/data/ppi_benchmark_complexes/10-patchdock/PatchDock/transOutput.pl"

# benchmark surfaces.
benchmark_list = open(benchmark_list_fn).readlines()
benchmark_pdbs_source = [x.rstrip() for x in benchmark_list]
benchmark_pdbs_source = benchmark_pdbs_source
benchmark_pdbs_target = [sys.argv[1]]

for target_pdb in benchmark_pdbs_target:
    # Load sc_labels.
    sc_labels = np.load(os.path.join(precomp_dir, target_pdb, "p1_sc_labels.npy"))
    center_point = np.argmax(np.median(np.nan_to_num(sc_labels[0]), axis=1))
    val = np.max(np.median(np.nan_to_num(sc_labels[0]), axis=1))

    center_point = np.argmax(np.median(np.nan_to_num(sc_labels[0]), axis=1))
    print(center_point)
    print(val)

    # Read the the point in the surface with the highest sc
    # Load vertices.
    X = np.load(os.path.join(precomp_dir, target_pdb, "p1_X.npy"))
    Y = np.load(os.path.join(precomp_dir, target_pdb, "p1_Y.npy"))
    Z = np.load(os.path.join(precomp_dir, target_pdb, "p1_Z.npy"))
    v_t = np.stack([X, Y, Z]).T

    pdbid = target_pdb.split("_")[0]
    t_chain = target_pdb.split("_")[1]

    # Load the pdb structure.
    parser = PDBParser()
    pdb_file_t = os.path.join(struct_dir, "{}_{}.pdb".format(pdbid, t_chain))

    target_struct = parser.get_structure(
        "{}_{}".format(pdbid, t_chain),
        os.path.join(struct_dir, "{}_{}.pdb".format(pdbid, t_chain)),
    )
    target_atom_coords = [atom.get_coord() for atom in target_struct.get_atoms()]
    target_atoms = [atom for atom in target_struct.get_atoms()]

    # Find the atom coords closest to the center point.
    dists = np.sqrt(np.sum(np.square(target_atom_coords - v_t[center_point]), axis=1))
    closest_at_ix = np.argmin(dists)
    target_res_id = target_atoms[closest_at_ix].get_parent().get_id()[1]
    target_chain_id = target_atoms[closest_at_ix].get_parent().get_parent().get_id()

    # For a few PDBs this is giving issues as these are not exposed according to PatchDock, so I use the second closest one.
    if target_pdb == "1t6b_X_Y":
        wrong_res_id = target_atoms[closest_at_ix].get_parent().get_id()[1]
        print(wrong_res_id)
        i = 1
        while target_res_id == wrong_res_id:
            closest_at_ix = np.argsort(dists)[i]
            target_res_id = target_atoms[closest_at_ix].get_parent().get_id()[1]
            print(target_res_id)
            i = i + 1

    outdir = os.path.join("run/", target_pdb)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    as_out_file = open(os.path.join(outdir, "site_p1.txt"), "w+")
    as_out_file.write("{} {}\n".format(target_res_id, target_chain_id))
    as_out_file.close()

    total_time = 0.0
    for source_pdb in benchmark_pdbs_source:

        s_pdbid = source_pdb.split("_")[0]
        s_chain = source_pdb.split("_")[2]
        pdb_file_s = os.path.join(struct_dir, "{}_{}.pdb".format(s_pdbid, s_chain))

        params_in_file = open("params_templ.txt", "r")
        params_out_fn = os.path.join(outdir, "params.txt")
        params_out_file = open(params_out_fn, "w+")

        for line in params_in_file:
            outline = line
            if "XXXX" in line:
                outline = "receptorPdb {}\n".format(pdb_file_t)
            elif "YYYY" in line:
                outline = "ligandPdb {}\n".format(pdb_file_s)
            params_out_file.write(outline)
        params_out_file.close()

        # Run patchdock. Time it (do not include the time to generate the PDBs)
        os.path.dirname(outdir)
        pd_out_fn = os.path.join(source_pdb)
        print([pd_bin, "params.txt", pd_out_fn])
        process = subprocess.Popen(
            ['/usr/bin/time', '-f', '%U', pd_bin, "params.txt", pd_out_fn],
            cwd=outdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        stderr_lines = stderr.splitlines()
        cpu_time = float(stderr_lines[-1])
        print("PatchDock took {:.3f}s".format(cpu_time))
        total_time += cpu_time
    # Save all runing times to a file called total_times.txt
    total_time_file = open("total_times.txt", "a+")
    total_time_file.write("{} {:.2f}\n".format(target_pdb, total_time))

