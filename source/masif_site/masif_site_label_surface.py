import pymesh
import os
import sys
import importlib
import numpy as np
import ipdb
from default_config.masif_opts import masif_opts

from masif_modules.read_data_from_matfile import load_matlab_file

"""
masif_site_label_surface.py: Color a protein ply surface file by the MaSIF-site interface score.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""


params = masif_opts["site"]
custom_params_file = sys.argv[1]
custom_params = importlib.import_module(custom_params_file, package=None)
custom_params = custom_params.custom_params

for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]

# Shape precomputation dir.
parent_in_dir = params["masif_precomputation_dir"]
eval_list = []

if len(sys.argv) == 3:
    # eval_list = [sys.argv[2].rstrip('_')]
    ppi_pair_ids = [sys.argv[2]]
# Read a list of pdb_chain entries to evaluate.
elif len(sys.argv) == 4 and sys.argv[2] == "-l":
    listfile = open(sys.argv[3])
    ppi_pair_ids = []
    for line in listfile:
        eval_list.append(line.rstrip())
    for mydir in os.listdir(parent_in_dir):
        ppi_pair_ids.append(mydir)
else:
    print("Not enough parameters")
    sys.exit(1)

for ppi_pair_id in ppi_pair_ids:
    shape_file = masif_opts["mat_dir"] + ppi_pair_id + "/" + ppi_pair_id + ".mat"
    pdbid = ppi_pair_id.split("_")[0]
    chains = [ppi_pair_id.split("_")[1], ppi_pair_id.split("_")[2]]

    for ix, pid in enumerate(["p1", "p2"]):
        pdb_chain_id = pdbid + "_" + chains[ix]

        if (
            pdb_chain_id not in eval_list
            and pdb_chain_id + "_" not in eval_list
            and len(eval_list) > 0
        ):
            continue

        try:
            p1 = load_matlab_file(shape_file, pid, True)
        except:
            print("File does not exist: {}".format(shape_file))
            continue
        try:
            scores = np.load(
                params["out_pred_dir"] + "/pred_" + pdbid + "_" + chains[ix] + ".npy"
            )
        except:
            print(
                "File does not exist: {}".format(
                    params["out_pred_dir"]
                    + "/pred_"
                    + pdbid
                    + "_"
                    + chains[ix]
                    + ".npy"
                )
            )
            continue

        vertices = np.stack([p1["X"][0], p1["Y"][0], p1["Z"][0]], axis=1)
        faces = p1["TRIV"].T
        # Matlab indexing starts at 1.
        faces = faces - 1

        mymesh = pymesh.form_mesh(vertices, faces)
        mymesh.add_attribute("charge")
        mymesh.set_attribute("charge", p1["charge"][0])

        normals = p1["normal"]
        n1 = normals[0, :]
        n2 = normals[1, :]
        n3 = normals[2, :]
        mymesh.add_attribute("vertex_nx")
        mymesh.set_attribute("vertex_nx", n1)
        mymesh.add_attribute("vertex_ny")
        mymesh.set_attribute("vertex_ny", n2)
        mymesh.add_attribute("vertex_nz")
        mymesh.set_attribute("vertex_nz", n3)

        mymesh.add_attribute("hphob")
        mymesh.set_attribute("hphob", p1["hphob"][0])

        mymesh.add_attribute("hbond")
        mymesh.set_attribute("hbond", p1["hbond"][0])

        mymesh.add_attribute("iface")
        mymesh.set_attribute("iface", scores[0])

        if not os.path.exists(params["out_surf_dir"]):
            os.makedirs(params["out_surf_dir"])

        print("Saving " + params["out_surf_dir"] + pdb_chain_id + ".ply")
        pymesh.save_mesh(
            params["out_surf_dir"] + pdb_chain_id + ".ply",
            mymesh,
            *mymesh.get_attribute_names(),
            use_float=True,
            ascii=True
        )

