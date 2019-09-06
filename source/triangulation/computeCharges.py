from Bio.PDB import *
import numpy as np
from sklearn.neighbors import KDTree

"""
computeCharges.py: Wrapper function to compute hydrogen bond potential (free electrons/protons) in the surface
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

from default_config.chemistry import (
    polarHydrogens,
    radii,
    acceptorAngleAtom,
    acceptorPlaneAtom,
    hbond_std_dev,
    donorAtom,
)

# Compute vertex charges based on hydrogen bond potential.
# pdb_filename: The filename of the protonated protein.
# vertices: The surface vertices of the protonated protein
# The name of each vertex in the format, example: B_125_x_ASN_ND2_Green
# where B is chain, 125 res id, x the insertion, ASN aatype, ND2 the name of the
# atom, and green is not used anymore.
def computeCharges(pdb_filename, vertices, names):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_filename, pdb_filename + ".pdb")
    residues = {}
    for res in struct.get_residues():
        chain_id = res.get_parent().get_id()
        if chain_id == "":
            chain_id = " "
        residues[(chain_id, res.get_id())] = res

    atoms = Selection.unfold_entities(struct, "A")
    satisfied_CO, satisfied_HN = computeSatisfied_CO_HN(atoms)

    charge = np.array([0.0] * len(vertices))
    # Go over every vertex
    for ix, name in enumerate(names):
        fields = name.split("_")
        chain_id = fields[0]
        if chain_id == "":
            chain_id = " "
        if fields[2] == "x":
            fields[2] = " "
        res_id = (" ", int(fields[1]), fields[2])
        aa = fields[3]
        atom_name = fields[4]
        # Ignore atom if it is BB and it is already satisfied.
        if atom_name == "H" and res_id in satisfied_HN:
            continue
        if atom_name == "O" and res_id in satisfied_CO:
            continue
        # Compute the charge of the vertex
        charge[ix] = computeChargeHelper(
            atom_name, residues[(chain_id, res_id)], vertices[ix]
        )

    return charge


# Compute the charge of a vertex in a residue.
def computeChargeHelper(atom_name, res, v):
    res_type = res.get_resname()
    # Check if it is a polar hydrogen.
    if isPolarHydrogen(atom_name, res):
        donor_atom_name = donorAtom[atom_name]
        a = res[donor_atom_name].get_coord()  # N/O
        b = res[atom_name].get_coord()  # H
        # Donor-H is always 180.0 degrees, = pi
        angle_deviation = computeAngleDeviation(a, b, v, np.pi)
        angle_penalty = computeAnglePenalty(angle_deviation)
        return 1.0 * angle_penalty
    # Check if it is an acceptor oxygen or nitrogen
    elif isAcceptorAtom(atom_name, res):
        acceptor_atom = res[atom_name]
        b = acceptor_atom.get_coord()
        a = res[acceptorAngleAtom[atom_name]].get_coord()
        # 120 degress for acceptor
        angle_deviation = computeAngleDeviation(a, b, v, 2 * np.pi / 3)
        # TODO: This should not be 120 for all atoms, i.e. for HIS it should be
        #       ~125.0
        angle_penalty = computeAnglePenalty(angle_deviation)
        plane_penalty = 1.0
        if atom_name in acceptorPlaneAtom:
            try:
                d = res[acceptorPlaneAtom[atom_name]].get_coord()
            except:
                return 0.0
            plane_deviation = computePlaneDeviation(d, a, b, v)
            plane_penalty = computeAnglePenalty(plane_deviation)
        return -1.0 * angle_penalty * plane_penalty
        # Compute the
    return 0.0


# Compute the absolute value of the deviation from theta
def computeAngleDeviation(a, b, c, theta):
    return abs(calc_angle(Vector(a), Vector(b), Vector(c)) - theta)


# Compute the angle deviation from a plane
def computePlaneDeviation(a, b, c, d):
    dih = calc_dihedral(Vector(a), Vector(b), Vector(c), Vector(d))
    dev1 = abs(dih)
    dev2 = np.pi - abs(dih)
    return min(dev1, dev2)


# angle_deviation from ideal value. TODO: do a more data-based solution
def computeAnglePenalty(angle_deviation):
    # Standard deviation: hbond_std_dev
    return max(0.0, 1.0 - (angle_deviation / (hbond_std_dev)) ** 2)


def isPolarHydrogen(atom_name, res):
    if atom_name in polarHydrogens[res.get_resname()]:
        return True
    else:
        return False


def isAcceptorAtom(atom_name, res):
    if atom_name.startswith("O"):
        return True
    else:
        if res.get_resname() == "HIS":
            if atom_name == "ND1" and "HD1" not in res:
                return True
            if atom_name == "NE2" and "HE2" not in res:
                return True
    return False


# Compute the list of backbone C=O:H-N that are satisfied. These will be ignored.
def computeSatisfied_CO_HN(atoms):
    ns = NeighborSearch(atoms)
    satisfied_CO = set()
    satisfied_HN = set()
    for atom1 in atoms:
        res1 = atom1.get_parent()
        if atom1.get_id() == "O":
            neigh_atoms = ns.search(atom1.get_coord(), 2.5, level="A")
            for atom2 in neigh_atoms:
                if atom2.get_id() == "H":
                    res2 = atom2.get_parent()
                    # Ensure they belong to different residues.
                    if res2.get_id() != res1.get_id():
                        # Compute the angle N-H:O, ideal value is 180 (but in
                        # helices it is typically 160) 180 +-30 = pi
                        angle_N_H_O_dev = computeAngleDeviation(
                            res2["N"].get_coord(),
                            atom2.get_coord(),
                            atom1.get_coord(),
                            np.pi,
                        )
                        # Compute angle H:O=C, ideal value is ~160 +- 20 = 8*pi/9
                        angle_H_O_C_dev = computeAngleDeviation(
                            atom2.get_coord(),
                            atom1.get_coord(),
                            res1["C"].get_coord(),
                            8 * np.pi / 9,
                        )
                        ## Allowed deviations: 30 degrees (pi/6) and 20 degrees
                        #       (pi/9)
                        if (
                            angle_N_H_O_dev - np.pi / 6 < 0
                            and angle_H_O_C_dev - np.pi / 9 < 0.0
                        ):
                            satisfied_CO.add(res1.get_id())
                            satisfied_HN.add(res2.get_id())
    return satisfied_CO, satisfied_HN


# Compute the charge of a new mesh, based on the charge of an old mesh.
# Use the top vertex in distance, for now (later this should be smoothed over 3
# or 4 vertices)
def assignChargesToNewMesh(new_vertices, old_vertices, old_charges, seeder_opts):
    dataset = old_vertices
    testset = new_vertices
    new_charges = np.zeros(len(new_vertices))
    if seeder_opts["feature_interpolation"]:
        num_inter = 4  # Number of interpolation features
        # Assign k old vertices to each new vertex.
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset, k=num_inter)
        # Square the distances (as in the original pyflann)
        dists = np.square(dists)
        # The size of result is the same as new_vertices
        for vi_new in range(len(result)):
            vi_old = result[vi_new]
            dist_old = dists[vi_new]
            # If one vertex is right on top, ignore the rest.
            if dist_old[0] == 0.0:
                new_charges[vi_new] = old_charges[vi_old[0]]
                continue

            total_dist = np.sum(1 / dist_old)
            for i in range(num_inter):
                new_charges[vi_new] += (
                    old_charges[vi_old[i]] * (1 / dist_old[i]) / total_dist
                )
    else:
        # Assign k old vertices to each new vertex.
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset)
        new_charges = old_charges[result]
    return new_charges

