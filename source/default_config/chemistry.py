# chemistry.py: Chemical parameters for MaSIF.
# Pablo Gainza - LPDI STI EPFL 2018-2019
# Released under an Apache License 2.0

import numpy as np

# radii for atoms in explicit case.
radii = {}
radii["N"] = "1.540000"
radii["N"] = "1.540000"
radii["O"] = "1.400000"
radii["C"] = "1.740000"
radii["H"] = "1.200000"
radii["S"] = "1.800000"
radii["P"] = "1.800000"
radii["Z"] = "1.39"
radii["X"] = "0.770000"  ## Radii of CB or CA in disembodied case.
# This  polar hydrogen's names correspond to that of the program Reduce. 
polarHydrogens = {}
polarHydrogens["ALA"] = ["H"]
polarHydrogens["GLY"] = ["H"]
polarHydrogens["SER"] = ["H", "HG"]
polarHydrogens["THR"] = ["H", "HG1"]
polarHydrogens["LEU"] = ["H"]
polarHydrogens["ILE"] = ["H"]
polarHydrogens["VAL"] = ["H"]
polarHydrogens["ASN"] = ["H", "HD21", "HD22"]
polarHydrogens["GLN"] = ["H", "HE21", "HE22"]
polarHydrogens["ARG"] = ["H", "HH11", "HH12", "HH21", "HH22", "HE"]
polarHydrogens["HIS"] = ["H", "HD1", "HE2"]
polarHydrogens["TRP"] = ["H", "HE1"]
polarHydrogens["PHE"] = ["H"]
polarHydrogens["TYR"] = ["H", "HH"]
polarHydrogens["GLU"] = ["H"]
polarHydrogens["ASP"] = ["H"]
polarHydrogens["LYS"] = ["H", "HZ1", "HZ2", "HZ3"]
polarHydrogens["PRO"] = []
polarHydrogens["CYS"] = ["H"]
polarHydrogens["MET"] = ["H"]

hbond_std_dev = np.pi / 3

# Dictionary from an acceptor atom to its directly bonded atom on which to
# compute the angle.
acceptorAngleAtom = {}
acceptorAngleAtom["O"] = "C"
acceptorAngleAtom["O1"] = "C"
acceptorAngleAtom["O2"] = "C"
acceptorAngleAtom["OXT"] = "C"
acceptorAngleAtom["OT1"] = "C"
acceptorAngleAtom["OT2"] = "C"
# Dictionary from acceptor atom to a third atom on which to compute the plane.
acceptorPlaneAtom = {}
acceptorPlaneAtom["O"] = "CA"
# Dictionary from an H atom to its donor atom.
donorAtom = {}
donorAtom["H"] = "N"
# Hydrogen bond information.
# ARG
# ARG NHX
# Angle: NH1, HH1X, point and NH2, HH2X, point 180 degrees.
# radii from HH: radii[H]
# ARG NE
# Angle: ~ 120 NE, HE, point, 180 degrees
donorAtom["HH11"] = "NH1"
donorAtom["HH12"] = "NH1"
donorAtom["HH21"] = "NH2"
donorAtom["HH22"] = "NH2"
donorAtom["HE"] = "NE"

# ASN
# Angle ND2,HD2X: 180
# Plane: CG,ND2,OD1
# Angle CG-OD1-X: 120
donorAtom["HD21"] = "ND2"
donorAtom["HD22"] = "ND2"
# ASN Acceptor
acceptorAngleAtom["OD1"] = "CG"
acceptorPlaneAtom["OD1"] = "CB"

# ASP
# Plane: CB-CG-OD1
# Angle CG-ODX-point: 120
acceptorAngleAtom["OD2"] = "CG"
acceptorPlaneAtom["OD2"] = "CB"

# GLU
# PLANE: CD-OE1-OE2
# ANGLE: CD-OEX: 120
# GLN
# PLANE: CD-OE1-NE2
# Angle NE2,HE2X: 180
# ANGLE: CD-OE1: 120
donorAtom["HE21"] = "NE2"
donorAtom["HE22"] = "NE2"
acceptorAngleAtom["OE1"] = "CD"
acceptorAngleAtom["OE2"] = "CD"
acceptorPlaneAtom["OE1"] = "CG"
acceptorPlaneAtom["OE2"] = "CG"

# HIS Acceptors: ND1, NE2
# Plane ND1-CE1-NE2
# Angle: ND1-CE1 : 125.5
# Angle: NE2-CE1 : 125.5
acceptorAngleAtom["ND1"] = "CE1"
acceptorAngleAtom["NE2"] = "CE1"
acceptorPlaneAtom["ND1"] = "NE2"
acceptorPlaneAtom["NE2"] = "ND1"

# HIS Donors: ND1, NE2
# Angle ND1-HD1 : 180
# Angle NE2-HE2 : 180
donorAtom["HD1"] = "ND1"
donorAtom["HE2"] = "NE2"

# TRP Donor: NE1-HE1
# Angle NE1-HE1 : 180
donorAtom["HE1"] = "NE1"

# LYS Donor NZ-HZX
# Angle NZ-HZX : 180
donorAtom["HZ1"] = "NZ"
donorAtom["HZ2"] = "NZ"
donorAtom["HZ3"] = "NZ"

# TYR acceptor OH
# Plane: CE1-CZ-OH
# Angle: CZ-OH 120
acceptorAngleAtom["OH"] = "CZ"
acceptorPlaneAtom["OH"] = "CE1"

# TYR donor: OH-HH
# Angle: OH-HH 180
donorAtom["HH"] = "OH"
acceptorPlaneAtom["OH"] = "CE1"

# SER acceptor:
# Angle CB-OG-X: 120
acceptorAngleAtom["OG"] = "CB"

# SER donor:
# Angle: OG-HG-X: 180
donorAtom["HG"] = "OG"

# THR acceptor:
# Angle: CB-OG1-X: 120
acceptorAngleAtom["OG1"] = "CB"

# THR donor:
# Angle: OG1-HG1-X: 180
donorAtom["HG1"] = "OG1"

