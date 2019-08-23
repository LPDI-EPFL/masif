from pymol import cmd, stored
import sys
import os, math, re
from pymol.cgo import *
import os.path
import numpy as np
"""
   loadPLY.py: This pymol function loads ply files into pymol.
    Pablo Gainza - LPDI STI EPFL 2016-2019
    This file is part of MaSIF.
    Released under an Apache License 2.0
"""
colorDict = {
    "sky": [COLOR, 0.0, 0.76, 1.0],
    "sea": [COLOR, 0.0, 0.90, 0.5],
    "yellowtint": [COLOR, 0.88, 0.97, 0.02],
    "hotpink": [COLOR, 0.90, 0.40, 0.70],
    "greentint": [COLOR, 0.50, 0.90, 0.40],
    "blue": [COLOR, 0.0, 0.0, 1.0],
    "green": [COLOR, 0.0, 1.0, 0.0],
    "yellow": [COLOR, 1.0, 1.0, 0.0],
    "orange": [COLOR, 1.0, 0.5, 0.0],
    "red": [COLOR, 1.0, 0.0, 0.0],
    "black": [COLOR, 0.0, 0.0, 0.0],
    "white": [COLOR, 1.0, 1.0, 1.0],
    "gray": [COLOR, 0.9, 0.9, 0.9],
}

# Create a gradient color from color 1 to whitish, to color 2. val goes from 0 (color1) to 1 (color2).
def color_gradient(vals, color1, color2):
    c1 = Color("white")
    c2 = Color("orange")
    ix = np.floor(vals * 100).astype(int)
    crange = list(c1.range_to(c2, 100))
    mycolor = []
    print(crange[0].get_rgb())
    for x in ix:
        myc = crange[x].get_rgb()
        mycolor.append([COLOR, myc[0], myc[1], myc[2]])
    return mycolor


def iface_color(iface):
    # max value is 1, min values is 0
    hp = iface.copy()
    hp = hp * 2 - 1
    mycolor = charge_color(-hp)
    return mycolor


# Returns the color of each vertex according to the charge.
# The most purple colors are the most hydrophilic values, and the most
# white colors are the most positive colors.
def hphob_color(hphob):
    # max value is 4.5, min values is -4.5
    hp = hphob.copy()
    # normalize
    hp = hp + 4.5
    hp = hp / 9.0
    # mycolor = [ [COLOR, 1.0, hp[i], 1.0]  for i in range(len(hp)) ]
    mycolor = [[COLOR, 1.0, 1.0 - hp[i], 1.0] for i in range(len(hp))]
    return mycolor


# Returns the color of each vertex according to the charge.
# The most red colors are the most negative values, and the most
# blue colors are the most positive colors.
def charge_color(charges):
    # Assume a std deviation equal for all proteins....
    max_val = 1.0
    min_val = -1.0

    norm_charges = charges
    blue_charges = np.array(norm_charges)
    red_charges = np.array(norm_charges)
    blue_charges[blue_charges < 0] = 0
    red_charges[red_charges > 0] = 0
    red_charges = abs(red_charges)
    red_charges[red_charges > max_val] = max_val
    blue_charges[blue_charges < min_val] = min_val
    red_charges = red_charges / max_val
    blue_charges = blue_charges / max_val
    # red_charges[red_charges>1.0] = 1.0
    # blue_charges[blue_charges>1.0] = 1.0
    green_color = np.array([0.0] * len(charges))
    mycolor = [
        [
            COLOR,
            0.9999 - blue_charges[i],
            0.9999 - (blue_charges[i] + red_charges[i]),
            0.9999 - red_charges[i],
        ]
        for i in range(len(charges))
    ]
    for i in range(len(mycolor)):
        for k in range(1, 4):
            if mycolor[i][k] < 0:
                mycolor[i][k] = 0

    return mycolor


def load_ply(
    filename, color="white", name="ply", dotSize=0.2, lineSize=0.5, doStatistics=False
):
    ## Pymesh should be faster and supports binary ply files. However it is difficult to install with pymol...
    #        import pymesh
    #        mesh = pymesh.load_mesh(filename)

    from simple_mesh import Simple_mesh

    mesh = Simple_mesh()
    mesh.load_mesh(filename)

    ignore_normal = False
    with_normal = False
    with_color = False

    group_names = ""

    verts = mesh.vertices
    try:
        charge = mesh.get_attribute("vertex_charge")
        color_array = charge_color(charge)
    except:
        print("Could not load vertex charges.")
        color_array = [colorDict["green"]] * len(verts)
    if "vertex_nx" in mesh.get_attribute_names():
        nx = mesh.get_attribute("vertex_nx")
        ny = mesh.get_attribute("vertex_ny")
        nz = mesh.get_attribute("vertex_nz")
        normals = np.vstack([nx, ny, nz]).T
        print(normals.shape)

    # Draw vertices
    obj = []
    color = "green"

    for v_ix in range(len(verts)):
        vert = verts[v_ix]
        colorToAdd = color_array[v_ix]
        # Vertices
        obj.extend(colorToAdd)
        obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])

    name = "vert_" + filename
    group_names = name
    cmd.load_cgo(obj, name, 1.0)
    obj = []

    faces = mesh.faces

    # Draw surface charges.
    if (
        "vertex_charge" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        color_array_surf = color_array
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "pb_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []
    # Draw hydrophobicity
    if (
        "vertex_hphob" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        hphob = mesh.get_attribute("vertex_hphob")
        color_array_surf = hphob_color(hphob)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "hphobic_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []
    # Draw shape index
    if (
        "vertex_si" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        si = mesh.get_attribute("vertex_si")
        color_array_surf = charge_color(si)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "si_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []
    # Draw shape index
    if (
        "vertex_si" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        si = mesh.get_attribute("vertex_si")
        color_array_surf = charge_color(si)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "si_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []

    obj = []
    # Draw ddc
    if (
        "vertex_ddc" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        ddc = mesh.get_attribute("vertex_ddc")
        # Scale to -1.0->1.0
        ddc = ddc * 1.4285
        color_array_surf = charge_color(ddc)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "ddc_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []

    # Draw iface
    if (
        "vertex_iface" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        iface = mesh.get_attribute("vertex_iface")
        color_array_surf = iface_color(iface)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "iface_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []
    # Draw hbond
    if (
        "vertex_hbond" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        hbond = mesh.get_attribute("vertex_hbond")
        color_array_surf = charge_color(hbond)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.6])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "hbond_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    # Draw triangles (faces)
    for tri in faces:
        pairs = [[tri[0], tri[1]], [tri[0], tri[2]], [tri[1], tri[2]]]
        colorToAdd = colorDict["gray"]
        for pair in pairs:
            vert1 = verts[pair[0]]
            vert2 = verts[pair[1]]
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
    name = "mesh_" + filename
    cmd.load_cgo(obj, name, 1.0)
    group_names = group_names + " " + name

    # Draw normals
    if with_normal and not ignore_normal:
        for v_ix in range(len(verts)):
            colorToAdd = colorDict["white"]
            vert1 = verts[v_ix]
            vert2 = [
                verts[v_ix][0] + nx[v_ix],
                verts[v_ix][1] + ny[v_ix],
                verts[v_ix][2] + nz[v_ix],
            ]
            obj.extend([LINEWIDTH, 2.0])
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
        cmd.load_cgo(obj, "normal_" + filename, 1.0)

    print(group_names)
    cmd.group(filename, group_names)


# Load the sillouete of an iface.
def load_giface(filename, color="white", name="giface", dotSize=0.2, lineSize=1.0):
    mesh = pymesh.load_mesh(filename)
    if "vertex_iface" not in mesh.get_attribute_names():
        return
    iface = mesh.get_attribute("vertex_iface")
    # Color an edge only if:
    # iface > 0 for its two edges
    # iface is zero for at least one of its edges.
    # Go through each face.
    faces = mesh.faces
    verts = mesh.vertices
    obj = []
    visited = set()
    colorToAdd = colorDict["green"]
    obj.extend([BEGIN, LINES])
    obj.extend([LINEWIDTH, 5.0])
    obj.extend(colorToAdd)
    for tri in faces:
        pairs = [
            [tri[0], tri[1], tri[2]],
            [tri[0], tri[2], tri[1]],
            [tri[1], tri[2], tri[0]],
        ]
        for pair in pairs:
            if iface[pair[0]] > 0 and iface[pair[1]] > 0 and iface[pair[2]] == 0:
                vert1 = verts[pair[0]]
                vert2 = verts[pair[1]]

                obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
                obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
    obj.append(END)
    name = "giface_" + filename
    cmd.load_cgo(obj, name, 1.0)
    colorToAdd = colorDict["green"]

    obj = []
    obj.extend(colorToAdd)
    for tri in faces:
        pairs = [
            [tri[0], tri[1], tri[2]],
            [tri[0], tri[2], tri[1]],
            [tri[1], tri[2], tri[0]],
        ]
        for pair in pairs:
            if iface[pair[0]] > 0 and iface[pair[1]] > 0 and iface[pair[2]] == 0:
                vert1 = verts[pair[0]]
                vert2 = verts[pair[1]]

                obj.extend([SPHERE, (vert1[0]), (vert1[1]), (vert1[2]), 0.4])
                obj.extend([SPHERE, (vert2[0]), (vert2[1]), (vert2[2]), 0.4])
    # obj.append(END)
    name = "giface_verts_" + filename
    cmd.load_cgo(obj, name, 1.0)

