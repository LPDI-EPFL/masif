# Pablo Gainza Cirauqui 2016
# This pymol function loads dot files into pymol.
from pymol import cmd, stored
from pymol.cgo import *
import os.path
import numpy as np

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


def load_dots(
    filename, color="white", name="ply", dotSize=0.2, lineSize=0.5, doStatistics=False
):
    lines = open(filename).readlines()
    lines = [line.rstrip() for line in lines]
    lines = [line.split(",") for line in lines]
    verts = [[float(x[0]), float(x[1]), float(x[2])] for x in lines]

    normals = None

    if len(lines[0]) > 3:
        # normal is the last column - draw it
        normals = [[float(x[3]), float(x[4]), float(x[5])] for x in lines]

    # Draw vertices
    obj = []

    for v_ix in range(len(verts)):
        colorToAdd = colorDict[color]
        vert = verts[v_ix]
        # Vertices
        obj.extend(colorToAdd)
        obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
    #    obj.append(END)
    name = "vert_" + filename
    group_names = name
    cmd.load_cgo(obj, name, 1.0)
    obj = []
    # Draw normals
    if normals is not None:
        colorToAdd = colorDict["white"]
        obj.extend([BEGIN, LINES])
        obj.extend([LINEWIDTH, 2.0])
        colorToAdd = colorDict[color]
        obj.extend(colorToAdd)
        for v_ix in range(len(verts)):
            vert1 = verts[v_ix]
            vert2 = np.array(verts[v_ix]) + np.array(normals[v_ix])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
        #        obj.append(END)
        name = "norm_" + filename
        group_names = name
        cmd.load_cgo(obj, name, 1.0)
    # Draw normals

