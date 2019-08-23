import pymesh
import numpy
"""
read_ply.py: Read a ply file from disk using pymesh and load the attributes used by MaSIF. 
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

def read_ply(filename):
    # Read a ply file from disk using pymesh and load the attributes used by MaSIF. 
    # filename: the input ply file. 
    # returns data as tuple.
    mesh = pymesh.load_mesh(filename)

    attributes = mesh.get_attribute_names()
    if "vertex_nx" in attributes:
        nx = mesh.get_attribute("vertex_nx")
        ny = mesh.get_attribute("vertex_ny")
        nz = mesh.get_attribute("vertex_nz")

        normals = numpy.column_stack((nx, ny, nz))
    else:
        normals = None
    if "vertex_charge" in attributes:
        charge = mesh.get_attribute("vertex_charge")
    else:
        charge = numpy.array([0.0] * len(mesh.vertices))

    if "vertex_cb" in attributes:
        vertex_cb = mesh.get_attribute("vertex_cb")
    else:
        vertex_cb = numpy.array([0.0] * len(mesh.vertices))

    if "vertex_hbond" in attributes:
        vertex_hbond = mesh.get_attribute("vertex_hbond")
    else:
        vertex_hbond = numpy.array([0.0] * len(mesh.vertices))

    if "vertex_hphob" in attributes:
        vertex_hphob = mesh.get_attribute("vertex_hphob")
    else:
        vertex_hphob = numpy.array([0.0] * len(mesh.vertices))

    return (
        mesh.vertices,
        mesh.faces,
        normals,
        charge,
        vertex_cb,
        vertex_hbond,
        vertex_hphob,
    )

