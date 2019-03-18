import pymesh
import numpy
def read_ply(filename):
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
        charge = numpy.array([0.0]*len(mesh.vertices))

    if "vertex_cb" in attributes:
        vertex_cb = mesh.get_attribute("vertex_cb")
    else:
        vertex_cb = numpy.array([0.0]*len(mesh.vertices))

    if "vertex_hbond" in attributes:
        vertex_hbond = mesh.get_attribute("vertex_hbond")
    else: 
        vertex_hbond = numpy.array([0.0]*len(mesh.vertices))

    if "vertex_hphob" in attributes:
        vertex_hphob = mesh.get_attribute("vertex_hphob")
    else: 
        vertex_hphob = numpy.array([0.0]*len(mesh.vertices))

    return mesh.vertices, mesh.faces, normals, charge, vertex_cb, vertex_hbond, vertex_hphob


