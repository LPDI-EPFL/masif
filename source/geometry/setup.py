from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("compute_fast_polar_coordinates.pyx"), 
    include_dirs=[numpy.get_include()]
)
