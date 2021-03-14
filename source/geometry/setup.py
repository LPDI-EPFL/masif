from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("fast_patches_cython2.pyx"), 
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize("theta0.pyx"), unraisable_tracebacks=True, 
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize("compute_fast_polar_coordinates.pyx"), 
    include_dirs=[numpy.get_include()]
)
