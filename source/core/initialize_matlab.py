import matlab.engine
import os


def initialize_matlab():
    eng = matlab.engine.start_matlab()
    matlab_root = os.environ["masif_matlab"]
    eng.addpath(matlab_root, nargout=0)
    eng.addpath(matlab_root + "/fmm/", nargout=0)
    eng.addpath(matlab_root + "/util/", nargout=0)
    eng.addpath(matlab_root + "/coords/", nargout=0)
    return eng

