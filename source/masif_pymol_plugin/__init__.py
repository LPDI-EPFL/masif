# Pablo Gainza Cirauqui 2016 LPDI IBI STI EPFL
# This pymol plugin for Masif just enables the load ply functions. 

from pymol import cmd
from loadPLY import *
from loadDOTS import *
import sys

def __init_plugin__(app):
    cmd.extend('loadply', load_ply)
    cmd.extend('loaddots', load_dots)
    cmd.extend('loadgiface', load_giface)

