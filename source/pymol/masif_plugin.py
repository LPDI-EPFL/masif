# Pablo Gainza Cirauqui 2016 LPDI IBI STI EPFL
# This pymol plugin for Masif just enables the load ply functions. 

import tkSimpleDialog
import tkMessageBox
import Tkinter
import Pmw
from pymol import cmd
from loadPLY import *
import sys, urllib2, zlib

cmd.extend('loadply', load_ply)
cmd.extend('loadgiface', load_giface)
