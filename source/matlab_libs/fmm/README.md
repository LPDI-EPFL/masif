# source/matlab_libs/fmm/

+ fast_marching.m: Wrapper function to call fast marching. 
+ *mex*: Mex files for fast marching for linux, windows and mac.

Contains code for the fast marching method (to approximate geodesic distances much better than Dijkstra). 
Since it is much slower than Dijkstra, this code is currently only used to compute the shape complementarity 
of patches, and will be removed in the near future from MaSIF.
