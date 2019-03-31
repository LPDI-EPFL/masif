# masif
## MaSIF- Molecular surface interaction fingerprints. Geometric deep learning to decipher patterns in molecular surfaces.

## About MaSIF

MaSIF is a proof-of-concept method for identifying patterns (fingerprints) in protein surfaces. It contains a protocol to prepare protein structure files into feature-rich surfaces, to decompose these into patches, and to identify patterns in these using deep geometric learning.

This code base is structured to reproduce the experiments found in: 

P. Gainza, F. Sverrisson, F Monti, E. Rodola, M. M. Bronstein, B.E. Correia. Deciphering interaction fingerprints from protein molecular surfaces. 2019. 

MaSIF it is meant to form as a base for any protein surface-oriented learning task. 

![MaSIF conceptual framework and method](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/Fig0_v11-01.png)

## Dependencies 
MaSIF has a long list of dependencies, as it relies on external software/libraries to  work with protein databank files and surface files, and to compute features and coordinates: 
* Python (>= 2.7)
* [reduce](http://kinemage.biochem.duke.edu/software/reduce.php) 
* [MSMS](http://mgltools.scripps.edu/packages/MSMS/)
* [biopython](https://github.com/biopython/biopython)
* [pyflann](https://github.com/primetang/pyflann)
* PDB2PQR, multivalue, and [APBS](http://www.poissonboltzmann.org/)
* [open3D](https://github.com/IntelVCL/Open3D)
* [matlab](https://ch.mathworks.com/products/matlab.html) 
* [Pymol](https://pymol.org/2/)(optional)
* [Tensorflow] (https://www.tensorflow.org/)
 
## Installation 

## Data preparation

## Training
