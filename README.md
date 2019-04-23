
![MaSIF banner and concept](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/Concept-01.png)

## MaSIF- Molecular Surface Interaction Fingerprints: Geometric deep learning to decipher patterns in molecular surfaces.

[![bioRxiv shield](https://img.shields.io/badge/bioRxiv-1709.01233-green.svg?style=flat)](https://www.biorxiv.org/content/10.1101/606202v1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2625420.svg)](https://doi.org/10.5281/zenodo.2625420)


## Table of Contents: 

- [Description](#description)
- [Software requirements](#software-requirements)
- [Installation](#Installation)
- [Data preparation](#Data-preparation)
- [MaSIF applications](#MaSIF-applications)
- [License](#License)
## Description

MaSIF is a proof-of-concept method to identify patterns (fingerprints)
in protein surfaces which may be important for specific biomolecular interactions. 
To achieve this, MaSIF exploits techniques from the field of geometric deep learning.
First, MaSIF decomposes a surface into overlapping radial patches with a fixed geodesic radius, wherein each 
point is assigned an array of geometric and chemical features. MaSIF then computes a descriptor 
for each surface patch, a vector that encodes a description of the features present in the patch. 
Then, this descriptor can be processed in a set of additional layers where different interactions 
can be classified. The features 
encoded in each descriptor and the final output depend on the application-specific training data and the 
optimization objective, meaning that the same architecture can be repurposed for various tasks.

This repository contains a protocol to prepare protein structure files into feature-rich surfaces 
(with both geometric and chemical features),
to decompose these into patches, and tensorflow-based neural network code
to identify patterns in these using deep geometric learning.
To show the potential of the approach, we showcase three proof-of-concept applications: 
a) ligand prediction for protein binding pockets (MaSIF-ligand); b) protein-protein interaction 
(PPI) site prediction in protein surfaces, to predict which surface patches on a protein are more 
likely to interact with other proteins (MaSIF-site); c) ultrafast scanning of surfaces, where we use 
surface fingerprints from binding partners to predict the structural configuration of protein-protein complexes (MaSIF-search). 

This repository reproduces the experiments of: 

Gainza, P., Sverrisson, F., Monti, F., Rodola, E., Bronstein, M. M., & Correia, B. E. (2019). Deciphering interaction fingerprints from protein molecular surfaces. bioRxiv, 606202.

MaSIF is distributed under an [Apache License](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/LICENSE). This 
code is meant to serve as a tutorial, and the basis for researchers to exploit MaSIF in protein surface-oriented learning tasks. 

## System requirements

MaSIF has been tested on both Linux (Red Hat Enterprise Linux Server release 7.4, with a Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz processesor and 16GB of memory allotment) 
and Mac OS environments (macOS High Sierra, processor 2.8 GHz Intel Core i7, 16GB memory).

Currently, MaSIF takes about 2 minutes to preprocess every protein. For this reason, we recommend a distributed cluster to 
preprocess the data for large datasets of proteins. Once data has been preprocessed, we strongly recommend using a GPU to 
train or evaluate the trained models as it can be up to 100 times faster than a CPU. 

## Software prerequisites 
MaSIF relies on external software/libraries to handle protein databank files and surface files, 
to compute chemical/geometric features and coordinates, and to perform neural network calculations. 
The following is the list of required libraries and programs, as well as the version on which it was tested (in parenthesis).
* Python (2.7)
* [reduce](http://kinemage.biochem.duke.edu/software/reduce.php) (3.23). To add protons to proteins. 
* [MSMS](http://mgltools.scripps.edu/packages/MSMS/) (2.6.1). To compute the surface of proteins. 
* [BioPython](https://github.com/biopython/biopython) (1.66) . To parse PDB files. 
* [PyMesh](https://github.com/PyMesh/PyMesh) (0.1.14). To handle ply surface files, attributes, and to regularize meshes.
* [pyflann](https://github.com/primetang/pyflann) (1.6.14). To perform nearest neighbor searches of vertices.
* PDB2PQR (2.1.1), multivalue, and [APBS](http://www.poissonboltzmann.org/) (1.5). These programs are necessary to compute electrostatics charges.
* [open3D](https://github.com/IntelVCL/Open3D) (0.5.0.0). Mainly used for RANSAC alignment.
* [matlab](https://ch.mathworks.com/products/matlab.html) (R2018a). Used to compute some geometric features and angular/radial coordinates.
* [Python bindings for matlab](https://www.mathworks.com/help/matlab/matlab_external/get-started-with-matlab-engine-for-python.html) - To call matlab functions from within Python.
* [Tensorflow](https://www.tensorflow.org/) (1.9). Use to model, train, and evaluate the actual neural networks. Models were trained and evaluated on a NVIDIA Tesla K40 GPU.
* [Pymol](https://pymol.org/2/). This optional plugin allows one to visualize surface files in PyMOL.
 
We are working to reduce this list of requirements for future versions.

## Installation 

After preinstalling dependencies, add the following environment variables to your path, changing the appropriate directories:

```
export APBS_BIN=/path/to/apbs/APBS-1.5-linux64/bin/apbs
export MULTIVALUE_BIN=/path/to/apbs/APBS-1.5-linux64/share/apbs/tools/bin/multivalue
export PDB2PQR_BIN=/path/to/apbs/apbs/pdb2pqr-linux-bin64-2.1.1/pdb2pqr
export PATH=$PATH:/path/to/reduce/
export REDUCE_HET_DICT=/path/to/reduce/reduce_wwPDB_het_dict.txt
export PYMESH_PATH=/path/to/PyMesh
export MSMS_BIN=/path/to/msms/msms
export PDB2XYZRN=/path/to/msms/pdb_to_xyzrn
```

Clone masif to a new directory:

```
git clone https://github.com/lpdi-epfl/masif
```


## Method overview 

From a protein structure 
MaSIF computes a molecular surface discretized as a mesh according to the solvent 
excluded surface (using MSMS), and assigns geometric and chemical features to every point (vertex) 
in the mesh. Then, MaSIF applies a geometric deep neural network to these features. 
The neural network consists of one or more layers applied sequentially; a key component 
of the architecture is the geodesic convolution, generalizing the classical convolution 
to surfaces and implemented as an operation on local patches. 

![MaSIF conceptual framework and method](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/Method-01.png)

Around each vertex of the mesh, we extract a patch with geodesic radius of r=9 Å or r=12 Å. 
For each vertex within the patch, we compute two geometric features (shape index and distance-dependent curvature) 
and three chemical features (hydropathy index, continuum electrostatics, and the location of free electrons and proton donors), 
further described in the Methods of the paper. The vertices within a patch are assigned geodesic 
polar coordinates: the radial coordinate, representing the geodesic distance 
to the center of the patch; and the angular coordinate, computed with respect to 
a random direction from the center of the patch, as the patch lacks a canonical orientation. 
In these coordinates, we then construct a family of learnable parametric kernels 
that locally average the vertex-wise patch features and produce an output of 
fixed dimension, which is correlated with a set of learnable filters. We refer 
to this family of learnable parametric kernels as a learned soft polar grid. 
Note that since the angular coordinates were computed with respect to a random 
direction, it becomes essential to compute information that is invariant to different 
directions (rotation invariance). To this end, we perform K rotations on the 
patch and compute the maximum over all rotations, producing the geodesic convolution 
output for the patch location. The procedure is repeated for different patch locations 
similarly to a sliding window operation on images, producing the surface fingerprint 
at each point, in the form of a vector that stores information about the surface patterns 
of the center point and its neighborhood. The learning procedure consists of finding the 
optimal parameter set of the local kernels and filter weights. The parameter set minimizes 
a cost function on the training dataset, which is specific to each application that we 
present here. 

We have thus created descriptors for surface patches that can be further 
processed in neural network architectures. Each descriptor can then be further
processed by each application. 

## MaSIF proof-of-concept applications

MaSIF was tested on three proof of concept applications. 

![MaSIF proof-of-concept applications](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/Applications-01.png)

### MaSIF-ligand

### Data preparation



MaSIF-ligand is run from the data/masif_ligand directory. 

The first step is to prepare the dataset by running data_prepare_one.sh on every structure identifier in the dataset. An example of how to do this and parallelise over multiple nodes can be found in data_prepare.slurm.

The next step is to combine the pre-processed data into a single TFRecords file by running the commands in make_tfrecord.slurm. 

The neural network is trained and finally evaluated by running the commands in train_model.slurm and evaluate_test.slurm respectively.

### MaSIF-site
### MaSIF-search

## License

MaSIF is released under an Apache v2 license. Copyright Gainza, P., Sverrisson, F., Monti, F., Rodola, E., Bronstein, M. M., & Correia, B. E.

## Reference

Please cite: 
[1] Gainza, P., Sverrisson, F., Monti, F., Rodola, E., Bronstein, M. M., & Correia, B. E. (2019). Deciphering interaction fingerprints from protein molecular surfaces. bioRxiv, 606202.
