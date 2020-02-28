![MaSIF banner and concept](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/img/Concept-01.png)

# Docker tutorial for MaSIF.

## Table of Contents: 

- [Installation](#Installation)
- [MaSIF-site](#MaSIF-site)
    * [Running MaSIF-site on a single protein from a PDB id or PDB file](#Running-MaSIF-site-on-a-single-protein-from-a-PDB-id)
    * [Running MaSIF-site on a single protein from a PDB file](#Running-MaSIF-site-on-a-single-protein-from-a-PDB-file)
    * [Reproducing the transient benchmark from the paper](#Reproducing-the-transient-benchmark-from-the-paper)
- [Building MaSIF from a Dockerfile](Dockerfile)
- [MaSIF-ligand](#MaSIF-ligand)
- [MaSIF-search](#MaSIF-search)


## Installation

```
docker pull pablogainza/masif:latest
docker run -it pablogainza/masif
```
You now start a local container with MaSIF. The first step should be to update the repository to make sure you have the latest version (in case the image has not been update):

```
root@b30c52bcb86f:/masif# git pull 
```

## MaSIF-site

### Running MaSIF-site on a single protein from a PDB id

Go into the MaSIF site data directory. 
```
root@b30c52bcb86f:/masif# cd data/masif_site/
root@b30c52bcb86f:/masif/data/masif_site# 
```

We will now run MaSIF site on chain A of PDB id 4ZQK. It is important to always input a chain and a PDB id. The first step consists of preparing the data and it is the slowest part of the process. It consists of downloading the pdb, extracting the chain, protonating it, computing the molecular surface and PB electrostatics, and decomposing the protein into patches (about 1 minute on a 120 residue protein): 

```
root@b30c52bcb86f:/masif/data/masif_site# ./data_prepare_one.sh 4ZQK_A
Downloading PDB structure '4ZQK'...
Removing degenerated triangles
Removing degenerated triangles
4ZQK_A
Reading data from input ply surface files.
Dijkstra took 2.28s
Only MDS time: 18.26s
Full loop time: 28.54s
MDS took 28.54s
```

If you want to run a prediction on multiple chains (e.g. a multidomain protein) you can do so by concatenting all chains (e.g., 4ZQK_AB). You can also run this command on a specific file (i.e. not on a downloaded file) using the --file flag: 

```
root@b30c52bcb86f:/masif/data/masif_site# ./data_prepare_one.sh --file /path/to/myfile/4ZQK.pdb 4ZQK_A
```

The next step consists of actually running the protein through the neural network to predict interaction sites: 

```
root@b30c52bcb86f:/masif/data/masif_site# ./predict_site.sh 4ZQK_A
Setting model_dir to nn_models/all_feat_3l/model_data/
Setting feat_mask to [1.0, 1.0, 1.0, 1.0, 1.0]
Setting n_conv_layers to 3
Setting out_pred_dir to output/all_feat_3l/pred_data/
Setting out_surf_dir to output/all_feat_3l/pred_surfaces/
(12, 2)
...
Total number of patches for which scores were computed: 2336

GPU time (real time, not actual GPU time): 1.890s
```

After this step you can find the predictions in numpy files:

```
root@b30c52bcb86f:/masif/data/masif_site# ls output/all_feat_3l/pred_data
pred_4ZQK_A.npy
root@b30c52bcb86f:/masif/data/masif_site/#
```

Finally you can run a command to output a ply file with the predicted interface for visualization. A ROC AUC is also computed, but it is only accurate if the protein was found in the original complex (the ground truth is extracted from there):

```
root@b30c52bcb86f:/masif/data/masif_site# ./color_site.sh 4ZQK_A
Setting model_dir to nn_models/all_feat_3l/model_data/
Setting feat_mask to [1.0, 1.0, 1.0, 1.0, 1.0]
Setting n_conv_layers to 3
Setting out_pred_dir to output/all_feat_3l/pred_data/
Setting out_surf_dir to output/all_feat_3l/pred_surfaces/
ROC AUC score for protein 4ZQK_A : 0.91
Saving output/all_feat_3l/pred_surfaces/4ZQK_A.ply
Computed 1 proteins
Median ROC AUC score: 0.9137235650273854
root@b30c52bcb86f:/masif/data/masif_site#
```

If you have installed the pymol plugin for MaSIF, you can now visualize the predictions. From your local computer run: 

``` 
docker cp b30c52bcb86f:/masif/data/masif_site/output/all_feat_3l/pred_surfaces/4ZQK_A.ply 
```

```
pymol
```

Then from the pymol command window run: 

```
loadply 4ZQK_A.ply 
```

Then deactivate all objects except the one with 'iface' as part of its name. You should see something like this: 

![MaSIF PyMOL plugin example](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/img/masif_plugin_example_2.png)



### Reproducing the transient benchmark from the paper

All the MaSIF-site experiments from the paper should be reproducible using the Docker container. For convenience, I have provided a script to reproduce the transient PPI interaction prediction benchmark, which is the one that is compared to state-of-the-art tools (SPPIDER, PSIVER).

```
cd data/masif_site
./reproduce_transient_benchmark.sh
```

## MaSIF-ligand

## MaSIF-search

## Dockerfile

To build MaSIF from a Dockerfile run the following steps: 

```
git clone https://github.com/LPDI-EPFL/masif-dockerfile
docker build -t masif_docker . 
```
