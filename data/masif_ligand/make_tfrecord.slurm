#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8192
#SBATCH --time 48:00:00
#SBATCH --partition=serial

source /work/upcorreia/bin/load_masif_environment.sh
masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab

python -u $masif_source/data_preparation/04b-make_ligand_tfrecords.py
