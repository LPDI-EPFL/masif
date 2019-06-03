#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16000
#SBATCH --time 20:00:00
#SBATCH --partition=serial

masif_root=$(git rev-parse --show-toplevel)
export masif_db_root=/home/gainza/lpdi_fs/masif/
source /home/gainza/lpdi_fs/masif/tensorflow-1.12_on_cpu/bin/activate
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export masif_root
export PYTHONPATH=$PYTHONPATH:$masif_source:`pwd`
python -u $masif_source/masif_seed_search/masif_seed_search_nn.py $1 $2
