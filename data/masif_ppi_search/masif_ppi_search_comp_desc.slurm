#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=40:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000

module purge
slmodules -r deprecated
module load gcc/5.4.0 cuda cudnn mvapich2 openblas
masif_root=$(git rev-parse --show-toplevel)
source /home/gainza/lpdi_fs/masif/tensorflow_on_gpu/bin/activate
./compute_descriptors.sh $1
