#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=20:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000

# Load your GPU environment here.
#source /work/upcorreia/bin/load_masif_environment_gpu.sh
./compute_descriptors.sh $1 $2
