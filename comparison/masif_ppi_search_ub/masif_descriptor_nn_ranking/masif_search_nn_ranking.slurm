#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=2:00:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 16000

#module purge
#slmodules -r deprecated
slmodules -s x86_E5v2_Mellanox_GPU
module load gcc cuda cudnn python mvapich2 openblas
source tensorflow-1.12/bin/activate
#python -v
./second_stage_masif.sh
