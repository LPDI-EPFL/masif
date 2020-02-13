#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=40:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000
#SBATCH --output=exelogs/gpu_monet_seeder.%A_%a.out
#SBATCH --error=exelogs/gpu_monet_seeder.%A_%a.err

module purge
module load gcc cuda cudnn mvapich2 openblas
deactivate
source ~/lpdi_fs/masif/tensorflow-1.12/bin/activate
./train_nn.sh nn_models.sc05.all_feat.custom_params.py
