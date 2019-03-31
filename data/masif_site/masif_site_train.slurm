#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=40:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000
#SBATCH --output=exelogs/out/gpu_masif_site.%A_%a.out
#SBATCH --error=exelogs/err/gpu_masif_site.%A_%a.err

module purge
slmodules -r deprecated
module load gcc/5.4.0 cuda cudnn mvapich2 openblas
source /home/gainza/lpdi_fs/seednet/monet_seeder/tensorflow_on_gpu/bin/activate
./train_nn.sh $1
