#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=8:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 16384
#SBATCH --array=101-354

#./predict_site.sh -l in/x${SLURM_ARRAY_TASK_ID} 
./compute_descriptors.sh -l in/x${SLURM_ARRAY_TASK_ID}
