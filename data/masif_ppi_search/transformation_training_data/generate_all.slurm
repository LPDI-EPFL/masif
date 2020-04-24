#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition=serial
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8192
#SBATCH --time 02:00:00
#SBATCH --array=1-1000

./generate_data.sh ${SLURM_ARRAY_TASK_ID}
