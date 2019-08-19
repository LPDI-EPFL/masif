#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16000
#SBATCH --time 10:0:0
#SBATCH --array=1-20

echo STARTING AT `date`
echo ${SLURM_ARRAY_TASK_ID}
echo FINISHED at `date`

# Run masif search second stage with 100 decoys and 2000 decoys. 
./second_stage_masif.sh 100 
./second_stage_masif.sh 2000 
