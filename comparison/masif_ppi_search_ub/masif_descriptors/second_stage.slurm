#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16000
#SBATCH --time 0:30:0
#SBATCH --array=1-20

echo STARTING AT `date`
echo ${SLURM_ARRAY_TASK_ID}
echo FINISHED at `date`

./second_stage_masif.sh 
