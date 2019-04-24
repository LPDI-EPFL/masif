#!/bin/bash
#SBATCH --partition=serial
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 03:00:00
#SBATCH --array=1-100
#SBATCH --output=exelogs/out/00_pd.%A_%a.out
#SBATCH --error=exelogs/err/00_pd.%A_%a.err

i=1
while read p; do
    if [ $(( i % 40 )) == ${SLURM_ARRAY_TASK_ID} ]; then
        python ./runPatchDockBenchmark.py $p
    fi
    i=$((i+1))
done < ../benchmark_list.txt
