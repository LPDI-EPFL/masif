#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition=serial
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8192
#SBATCH --time 01:00:00
#SBATCH --array=1-40

i=1
while read p; do
    if [ $(( i % 41)) == ${SLURM_ARRAY_TASK_ID} ]; then
        FIELD1=$(echo $p| cut -d" " -f1)
        PDBID=$(echo $FIELD1| cut -d"_" -f1)
        CHAIN1=$(echo $FIELD1| cut -d"_" -f2)
        CHAIN2=$(echo $FIELD1| cut -d"_" -f3)
        ./data_prepare_one.sh $PDBID\_$CHAIN1\_$CHAIN2
    fi
    i=$((i+1))
done < lists/benchmark_list_ub.txt
