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

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gainza/lpdi_fs/programs/python/lib/python2.7/site-packages/pymesh2-0.1.14-py2.7-linux-x86_64.egg/pymesh/lib:/home/gainza/lpdi_fs/programs/python/lib/python2.7/site-packages/pymesh2-0.1.14-py2.7-linux-x86_64.egg/pymesh/third_party/lib/
i=1
while read p; do
    if [ $(( i % 100 )) == ${SLURM_ARRAY_TASK_ID} ]; then
        python ./runPatchDockBenchmark.py $p
    fi
    i=$((i+1))
done < ../benchmark_list.txt
