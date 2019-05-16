#!/bin/bash
./data_prepare_one.sh $1
sbatch masif_site_eval.slurm $1
sbatch masif_ppi_search_comp_desc.slurm $1
