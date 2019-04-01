masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
# Compute geometric invariant descriptors, as implemented in Yin et al. PNAS 2009
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_ppi_search/
python $masif_source/gif_descriptors/compute_gif_descriptors.py lists/ransac_benchmark_list.txt
