masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/
python $masif_source/masif_ppi_search/second_stage_alignment.py ../../../data/masif_ppi_search_ub 1000 4000 1000 gif
