masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export masif_root
export PYTHONPATH=$PYTHONPATH:$masif_source:`pwd`
python $masif_source/masif_seed_search/masif_seed_search_debug.py params_small_proteins 4ZQK_A
