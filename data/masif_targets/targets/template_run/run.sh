masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export masif_root
export PYTHONPATH=$PYTHONPATH:$masif_source:`pwd`
python $masif_source/masif_seed_search/masif_seed_search_nn.py params 4ZQK_A
#echo $masif_source/masif_seed_search/masif_seed_search_nn.py params 4ZQK_A
