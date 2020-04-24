masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_ppi_search/
python -W ignore $masif_source/masif_ppi_search/masif_ppi_search_comp_desc.py nn_models.sc05.all_feat.custom_params  -l lists/benchmark_list_ub.txt
