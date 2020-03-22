masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_ppi_search/
python $masif_source/masif_ppi_search/masif_ppi_search_comp_desc.py nn_models.sc05.all_feat.custom_params $1 $2
