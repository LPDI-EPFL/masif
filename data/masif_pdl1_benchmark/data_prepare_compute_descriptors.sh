masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
export PYTHONPATH=$PYTHONPATH:$masif_source:.
PDB_ID=$(echo $1| cut -d"_" -f1)
CHAIN1=$(echo $1| cut -d"_" -f2)
CHAIN2=$(echo $1| cut -d"_" -f3)
# Load your environment here. 
python -W ignore $masif_source/data_preparation/00-pdb_download.py $1 
python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1 
python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN2
python -W ignore $masif_source/data_preparation/04-masif_precompute.py masif_site $1
python -W ignore $masif_source/data_preparation/04-masif_precompute.py masif_ppi_search $1
python -W ignore $masif_source/masif_site/masif_site_predict.py nn_models.all_feat_3l.custom_params $1 $2
python -W ignore $masif_source/masif_site/masif_site_label_surface.py nn_models.all_feat_3l.custom_params $1 $2
python -W ignore $masif_source/masif_ppi_search/masif_ppi_search_comp_desc.py nn_models.sc05.all_feat.custom_params $1 $2
