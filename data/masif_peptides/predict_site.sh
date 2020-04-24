masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
source /work/upcorreia/bin/load_masif_environment.sh
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/
python -W ignore $masif_source/masif_site/masif_site_predict.py nn_models.all_feat_3l.custom_params $1 $2
