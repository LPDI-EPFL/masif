masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/
python $masif_source/masif_site/masif_site_label_surface.py nn_models.all_feat_3l.custom_params -l lists/testing_transient.txt
