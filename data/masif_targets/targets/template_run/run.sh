masif_root=$(git rev-parse --show-toplevel)
export masif_db_root=/home/gainza/lpdi_fs/masif/
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export masif_root
export PYTHONPATH=$PYTHONPATH:$masif_source:`pwd`
source /home/gainza/lpdi_fs/masif/tensorflow-1.12_on_cpu/bin/activate
python $masif_source/masif_seed_search/masif_seed_search_nn.py params_small_proteins 4ZQK_A
