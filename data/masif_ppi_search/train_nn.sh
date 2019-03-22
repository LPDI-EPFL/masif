masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
#source /home/gainza/lpdi_fs/seeder/data/ppi_benchmark_complexes/tensorflow-1.9/bin/activate
#python $masif_source/05-masif_ppi_cache_training_data.py 
python $masif_source/06-masif_ppi_search_train.py $1
