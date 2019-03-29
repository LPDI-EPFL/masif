masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/
source /home/gainza/lpdi_fs/seeder/data/ppi_benchmark_complexes/tensorflow-1.9/bin/activate
python $masif_source/masif_site/masif_site_predict.py $1 -l lists/testing_transient.txt
