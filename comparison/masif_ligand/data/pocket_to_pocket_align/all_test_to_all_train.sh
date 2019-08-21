# all_test_to_all_train.sh: Align each of the testing set cofactors to the training set cofactors.
# Pablo Gainza - LPDI STI EPFL 2019
# Released under an Apache License 2.0

for i in {1..489}
do
	python3 ./pocket_to_pocket_align.py $i &
	sleep 1
done
