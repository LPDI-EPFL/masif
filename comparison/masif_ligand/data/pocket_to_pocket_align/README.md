### Scripts to generate a pocket-to-pocket split for benchmarking MaSIF-ligand, ProBiS and KRIPO.

+ *all_test_to_all_train.sh* - Script to align each pocket in the test set to all pockets in the training set. 
+ *get_split.py* - Once pockets have been aligned, call this script to get the split. 
+ *pocket_to_pocket_align.py*	- Script to align one pocket to all pockets in the training set and output the TM-score
+ *testset_pocket_split_tmscore0.50.txt* - List of testset pockets that do not align with a TM-score > 0.50 to any pocket in the training set.
