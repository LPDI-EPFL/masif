### comparison/masif_ligand/Kripo

Code for running KRIPO benchmark

First run add_pdb_header.py to format PDB files for KRIPO. 

Then go to training_set/ and testing_set/ and run generate_fingerprints_multithreaded.py to generate fingerprints for the training and testing sets.

Run compare_test_training.py to generate similarity scores between the training and testing set. 

Go to similarity_results/ and run generate_ROCs.py to generate ROC curve for KRIPO and compare to results from MaSIF and ProBiS.
