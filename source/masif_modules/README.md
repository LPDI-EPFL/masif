### source/masif_modules

Contains most the files that define the neural network, files to train and evaluate the neural network
and files necessary to extract files for the neural network.

The neural networks models for each application are defined in: 
- MaSIF-ligand neural network: MaSIF_ligand.py 
- MaSIF-site neural network: MaSIF_site.py 
- MaSIF-search neural network: MaSIF_ppi_search.py 

Navigating this code: 

+ *MaSIF_ligand.py*: MaSIF-ligand neural network class and definition.
+ *MaSIF_ppi_search.py*: MaSIF-search neural network class and definition.
+ *MaSIF_site.py*: MaSIF-site neural network class and definition.
+ *compute_input_feat.py*: precompute the input features to the neural network in the format used for input (with, for example, padding)
+ *extract_features.py*: Precomputation step: extract features from matlab files, extract patches, and compute the distant dependend curvature.
+ *read_data_from_matfile.py*: Read the data from matlab files. 
+ *read_ligand_tfrecords.py*: Read the tf records for MaSIF-ligand.
+ *train_masif_site.py*: Train, test, and evaluate MaSIF-site.
+ *train_ppi_search.py*: Train test and evaluate MaSIF-search
