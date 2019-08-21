### data_preparation

Contains the protocol followed to prepare protein structures for MaSIF, in order of execution:

+ *00-pdb_download.py*: Download pdb file from Protein DataBank 

+ *00b-generate_assembly.py*: Generate a biological assembly (used by MaSIF-ligand only) 

+ *00c-save_ligand_coords.py*: Save the coordinates of small molecules (used by MaSIF-ligand only) 

+ *01-pdb_extract_and_triangulate.py*: Extract the PDB chains analyzed and triangulate them

+ *02-compute_matlab_matrix.py*: Compute the matlab matrices with the shape index and shape complementarity values.

+ *03-compute_coords.py*: Compute the angular and radial coordinates of each patch. 

+ *03b-convert_mat2npy.py*: Convert the matlab angular and radial coordinates to numpy (for faster access)

+ *04-masif_precompute.py*: Decompose proteins into patches for input into the neural network.

+ *04b-make_ligand_tfrecords.py*: Make tensorflow records (used by MaSIF-ligand only)

