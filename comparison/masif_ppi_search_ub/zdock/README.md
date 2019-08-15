				ZDOCK 3.0.2

This is the README file for release 3.0.2 of the ZDOCK program. ZDOCK is 
an initial stage protein-protein docking program, developed initially by
Rong Chen and Zhiping Weng in 2002. It optimizes pairwise shape 
complementarity, an interface contact potential called IFACE, and 
electrostatic energies using the Fast Fourier Transform algorithm. The 
FFT is optimized for speed and memory usage using the Conv3D library,
along with other improvements to the molecular representation on the 3D
grid. 

This distribution includes an executable file (zdock) of the ZDOCK program,
PDB processing files (mark_sur, uniCHARMM, block.pl), and auxiliary files
(create.pl, create_lig) to create predicted complex structures from ZDOCK
output. 

The create_lig executable has been updated for ZDOCK 3.0.2, so it is 
necessary to use create_lig and create.pl from this distribution and 
not from earlier ZDOCK versions.

Example:
mark_sur receptor.pdb receptor_m.pdb
mark_sur ligand.pdb ligand_m.pdb
zdock -R receptor_m.pdb -L ligand_m.pdb -o zdock.out
create.pl zdock.out

Please note: The file uniCHARMM should be in the current directory when
executing mark_sur. Also, receptor_m.pdb, ligand_m.pdb and create_lig must 
be in your current directory when you create all predicted structures 
using create.pl.

Standard PDB format files must be processed by mark_sur before being used as 
the input to ZDOCK. Formatted PDB files of docking benchmark can be downloaded 
at http://zlab.umassmed.edu/benchmark. If you know that some atoms
are not in the binding site, you can block them by changing their ACE type
(column 55-56) to 19. This blocking procedure can improve docking
performance significantly. A blocking script block.pl is included, type
"block.pl" for usage information.

More options can be found by typing "zdock".
