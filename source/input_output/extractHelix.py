from Bio.PDB import * 

# Exclude disordered atoms. 
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == 'A'

def extractHelix(helix, infilename, outfilename, chain_ids=None, includeWaters=False,\
                            invert=False):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, 'M')[0]
    chains = Selection.unfold_entities(struct, 'C')
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()
    for chain in model:
        if chain.get_id() in chain_ids:
            structBuild.init_chain(chain.get_id())
            for residue in chain: 
                het = residue.get_id()
                if het[0] == ' ' and het in helix:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

