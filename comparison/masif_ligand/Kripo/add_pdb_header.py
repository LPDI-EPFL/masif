import glob

# Kripo needs a header line to be able to read pdbs
# We add a dummy header to every file

file_list = glob.glob(
    "../../../data/masif_ligand/data_preparation/00b-pdbs_assembly/*.pdb"
)
header = (
    "HEADER    DEHYDROGENASE                           16-JAN-98   {}              \n"
)
for f in file_list:
    pdb = f.split("/")[-1].split(".")[0]
    with open(f) as fr:
        lines = fr.read()

    lines = header.format(pdb) + lines
    with open(f[:-4] + "_header.pdb", "w+") as fw:
        fw.write(lines)
