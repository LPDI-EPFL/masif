import multiprocessing
import dask
import subprocess

# Generates fingerprints for training set

n_cores = multiprocessing.cpu_count()
print("Number of cpus:", n_cores)
with open("all_training_pdbs.list") as f:
    all_training_pdbs = f.read().splitlines()

files_per_core = len(all_training_pdbs) // n_cores + 1
for core in range(n_cores):
    with open("training_pdbs_{}.list".format(core), "w+") as f:
        for pdb in all_training_pdbs[
            core * files_per_core : (core + 1) * files_per_core
        ]:
            f.write(pdb + "\n")


@dask.delayed
def generate_fingerprints(core_n):
    with open("logs/stdout_{}.txt".format(core_n), "w+") as fout:
        with open("logs/stderr_{}.txt".format(core_n), "w+") as ferr:
            response = subprocess.run(
                [
                    "kripo",
                    "generate",
                    "--no-fragmentation",
                    "--ligand-expo",
                    "../ligand-expo.db",
                    "training_pdbs_{}.list".format(core_n),
                    "training_frags_{}.db".format(core_n),
                    "training_phars_{}.h5".format(core_n),
                    "training_fingerprints_{}.db".format(core_n),
                ],
                stdout=fout,
                stderr=ferr,
            )


results = []
for core in range(n_cores):
    results.append(generate_fingerprints(core))

results = dask.compute(*results)
