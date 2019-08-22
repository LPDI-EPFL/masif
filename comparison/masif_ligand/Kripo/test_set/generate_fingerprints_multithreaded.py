import multiprocessing
import dask
import subprocess

# Generates fingerprints for testing set

n_cores = multiprocessing.cpu_count()
print("Number of cpus:", n_cores)
with open("all_test_pdbs.list") as f:
    all_test_pdbs = f.read().splitlines()

files_per_core = len(all_test_pdbs) // n_cores + 1
for core in range(n_cores):
    with open("test_pdbs_{}.list".format(core), "w+") as f:
        for pdb in all_test_pdbs[core * files_per_core : (core + 1) * files_per_core]:
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
                    "test_pdbs_{}.list".format(core_n),
                    "test_frags_{}.db".format(core_n),
                    "test_phars_{}.h5".format(core_n),
                    "test_fingerprints_{}.db".format(core_n),
                ],
                stdout=fout,
                stderr=ferr,
            )


results = []
for core in range(n_cores):
    results.append(generate_fingerprints(core))

results = dask.compute(*results)
