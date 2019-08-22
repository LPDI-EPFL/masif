import subprocess
import multiprocessing

n_test_files = multiprocessing.cpu_count()
n_training_files = multiprocessing.cpu_count()

# Compares fingerprints between training and testing sets
for i in range(n_test_files):
    print(i)
    for j in range(n_training_files):
        with open(
            "similarity_results/logs/stdout_{}_{}.txt".format(i, j), "w+"
        ) as fout:
            with open(
                "similarity_results/logs/stderr_{}_{}.txt".format(i, j), "w+"
            ) as ferr:
                response = subprocess.run(
                    [
                        "kripodb",
                        "fingerprints",
                        "similarities",
                        "--out_format",
                        "tsv",
                        "--cutoff",
                        "0.0",
                        "--fragmentsdbfn",
                        "frags.db",
                        "test_set/test_fingerprints_{}.db".format(i),
                        "training_set/training_fingerprints_{}.db".format(j),
                        "similarity_results/test_train_similarities_{}_{}.tsv".format(
                            i, j
                        ),
                    ],
                    stdout=fout,
                    stderr=ferr,
                )

