import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import multiprocessing


n_train = multiprocessing.cpu_count()
n_test = multiprocessing.cpu_count()

ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
lig2num = {l: i for i, l in enumerate(ligands)}

with open("../lists/all_masif_test_names.txt") as f:
    test_set_pdb_list = f.read().splitlines()
test_set_pdb_list = sorted(test_set_pdb_list)

with open("../lists/testset_pocket_split_tmscore0.40.txt") as f:
    test_set_pdb_list040 = f.read().splitlines()
    test_set_pdb_list040 = [t.split(",")[0] for t in test_set_pdb_list040]
test_set_pdb_list040 = sorted(test_set_pdb_list040)

with open("../lists/testset_pocket_split_tmscore0.25.txt") as f:
    test_set_pdb_list025 = f.read().splitlines()
    test_set_pdb_list025 = [t.split(",")[0] for t in test_set_pdb_list025]
test_set_pdb_list025 = sorted(test_set_pdb_list025)


def get_df(test_idx, train_idx):
    df = pd.read_csv(
        "test_train_similarities_{}_{}.tsv".format(test_idx, train_idx),
        sep="\t",
        header=None,
    )

    test_pdbs = list(set(df[0].values))
    test_pdbs = sorted([t for t in test_pdbs if t.split("_")[1] in ligands])
    train_pdbs = list(set(df[1].values))
    train_pdbs = sorted([t for t in train_pdbs if t.split("_")[1] in ligands])

    clean_df = np.empty((len(test_pdbs), len(train_pdbs)))
    clean_df[:] = np.nan
    clean_df = pd.DataFrame(clean_df, index=test_pdbs, columns=train_pdbs)

    for i, r in df.iterrows():
        if (r[0] in clean_df.index) and (r[1] in clean_df.columns):
            clean_df.loc[r[0], r[1]] = r[2]

    return clean_df


i = 0
full_df = pd.DataFrame()
for i in range(n_test):
    test_df = pd.DataFrame()
    for j in range(n_train):
        df_part = get_df(i, j)
        test_df = pd.concat([test_df, df_part], axis=1, sort=False)
    full_df = pd.concat([full_df, test_df], axis=0, sort=False)

ligand_array = False * np.empty((len(ligands), len(full_df.columns)), dtype=bool)
for i, ind in enumerate(full_df.columns):
    train_ligand = ind.split("_")[1]
    ligand_array[lig2num[train_ligand], i] = True

pos_scores = []
neg_scores = []
failed_structures = []
for tested in test_set_pdb_list:
    pdb_id = tested.split(".")[0].split("_")
    if pdb_id[2] == "HEM":
        continue
    true_ligand = lig2num[pdb_id[2]]
    pdb_id = "_".join([pdb_id[0].lower(), pdb_id[2], "frag1"])
    if true_ligand > 3:
        true_ligand -= 1
    try:
        r = full_df.loc[pdb_id]
    except:
        failed_structures.append(pdb_id)
    false_ligands = [l for l in range(len(ligands) - 1) if l != true_ligand]
    scores = [np.max(r[ligand_array[lig]]) for lig in range(len(ligands))]
    scores = scores[:3] + scores[4:]
    scores = scores / np.sum(scores)
    pos_scores.append(scores[true_ligand])
    neg_scores += list(scores[false_ligands])

print(
    "No cutoff ROC AUC",
    roc_auc_score(
        [1] * len(pos_scores) + [0] * len(neg_scores), pos_scores + neg_scores
    ),
)

fpr100, tpr100, thresholds100 = roc_curve(
    [1] * len(pos_scores) + [0] * len(neg_scores), pos_scores + neg_scores
)
np.save("kripo_fpr100.npy", fpr100)
np.save("kripo_tpr100.npy", tpr100)

pos_scores040 = []
neg_scores040 = []
failed_structures2_040 = []
for tested in test_set_pdb_list040:
    pdb_id = tested.split(".")[0].split("_")
    if pdb_id[2] == "HEM":
        continue
    true_ligand = lig2num[pdb_id[2]]
    pdb_id = "_".join([pdb_id[0].lower(), pdb_id[2], "frag1"])
    if true_ligand > 3:
        true_ligand -= 1
    try:
        r = full_df.loc[pdb_id]
    except:
        failed_structures2_040.append(pdb_id)
    false_ligands = [l for l in range(len(ligands) - 1) if l != true_ligand]
    scores = [np.max(r[ligand_array[lig]]) for lig in range(len(ligands))]
    scores = scores[:3] + scores[4:]
    scores = scores / np.sum(scores)
    pos_scores040.append(scores[true_ligand])
    neg_scores040 += list(scores[false_ligands])

print(
    "0.4 cutoff ROC AUC",
    roc_auc_score(
        [1] * len(pos_scores040) + [0] * len(neg_scores040),
        pos_scores040 + neg_scores040,
    ),
)
fpr040, tpr040, thresholds040 = roc_curve(
    [1] * len(pos_scores040) + [0] * len(neg_scores040), pos_scores040 + neg_scores040
)
np.save("kripo_fpr040.npy", fpr040)
np.save("kripo_tpr040.npy", tpr040)


pos_scores025 = []
neg_scores025 = []
failed_structures2_025 = []
for tested in test_set_pdb_list025:
    pdb_id = tested.split(".")[0].split("_")
    if pdb_id[2] == "HEM":
        continue
    true_ligand = lig2num[pdb_id[2]]
    pdb_id = "_".join([pdb_id[0].lower(), pdb_id[2], "frag1"])
    if true_ligand > 3:
        true_ligand -= 1
    try:
        r = full_df.loc[pdb_id]
    except:
        failed_structures2_025.append(pdb_id)
    false_ligands = [l for l in range(len(ligands) - 1) if l != true_ligand]
    scores = [np.max(r[ligand_array[lig]]) for lig in range(len(ligands))]
    scores = scores[:3] + scores[4:]
    scores = scores / np.sum(scores)
    pos_scores025.append(scores[true_ligand])
    neg_scores025 += list(scores[false_ligands])

print(
    roc_auc_score(
        "0.25 cutoff ROC AUC",
        [1] * len(pos_scores025) + [0] * len(neg_scores025),
        pos_scores025 + neg_scores025,
    )
)
fpr025, tpr025, thresholds025 = roc_curve(
    [1] * len(pos_scores025) + [0] * len(neg_scores025), pos_scores025 + neg_scores025
)
np.save("kripo_fpr025.npy", fpr025)
np.save("kripo_tpr025.npy", tpr025)

