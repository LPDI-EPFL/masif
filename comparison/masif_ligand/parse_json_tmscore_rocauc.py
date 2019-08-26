import json 
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics 
import os
from IPython.core.debugger import set_trace

"""
parse_json_tmscore_rocauc.py: Compute the ROC AUC values for MaSIF-ligand and ProBiS
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

# Compute the ROC AUC for cases where the testset pocket does not align to any of the pockets in the training set at at a tmscore greater than X

# Return whether the correct prediction was given or not, as well as the predicted score of the correct value.
def get_masif_ligand_prediction(pdbid, probis_test_id, cofid):
    masif_ligand_dir = \
            "masif_pred/test_set_predictions/"

    label_names_order = ['ADP', 'COA', 'FAD', 'HEM', 'NAD', 'NAP', 'SAM']
    labels = None
    logits = None
    names = None
    for myfile in os.listdir(masif_ligand_dir):
        if myfile.startswith(pdbid):
            if 'labels' in myfile:
                labels = np.load(os.path.join(masif_ligand_dir, myfile))
            if 'logits' in myfile:
                logits = np.load(os.path.join(masif_ligand_dir, myfile))
            if 'names.npy' in myfile:
                names = np.load(os.path.join(masif_ligand_dir, myfile))
    if labels is None or names is None:
        return -1, -1
    # pocket_ix: numbering of the pocket within all pockets of teh protein.
    pocket_ix = list(names).index(probis_test_id)
    true_val = labels[pocket_ix] 
    pred_val = logits[pocket_ix]
    pred_array = np.squeeze(np.mean(pred_val, axis=0))
    pred_val = np.argmax(pred_array)
    score_for_true = pred_array[true_val]
    ## Ignore heme
    score_for_false = [x for ix, x in enumerate(pred_array) if ix != true_val and x != 3]
    score_for_false = [x for ix, x in enumerate(pred_array) if ix != true_val ]

    if pred_val == true_val:
        print("Success")
        return [1.0, score_for_true, score_for_false]
    else:
        print("Fail")
        return [0.0, score_for_true, score_for_false]

# Parse the list of testset results under the cutoff.
testset_list = open('data/pocket_to_pocket_align/testset_pocket_split_tmscore0.50.txt').readlines()
testset_members = [x.split(',')[0] for x in testset_list]

training_list = open('data/training_srfs/training_srfs.txt').readlines()
training_set = [x.split()[0] for x in training_list]

# Open the list of predictions that were actually read in MaSIF
masif_list = open('masif_pred/test_set_predictions/all_masif_kripo_names.txt').readlines()
masif_list = [x.rstrip() for x in masif_list]

correct_all = 0.0
total_all = 0.0

correct_by_class_probis = {}
total_by_class = {}
correct_by_class_masif = {}
correct_masif = 0.0
total_masif = 0.0
masif_true_scores = []
masif_false_scores = []

probis_true_scores = []
probis_false_scores = []
totally_wrong = 0

# Read probis output for each pocket.  
for myfile in os.listdir('out_probis/json'):
    myjson = json.load(open('out_probis/json/'+myfile))

    # Check that this pocket was actually tested by MaSIF.
    probis_test_id = myfile.replace('.json', '')
    # Ensure that this specific member is in the testset. 
#    if probis_test_id not in testset_members:
#        continue
    if probis_test_id not in masif_list:
        print('{} not in masif predictions.'.format( probis_test_id ))
        continue
   
    cofid_test = myfile.split('_')[2].split('.')[0]

    # Ignore HEME
    if cofid_test == 'HEM':
        continue


    # Initialize the statistics dictionaries..
    if cofid_test not in correct_by_class_probis:
        correct_by_class_probis[cofid_test] = 0.0
        total_by_class[cofid_test] = 0.0
        correct_by_class_masif[cofid_test] = 0.0
    pdbid = myfile.split('_')[0]

    # First evaluate with MaSIF.
    print ('Evaluating {} {} with MaSIF'.format(pdbid, cofid_test))
    masif_pred = get_masif_ligand_prediction(pdbid, probis_test_id, cofid_test)
#    if masif_pred[0] < 0:
#        continue
    if masif_pred[0] > 0: 
        # MaSIF predicted it correctly.
        correct_by_class_masif[cofid_test] += 1.0
        correct_masif += 1
    # Store true and false predictions for ROC AUC.
    masif_true_scores.append(masif_pred[1])
    masif_false_scores.append(masif_pred[2])
    total_masif+= 1

    print('Evaluating with ProBIS {}'.format(pdbid, cofid_test))

    total_by_class[cofid_test] += 1.0
    total_all += 1.0

    found = False
    # Check whether the cofactor of the top result corresponds to the testing cofactor.
    pdbid_train = myjson[0]['pdb_id']
    cofid_train = pdbid_train.split('_')[2].split('.')[0]
    if cofid_test == cofid_train:
        # ProBIS was successful since the top result corresponds in its ID.
        found=True
        print('{} {} {} {} {}'.format(pdbid, cofid_test, pdbid_train, cofid_train, myjson[0]['alignment'][0]['scores']['z_score']))
        correct_by_class_probis[cofid_test] +=1.0
        correct_all += 1.0
    else:
        print('Wrong: {} {} {} {} {}'.format(pdbid, cofid_test, pdbid_train, cofid_train, myjson[0]['alignment'][0]['scores']['z_score']))

    # Go through every one of the results and check the highest zscore for each cofactor
    # (i.e. we check which structure for each cofactor has the highest zscore)
#    max_probis_zscore= {'ADP': -1000.0, 'COA':-1000.0, 'FAD':-1000.0, 'HEM':-1000.0, 'NAD':-1000.0, 'NAP':-1000.0, 'SAM':-1000.0}
    # Ignore HEME.
    max_probis_zscore= {'ADP': -3.0, 'COA':-3.0, 'FAD':-3.0, 'NAD':-3.0, 'NAP':-3.0, 'SAM':-3.0}
    for probis_pred in myjson:
        pdbid_train = probis_pred['pdb_id']
        zscore = probis_pred['alignment'][0]['scores']['z_score']

        cofid_train = pdbid_train.split('_')[2].split('.')[0]
        if cofid_train == 'HEM':
            continue
        # Assign to this cofactor the max score between previously seen and this one.
        max_probis_zscore[cofid_train] = max(max_probis_zscore[cofid_train], zscore)

    # Normalize the scores.
    Z = 1e-8
    for key in max_probis_zscore:
        max_probis_zscore[key] += 3.0
        Z += max_probis_zscore[key]
    for key in max_probis_zscore:
        max_probis_zscore[key] = max_probis_zscore[key]/Z

    probis_true_scores.append(max_probis_zscore[cofid_test])
    # For every cofactor not equal to the test set one, assign it to the negatives
    for cof in max_probis_zscore: 
        if cof != cofid_test:
            probis_false_scores.append(max_probis_zscore[cof])

print('Total number of samples Probis: {}'.format(total_all))
print('Total number of samples Masif: {}'.format(total_masif))
print('Accuracy MaSIF = {:.2f}'.format(100*correct_masif/total_all))
print('Accuracy = {:.2f}'.format(100*correct_all/total_all))
print ('Correct MaSIF: {}'.format(correct_masif))
print ('Correct ProBIS: {}'.format(correct_all))

balanced_acc = 0.0
num_classes = 0.0
for key in correct_by_class_probis:
    if total_by_class[key] > 0:
        balanced_acc += correct_by_class_probis[key]/total_by_class[key]
        num_classes += 1.0
        print('{}: {:.2f}'.format(key, correct_by_class_probis[key]/total_by_class[key]))
print('Num classes: {}'.format(num_classes))
print('Balanced accuracy = {:.2f}'.format(100*balanced_acc/num_classes))

balanced_acc = 0.0
num_classes = 0.0
print ("Now for MaSIF: ")
for key in correct_by_class_probis:
    if total_by_class[key] > 0:
        balanced_acc += correct_by_class_masif[key]/total_by_class[key]
        num_classes += 1.0
        print('{}: {:.2f}'.format(key, correct_by_class_masif[key]/total_by_class[key]))
print('Num classes: {}'.format(num_classes))
print('Balanced accuracy = {:.2f}'.format(100*balanced_acc/num_classes))

# Compute ROC AUC scores for MaSIF
masif_false_scores = np.concatenate(masif_false_scores, axis=0)
labels = np.concatenate([np.ones_like(masif_true_scores), np.zeros_like(masif_false_scores)], axis=0)
preds = np.concatenate([masif_true_scores, masif_false_scores], axis=0)

roc_auc_score_masif = sklearn.metrics.roc_auc_score(labels, preds)
print(roc_auc_score_masif)
fpr_masif, tpr_masif, threshold = sklearn.metrics.roc_curve(labels, preds)
#np.save('masif_fpr40.npy', fpr_masif)
#np.save('masif_tpr40.npy', tpr_masif)

plt.title('Receiver Operating Characteristic MaSIF vs ProBIS pocket split')
plt.plot(fpr_masif, tpr_masif, 'b', label = 'MaSIF AUC = %0.2f' % roc_auc_score_masif)

# Compute ROC AUC scores for ProBIS 
labels = np.concatenate([np.ones_like(probis_true_scores), np.zeros_like(probis_false_scores)], axis=0)
preds = np.concatenate([probis_true_scores, probis_false_scores], axis=0)
fpr_probis, tpr_probis, threshold = sklearn.metrics.roc_curve(labels, preds)
roc_auc_score_probis= sklearn.metrics.roc_auc_score(labels, preds)
print(roc_auc_score_probis)

plt.plot(fpr_probis, tpr_probis, 'g', label = 'ProBIS AUC = %0.2f' % roc_auc_score_probis)
#np.save('probis_fpr40.npy', fpr_probis)
#np.save('probis_tpr40.npy', tpr_probis)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig('roc_curve_masif_probis_pocket_split.png')

print('Totally wrong = {}'.format(totally_wrong))



