import numpy as np 
from sklearn.metrics import roc_auc_score

posd = np.load('pos_dists.npy')
negd = np.load('neg_dists.npy')

ytrue = np.concatenate([np.ones_like(posd), np.zeros_like(negd)])
ypred = 1.0/np.concatenate([posd, negd])
print(roc_auc_score(ytrue, ypred))
