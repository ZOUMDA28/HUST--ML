import numpy as np
import os
import csv

def load_data(data_path, target_path):
    X = np.genfromtxt(data_path, delimiter=',')
    y_original = np.genfromtxt(target_path, delimiter=',', dtype=int) # 标签是0或1
    return X, y_original

def k_fold_split(n_samples, n_folds=10, shuffle=True, random_state=None):

    indices = np.arange(n_samples)
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1  
    
    cv_splits = []
    current_idx = 0
    all_folds_indices_list = [] #存储每一折的索引

    for fold_size in fold_sizes:
        start, stop = current_idx, current_idx + fold_size
        all_folds_indices_list.append(indices[start:stop])
        current_idx = stop
    
    for i in range(n_folds):
        test_indices = all_folds_indices_list[i]
        train_indices_list = [all_folds_indices_list[j] for j in range(n_folds) if j != i]
        
        if not train_indices_list: 
             train_indices = np.array([], dtype=int)
        else:
            train_indices = np.concatenate(train_indices_list)
        cv_splits.append((train_indices, test_indices))
    return cv_splits

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))