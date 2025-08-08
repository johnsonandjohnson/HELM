import torch
import numpy as np
from sklearn.metrics import f1_score, r2_score, accuracy_score
from scipy.stats import spearmanr, pearsonr


def calculate_pearson_r(y_true, y_predictions):
    pearson_r, _ = pearsonr(y_true.flatten(), y_predictions.flatten())
    return pearson_r


def calculate_spearman_rank_corr(y_true, y_predictions):
    spearman_metric = 0.0
    if y_predictions.ndim == 1 or (y_predictions.ndim == 2 and y_predictions.shape[1] > 1):
        spearman_metric = [spearmanr(y_true[:, i], y_predictions[:, i]).correlation for i in
                           range(y_predictions.shape[1])]
    else:
        spearman_metric += spearmanr(y_true, y_predictions).correlation
    #     spearman_r, _ = spearmanr(y_true, y_predictions)
    return np.mean(spearman_metric)


def calculate_r_squared(y_true, y_predictions):
    r_squared = r2_score(y_true, y_predictions)
    return r_squared


def calculate_accuracy(y_true, y_predictions):
    correct_predictions = sum(1 for true, pred in zip(y_true, y_predictions) if true == pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy * 100

def calculate_accuracy_multi(y_true, y_predictions):
    y_true = np.array(y_true)
    y_predictions = np.array(y_predictions)
    accuracy = []
    for label in range(3):
        acc = accuracy_score(y_true[y_true==label], y_predictions[y_true==label])
        accuracy.append(acc)
    return accuracy 

def calculate_f1_score_multi(y_true, y_predictions):
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_predictions = np.array(y_predictions)
    f1_per_label = []
    for label in range(3):
        f1 = f1_score(y_true[y_true==label], y_predictions[y_true==label], average=None)
        f1_per_label.append(f1[label].item())
    
    return f1_per_label
