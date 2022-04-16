from sklearn.metrics import *

import sys, os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The `multi_label_metric` function is what was used in the original paper.
from src.util import multi_label_metric


# File Description:
# This file contains various helper functions.

# Computes various metrics based on predicted and true labels.
def classification_metrics(Y_score, Y_pred, Y_true, print_metrics=False):
    acc, auc, precision, recall, f1score, jaccard = accuracy_score(Y_true, Y_pred), \
                                           roc_auc_score(Y_true, Y_score), \
                                           precision_score(Y_true, Y_pred), \
                                           recall_score(Y_true, Y_pred), \
                                           f1_score(Y_true, Y_pred, average='binary'), \
                                           jaccard_score(Y_true, Y_pred)
    if print_metrics:
        print(f"[LUKE'S SIMPLE SKLEARN METRICS] acc: {acc:.3f}, auc: {auc:.3f}, precision: {precision:.3f}, "
              f"recall: {recall:.3f}, f1: {f1score:.3f}, jaccard: {jaccard:.3f}")

    return acc, auc, precision, recall, f1score, jaccard

# The results_array should have one entry for each patient. The results are averaged across each patient.
def print_simple_metric_results(results_array):
    print("[LUKE'S SIMPLE SKLEARN METRICS]")
    print('  acc:', results_array[:, 0].mean())
    print('  auc:', results_array[:, 1].mean())
    print('  precision:', results_array[:, 2].mean())
    print('  recall:', results_array[:, 3].mean())
    print('  f1:', results_array[:, 4].mean())
    print('  jaccard:', results_array[:, 5].mean())

# Computes metrics using the original author's util function `multi_label_metric`. Results from this function were
# quoted in the original paper.
def original_metrics(Y_true, Y_pred, Y_score, print_metrics=False):
     adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(Y_true, Y_pred, Y_score)
     if print_metrics:
         print(f"[ORIGINAL CODEBASE METRICS] prauc: {adm_prauc:.3f}, precision: {adm_avg_p:.3f}, "
               f"recall: {adm_avg_r:.3f}, f1: {adm_avg_f1:.3f}, jaccard: {adm_ja:.3f}")
     return adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1

# The results_array should have one entry for each patient. The results are averaged across each patient.
def print_original_metric_results(results_array):
    print("[ORIGINAL CODEBASE METRICS]")
    print('  jaccard:', results_array[:, 0].mean())
    print('  prauc:', results_array[:, 1].mean())
    print('  precision:', results_array[:, 2].mean())
    print('  recall:', results_array[:, 3].mean())
    print('  f1:', results_array[:, 4].mean())