from sklearn.metrics import *

import sys, os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The `multi_label_metric` function is what was used in the original paper.
from src.util import multi_label_metric

# Computes various metrics based on predicted and true labels.
def classification_metrics(Y_score, Y_pred, Y_true):
    acc, auc, precision, recall, f1score, jaccard = accuracy_score(Y_true, Y_pred), \
                                           roc_auc_score(Y_true, Y_score), \
                                           precision_score(Y_true, Y_pred), \
                                           recall_score(Y_true, Y_pred), \
                                           f1_score(Y_true, Y_pred, average='binary'), \
                                           jaccard_score(Y_true, Y_pred)
    print(f"[LUKE'S SIMPLE SKLEARN METRICS] acc: {acc:.3f}, auc: {auc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, "
          f"f1: {f1score:.3f}, jaccard: {jaccard:.3f}")

    return acc, auc, precision, recall, f1score, jaccard

# Computes metrics using the original author's util function `multi_label_metric`. Results from this function were
# quoted in the original paper.
def original_metrics(Y_true, Y_pred, Y_score):
     adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(Y_true, Y_pred, Y_score)
     print(f"[ORIGINAL CODEBASE METRICS] prauc: {adm_prauc:.3f}, precision: {adm_avg_p:.3f}, "
           f"recall: {adm_avg_r:.3f}, f1: {adm_avg_f1:.3f}, jaccard: {adm_ja:.3f}")
     return adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1
