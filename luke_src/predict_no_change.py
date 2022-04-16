import torch
import numpy as np

from data_loader import get_voc_sizes, get_train_test_data, get_vector_representation
from utils import classification_metrics, original_metrics, print_original_metric_results, print_simple_metric_results


# File Description: Baseline Model 1.
# This is a script that evaluates a simple strategy for predicting patient medication. This strategy is simply
# predicting the previous visit's medication (no change).

# Evaluate predictions of no change in meds. This is run on the test samples using both (1) the original metrics
# from the paper and (2) simple sklearn classification metrics. Each patient is scored, and weighted equally in the
# aggregated scores.
def evaluate_no_change_in_meds(test_samples, num_med_codes):
    original_patient_metrics = []
    lukes_simple_patient_metrics = []
    patient_num = 0
    for sample_patient in test_samples:
        # Print evaluation progress.
        if patient_num % 500 == 0:
            print('Evaluating test patient number:', patient_num)
        patient_num += 1

        _, _, prev_meds, cur_meds = sample_patient

        # Predict that the prescription will not change. The target is in binary vector form, so we need to transform
        # the prev_meds into the same format with get_vector_representation().
        y_hat = torch.LongTensor([get_vector_representation(p, num_med_codes) for p in prev_meds])
        y_pred = y_hat > .5
        y = torch.stack(cur_meds)

        # Compute metrics for the current patient, and append to metrics array.
        original_patient_metrics.append(np.array(original_metrics(y.numpy(), y_pred.numpy(), y_hat.numpy())))
        lukes_simple_patient_metrics.append(np.array(classification_metrics(y_hat.numpy().reshape(-1),
                                                                            y_pred.numpy().reshape(-1),
                                                                            y.numpy().reshape(-1))))

    # Print aggregated metrics, with each patient equally weighted.
    print_original_metric_results(np.array(original_patient_metrics))
    print_simple_metric_results(np.array(lukes_simple_patient_metrics))


def run_model():
    _, _, num_med_codes = get_voc_sizes()
    _, _, test_samples = get_train_test_data(num_med_codes)

    # Evaluate predictions of no change in meds. Meds from the previous visit are used as the prediction for meds of
    # the current visit.
    evaluate_no_change_in_meds(test_samples, num_med_codes)


if __name__ == '__main__':
    run_model()
