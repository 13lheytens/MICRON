import torch
import numpy as np
from data_loader import get_voc_sizes, get_train_val_data, get_vector_representation
from utils import classification_metrics, original_metrics


# Evaluate on the validation samples using the metrics function in the original codebase.
def original_evaluate(val_samples, num_med_codes):
    all_cur_meds_true = []
    all_cur_meds_pred = []
    all_cur_meds_score = []
    for sample in val_samples:
        _, _, prev_meds, cur_meds = sample

        # Predict that the prescription will not change.
        y_hat = torch.LongTensor(get_vector_representation(prev_meds, num_med_codes))
        y_pred = y_hat > .5
        y = torch.LongTensor(cur_meds)

        all_cur_meds_true.append(y.to('cpu').long().numpy())
        all_cur_meds_pred.append(y_pred.to('cpu').long().numpy())
        all_cur_meds_score.append(y_hat.to('cpu').long().numpy())

    return original_metrics(np.array(all_cur_meds_true), np.array(all_cur_meds_pred), np.array(all_cur_meds_score))


# Evaluate the model on the validation samples using simple sklearn classification metrics, where each medication for
# each visit is equally weighted.
def evaluate(val_samples, num_med_codes):
    all_cur_meds_true = torch.LongTensor()
    all_cur_meds_pred = torch.LongTensor()
    all_cur_meds_score = torch.FloatTensor()
    for sample in val_samples:
        _, _, prev_meds, cur_meds = sample

        # Predict that the prescription will not change.
        y_hat = torch.LongTensor(get_vector_representation(prev_meds, num_med_codes))
        y_pred = y_hat > .5
        y = torch.LongTensor(cur_meds)

        all_cur_meds_true = torch.cat((all_cur_meds_true, y.to('cpu').long()), dim=0)
        all_cur_meds_pred = torch.cat((all_cur_meds_pred, y_pred.to('cpu').long()), dim=0)
        all_cur_meds_score = torch.cat((all_cur_meds_score, y_hat.to('cpu')), dim=0)

    return classification_metrics(all_cur_meds_score.detach().numpy(),
                                  all_cur_meds_pred.detach().numpy(),
                                  all_cur_meds_true.detach().numpy())


def run_model():
    _, _, num_med_codes = get_voc_sizes()
    _, val_samples = get_train_val_data(num_med_codes)

    evaluate(val_samples, num_med_codes)
    original_evaluate(val_samples, num_med_codes)


if __name__ == '__main__':
    run_model()
