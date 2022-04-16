import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

from data_loader import get_voc_sizes, get_train_test_data
from utils import classification_metrics, original_metrics, print_original_metric_results, print_simple_metric_results


# File Description:
# This file contains a NN implementation for predicting patient medication. It is also a script that will train and
# periodically evaluate the model.

# Neural Network that uses current diagnosis codes and procedural codes to predict medication prescriptions.
class NeuralNetNoHistory(nn.Module):
    def __init__(self, diag_len, proc_len, med_len):
        super(NeuralNetNoHistory, self).__init__()

        # Diagnosis codes embedding layer.
        self.diag_embedding = nn.Embedding(diag_len, 128)

        # Procedural codes embedding layer.
        self.proc_embedding = nn.Embedding(proc_len, 128)

        # Map combined embeddings 256 -> 64.
        self.linear_1 = nn.Linear(256, 64)

        # Map to the same shape as the target.
        self.linear_2 = nn.Linear(64, med_len)

        self.dropout = nn.Dropout(.5)
        self.nn_steps = nn.Sequential(self.linear_1,
                                      nn.ReLU(),
                                      self.dropout,
                                      self.linear_2,
                                      nn.Sigmoid()
                                      )

    # Uses diagnosis and procedural codes to predict the current visit medical code prescriptions.
    def forward(self, diag_codes, proc_codes):
        # Sum the embedded diagnosis codes that are set for a single visit, then stack into a tensor with
        # shape (num_visits, 128).
        embedded_diag = torch.stack(
            [torch.sum(self.diag_embedding(visit_diag_codes), dim=0) for visit_diag_codes in diag_codes])
        # Sum the embedded procedural codes that are set for a single visit, then stack into a tensor with
        # shape (num_visits, 128).
        embedded_proc = torch.stack(
            [torch.sum(self.proc_embedding(visit_proc_codes), dim=0) for visit_proc_codes in proc_codes])

        # Combine the embedded codes into one tensor with shape (num_visits, 256).
        combined_inputs = torch.cat([embedded_diag, embedded_proc], dim=1)

        # Predict the medicine prescriptions for the batch.
        return self.nn_steps(combined_inputs)


# Evaluate the model on the test samples using both (1) the original metrics from the paper and (2) simple sklearn
# classification metrics. Each patient is scored, and weighted equally in the aggregated scores.
def evaluate(model, test_samples):
    model.eval()

    original_patient_metrics = []
    lukes_simple_patient_metrics = []
    patient_num = 0
    for sample_patient in test_samples:
        # Print evaluation progress.
        if patient_num % 500 == 0:
            print('Evaluating test patient number:', patient_num)
        patient_num += 1

        diag_codes, proc_codes, _, cur_meds = sample_patient

        # y_hat holds the predicted medication probabilities.
        y_hat = model(diag_codes, proc_codes)
        y_pred = y_hat > .5
        y = torch.stack(cur_meds)

        # Compute metrics for the current patient, and append to metrics array.
        original_patient_metrics.append(np.array(original_metrics(y.detach().numpy(),
                                                                  y_pred.detach().numpy(),
                                                                  y_hat.detach().numpy())))
        lukes_simple_patient_metrics.append(np.array(classification_metrics(y_hat.detach().numpy().reshape(-1),
                                                                            y_pred.detach().numpy().reshape(-1),
                                                                            y.detach().numpy().reshape(-1))))

    # Print aggregated metrics, with each patient equally weighted.
    print_original_metric_results(np.array(original_patient_metrics))
    print_simple_metric_results(np.array(lukes_simple_patient_metrics))


# Trains the model, and checks the test results every 5 epochs.
def train_model(model, train_samples, test_samples, optimizer, num_epochs, criterion):
    for epoch in range(num_epochs):
        model.train()
        curr_epoch_loss = []
        for diag_codes, proc_codes, _, cur_meds in train_samples:
            # Zero out the currently accumulated gradient.
            optimizer.zero_grad()

            # Predict medications.
            y_hat = model.forward(diag_codes, proc_codes)

            # Batch the target values into the expected tensor format.
            y = torch.stack(cur_meds)

            # Compute the loss.
            loss = criterion(y_hat, y)

            # Backprop the gradients.
            loss.backward()
            optimizer.step()
            curr_epoch_loss.append(loss.cpu().data.numpy())

        print(f"epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
        # Evaluate the model on the test set every 5 epochs.
        if epoch % 5 == 0:
            # Evaluate the partially-trained model.
            evaluate(model, test_samples)


# Trains and periodically evaluates the model.
def run_model():
    # Fetch the max index for diag, proc, and med codes.
    num_diag_codes, num_proc_codes, num_med_codes = get_voc_sizes()

    # Initialize the model.
    model = NeuralNetNoHistory(num_diag_codes, num_proc_codes, num_med_codes)
    print(model)

    # Fetch the train and test samples.
    train_samples, batched_train_samples, test_samples = get_train_test_data(num_med_codes)

    # Adam optimizer outperforms SGD for this model.
    optimizer = Adam(model.parameters(), lr=.001)

    # BCELoss is used here because the target a binary vector (ex: [0, 0, 1, 0, 1, ..., 1]). This vector represents the
    # medications that are prescribed to a patient for a given visit.
    criterion = torch.nn.BCELoss()

    # Train the model.
    train_model(model, batched_train_samples, test_samples, optimizer, 15, criterion)

    # Evaluate the fully trained model.
    evaluate(model, test_samples)

if __name__ == '__main__':
    run_model()
