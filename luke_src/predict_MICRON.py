import torch
import torch.nn as nn
import numpy as np
import argparse
import datetime

from torch.optim import Adam

from data_loader import get_voc_sizes, get_micron_train_test_data
from utils import classification_metrics, original_metrics, print_original_metric_results, print_simple_metric_results


# File Description: MICRON implementation.
# This file contains my MICRON implementation for predicting patient medication. It is also a script that will train and
# periodically evaluate the model.

# Luke's implementation of the MICRON model for prediction patient visit medications. This differs in a few ways from
# the original implementation: (1) the DDI adjacency matrix is not used, and (2) the model only returns the predicted
# medical prescriptions. The original model's DDI adj matrix was only used as an addition to the loss function (it
# wasn't used in the actual medication prediction). I simplified the loss function in my model (while still maintaining
# roughly the same F1 score), so the DDI info was no longer used and I removed it. The original model returned several
# terms as outputs of its model (drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss), and many were only
# used as inputs into a complex loss function. My model only outputs the predicted medical codes for the current visit.
class LukeMicron(nn.Module):
    def __init__(self, diag_len, proc_len, med_len):
        super(LukeMicron, self).__init__()

        # Diagnosis codes embedding layer.
        self.diag_embedding = nn.Embedding(diag_len, 128)

        # Procedural codes embedding layer.
        self.proc_embedding = nn.Embedding(proc_len, 128)

        # Map previous meds from med_len -> 128.
        self.med_embedding = nn.Embedding(med_len, 128)

        # Map combined diagnosis and procedural code embeddings 128*2 -> 128.
        self.linear_1 = nn.Linear(128 * 2, 128)
        self.dropout = nn.Dropout(.5)

        # Map current diag/proc codes (128) previous diag/proc codes (128) and previous med codes (128) -> 256.
        self.linear_2 = nn.Linear(128 * 3, 256)
        # Transform 256 -> medical code encoding.
        self.linear_3 = nn.Linear(256, med_len)

        # Combines diagnosis and procedural codes for a single visit.
        self.health_net = nn.Sequential(self.linear_1,
                                        self.dropout)

        # Uses current diagnosis/procedural codes, the previous visit diagnosis/procedural codes, and the previous
        # visit medical codes to make a prediction about current visit medical codes. Prescription net is where we infer
        # CHANGE in the medical prescriptions (by passing in the previous visit's medical codes).
        self.prescription_net = nn.Sequential(self.linear_2,
                                              nn.ReLU(),
                                              self.linear_3,
                                              nn.Sigmoid())

    # Uses diagnosis and procedural codes, along with the previous visit's prescription to predict the current visit
    # medical code prescriptions.
    def forward(self, diag_codes, proc_codes, prev_diag_codes, prev_proc_codes, prev_med_codes):
        # Sum the embedded diagnosis codes that are set for a single visit, then stack into a tensor with shape (num_visits, 128).
        embedded_diag = torch.stack(
            [torch.sum(self.diag_embedding(visit_diag_codes), dim=0) for visit_diag_codes in diag_codes])
        # Sum the embedded procedural codes that are set for a single visit, then stack into a tensor with shape (num_visits, 128).
        embedded_proc = torch.stack(
            [torch.sum(self.proc_embedding(visit_proc_codes), dim=0) for visit_proc_codes in proc_codes])

        # Create the previous visit diagnosis and procedural code embeddings.
        prev_embedded_diag = torch.stack(
            [torch.sum(self.diag_embedding(prev_visit_diag_codes), dim=0) for prev_visit_diag_codes in prev_diag_codes])
        prev_embedded_proc = torch.stack(
            [torch.sum(self.proc_embedding(prev_visit_proc_codes), dim=0) for prev_visit_proc_codes in prev_proc_codes])

        # Current visit health net representation based on diagnosis and procedural codes.
        cur_health_net_result = self.health_net(torch.cat([embedded_diag, embedded_proc], dim=1))
        # Previous visit health net representation based on last visit diagnosis and procedural codes.
        prev_health_net_result = self.health_net(torch.cat([prev_embedded_diag, prev_embedded_proc], dim=1))

        # Sum the embedded medical codes that are set from the previous visit, then stack into a tensor with shape
        # (num_visits, 128).
        embedded_prev_codes = torch.stack(
            [torch.sum(self.med_embedding(med_codes), dim=0) for med_codes in prev_med_codes])

        # Combine the current health net result, previous health net result, and previous medical codes into a single
        # representation. This allows the prescription net to infer CHANGE in medication from the medical codes from the
        # previous visit.
        prescription_representation = torch.cat([cur_health_net_result, prev_health_net_result, embedded_prev_codes],
                                                dim=1)

        # In my implementation, I use the (embedded) previous visit's medical codes as input into this final stage of
        # the model. In this way, the CHANGE in prescriptions is inferred by the model, and the output is the current
        # visit prescriptions. This simplifies the loss function greatly, while still maintaining the original paper's
        # emphasis on the importance of CHANGE in medical prescription.
        return self.prescription_net(prescription_representation)


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
            print('Evaluating test patient number:', patient_num, '-', patient_num + 500)
        patient_num += 1

        diag_codes, proc_codes, prev_diag, prev_proc, prev_med_codes, cur_meds = sample_patient

        # y_hat holds the predicted medication probabilities. This differs from the original model, where the output of
        # the model was several different values (drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss). Some
        # of those values are only used for training as inputs into a complex loss function. I chose to simplify the
        # logic by only returning the current visit medical prescriptions.
        y_hat = model(diag_codes, proc_codes, prev_diag, prev_proc, prev_med_codes)
        y_pred = y_hat > .5
        y = torch.stack(cur_meds)

        # Compute metrics for the current patient, and append to metrics array.
        original_patient_metrics.append(
            np.array(original_metrics(y.detach().numpy(), y_pred.detach().numpy(), y_hat.detach().numpy())))
        lukes_simple_patient_metrics.append(np.array(classification_metrics(y_hat.detach().numpy().reshape(-1),
                                                                            y_pred.detach().numpy().reshape(-1),
                                                                            y.detach().numpy().reshape(-1))))

    # Print aggregated metrics, with each patient equally weighted.
    print_original_metric_results(np.array(original_patient_metrics))
    print_simple_metric_results(np.array(lukes_simple_patient_metrics))


# Trains the model.
def train_model(model, train_samples, optimizer, num_epochs, criterion):
    start_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        model.train()
        curr_epoch_loss = []
        for diag_codes, proc_codes, prev_diag, prev_proc, prev_meds, cur_meds in train_samples:
            # Zero out the currently accumulated gradient.
            optimizer.zero_grad()

            # Predict medications. This differs from the original model, where the output of the model was several
            # different values (drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss). Some of those values
            # are only used for training as inputs into a complex loss function. I chose to simplify the logic by only
            # returning the current visit medical prescription probabilities.
            y_hat = model.forward(diag_codes, proc_codes, prev_diag, prev_proc, prev_meds)

            # Batch the target values into the expected tensor format.
            y = torch.stack(cur_meds)

            # Compute the loss.
            loss = criterion(y_hat, y)

            # Backprop the gradients.
            loss.backward()
            optimizer.step()
            curr_epoch_loss.append(loss.cpu().data.numpy())

        print(f"epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")

    print('Total Training time', (datetime.datetime.now() - start_time).total_seconds())


# Trains and/or evaluates the model.
def run_model(test_only, save_trained_model):
    # Fetch the max index for diag, proc, and med codes.
    num_diag_codes, num_proc_codes, num_med_codes = get_voc_sizes()

    # Fetch the train and test samples.
    train_samples, batch_train_samples, test_samples = get_micron_train_test_data(num_med_codes, batch_size=128)

    # Initialize the model.
    model = LukeMicron(num_diag_codes, num_proc_codes, num_med_codes)
    print(model)

    if test_only:
        print('Loading Pretrained Model...')
        # Load the pretrained model.
        model.load_state_dict(torch.load(open('pretrained_models/MICRON.model', 'rb')))

        # Evaluate the pre-trained model.
        evaluate(model, test_samples)
        return

    # Adam optimizer outperforms SGD for this model.
    optimizer = Adam(model.parameters(), lr=.001)

    # BCELoss is used here because the target a binary vector (ex: [0, 0, 1, 0, 1, ..., 1]). This vector represents the
    # medications that are prescribed to a patient for a given visit.
    criterion = torch.nn.BCELoss()

    # Train the model.
    train_model(model, batch_train_samples, optimizer, 15, criterion)

    # Evaluate the fully trained model.
    evaluate(model, test_samples)

    if save_trained_model:
        # Save the fully trained model.
        torch.save(model.state_dict(), open('pretrained_models/MICRON.model', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--save_trained", action='store_true')
    args = parser.parse_args()

    run_model(args.test_only, args.save_trained)
