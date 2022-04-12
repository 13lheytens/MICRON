import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

from data_loader import get_voc_sizes, get_train_val_data
from utils import classification_metrics, original_metrics


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
        # Sum the embedded diagnosis codes that are set for a single visit, then stack into a tensor with shape (batch_size, 128).
        embedded_diag = torch.stack(
            [torch.sum(self.diag_embedding(visit_diag_codes), dim=0) for visit_diag_codes in diag_codes])
        # Sum the embedded procedural codes that are set for a single visit, then stack into a tensor with shape (batch_size, 128).
        embedded_proc = torch.stack(
            [torch.sum(self.proc_embedding(visit_proc_codes), dim=0) for visit_proc_codes in proc_codes])

        # Combine the embedded codes into one tensor with shape (batch_size, 256).
        combined_inputs = torch.cat([embedded_diag, embedded_proc], dim=1)

        # Predict the medicine prescriptions for the batch.
        return self.nn_steps(combined_inputs)


# Evaluate the model on the validation samples using the metrics function in the original codebase.
def original_evaluate(model, val_samples):
    model.eval()
    all_cur_meds_true = []
    all_cur_meds_pred = []
    all_cur_meds_score = []
    for sample in val_samples:
        diag_codes, proc_codes, _, cur_meds = sample

        # y_hat holds the predicted medication probabilities.
        y_hat = model(torch.LongTensor(diag_codes).unsqueeze(0), torch.LongTensor(proc_codes).unsqueeze(0)).squeeze()
        y_pred = y_hat > .5
        y = torch.LongTensor(cur_meds)

        all_cur_meds_true.append(y.to('cpu').long().numpy())
        all_cur_meds_pred.append(y_pred.to('cpu').long().numpy())
        all_cur_meds_score.append(y_hat.to('cpu').long().numpy())

    return original_metrics(np.array(all_cur_meds_true), np.array(all_cur_meds_pred), np.array(all_cur_meds_score))


# Evaluate the model on the validation samples using simple sklearn classification metrics.
def evaluate(model, val_samples):
    model.eval()
    all_cur_meds_true = torch.LongTensor()
    all_cur_meds_pred = torch.LongTensor()
    all_cur_meds_score = torch.FloatTensor()
    for sample in val_samples:
        diag_codes, proc_codes, _, cur_meds = sample

        # y_hat holds the predicted medication probabilities.
        y_hat = model(torch.LongTensor(diag_codes).unsqueeze(0), torch.LongTensor(proc_codes).unsqueeze(0)).squeeze()
        y_pred = y_hat > .5
        y = torch.LongTensor(cur_meds)

        all_cur_meds_true = torch.cat((all_cur_meds_true, y.to('cpu').long()), dim=0)
        all_cur_meds_pred = torch.cat((all_cur_meds_pred, y_pred.to('cpu').long()), dim=0)
        all_cur_meds_score = torch.cat((all_cur_meds_score, y_hat.to('cpu')), dim=0)

    return classification_metrics(all_cur_meds_score.detach().numpy(),
                                  all_cur_meds_pred.detach().numpy(),
                                  all_cur_meds_true.detach().numpy())


# Trains the model, and checks the validation results every 3 epochs.
def train_model(model, train_samples, val_samples, optimizer, num_epochs, criterion):
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
        # Evaluate the model on the validation set every 3 epochs.
        if epoch % 3 == 0:
            # Evaluate with simple sklearn classification metrics.
            evaluate(model, val_samples)
            # Evaluate with the original codebase metrics function.
            original_evaluate(model, val_samples)


# Trains and periodically evaluates the model.
def run_model():
    # Fetch the max index for diag, proc, and med codes.
    num_diag_codes, num_proc_codes, num_med_codes = get_voc_sizes()

    # Initialize the model.
    model = NeuralNetNoHistory(num_diag_codes, num_proc_codes, num_med_codes)
    print(model)

    # Fetch the train and test samples.
    train_samples, val_samples = get_train_val_data(num_med_codes)

    # Adam optimizer outperforms SGD for this model.
    optimizer = Adam(model.parameters(), lr=.001)

    # BCELoss is used here because the target a binary vector (ex: [0, 0, 1, 0, 1, ..., 1]). This vector represents the
    # medications that are prescribed to a patient for a given visit.
    criterion = torch.nn.BCELoss()

    # Train the model.
    train_model(model, train_samples, val_samples, optimizer, 50, criterion)


if __name__ == '__main__':
    run_model()
