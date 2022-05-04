import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dill
import argparse
import datetime

from torch.optim import Adam

from data_loader import get_voc_sizes, get_train_test_data
from utils import classification_metrics, original_metrics, print_original_metric_results, print_simple_metric_results
from gcn import GraphConvolution


# File Description: GameNet implementation.
# This file contains a GameNet implementation for predicting patient medication. It is also a script that will train and
# evaluate the model.

# Transforms visit medical codes into multi-hot representation.
def get_multihot_visit_med_codes(med_len, visit_codes_med):
    res = [0] * med_len
    for code in visit_codes_med:
        res[code] = 1
    return res


# Transforms patient medical codes history into multi-hot encoded representation. This is used during the Dynamic Memory
# stage of the GameNet model.
def get_encoded_hist_medical_codes(med_len, num_visits, med_codes):
    res = np.zeros((num_visits, med_len))
    for i, visit_codes in enumerate(med_codes):
        res[i] = get_multihot_visit_med_codes(med_len, visit_codes)
    return torch.FloatTensor(res)


# This class embeds diagnosis and procedural codes and applies a multi-layer gated recurrent unit (GRU) RNN to the
# input sequence.
class EmbedAndRNN(nn.Module):
    def __init__(self, codes_len, output_dim):
        super(EmbedAndRNN, self).__init__()

        # Embed layer to transform input into size 128.
        self.embedding = nn.Embedding(codes_len, 128)

        # RNN layer with gru.
        self.gru = nn.GRU(128, output_dim, batch_first=True)

    def forward(self, codes):
        embedded_codes = torch.stack(
            [torch.sum(self.embedding(visit_codes), dim=0) for visit_codes in codes]).unsqueeze(dim=0)

        result, _ = self.gru(embedded_codes)
        return result


# This class takes in a patient's current and previous visit diagnosis and procedural codes, embeds them, runs them
# through a RNN, then combines them into a total patient history, and passes it through a linear layer.
class PatientDiagAndProcHistory(nn.Module):
    def __init__(self, diag_len, proc_len, output_dim=16):
        super(PatientDiagAndProcHistory, self).__init__()

        # Diagnosis codes embedding layer.
        self.diag_embedding_rnn = EmbedAndRNN(diag_len, 32)

        # Procedural codes embedding layer.
        self.proc_embedding_rnn = EmbedAndRNN(proc_len, 32)

        # Linear layer for combining the diagnosis and procedural code representations.
        self.linear_1 = nn.Linear(2 * 32, output_dim)

    def forward(self, diag_codes, proc_codes):
        embedded_diag = self.diag_embedding_rnn(diag_codes)
        embedded_proc = self.proc_embedding_rnn(proc_codes)

        # Place both embedded codes into one tensor.
        combined_inputs = torch.cat([embedded_diag, embedded_proc], dim=-1).squeeze(0)
        return self.linear_1(combined_inputs)


# This class uses Graph Convolution to learn weights to attach to a static adjacency matrix. This will be used for the
# static EHR and DDI adjacency matrices.
class GCN(nn.Module):
    def __init__(self, med_len, adj_matrix, emb_dim=16):
        super(GCN, self).__init__()
        self.eye_med = torch.eye(med_len)
        self.gcn_1 = GraphConvolution(med_len, emb_dim)
        self.gcn_2 = GraphConvolution(emb_dim, emb_dim)
        self.normalized_adjacency_matrix = self.normalize_adjacency_matrix(adj_matrix)

    def forward(self):
        # This GCN implementation uses 2 layers of GraphConvolution.
        first_pass_gcn = self.gcn_1(self.eye_med, self.normalized_adjacency_matrix)
        return self.gcn_2(F.relu(first_pass_gcn), self.normalized_adjacency_matrix)

    def normalize_adjacency_matrix(self, adj_matrix):
        return torch.FloatTensor(adj_matrix / (adj_matrix +
                                               np.eye(adj_matrix.shape[0])  # add the identity matrix
                                               ).sum(1))


# This class holds the static memory associated with EHR and DDI static information (adjacency matrices). The GCN
# member variables are used to learn weights to attach to this adjacency matrix info. nn.Parameters are also used as
# weights for each adj matrix as they are combined.
class StaticMemoryBank(nn.Module):
    def __init__(self, med_len, adj_ehr, adj_ddi, emb_dim=16):
        super(StaticMemoryBank, self).__init__()
        self.gcn_ehr = GCN(med_len, adj_ehr, emb_dim)
        self.gcn_ddi = GCN(med_len, adj_ddi, emb_dim)

        # Initialize parameters for ehr weight and ddi weight.
        # .8 and .05 are random numbers used as initial values. I hard-coded these to make the results more
        # deterministic.
        self.ehr_weight = nn.Parameter(torch.FloatTensor([[.80]]))
        self.ddi_weight = nn.Parameter(torch.FloatTensor([[.05]]))

    def forward(self):
        info_ehr = self.gcn_ehr.forward()
        info_ddi = self.gcn_ddi.forward()
        return info_ehr * self.ehr_weight - info_ddi * self.ddi_weight


# Combines 3 facts (patient query, static memory result, dynamic memory result) into a single output. Uses 2 different
# linear layers.
class OutNet(nn.Module):
    def __init__(self, med_len, emb_dim=16):
        super(OutNet, self).__init__()
        self.linear_1 = nn.Linear(3 * emb_dim, 2 * emb_dim)
        self.linear_2 = nn.Linear(2 * emb_dim, med_len)

    def forward(self, fact1, fact2, fact3):
        return self.linear_2(F.relu(self.linear_1(torch.cat([fact1, fact2, fact3], -1))))


# Parent GameNet module that executes all steps.
class GameNet(nn.Module):
    def __init__(self, diag_len, proc_len, med_len, adj_ehr, adj_ddi, emb_dim=128):
        super(GameNet, self).__init__()
        self.med_len = med_len
        self.patient_diag_proc_history = PatientDiagAndProcHistory(diag_len, proc_len, emb_dim)
        self.static_memory = StaticMemoryBank(med_len, adj_ehr, adj_ddi, emb_dim)
        self.outnet = OutNet(med_len, emb_dim)

    def forward(self, diag_codes, proc_codes, hist_med_codes):
        diag_proc_history = self.patient_diag_proc_history(diag_codes, proc_codes)
        static_memory_ehr_ddi = self.static_memory()

        # Multi-hot encoded historical medical codes.
        encoded_hist_medical_codes = get_encoded_hist_medical_codes(self.med_len,
                                                                    len(diag_proc_history),
                                                                    hist_med_codes)

        # Fact 1: the current visit's vector representation of diagnosis and procedural codes.
        cur_visit_query = diag_proc_history[[-1]]

        # Fact 2: the static memory result for the current visit query.
        static_memory_result = torch.mm(F.softmax(torch.mm(cur_visit_query,
                                                           static_memory_ehr_ddi.t()), dim=1),
                                        static_memory_ehr_ddi)

        # Fact 3: the dynamic memory result for the current visit query. This is the result of combining the current
        # visit, all historical visits, the historical medical codes, and the static memory (ehr and ddi info).
        dynamic_memory_result = torch.mm(torch.mm(F.softmax(torch.mm(cur_visit_query,
                                                                     diag_proc_history.t()), dim=1),
                                                  encoded_hist_medical_codes),
                                         static_memory_ehr_ddi)

        # Combine all facts into the result.
        return self.outnet(cur_visit_query, static_memory_result, dynamic_memory_result)


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

        diag_codes, proc_codes, prev_med_codes, cur_meds = sample_patient
        y_hat = []
        y = []

        # Each visit's prediction requires the entire patient medical history.
        for i, visit_meds in enumerate(cur_meds):
            # All hist diagnosis codes, including the current visit.
            all_hist_diag_codes = diag_codes[:i + 1]
            # All hist procedural codes, including the current visit.
            all_hist_proc_codes = proc_codes[:i + 1]
            # All hist medical codes, NOT including the current visit.
            # Start at 1 because prev_med_codes[i] holds the med codes from visit i-1.
            all_hist_med_codes = prev_med_codes[1:i + 1]

            # Predict current visit medications.
            y_hat.append(model.forward(all_hist_diag_codes,
                                       all_hist_proc_codes,
                                       all_hist_med_codes).squeeze().detach().numpy())
            y.append(visit_meds.squeeze().detach().numpy())

        # y_hat holds the predicted medication probabilities.
        y_hat = torch.LongTensor(y_hat)
        y = torch.LongTensor(y)
        y_pred = y_hat > .5

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
def train_model(model, train_samples, test_samples, optimizer, num_epochs, criterion):
    start_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        model.train()
        curr_epoch_loss = []
        patient_num = 1
        for diag_codes, proc_codes, prev_meds, cur_meds in train_samples:
            patient_num += 1

            # Zero out the currently accumulated gradient.
            optimizer.zero_grad()
            loss = 0

            # Each visit's prediction requires the entire patient medical history.
            for i, visit_meds in enumerate(cur_meds):
                # All hist diagnosis codes, including the current visit.
                all_hist_diag_codes = diag_codes[:i + 1]
                # All hist procedural codes, including the current visit.
                all_hist_proc_codes = proc_codes[:i + 1]
                # All hist medical codes, NOT including the current visit.
                # Start at 1 because prev_med_codes[i] holds the med codes from visit i-1.
                all_hist_med_codes = prev_meds[1:i + 1]

                # Predict medications.
                y_hat = model.forward(all_hist_diag_codes, all_hist_proc_codes, all_hist_med_codes)
                y = visit_meds.reshape(1, -1)

                # Compute the loss.
                loss += criterion(y_hat, y)

            # Take the mean loss.
            loss /= len(cur_meds)

            # Backprop the gradients.
            loss.backward()
            optimizer.step()
            curr_epoch_loss.append(loss.cpu().data.numpy())

            if patient_num % 500 == 0:
                print(f"epoch {epoch}, patient_num {patient_num}: curr_epoch_loss={np.mean(curr_epoch_loss)}")

        # Evaluate the model after each epoch.
        evaluate(model, test_samples)

    print('Total Training time', (datetime.datetime.now() - start_time).total_seconds())


# Trains and/or evaluates the model.
def run_model(test_only, save_trained_model):
    # Fetch the max index for diag, proc, and med codes.
    num_diag_codes, num_proc_codes, num_med_codes = get_voc_sizes()

    # Load static EHR and DDI information.
    adj_ehr = dill.load(open('../data/ehr_adj_final.pkl', 'rb'))
    adj_ddi = dill.load(open('../data/ddi_A_final.pkl', 'rb'))

    # Initialize the model.
    model = GameNet(num_diag_codes, num_proc_codes, num_med_codes, adj_ehr, adj_ddi)
    print(model)

    # Fetch the train and test samples.
    train_samples, _, test_samples = get_train_test_data(num_med_codes)

    if test_only:
        print('Loading Pretrained Model...')
        # Load the pretrained model.
        model.load_state_dict(torch.load(open('pretrained_models/GameNet.model', 'rb')))

        # Evaluate the pre-trained model.
        evaluate(model, test_samples)
        return

    # Adam optimizer outperforms SGD for this model.
    optimizer = Adam(model.parameters(), lr=.001)

    # Use binary cross entropy as the loss function.
    criterion = F.binary_cross_entropy_with_logits

    # Train the model.
    train_model(model, train_samples, test_samples, optimizer, 5, criterion)

    # Evaluate the fully trained model.
    evaluate(model, test_samples)

    if save_trained_model:
        # Save the fully trained model.
        torch.save(model.state_dict(), open('pretrained_models/GameNet.model', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--save_trained", action='store_true')
    args = parser.parse_args()

    run_model(args.test_only, args.save_trained)
