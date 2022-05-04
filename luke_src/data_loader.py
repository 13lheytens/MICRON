import pandas as pd
import dill
import random
import torch


# File Description:
# This file contains functions for fetching data from the "../data" directory.

# Transforms the input indices into a binary vector representation.
def get_vector_representation(input_indices, max_value):
    res = [0] * max_value
    for i in input_indices:
        res[i] = 1
    return res


# Retrieves all records and splits into train/test sets. Data is of shape (num_patients, num_patient_visits, 4). Also
# batches train samples in batch_size visits for faster training.
def get_train_test_data(num_med_codes, train_test_split=.8, batch_size=32):
    records = dill.load(open('../data/records_final.pkl', 'rb'))
    all_samples = []
    for patient in records:
        visit_samples = [[], [], [], []]
        # Skip the first visit because there are no previous medications.
        for visit in range(1, len(patient)):
            # Get the codes for the current visit, and the previous visit.
            cur_diag_codes, prev_diag_codes = patient[visit][0], patient[visit - 1][0]
            cur_proc_codes, prev_proc_codes = patient[visit][1], patient[visit - 1][1]
            cur_med_codes, prev_med_codes = patient[visit][2], patient[visit - 1][2]

            visit_samples[0].append(torch.LongTensor(cur_diag_codes))  # diag codes
            visit_samples[1].append(torch.LongTensor(cur_proc_codes))  # proc codes
            visit_samples[2].append(torch.LongTensor(prev_med_codes))  # previous meds
            # Transform the target "cur_med_codes" into a binary vector representation.
            visit_samples[3].append(
                torch.FloatTensor(get_vector_representation(cur_med_codes, num_med_codes)))  # current meds (target)

        if len(visit_samples[0]) > 0:
            all_samples.append(visit_samples)

    # Shuffle all samples.
    random.seed(10)
    random.shuffle(all_samples)

    # Split into train and test sets.
    split = int(len(all_samples) * train_test_split)
    train, test = all_samples[:split], all_samples[split:]
    print('Number of train samples:     ', len(train))
    print('Number of train patient visits:     ', sum([len(patient[0]) for patient in train]))
    print('Number of test samples:', len(test))
    print('Number of test patient visits:', sum([len(patient[0]) for patient in test]))

    # Batch the train samples into batch_size for faster training. Each batch sample will contain batch_size number of
    # patient visits.
    batched_train = []
    cur_batch = [[], [], [], []]
    for patient in train:
        diag_codes, proc_codes, prev_meds, cur_meds = patient
        for i in range(len(diag_codes)):
            cur_batch[0].append(diag_codes[i])
            cur_batch[1].append(proc_codes[i])
            cur_batch[2].append(prev_meds[i])
            cur_batch[3].append(cur_meds[i])
            if len(cur_batch[0]) == batch_size:
                batched_train.append(cur_batch)
                cur_batch = [[], [], [], []]
    if len(cur_batch[0]) > 0:
        batched_train.append(cur_batch)
    print('Number of train batches:', len(batched_train))

    return train, batched_train, test


# The Micron model takes in the current visit diagnosis and procedural codes, as well as the previous visit's diagnosis,
# procedural, and medical codes. This function is used specifically for the MICRON model.
def get_micron_train_test_data(num_med_codes, train_test_split=.8, batch_size=32):
    records = dill.load(open('../data/records_final.pkl', 'rb'))
    all_samples = []
    for patient in records:
        visit_samples = [[], [], [], [], [], []]
        # Skip the first visit because there are no previous medications.
        for visit in range(1, len(patient)):
            # Get the codes for the current visit, and the previous visit.
            cur_diag_codes, prev_diag_codes = patient[visit][0], patient[visit - 1][0]
            cur_proc_codes, prev_proc_codes = patient[visit][1], patient[visit - 1][1]
            cur_med_codes, prev_med_codes = patient[visit][2], patient[visit - 1][2]

            visit_samples[0].append(torch.LongTensor(cur_diag_codes))  # diag codes
            visit_samples[1].append(torch.LongTensor(cur_proc_codes))  # proc codes
            visit_samples[2].append(torch.LongTensor(prev_diag_codes))  # previous diag
            visit_samples[3].append(torch.LongTensor(prev_proc_codes))  # previous proc
            visit_samples[4].append(torch.LongTensor(prev_med_codes))  # previous meds
            visit_samples[5].append(
                torch.FloatTensor(get_vector_representation(cur_med_codes, num_med_codes)))  # current meds (target)

        if len(visit_samples[0]) > 0:
            all_samples.append(visit_samples)

    # Shuffle all samples.
    random.seed(10)
    random.shuffle(all_samples)

    # Split into train and test sets.
    split = int(len(all_samples) * train_test_split)
    train, test = all_samples[:split], all_samples[split:]
    print('Number of train samples:     ', len(train))
    print('Number of train patient visits:     ', sum([len(patient[0]) for patient in train]))
    print('Number of test samples:', len(test))
    print('Number of test patient visits:', sum([len(patient[0]) for patient in test]))

    # Batch the train samples into batch_size for faster training. Each batch sample will contain batch_size number of
    # patient visits.
    batched_train = []
    cur_batch = [[], [], [], [], [], []]
    for patient in train:
        diag_codes, proc_codes, prev_diag, prev_proc, prev_meds, cur_meds = patient
        for i in range(len(diag_codes)):
            cur_batch[0].append(diag_codes[i])
            cur_batch[1].append(proc_codes[i])
            cur_batch[2].append(prev_diag[i])
            cur_batch[3].append(prev_proc[i])
            cur_batch[4].append(prev_meds[i])
            cur_batch[5].append(cur_meds[i])
            if len(cur_batch[0]) == batch_size:
                batched_train.append(cur_batch)
                cur_batch = [[], [], [], [], [], []]
    if len(cur_batch[0]) > 0:
        batched_train.append(cur_batch)
    print('Number of train batches:', len(batched_train))

    return train, batched_train, test


# Returns the maximum index for diagnosis codes, procedural codes, and medical codes.
def get_voc_sizes():
    # Load diagnosis mapping, and ensure that the length matches the max numerical value.
    diag_voc = pd.read_csv('../data/diag_voc.csv', index_col='idx').index
    num_diag_codes = len(diag_voc)
    assert (diag_voc.max() == num_diag_codes - 1)

    # Load procedure mapping, and ensure that the length matches the max numerical value.
    proc_voc = pd.read_csv('../data/pro_voc.csv', index_col='idx').index
    num_proc_codes = len(proc_voc)
    assert (proc_voc.max() == num_proc_codes - 1)

    # Load medical code mapping, and ensure that the length matches the max numerical value.
    med_voc = pd.read_csv('../data/med_voc.csv', index_col='idx').index
    num_med_codes = len(med_voc)
    assert (med_voc.max() == num_med_codes - 1)

    return num_diag_codes, num_proc_codes, num_med_codes
