import pandas as pd
import dill
import random
import torch


# Transforms the input indices into a binary vector representation.
def get_vector_representation(input_indices, max_value):
    res = [0] * max_value
    for i in input_indices:
        res[i] = 1
    return res


# Retrieves all records and splits into train/validation sets. Transforms train samples into batches.
def get_train_val_data(num_med_codes, batch_size=32, train_val_split=.8):
    records = dill.load(open('../data/records_final.pkl', 'rb'))
    all_samples = []
    for patient in records:
        # Skip the first visit because there are no previous medications.
        for visit in range(1, len(patient)):
            # Get the codes for the current visit, and the previous visit.
            cur_diag_codes, prev_diag_codes = patient[visit][0], patient[visit - 1][0]
            cur_proc_codes, prev_proc_codes = patient[visit][1], patient[visit - 1][1]
            cur_med_codes, prev_med_codes = patient[visit][2], patient[visit - 1][2]

            # Append sample. Transform the target "cur_med_codes" into a binary vector representation.
            all_samples.append([cur_diag_codes, cur_proc_codes, prev_med_codes,
                                get_vector_representation(cur_med_codes, num_med_codes)])

    # Shuffle all samples.
    random.seed(10)
    random.shuffle(all_samples)

    # Split into train and validation sets.
    split = int(len(all_samples) * train_val_split)
    split = split - (split % batch_size)
    print('Number of train samples:     ', split)
    print('Number of validation samples:', len(all_samples) - split)
    train, val = all_samples[:split], all_samples[split:]

    # Batch the train samples.
    batched_train = []
    cur_batch = [[], [], [], []]
    for t_sample in train:
        cur_batch[0].append(torch.LongTensor(t_sample[0]))  # diag codes
        cur_batch[1].append(torch.LongTensor(t_sample[1]))  # proc codes
        cur_batch[2].append(torch.LongTensor(t_sample[2]))  # previous meds
        cur_batch[3].append(torch.FloatTensor(t_sample[3]))  # current meds (target)
        if len(cur_batch[0]) == batch_size:
            batched_train.append(cur_batch)
            cur_batch = [[], [], [], []]
    print('Total training batches', len(batched_train))
    return batched_train, val


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
