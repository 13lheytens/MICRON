import pandas as pd

# @heytens: I copied this Voc class from ../data/preprocessing.py.
##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

# @heytens: Loads voc from csvs.
def load_voc_from_csvs(datadir_path):
    diag_voc = Voc()
    pro_voc = Voc()
    med_voc = Voc()

    diag_voc.idx2word = pd.read_csv(datadir_path + 'diag_voc.csv', index_col='idx').word.to_dict()
    pro_voc.idx2word = pd.read_csv(datadir_path + 'pro_voc.csv', index_col='idx').word.to_dict()
    med_voc.idx2word = pd.read_csv(datadir_path + 'med_voc.csv', index_col='idx').word.to_dict()

    def add_word_to_idx(voc):
        for idx, word in voc.idx2word.items():
            print(idx, word)
            if word not in voc.word2idx:
                voc.word2idx[word] = idx
        return voc

    diag_voc = add_word_to_idx(diag_voc)
    pro_voc = add_word_to_idx(pro_voc)
    med_voc = add_word_to_idx(med_voc)

    return diag_voc, pro_voc, med_voc