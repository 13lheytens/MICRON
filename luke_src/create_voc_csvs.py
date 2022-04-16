import dill
import pandas as pd


# File Description:
# This is a script for transforming the "../data/voc_final.pkl" file into csvs.

# @heytens: On my laptop, I was having issues loading the `voc_final.pkl` file. The dill package was throwing an
#           exception as it loaded the binary data (only for the `voc_final.pkl` file). After spending too much time
#           trying to fix it, I decided to use an AWS server to generate voc CSVs instead.
#
#           This file creates 3 voc csvs from the binary data in `voc_final.pkl`.
def main():
    # load data
    data_dir = '../data/'
    voc_path = data_dir + 'voc_final.pkl'
    voc = dill.load(open(voc_path, 'rb'))
    print(voc.keys())
    
    for k, v in voc.items():
        df = pd.DataFrame.from_dict(v.idx2word, orient='index').reset_index()
        df.columns = ['idx', 'word']
        df.to_csv(data_dir + k + '.csv', index=False)
        
if __name__ == '__main__':
    main()
