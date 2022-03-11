import dill
import pandas as pd

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
