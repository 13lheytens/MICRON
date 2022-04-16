# Code structure
- predict_*
    -   Each file contains its own model.
    -   To train and evaluate the model, simply invoke the python script.
        - Ex: `python3 predict_no_change.py`
- create_voc_csvs.py
    - A script for transforming the "../data/voc_final.pkl" file into csvs.
- data_loader.py
    - Contains helper functions for loading preprocessed train/test data.
- utils.py
    - Contains helper functions for evaluating models (F1 score, precision, recall, etc.).

