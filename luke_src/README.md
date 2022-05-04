See parent directory README for more information.

# Code structure
- predict_*
    -   Each file contains its own model.
    -   To train and evaluate the model, simply invoke the python script.
        - Ex: `python3 predict_MICRON.py`
    -   To only evaluate a pretrained model:
        - Ex: `python3 predict_MICRON.py --test_only`
    -   To train and save a model as a pretrained model:
        - Ex: `python3 predict_MICRON.py --save_trained`
- create_voc_csvs.py
    - A script for transforming the "../data/voc_final.pkl" file into csvs.
- data_loader.py
    - Contains helper functions for loading preprocessed train/test data.
- gcn.py
    - Contains the GraphConvolution implementation of the original author, which is used in the GameNet model.
- utils.py
    - Contains helper functions for evaluating models (F1 score, precision, recall, etc.).
- data_exploration/
    - Contains jupyter notebooks for evaluating patient data.
- pretrained_models/
    - Contains saved trained model state output.

### Baseline Models:
  - Baseline Model 1 - predict_no_change.py
  - Baseline Model 2 - predict_meds_no_history_nn.py
  - Baseline Model 3 - predict_meds_with_history_nn.py