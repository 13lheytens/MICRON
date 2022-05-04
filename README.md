## Luke's Reproduction of MICRON findings
- `src/` contains the original author's code.
- `luke_src/` contains luke's code as he reproduces the findings of the original author. 
- `data/` contains the data for training and testing, along with a preprocessing script from the original author.

  ### Citation to the original paper and repo
  - This repo is a fork of the original author's (Chaoqi Yang) implementation found here: https://github.com/ycq091044/MICRON
  - The original paper can be found here: https://arxiv.org/pdf/2105.01876.pdf

    ```
    Chaoqi Yang, Cao Xiao, Lucas Glass, and Jimeng Sun.
      2021. Change matters: Medication change prediction 
      with recurrent residual networks.
    ```
    
  ### Dependencies
  I used:
  - MacOs 11.6.2
  - Python 3.9.4
  - The python dependencies I used for this project can be found in `requirements.txt`.
  - The dependencies can be installed using the following command:
        
    ```
    pip3 install -r requirments.txt
    ```
    
  ### Raw data download instruction
  - Download the following MIMIC-III Patient data files from PhysioNet. https://physionet.org/content/mimiciii/1.4/
    - PRESCIPTIONS.csv
    - DIAGNOSES_ICD.csv
    - PROCEDURES_ICD.csv
  - Place the above files in the `data/` directory.
  - The preprocessed data is already available in the `data/records_final.pkl` file.

  ### Preprocessing code
  - I used the (modified) preprocessing script of the original author to ensure that I was using the same filtered dataset.
  - This script filters and combines MIMIC-III patient prescriptions, procedure ICD codes, diagnosis codes, as well as drug code mappings into the `records_final.pkl` file.
  - To process:
    
    ```
    cd data
    python3 preprocessing.py
    ```

  ### Training code
  - Each luke_src/predict_* file contains the model implementation, as well as the script for training and evaluating the model.
  - To train MICRON model:
    
    ```
    cd luke_src
    python3 predict_MICRON.py --save_trained
    ```
  - The "--save_trained" flag will save the trained model to the "pretrained_models/" folder.
  - Other models are trained in the same way.
    
  ### Evaluation code
  - To evaluate a pretrained model, use the flag "--test_only".
  - The models are also evaluated at the end of training.
  - To evaluate the pretrained MICRON model:
    ```
    cd luke_src
    python3 predict_MICRON.py --test_only
    ```
  - Other models are evaluated in the same way.
    
  ### Pretrained model
  - Pretrained models are found in the "luke_src/pretrained_models/" folder.
  - These are the result of training each model with the "--save_trained" flag. 

  ### Table of results 
  
    | Model  | F1 Score | Jaccard |
    | ------------- | ------------- | ------------- |
    | Baseline Model 1  | .603  | .444 |
    | Baseline Model 2  | .638  | .479 |
    | Baseline Model 3  | .661  | .505 |
    | GameNet  | .499*  | .347* |
    | MICRON  | .669  | .513 |
    | MICRON Ablation  | .662  | .505 |

    ** GameNet results are lower than expected, and could very likely be improved with more compute resources and iterations on the model.