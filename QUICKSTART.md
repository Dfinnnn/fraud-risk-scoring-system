# Quickstart

## 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Run notebooks

Open the notebooks in order:

notebooks/DataCleaning2.ipynb
notebooks/FeatureEngineering3.ipynb

These notebooks prepare the cleaned and feature-engineered datasets.

4. Train models

From the project root folder, run:

cd fraud_system
python models/catboost_train.py
python models/dnn_train.py
python models/autoencoder_train.py

5. Run model check
python evaluation/model_check.py

Notes

The required data files and trained model artifacts are not included in this GitHub repository. They must be generated locally by running the notebooks and training scripts.
