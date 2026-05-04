# Hybrid AI Fraud Detection and Risk Scoring System

This project develops a hybrid AI-powered fraud detection and risk scoring system using the IEEE-CIS Fraud Detection dataset.

The system currently focuses on data cleaning, feature engineering, individual model training, model inference, and preliminary model checking.

## Current Development Status

Completed:
- Data cleaning notebook
- Feature engineering notebook
- CatBoost training and inference
- DNN training and inference
- Autoencoder training and inference
- Individual model checking script

In progress:
- Fusion logic
- Final transaction risk scoring
- Dynamic entity risk profiling
- Escalation logic
- Early warning system
- SHAP explainability
- Streamlit dashboard

## Current Repository Structure

`	ext
fraud-risk-scoring-system/
+-- notebooks/
¦   +-- DataCleaning2.ipynb
¦   +-- FeatureEngineering3.ipynb
¦
+-- fraud_system/
¦   +-- plot_ch4_outputs.py
¦   +-- pipeline.py
¦   +-- utils/
¦   +-- models/
¦   +-- evaluation/
¦   +-- data/
¦
+-- README.md
+-- QUICKSTART.md
+-- requirements.txt
+-- .gitignore
Important Notes

Raw datasets, processed CSV files, trained models, pickle files, prediction outputs, logs, and zip files are not uploaded to GitHub.

These files are ignored using .gitignore because they may be large, generated, or environment-specific.

Main Models
CatBoost: structured/tabular fraud probability model
Deep Neural Network: behavioural pattern learning model
Autoencoder: anomaly detection model
Current Scope

This repository currently contains individual model development and checking only. Fusion, transaction risk scoring, entity profiling, escalation, early warning, and dashboard modules will be added later.
