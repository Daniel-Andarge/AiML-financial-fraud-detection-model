import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, ttest_ind
import mlflow
import mlflow.sklearn

sys.path.append('../scripts')
from data_loader import load_data
from mlops_tracking import log_model_experiment

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from data_loader import load_data

def train_models():
    # Paths to the dataset files
    filename1 = 'final_fraud_df.parquet'
    filename2 = 'final_credit_df.parquet'

    path1 = os.path.join('..', 'data/model_input', filename1)
    path2 = os.path.join('..', 'data/model_input', filename2)

    # Load datasets
    fraud_ip_data = load_data(path1)
    credit_card_data = load_data(path2)

    X = fraud_ip_data.drop('class', axis=1)
    y = fraud_ip_data['class']

    # Split the data into training, validation, and test sets
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test)

    # Train Random Forest Classifier
    rf = RandomForestClassifier(max_depth=2, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = classification_report(y_val, rf.predict(X_val), target_names=['Not Fraud', 'Fraud'], output_dict=True)

    # Train MLP Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    mlp_acc = classification_report(y_val, mlp.predict(X_val), output_dict=True)

    # Train XGBoost Classifier
    xgb = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0)
    xgb.fit(X_train, y_train)
    y_val_pred = xgb.predict(X_val)
    xgb_acc = classification_report(y_val, y_val_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)

    # Log the models and their evaluation metrics to MLflow
    models = [xgb, mlp, rf]
    metrics = [xgb_acc, mlp_acc, rf_acc]

    run_ids = log_model_experiment(models, metrics)

    for i, run_id in enumerate(run_ids):
        print(f"Logged model {i+1} to MLflow with run ID: {run_id}")

    return models, metrics
