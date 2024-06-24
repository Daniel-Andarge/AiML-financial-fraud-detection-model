import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from data_loader import load_data
from mlops_tracking import log_model_experiment
import mlflow
import mlflow.sklearn

class ModelTrainingPipeline:
    def __init__(self):
        self.trained_models = None
        self.model_metrics = None
        self.run_ids = None
        self.model_names = ["XGBoost", "MLP", "RandomForest"]
        self.mlflow_experiment_id = None

    def run_pipeline(self):
        with mlflow.start_run() as run:
            self.mlflow_experiment_id = run.info.experiment_id

            # Train models
            self.trained_models, self.model_metrics = self.train_models()

            # Log parameters
            self.log_parameters()

            # Log experiments and version models
            self.run_ids = log_model_experiment(self.trained_models, self.model_metrics, self.mlflow_experiment_id)

            for i, run_id in enumerate(self.run_ids):
                print(f"Experiment logged for model {i+1} with run ID: {run_id}")

            # Visualize the metrics
            self.visualize_metrics()
            
            # Display the metrics in tabular format
            self.display_metrics_table()

            # Save the best model
            self.save_best_model()

            # Log models
            self.log_models()

        return self.trained_models, self.model_metrics, self.run_ids

    def train_models(self):
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

        return models, metrics

    def visualize_metrics(self):
        if self.model_metrics is None:
            raise ValueError("Model metrics not available. Run the pipeline first.")

        # Prepare data for visualization
        metrics_data = []
        for i, metrics in enumerate(self.model_metrics):
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        metrics_data.append((self.model_names[i], f"{metric_name}_{sub_metric_name}", sub_metric_value))
                else:
                    metrics_data.append((self.model_names[i], metric_name, metric_value))

        # Create a DataFrame
        metrics_df = pd.DataFrame(metrics_data, columns=["Model", "Metric", "Value"])

        # Plot the metrics
        plt.figure(figsize=(14, 7))
        sns.barplot(x="Metric", y="Value", hue="Model", data=metrics_df)
        plt.title("Model Metrics Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def display_metrics_table(self):
        if self.model_metrics is None:
            raise ValueError("Model metrics not available. Run the pipeline first.")

        for i, metrics in enumerate(self.model_metrics):
            print(f"\nMetrics for {self.model_names[i]}:")
            metrics_data = []
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        metrics_data.append((f"{metric_name}_{sub_metric_name}", sub_metric_value))
                else:
                    metrics_data.append((metric_name, metric_value))

            # Create a DataFrame for the current model
            metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
            print(metrics_df)

    def save_best_model(self):
        if self.trained_models is None or self.model_metrics is None:
            raise ValueError("Models and metrics not available. Run the pipeline first.")

     
        best_model_idx = -1
        best_metric_value = -1
        for i, metrics in enumerate(self.model_metrics):
            if '1' in metrics and 'f1-score' in metrics['1']:
                f1_score_fraud = metrics['1']['f1-score']
                if f1_score_fraud > best_metric_value:
                    best_metric_value = f1_score_fraud
                    best_model_idx = i

        # Save the best model
        best_model = self.trained_models[best_model_idx]
        if not os.path.exists('../models'):
            os.makedirs('../models')
        joblib.dump(best_model, '../models/best_model.pkl')
        print(f"Best model saved with F1-score for 'Fraud' class: {best_metric_value}")

    def log_parameters(self):
        mlflow.log_param("Data Path 1", "../data/model_input/final_fraud_df.parquet")
        mlflow.log_param("Data Path 2", "../data/model_input/final_credit_df.parquet")
        mlflow.log_param("Random State", 42)

    def log_models(self):
        for i, model in enumerate(self.trained_models):
            mlflow.sklearn.log_model(model, f"Model_{i+1}")

