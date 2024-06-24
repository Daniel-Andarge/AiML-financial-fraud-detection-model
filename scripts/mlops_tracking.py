import uuid
import mlflow

def log_model_experiment(trained_models, model_metrics, mlflow_experiment_id):
    """
    Logs the trained models and their associated metrics to an MLflow experiment.

    Args:
        trained_models (list): A list of trained model objects.
        model_metrics (list): A list of dictionaries containing the metrics for each model.
        mlflow_experiment_id (str): The ID of the MLflow experiment to log into.

    Returns:
        list: The list of run IDs of the logged experiments.
    """
    run_ids = []

    # Start a new MLflow run for each model
    for model, metrics in zip(trained_models, model_metrics):
        with mlflow.start_run(experiment_id=mlflow_experiment_id, nested=True):
            # Log the model
            mlflow.sklearn.log_model(model, "model")

            # Log the metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        mlflow.log_metric(f"{metric_name}_{sub_metric_name}", sub_metric_value)
                else:
                    mlflow.log_metric(metric_name, metric_value)

        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        run_ids.append(run_id)

    return run_ids
