import mlflow
import yaml
import os
from contextlib import contextmanager
from typing import Dict, Any

class ExperimentLogger:
    """A wrapper for MLflow to handle experiment tracking."""

    def __init__(self, config_path="management/configurations/project/tracking.yaml"):
        """
        Initializes the logger by loading the configuration.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Tracking configuration not found at: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.tool = config.get('tool')
        if self.tool != 'mlflow':
            raise ValueError(f"Unsupported tracking tool: {self.tool}. Only 'mlflow' is currently supported.")

        mlflow_config = config.get('mlflow', {})
        self.tracking_uri = mlflow_config.get('tracking_uri', './mlruns')
        self.default_experiment_name = mlflow_config.get('default_experiment_name', 'QuarkBrainSimulation')

        mlflow.set_tracking_uri(self.tracking_uri)

    def set_experiment(self, experiment_name: str):
        """
        Sets the active experiment in MLflow.

        Args:
            experiment_name (str): The name of the experiment.
        """
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str = None, experiment_name: str = None):
        """
        Starts an MLflow run as a context manager.

        Args:
            run_name (str, optional): The name for the run.
            experiment_name (str, optional): The experiment to log to. Defaults to the configured default.
        """
        if experiment_name:
            self.set_experiment(experiment_name)
        else:
            self.set_experiment(self.default_experiment_name)
        
        try:
            with mlflow.start_run(run_name=run_name) as run:
                yield run
        finally:
            # The run is automatically ended by the context manager
            pass

    def log_params(self, params: Dict[str, Any]):
        """
        Logs a dictionary of parameters.

        Args:
            params (Dict[str, Any]): The parameters to log.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Logs a dictionary of metrics.

        Args:
            metrics (Dict[str, float]): The metrics to log.
        """
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Logs a local file or directory as an artifact.

        Args:
            local_path (str): Path to the file or directory to log.
            artifact_path (str, optional): Directory within the run's artifact URI to log to.
        """
        mlflow.log_artifact(local_path, artifact_path)

# Example usage:
if __name__ == '__main__':
    print("Running ExperimentLogger example...")
    
    # Initialize the logger
    logger = ExperimentLogger()

    # Define some example parameters and metrics
    params = {
        "learning_rate": 0.01,
        "epochs": 10,
        "model_type": "SimpleRNN"
    }
    metrics = {
        "accuracy": 0.95,
        "loss": 0.04,
        "neural_synchrony": 0.88
    }

    # Start a run and log
    with logger.start_run(run_name="ExampleRun") as run:
        print(f"Started run with ID: {run.info.run_id}")
        
        print("Logging parameters...")
        logger.log_params(params)
        
        print("Logging metrics...")
        logger.log_metrics(metrics)
        
        # Create a dummy artifact to log
        dummy_file_path = "dummy_artifact.txt"
        with open(dummy_file_path, "w") as f:
            f.write("This is a test artifact.")
        
        print("Logging artifact...")
        logger.log_artifact(dummy_file_path)
        os.remove(dummy_file_path)

    print("Example run completed. Check the 'mlruns' directory for results.")
