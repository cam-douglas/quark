from abc import ABC, abstractmethod
import time
from typing import Dict, Any

class BaseBenchmark(ABC):
    """
    Abstract Base Class for all cognitive benchmarks.

    Each benchmark test must inherit from this class and implement the abstract methods.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def setup(self):
        """
        Prepare the environment, data, and parameters for the benchmark.
        This method is called once before the benchmark runs.
        """
        pass

    @abstractmethod
    def run(self) -> Any:
        """
        Execute the core logic of the benchmark test.
        This method should return the raw output of the test.
        """
        pass

    @abstractmethod
    def evaluate(self, data: Any) -> Dict[str, Any]:
        """
        Analyze the raw output from the run method and compute performance metrics.

        Args:
            data (Any): The output from the run() method.

        Returns:
            Dict[str, Any]: A dictionary of performance metrics (e.g., accuracy, latency).
        """
        pass

    def execute(self) -> Dict[str, Any]:
        """
        Run the full benchmark pipeline: setup, run, evaluate.
        This method orchestrates the benchmark and records timing information.
        """
        print(f"Setting up benchmark: {self.name}...")
        self.setup()

        print(f"Running benchmark: {self.name}...")
        start_time = time.time()
        raw_data = self.run()
        end_time = time.time()
        print(f"Finished benchmark: {self.name}.")

        latency = end_time - start_time

        print(f"Evaluating results for {self.name}...")
        metrics = self.evaluate(raw_data)
        metrics['latency_seconds'] = round(latency, 4)
        
        self.results = {
            "name": self.name,
            "description": self.description,
            "metrics": metrics
        }

        print(f"Evaluation complete for {self.name}.")
        return self.results
