"""A connector for the Hugging Face Hub API to search for models and datasets.

Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Operational tooling invoked by agents/simulators when required.
"""

from huggingface_hub import HfApi
from typing import List, Dict

class HuggingFaceConnector:
    """
    Handles interaction with the Hugging Face Hub for model and dataset discovery.
    """
    def __init__(self):
        """Initializes the Hugging Face Hub connector."""
        try:
            self.api = HfApi()
            print("Hugging Face API initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Hugging Face API: {e}")
            self.api = None

    def is_available(self) -> bool:
        """Checks if the Hugging Face API is available."""
        return self.api is not None

    def search_models(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Searches for models on the Hugging Face Hub.
        Args:
            query: The search term.
            max_results: The maximum number of results to return.
        Returns:
            A list of dictionaries, each containing info about a model.
        """
        if not self.is_available():
            return []

        try:
            models = self.api.list_models(search=query, sort="likes", direction=-1, limit=max_results)
            results = []
            for model in models:
                results.append({
                    "id": model.modelId,
                    "pipeline_tag": model.pipeline_tag,
                    "likes": model.likes,
                    "author": model.author
                })
            return results
        except Exception as e:
            print(f"Error searching Hugging Face models: {e}")
            return []

    def search_datasets(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Searches for datasets on the Hugging Face Hub.
        Args:
            query: The search term.
            max_results: The maximum number of results to return.
        Returns:
            A list of dictionaries, each containing info about a dataset.
        """
        if not self.is_available():
            return []

        try:
            datasets = self.api.list_datasets(search=query, sort="likes", direction=-1, limit=max_results)
            results = []
            for ds in datasets:
                results.append({
                    "id": ds.id,
                    "author": ds.author,
                    "likes": ds.likes
                })
            return results
        except Exception as e:
            print(f"Error searching Hugging Face datasets: {e}")
            return []
