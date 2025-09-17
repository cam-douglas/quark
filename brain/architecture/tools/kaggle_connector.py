"""A connector for the Kaggle API to enable dataset search and download.

Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Operational tooling invoked by agents/simulators when required.
"""

import os
from typing import List, Dict, Optional

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except (ImportError, OSError):
    KAGGLE_AVAILABLE = False

class KaggleConnector:
    """
    Handles interaction with the Kaggle API for dataset discovery and download.
    """
    def __init__(self):
        """Initializes the Kaggle API connector."""
        self.api = None
        if KAGGLE_AVAILABLE:
            try:
                self.api = KaggleApi()
                self.api.authenticate()
                print("Kaggle API authenticated successfully.")
            except Exception as e:
                print(f"Kaggle authentication failed: {e}")
                print("Please ensure your kaggle.json is correctly placed in ~/.kaggle/")
                self.api = None
        else:
            print("Kaggle library not found. KaggleConnector will be disabled.")

    def is_available(self) -> bool:
        """Checks if the Kaggle API is authenticated and available."""
        return self.api is not None

    def search_datasets(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Searches for datasets on Kaggle.
        Args:
            query: The search term.
            max_results: The maximum number of results to return.
        Returns:
            A list of dictionaries, each containing info about a dataset.
        """
        if not self.is_available():
            return []

        try:
            datasets = self.api.dataset_list(search=query, max_size=1000000) # Max size in bytes
            # Limit and format results
            results = []
            for ds in datasets[:max_results]:
                results.append({
                    "ref": ds.ref,
                    "title": ds.title,
                    "size": ds.totalBytes,
                    "url": ds.url
                })
            return results
        except Exception as e:
            print(f"Error searching Kaggle datasets: {e}")
            return []

    def download_dataset(self, dataset_ref: str, download_path: str = "datasets") -> Optional[str]:
        """
        Downloads a dataset from Kaggle.
        Args:
            dataset_ref: The reference of the dataset (e.g., 'user/dataset-name').
            download_path: The local path to download the dataset to.
        Returns:
            The path to the downloaded files, or None on failure.
        """
        if not self.is_available():
            return None

        try:
            target_path = os.path.join(download_path, dataset_ref.replace('/', '_'))
            os.makedirs(target_path, exist_ok=True)
            self.api.dataset_download_files(dataset_ref, path=target_path, unzip=True)
            print(f"Dataset '{dataset_ref}' downloaded and unzipped to '{target_path}'")
            return target_path
        except Exception as e:
            print(f"Error downloading Kaggle dataset '{dataset_ref}': {e}")
            return None
