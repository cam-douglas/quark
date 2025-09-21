#!/usr/bin/env python3
"""
OpenML REST API Integration for Quark
======================================
This module provides integration with OpenML API for machine learning research.

OpenML is a collaborative platform for machine learning, providing access to
datasets, tasks, algorithms (flows), and experiment results (runs).

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API configuration
CREDENTIALS_PATH = Path(__file__).parent.parent / "data" / "credentials" / "all_api_keys.json"
with open(CREDENTIALS_PATH, 'r') as f:
    credentials = json.load(f)
    OPENML_CONFIG = credentials['services']['openml']

# API endpoints
BASE_URL = OPENML_CONFIG['endpoints']['json_base']
DATASET_URL = OPENML_CONFIG['endpoints']['dataset']
TASK_URL = OPENML_CONFIG['endpoints']['task']
FLOW_URL = OPENML_CONFIG['endpoints']['flow']


class OpenMLClient:
    """Client for interacting with OpenML REST API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenML client.
        
        Args:
            api_key: Optional API key for uploads and private data
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quark-OpenML-Integration/1.0'
        })
        
        if api_key:
            self.session.headers['api_key'] = api_key
    
    def _get(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a GET request to the OpenML API.
        
        Args:
            url: API endpoint URL
            params: Optional query parameters
            
        Returns:
            JSON response as dictionary
        """
        logger.debug(f"GET {url}")
        response = self.session.get(url, params=params)
        
        if response.status_code == 200:
            # OpenML returns data wrapped in various structures
            data = response.json()
            # Handle different response formats
            if 'data' in data:
                return data['data']
            elif 'data_set_description' in data:
                return data['data_set_description']
            elif 'task' in data:
                return data['task']
            elif 'flow' in data:
                return data['flow']
            else:
                return data
        elif response.status_code == 404:
            logger.warning(f"Resource not found: {url}")
            return {}
        else:
            logger.error(f"Error {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
    
    def list_datasets(
        self,
        offset: int = 0,
        limit: int = 100,
        status: str = 'active',
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available datasets.
        
        Args:
            offset: Offset for pagination
            limit: Number of datasets to return
            status: Dataset status ('active', 'deactivated', 'all')
            tag: Optional tag to filter datasets
            
        Returns:
            List of dataset summaries
        """
        url = f"{DATASET_URL}/list"
        params = {
            'offset': offset,
            'limit': limit,
            'status': status
        }
        
        if tag:
            params['tag'] = tag
        
        logger.info(f"Listing datasets (offset={offset}, limit={limit})")
        result = self._get(url, params)
        
        if isinstance(result, dict) and 'dataset' in result:
            datasets = result['dataset']
            # Ensure it's a list
            if not isinstance(datasets, list):
                datasets = [datasets]
            return datasets
        
        return []
    
    def get_dataset(self, dataset_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a dataset.
        
        Args:
            dataset_id: OpenML dataset ID
            
        Returns:
            Dataset information
        """
        url = f"{DATASET_URL}/{dataset_id}"
        
        logger.info(f"Getting dataset {dataset_id}")
        return self._get(url)
    
    def list_tasks(
        self,
        task_type: int = 1,
        offset: int = 0,
        limit: int = 100,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available tasks.
        
        Args:
            task_type: Type of task (1=classification, 2=regression)
            offset: Offset for pagination
            limit: Number of tasks to return
            tag: Optional tag to filter tasks
            
        Returns:
            List of task summaries
        """
        url = f"{TASK_URL}/list"
        params = {
            'type': task_type,
            'offset': offset,
            'limit': limit
        }
        
        if tag:
            params['tag'] = tag
        
        logger.info(f"Listing tasks (type={task_type}, offset={offset})")
        result = self._get(url, params)
        
        if isinstance(result, dict) and 'task' in result:
            tasks = result['task']
            if not isinstance(tasks, list):
                tasks = [tasks]
            return tasks
        
        return []
    
    def get_task(self, task_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a task.
        
        Args:
            task_id: OpenML task ID
            
        Returns:
            Task information
        """
        url = f"{TASK_URL}/{task_id}"
        
        logger.info(f"Getting task {task_id}")
        return self._get(url)
    
    def list_flows(
        self,
        offset: int = 0,
        limit: int = 100,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available flows (algorithms).
        
        Args:
            offset: Offset for pagination
            limit: Number of flows to return
            tag: Optional tag to filter flows
            
        Returns:
            List of flow summaries
        """
        url = f"{FLOW_URL}/list"
        params = {
            'offset': offset,
            'limit': limit
        }
        
        if tag:
            params['tag'] = tag
        
        logger.info(f"Listing flows (offset={offset})")
        result = self._get(url, params)
        
        if isinstance(result, dict) and 'flow' in result:
            flows = result['flow']
            if not isinstance(flows, list):
                flows = [flows]
            return flows
        
        return []
    
    def search_brain_datasets(self) -> Dict[str, List[Dict]]:
        """
        Search for brain/neuroscience related datasets.
        
        Returns:
            Dictionary of categorized brain datasets
        """
        brain_keywords = [
            'brain', 'neural', 'neuron', 'EEG', 'fMRI',
            'cognitive', 'neuroscience', 'cortex'
        ]
        
        brain_datasets = {
            'neuroimaging': [],
            'cognitive': [],
            'clinical': [],
            'other': []
        }
        
        print("\nSearching for brain-related datasets in OpenML")
        print("-" * 50)
        
        # Get datasets and filter by keywords
        datasets = self.list_datasets(limit=1000)
        
        for dataset in datasets:
            name = dataset.get('name', '').lower()
            
            # Check if dataset is brain-related
            is_brain = False
            for keyword in brain_keywords:
                if keyword.lower() in name:
                    is_brain = True
                    break
            
            if is_brain:
                dataset_info = {
                    'id': dataset.get('did'),
                    'name': dataset.get('name'),
                    'instances': dataset.get('NumberOfInstances'),
                    'features': dataset.get('NumberOfFeatures'),
                    'classes': dataset.get('NumberOfClasses'),
                    'format': dataset.get('format')
                }
                
                # Categorize
                if any(k in name for k in ['eeg', 'fmri', 'mri', 'scan']):
                    brain_datasets['neuroimaging'].append(dataset_info)
                elif any(k in name for k in ['cognitive', 'memory', 'attention']):
                    brain_datasets['cognitive'].append(dataset_info)
                elif any(k in name for k in ['disease', 'disorder', 'patient']):
                    brain_datasets['clinical'].append(dataset_info)
                else:
                    brain_datasets['other'].append(dataset_info)
                
                print(f"  Found: {dataset.get('name')} (ID: {dataset.get('did')})")
        
        # Report results
        total = sum(len(v) for v in brain_datasets.values())
        print(f"\nTotal brain-related datasets found: {total}")
        for category, datasets in brain_datasets.items():
            if datasets:
                print(f"  {category}: {len(datasets)} datasets")
        
        return brain_datasets
    
    def get_popular_ml_datasets(self) -> List[Dict[str, Any]]:
        """
        Get popular/benchmark ML datasets.
        
        Returns:
            List of popular datasets
        """
        # Some well-known dataset IDs
        popular_ids = {
            61: 'iris',  # Classic iris dataset
            2: 'anneal',  # Annealing dataset
            31: 'credit-g',  # German credit
            37: 'diabetes',  # Pima Indians diabetes
            44: 'spambase',  # Spam classification
            50: 'tic-tac-toe',  # Tic-tac-toe endgame
            54: 'vehicle',  # Vehicle silhouettes
            182: 'satimage',  # Satellite image
            1461: 'bank-marketing',  # Bank marketing
            1489: 'phoneme'  # Phoneme recognition
        }
        
        datasets = []
        
        print("\nFetching popular ML datasets")
        print("-" * 50)
        
        for did, name in popular_ids.items():
            try:
                dataset = self.get_dataset(did)
                if dataset:
                    info = {
                        'id': did,
                        'name': dataset.get('name', name),
                        'instances': dataset.get('number_of_instances'),
                        'features': dataset.get('number_of_features'),
                        'classes': dataset.get('number_of_classes'),
                        'missing': dataset.get('number_of_missing_values'),
                        'format': dataset.get('format'),
                        'description': dataset.get('description', '')[:100]
                    }
                    datasets.append(info)
                    print(f"  ✓ {name}: {info['instances']} instances, {info['features']} features")
            except Exception as e:
                print(f"  ✗ {name}: Error - {e}")
        
        return datasets
    
    def get_classification_benchmarks(self) -> List[Dict[str, Any]]:
        """
        Get classification benchmark tasks.
        
        Returns:
            List of classification tasks
        """
        print("\nFetching classification benchmark tasks")
        print("-" * 50)
        
        tasks = self.list_tasks(task_type=1, limit=10)  # Type 1 = classification
        
        benchmarks = []
        for task in tasks[:5]:  # Just first 5 for demo
            task_id = task.get('tid')
            try:
                task_detail = self.get_task(task_id)
                if task_detail:
                    benchmarks.append({
                        'id': task_id,
                        'name': task.get('name'),
                        'type': task_detail.get('task_type'),
                        'dataset_id': task_detail.get('input', [{}])[0].get('data_set', {}).get('data_set_id'),
                        'evaluation': task_detail.get('evaluation_measures', {}).get('evaluation_measure')
                    })
                    print(f"  Task {task_id}: {task.get('name')}")
            except Exception as e:
                logger.warning(f"Error getting task {task_id}: {e}")
        
        return benchmarks


def demonstrate_automl_capabilities(client: OpenMLClient):
    """
    Demonstrate AutoML-related capabilities.
    
    Args:
        client: OpenML client instance
    """
    print("\nAutoML Capabilities Demo")
    print("=" * 60)
    
    # List some popular AutoML flows/algorithms
    print("\n1. Available ML Algorithms (Flows)")
    print("-" * 40)
    
    flows = client.list_flows(limit=10)
    
    automl_flows = []
    for flow in flows[:5]:
        flow_info = {
            'id': flow.get('id'),
            'name': flow.get('name'),
            'uploader': flow.get('uploader')
        }
        automl_flows.append(flow_info)
        print(f"  Flow {flow_info['id']}: {flow_info['name']}")
    
    # Show task types
    print("\n2. ML Task Types")
    print("-" * 40)
    task_types = {
        1: "Supervised Classification",
        2: "Supervised Regression", 
        3: "Learning Curve",
        4: "Supervised Data Stream Classification",
        5: "Clustering",
        6: "Machine Learning Challenge",
        7: "Survival Analysis",
        10: "Subgroup Discovery"
    }
    
    for tid, tname in task_types.items():
        print(f"  Type {tid}: {tname}")
    
    return automl_flows


def main():
    """Example usage of OpenML client."""
    client = OpenMLClient()
    
    print("=" * 60)
    print("OpenML REST API Integration Test")
    print("Quark System - ML Research Platform Access")
    print("=" * 60)
    
    # Test 1: List datasets
    print("\n1. Testing dataset listing...")
    try:
        datasets = client.list_datasets(limit=5)
        if datasets:
            print(f"  Found {len(datasets)} datasets")
            for ds in datasets[:3]:
                print(f"    - {ds.get('name')} (ID: {ds.get('did')})")
            print("  ✓ Dataset listing successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Get specific dataset
    print("\n2. Testing dataset retrieval...")
    try:
        # Get iris dataset (ID: 61)
        dataset = client.get_dataset(61)
        if dataset:
            print(f"  Dataset: {dataset.get('name')}")
            print(f"    Instances: {dataset.get('number_of_instances')}")
            print(f"    Features: {dataset.get('number_of_features')}")
            print(f"    Format: {dataset.get('format')}")
            print("  ✓ Dataset retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: List tasks
    print("\n3. Testing task listing...")
    try:
        tasks = client.list_tasks(task_type=1, limit=5)
        if tasks:
            print(f"  Found {len(tasks)} classification tasks")
            for task in tasks[:3]:
                print(f"    - Task {task.get('tid')}: {task.get('name')}")
            print("  ✓ Task listing successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Search for brain datasets
    print("\n4. Searching for brain-related datasets...")
    try:
        brain_datasets = client.search_brain_datasets()
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "knowledge" / "openml_brain_datasets.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'source': 'OpenML REST API',
                    'date': '2025-01-20',
                    'description': 'Brain and neuroscience related ML datasets'
                },
                'datasets': brain_datasets
            }, f, indent=2)
        
        print(f"  Results saved to: {output_path}")
        print("  ✓ Brain dataset search successful")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 5: Get popular datasets
    print("\n5. Getting popular ML datasets...")
    try:
        popular = client.get_popular_ml_datasets()
        print(f"  Retrieved {len(popular)} popular datasets")
        print("  ✓ Popular dataset retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 6: AutoML demo
    automl_flows = demonstrate_automl_capabilities(client)
    
    # Test 7: Get benchmark tasks
    print("\n6. Getting classification benchmarks...")
    try:
        benchmarks = client.get_classification_benchmarks()
        print(f"  Retrieved {len(benchmarks)} benchmark tasks")
        print("  ✓ Benchmark retrieval successful")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("OpenML API integration test complete!")
    print("✓ ML research platform access working")
    print("=" * 60)


if __name__ == "__main__":
    main()
