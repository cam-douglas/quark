#!/usr/bin/env python3
"""
S3 Streaming Manager for Quark Brain Simulation
Enables direct streaming of models and datasets from S3 without full local download
"""

import os
import boto3
import torch
import numpy as np
import h5py
import io
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Iterator
import pickle
from contextlib import contextmanager
from tqdm import tqdm
import json

class S3StreamingManager:
    """Manages streaming access to models and datasets in S3"""
    
    def __init__(self, bucket_name: str = "quark-tokyo-bucket", region: str = "ap-northeast-1"):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.s3_resource = boto3.resource('s3', region_name=region)
        self.bucket = self.s3_resource.Bucket(bucket_name)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Streaming configuration
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks for streaming
        self.cache_size_mb = 100  # Small local cache for frequently accessed data
        
    def upload_model_to_s3(self, model_name: str, local_model_path: str, s3_key: str = None) -> bool:
        """Upload a model file to S3"""
        if not s3_key:
            s3_key = f"models/{model_name}/model.bin"
        
        try:
            file_size = os.path.getsize(local_model_path)
            
            with tqdm(desc=f"Uploading {model_name}", total=file_size, unit='B', unit_scale=True) as pbar:
                def upload_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                self.s3_client.upload_file(
                    local_model_path,
                    self.bucket_name,
                    s3_key,
                    Callback=upload_callback
                )
            
            self.logger.info(f"✅ Uploaded model {model_name} to S3: {s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to upload model {model_name}: {e}")
            return False
    
    def upload_dataset_to_s3(self, dataset_name: str, local_dataset_path: str, s3_key: str = None) -> bool:
        """Upload a dataset file to S3"""
        if not s3_key:
            s3_key = f"datasets/{dataset_name}/data.bin"
        
        try:
            file_size = os.path.getsize(local_dataset_path)
            
            with tqdm(desc=f"Uploading {dataset_name}", total=file_size, unit='B', unit_scale=True) as pbar:
                def upload_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                self.s3_client.upload_file(
                    local_dataset_path,
                    self.bucket_name,
                    s3_key,
                    Callback=upload_callback
                )
            
            self.logger.info(f"✅ Uploaded dataset {dataset_name} to S3: {s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to upload dataset {dataset_name}: {e}")
            return False
    
    @contextmanager
    def stream_model_from_s3(self, s3_key: str):
        """Stream a PyTorch model directly from S3"""
        try:
            # Get object from S3
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            # Create a BytesIO buffer from the S3 object
            model_buffer = io.BytesIO(obj['Body'].read())
            
            # Load PyTorch model from buffer
            model = torch.load(model_buffer, map_location='cpu')
            
            self.logger.info(f"✅ Streamed model from S3: {s3_key}")
            yield model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stream model from S3: {e}")
            yield None
        finally:
            # Cleanup
            if 'model_buffer' in locals():
                model_buffer.close()
    
    @contextmanager
    def stream_numpy_from_s3(self, s3_key: str):
        """Stream a NumPy array directly from S3"""
        try:
            # Get object from S3
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            # Create a BytesIO buffer
            data_buffer = io.BytesIO(obj['Body'].read())
            
            # Load NumPy array
            data = np.load(data_buffer, allow_pickle=True)
            
            self.logger.info(f"✅ Streamed NumPy data from S3: {s3_key}")
            yield data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stream NumPy from S3: {e}")
            yield None
        finally:
            if 'data_buffer' in locals():
                data_buffer.close()
    
    @contextmanager
    def stream_hdf5_from_s3(self, s3_key: str, temp_dir: str = None):
        """Stream HDF5 data from S3 (requires temporary file)"""
        temp_file = None
        try:
            # HDF5 requires a file path, so we need a temporary file
            if not temp_dir:
                temp_dir = tempfile.gettempdir()
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.h5', dir=temp_dir, delete=False)
            
            # Download to temporary file with progress
            file_size = self._get_s3_object_size(s3_key)
            
            with tqdm(desc=f"Streaming HDF5", total=file_size, unit='B', unit_scale=True) as pbar:
                def download_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    temp_file.name,
                    Callback=download_callback
                )
            
            # Open HDF5 file
            h5_file = h5py.File(temp_file.name, 'r')
            
            self.logger.info(f"✅ Streamed HDF5 from S3: {s3_key}")
            yield h5_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stream HDF5 from S3: {e}")
            yield None
        finally:
            # Cleanup
            if 'h5_file' in locals():
                h5_file.close()
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def stream_text_from_s3(self, s3_key: str, chunk_size: int = None) -> Iterator[str]:
        """Stream text data from S3 in chunks"""
        if not chunk_size:
            chunk_size = self.chunk_size
        
        try:
            # Get streaming body
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            for chunk in obj['Body'].iter_chunks(chunk_size=chunk_size):
                yield chunk.decode('utf-8')
                
            self.logger.info(f"✅ Streamed text from S3: {s3_key}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stream text from S3: {e}")
    
    def _get_s3_object_size(self, s3_key: str) -> int:
        """Get the size of an S3 object"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response['ContentLength']
        except Exception as e:
            self.logger.error(f"❌ Failed to get object size: {e}")
            return 0
    
    def create_sample_models_and_datasets(self) -> Dict[str, str]:
        """Create sample models and datasets for demonstration"""
        sample_files = {}
        temp_dir = Path(tempfile.gettempdir()) / "quark_samples"
        temp_dir.mkdir(exist_ok=True)
        
        # 1. Create a sample PyTorch model
        model_path = temp_dir / "neural_dynamics_model.pt"
        sample_model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=1)
        )
        torch.save(sample_model.state_dict(), model_path)
        sample_files['neural_dynamics_model'] = str(model_path)
        
        # 2. Create a sample NumPy dataset
        numpy_path = temp_dir / "cognitive_benchmarks.npy"
        sample_data = np.random.randn(1000, 100).astype(np.float32)
        np.save(numpy_path, sample_data)
        sample_files['cognitive_benchmarks'] = str(numpy_path)
        
        # 3. Create a sample HDF5 dataset
        h5_path = temp_dir / "brain_connectivity_data.h5"
        with h5py.File(h5_path, 'w') as f:
            # Sample brain connectivity matrix
            connectivity = np.random.rand(100, 100).astype(np.float32)
            f.create_dataset('connectivity_matrix', data=connectivity)
            f.create_dataset('node_labels', data=[f'node_{i}'.encode() for i in range(100)])
            f.attrs['description'] = 'Sample brain connectivity data'
        sample_files['brain_connectivity_data'] = str(h5_path)
        
        # 4. Create a larger training dataset
        training_path = temp_dir / "neural_training_data.pt"
        training_data = {
            'inputs': torch.randn(5000, 100),
            'targets': torch.randint(0, 10, (5000,)),
            'metadata': {'created': 'quark_demo', 'version': '1.0'}
        }
        torch.save(training_data, training_path)
        sample_files['neural_training_data'] = str(training_path)
        
        self.logger.info(f"✅ Created {len(sample_files)} sample files in {temp_dir}")
        return sample_files
    
    def upload_all_samples_to_s3(self) -> Dict[str, Any]:
        """Create and upload sample models and datasets to S3"""
        results = {
            'uploaded': {},
            'failed': {},
            'total_size_mb': 0
        }
        
        # Create sample files
        sample_files = self.create_sample_models_and_datasets()
        
        # Upload each file
        for name, local_path in sample_files.items():
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            results['total_size_mb'] += file_size_mb
            
            if 'model' in name:
                s3_key = f"models/{name.split('_')[0]}/{name}"
                success = self.upload_model_to_s3(name, local_path, s3_key)
            else:
                s3_key = f"datasets/{name.split('_')[0]}/{name}"
                success = self.upload_dataset_to_s3(name, local_path, s3_key)
            
            if success:
                results['uploaded'][name] = {
                    'local_path': local_path,
                    's3_key': s3_key,
                    'size_mb': file_size_mb
                }
            else:
                results['failed'][name] = local_path
        
        return results
    
    def test_streaming_capabilities(self) -> Dict[str, Any]:
        """Test all streaming capabilities with uploaded files"""
        test_results = {
            'pytorch_model': 'not_tested',
            'numpy_data': 'not_tested', 
            'hdf5_data': 'not_tested',
            'pytorch_dataset': 'not_tested',
            'errors': []
        }
        
        try:
            # Test PyTorch model streaming
            with self.stream_model_from_s3('models/neural/neural_dynamics_model.pt') as model:
                if model is not None:
                    test_results['pytorch_model'] = 'success'
                    self.logger.info("✅ PyTorch model streaming works")
                else:
                    test_results['pytorch_model'] = 'failed'
        except Exception as e:
            test_results['pytorch_model'] = 'error'
            test_results['errors'].append(f"PyTorch model: {e}")
        
        try:
            # Test NumPy streaming
            with self.stream_numpy_from_s3('datasets/cognitive/cognitive_benchmarks.npy') as data:
                if data is not None:
                    test_results['numpy_data'] = 'success'
                    self.logger.info(f"✅ NumPy streaming works - shape: {data.shape}")
                else:
                    test_results['numpy_data'] = 'failed'
        except Exception as e:
            test_results['numpy_data'] = 'error'
            test_results['errors'].append(f"NumPy data: {e}")
        
        try:
            # Test HDF5 streaming
            with self.stream_hdf5_from_s3('datasets/brain/brain_connectivity_data.h5') as h5_file:
                if h5_file is not None:
                    test_results['hdf5_data'] = 'success'
                    keys = list(h5_file.keys())
                    self.logger.info(f"✅ HDF5 streaming works - keys: {keys}")
                else:
                    test_results['hdf5_data'] = 'failed'
        except Exception as e:
            test_results['hdf5_data'] = 'error'
            test_results['errors'].append(f"HDF5 data: {e}")
        
        try:
            # Test PyTorch dataset streaming
            with self.stream_model_from_s3('datasets/neural/neural_training_data.pt') as dataset:
                if dataset is not None:
                    test_results['pytorch_dataset'] = 'success'
                    self.logger.info(f"✅ PyTorch dataset streaming works - keys: {list(dataset.keys())}")
                else:
                    test_results['pytorch_dataset'] = 'failed'
        except Exception as e:
            test_results['pytorch_dataset'] = 'error'
            test_results['errors'].append(f"PyTorch dataset: {e}")
        
        return test_results
    
    def get_s3_inventory(self) -> Dict[str, Any]:
        """Get complete inventory of what's in S3"""
        inventory = {
            'models': [],
            'datasets': [],
            'registries': [],
            'total_objects': 0,
            'total_size_mb': 0
        }
        
        try:
            # List all objects in bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        size_mb = obj['Size'] / (1024 * 1024)
                        
                        inventory['total_objects'] += 1
                        inventory['total_size_mb'] += size_mb
                        
                        obj_info = {
                            'key': key,
                            'size_mb': round(size_mb, 2),
                            'last_modified': obj['LastModified'].isoformat()
                        }
                        
                        if key.startswith('models/'):
                            inventory['models'].append(obj_info)
                        elif key.startswith('datasets/'):
                            inventory['datasets'].append(obj_info)
                        elif key.startswith('quark-registry/'):
                            inventory['registries'].append(obj_info)
            
            inventory['total_size_mb'] = round(inventory['total_size_mb'], 2)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get S3 inventory: {e}")
            inventory['error'] = str(e)
        
        return inventory

def main():
    """Main function for testing S3 streaming"""
    print("🌊 S3 Streaming Manager - Setup & Test")
    print("=" * 45)
    
    streaming_manager = S3StreamingManager()
    
    # Upload sample files to S3
    print("📤 Creating and uploading sample models/datasets...")
    upload_results = streaming_manager.upload_all_samples_to_s3()
    
    print(f"✅ Uploaded: {len(upload_results['uploaded'])} files")
    print(f"❌ Failed: {len(upload_results['failed'])} files")
    print(f"📊 Total size: {upload_results['total_size_mb']:.1f}MB")
    
    # Test streaming capabilities
    print("\n🧪 Testing streaming capabilities...")
    test_results = streaming_manager.test_streaming_capabilities()
    
    print("📋 Streaming Test Results:")
    for test_name, result in test_results.items():
        if test_name != 'errors':
            icon = "✅" if result == "success" else "❌" if result == "failed" else "⏭️"
            print(f"   {icon} {test_name}: {result}")
    
    if test_results['errors']:
        print("\n⚠️ Errors encountered:")
        for error in test_results['errors']:
            print(f"   • {error}")
    
    # Show S3 inventory
    print("\n📦 S3 Inventory:")
    inventory = streaming_manager.get_s3_inventory()
    print(f"   Models: {len(inventory['models'])}")
    print(f"   Datasets: {len(inventory['datasets'])}")
    print(f"   Registries: {len(inventory['registries'])}")
    print(f"   Total: {inventory['total_objects']} objects ({inventory['total_size_mb']}MB)")
    
    return streaming_manager

if __name__ == "__main__":
    main()
