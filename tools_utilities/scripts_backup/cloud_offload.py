#!/usr/bin/env python3
"""
Cloud Offload System for Heavy Cognitive Tasks
Purpose: Offload computationally intensive tasks to AWS via SkyPilot
Inputs: Task type, parameters, local agent state
Outputs: Processed results from cloud computation
Seeds: Deterministic task execution for reproducibility
Deps: boto3, skypilot, json, time, threading
"""

import boto3
import json
import time
import uuid
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkyOffloader:
    """Offloads heavy cognitive tasks to AWS via SkyPilot"""
    
    def __init__(self, bucket: str = "quark-offload-us-west-2", 
                 cluster_prefix: str = "quark-offload",
                 region: str = "us-west-2"):
        self.bucket = bucket
        self.cluster_prefix = cluster_prefix
        self.region = region
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        # Track active jobs
        self.active_jobs = {}
        
        logger.info(f"✅ SkyOffloader initialized for bucket: {bucket}")
    
    def _ensure_bucket_exists(self):
        """Ensure S3 bucket exists for offloading"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"✅ Bucket {self.bucket} exists")
        except:
            logger.info(f"Creating bucket {self.bucket}...")
            self.s3_client.create_bucket(
                Bucket=self.bucket,
                CreateBucketConfiguration={'LocationConstraint': self.region}
            )
            logger.info(f"✅ Created bucket {self.bucket}")
    
    def submit(self, task_type: str, parameters: Dict[str, Any], 
               timeout: int = 300) -> Tuple[str, Dict[str, Any]]:
        """
        Submit a cognitive task for cloud processing
        
        Args:
            task_type: Type of task (neural_simulation, memory_consolidation, etc.)
            parameters: Task parameters
            timeout: Maximum wait time in seconds
            
        Returns:
            Tuple of (job_id, result_dict)
        """
        job_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
        cluster_name = f"{self.cluster_prefix}-{job_id}"
        
        logger.info(f"🚀 Submitting {task_type} task with ID: {job_id}")
        
        try:
            # Create payload
            payload = {
                'job_id': job_id,
                'task_type': task_type,
                'parameters': parameters,
                'timestamp': time.time(),
                'region': self.region
            }
            
            # Upload payload to S3
            payload_key = f"payloads/{job_id}.json"
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=payload_key,
                Body=json.dumps(payload, indent=2)
            )
            
            # Create SkyPilot YAML for this job
            yaml_content = self._create_skypilot_yaml(job_id, payload_key)
            
            # Save YAML to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_path = f.name
            
            # Launch SkyPilot job
            logger.info(f"Launching SkyPilot cluster: {cluster_name}")
            launch_cmd = [
                'sky', 'launch', '-c', cluster_name, yaml_path, '--down'
            ]
            
            result = subprocess.run(launch_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"SkyPilot launch failed: {result.stderr}")
            
            # Wait for results
            logger.info(f"Waiting for results from {job_id}...")
            result_data = self._wait_for_results(job_id, timeout)
            
            # Cleanup
            os.unlink(yaml_path)
            
            logger.info(f"✅ Task {job_id} completed successfully")
            return job_id, result_data
            
        except Exception as e:
            logger.error(f"❌ Task {job_id} failed: {e}")
            # Cleanup on failure
            self._cleanup_job(job_id)
            raise
    
    def _create_skypilot_yaml(self, job_id: str, payload_key: str) -> str:
        """Create SkyPilot YAML configuration for the job"""
        result_key = f"results/{job_id}.json"
        
        yaml_content = f"""# SkyPilot configuration for cognitive task offload
name: {job_id}

resources:
  # Use small CPU instance for cost efficiency
  cpus: 2
  memory: 8GB
  disk_size: 20

setup: |
  # Install dependencies
  pip install boto3 numpy scipy matplotlib
  
  # Clone or copy project if needed
  # git clone <your-repo> quark || echo "Repo already exists"

run: |
  # Download payload from S3
  aws s3 cp s3://{self.bucket}/{payload_key} payload.json
  
  # Run the offload worker
  python cloud_computing/offload_worker.py \\
    --payload payload.json \\
    --result s3://{self.bucket}/{result_key} \\
    --bucket {self.bucket}
"""
        return yaml_content
    
    def _wait_for_results(self, job_id: str, timeout: int) -> Dict[str, Any]:
        """Wait for results to appear in S3"""
        result_key = f"results/{job_id}.json"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if result exists
                response = self.s3_client.head_object(Bucket=self.bucket, Key=result_key)
                
                # Download result
                response = self.s3_client.get_object(Bucket=self.bucket, Key=result_key)
                result_data = json.loads(response['Body'].read().decode('utf-8'))
                
                logger.info(f"📥 Downloaded results for {job_id}")
                return result_data
                
            except self.s3_client.exceptions.NoSuchKey:
                # Result not ready yet
                time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Error checking results: {e}")
                time.sleep(5)
                continue
        
        raise TimeoutError(f"Timeout waiting for results from {job_id}")
    
    def _cleanup_job(self, job_id: str):
        """Cleanup job artifacts from S3"""
        try:
            # Delete payload and result files
            for key in [f"payloads/{job_id}.json", f"results/{job_id}.json"]:
                try:
                    self.s3_client.delete_object(Bucket=self.bucket, Key=key)
                except:
                    pass  # Ignore if file doesn't exist
        except Exception as e:
            logger.warning(f"Cleanup warning for {job_id}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get offload system status"""
        return {
            'bucket': self.bucket,
            'region': self.region,
            'active_jobs': len(self.active_jobs),
            'bucket_exists': True  # We ensure it exists on init
        }

def create_offload_hook(agent_instance) -> SkyOffloader:
    """Create an offload hook integrated with the conscious agent"""
    offloader = SkyOffloader()
    
    # Add offload method to agent
    def offload_heavy_task(task_type: str, parameters: Dict[str, Any]):
        """Offload a heavy cognitive task to cloud"""
        try:
            job_id, result = offloader.submit(task_type, parameters)
            
            # Integrate result into agent state
            if task_type == "neural_simulation":
                agent_instance.unified_state['neural_activity'] = result.get('activity_level', 0.0)
            elif task_type == "memory_consolidation":
                agent_instance.unified_state['memory_consolidation'] = result.get('consolidation_level', 0.0)
            elif task_type == "attention_modeling":
                agent_instance.unified_state['attention_focus'] = result.get('focus_level', 0.0)
            
            logger.info(f"✅ Integrated cloud result for {task_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Cloud offload failed: {e}")
            return None
    
    # Attach to agent
    agent_instance.offload_heavy_task = offload_heavy_task
    agent_instance.cloud_offloader = offloader
    
    return offloader

if __name__ == "__main__":
    # Test the offloader
    offloader = SkyOffloader()
    
    # Test with a simple task
    test_params = {
        'duration': 1000,
        'num_neurons': 50,
        'scale': 0.5
    }
    
    try:
        job_id, result = offloader.submit("neural_simulation", test_params)
        print(f"✅ Test successful: {job_id}")
        print(f"Result: {result}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
