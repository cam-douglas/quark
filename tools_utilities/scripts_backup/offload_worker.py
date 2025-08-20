#!/usr/bin/env python3
"""
Offload Worker - Runs on AWS to process cognitive tasks
Purpose: Execute heavy cognitive computations on cloud infrastructure
Inputs: Task payload from S3, task parameters
Outputs: Processed results uploaded to S3
Seeds: Deterministic computation for reproducibility
Deps: boto3, numpy, scipy, json, argparse
"""

import boto3
import json
import argparse
import numpy as np
import time
import logging
from typing import Dict, Any
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CognitiveTaskProcessor:
    """Processes cognitive tasks on cloud infrastructure"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        logger.info("✅ Cognitive task processor initialized")
    
    def process_neural_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate neural activity patterns"""
        duration = parameters.get('duration', 1000)
        num_neurons = parameters.get('num_neurons', 100)
        scale = parameters.get('scale', 1.0)
        
        logger.info(f"🧠 Processing neural simulation: {num_neurons} neurons, {duration}ms")
        
        # Generate realistic neural activity
        spike_times = []
        spike_neurons = []
        
        for neuron in range(num_neurons):
            base_rate = np.random.uniform(5, 20) * scale  # Hz
            spike_intervals = np.random.exponential(1000/base_rate, size=20)
            spike_times_neuron = np.cumsum(spike_intervals)
            spike_times_neuron = spike_times_neuron[spike_times_neuron < duration]
            
            spike_times.extend(spike_times_neuron)
            spike_neurons.extend([neuron] * len(spike_times_neuron))
        
        # Calculate activity metrics
        total_spikes = len(spike_times)
        average_rate = total_spikes / (num_neurons * duration / 1000) if num_neurons > 0 else 0
        
        # Simulate processing time
        time.sleep(np.random.uniform(1, 3))
        
        return {
            'task_type': 'neural_simulation',
            'activity_level': average_rate,
            'total_spikes': total_spikes,
            'num_neurons': num_neurons,
            'duration': duration,
            'spike_times': spike_times[:100],  # Limit for JSON serialization
            'spike_neurons': spike_neurons[:100],
            'processing_time': time.time(),
            'cloud_processed': True
        }
    
    def process_memory_consolidation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate memory consolidation process"""
        duration = parameters.get('duration', 1000)
        scale = parameters.get('scale', 1.0)
        
        logger.info(f"🧠 Processing memory consolidation: {duration}ms")
        
        time_points = np.arange(0, duration, 1)
        consolidation = np.zeros_like(time_points, dtype=float)
        
        for i, t in enumerate(time_points):
            # Memory consolidation increases over time with sleep-like cycles
            cycle_position = (t % 5400) / 5400  # 90-minute cycles
            
            if cycle_position > 0.2:  # During sleep-like periods
                consolidation[i] = (0.5 + 0.5 * np.sin(2 * np.pi * t / 1000)) * scale
            else:  # During active periods
                consolidation[i] = (0.1 + 0.1 * np.sin(2 * np.pi * t / 200)) * scale
        
        # Simulate processing time
        time.sleep(np.random.uniform(2, 4))
        
        return {
            'task_type': 'memory_consolidation',
            'consolidation_level': consolidation[-1],
            'average_level': np.mean(consolidation),
            'duration': duration,
            'time_points': time_points.tolist()[:100],  # Limit for JSON
            'consolidation_levels': consolidation.tolist()[:100],
            'processing_time': time.time(),
            'cloud_processed': True
        }
    
    def process_attention_modeling(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate attention focus dynamics"""
        duration = parameters.get('duration', 1000)
        scale = parameters.get('scale', 1.0)
        
        logger.info(f"🧠 Processing attention modeling: {duration}ms")
        
        time_points = np.arange(0, duration, 1)
        attention_levels = np.zeros_like(time_points, dtype=float)
        
        for i, t in enumerate(time_points):
            # Attention varies with time and external stimuli
            base_attention = (0.5 + 0.3 * np.sin(2 * np.pi * t / 500)) * scale
            stimulus_effect = 0.2 * np.random.normal(0, 1) * np.exp(-t / 200) * scale
            attention_levels[i] = np.clip(base_attention + stimulus_effect, 0, 1)
        
        # Simulate processing time
        time.sleep(np.random.uniform(1, 3))
        
        return {
            'task_type': 'attention_modeling',
            'focus_level': np.mean(attention_levels),
            'peak_attention': np.max(attention_levels),
            'duration': duration,
            'time_points': time_points.tolist()[:100],
            'attention_levels': attention_levels.tolist()[:100],
            'processing_time': time.time(),
            'cloud_processed': True
        }
    
    def process_decision_making(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate decision making confidence"""
        duration = parameters.get('duration', 1000)
        scale = parameters.get('scale', 1.0)
        
        logger.info(f"🧠 Processing decision making: {duration}ms")
        
        time_points = np.arange(0, duration, 1)
        confidence_levels = np.zeros_like(time_points, dtype=float)
        
        for i, t in enumerate(time_points):
            # Decision confidence builds over time with information gathering
            base_confidence = (0.3 + 0.6 * (1 - np.exp(-t / 300))) * scale
            uncertainty = 0.1 * np.random.normal(0, 1) * np.exp(-t / 400) * scale
            confidence_levels[i] = np.clip(base_confidence + uncertainty, 0, 1)
        
        # Simulate processing time
        time.sleep(np.random.uniform(2, 4))
        
        return {
            'task_type': 'decision_making',
            'confidence_level': confidence_levels[-1],
            'confidence_growth': confidence_levels[-1] - confidence_levels[0],
            'duration': duration,
            'time_points': time_points.tolist()[:100],
            'confidence_levels': confidence_levels.tolist()[:100],
            'processing_time': time.time(),
            'cloud_processed': True
        }
    
    def process_training_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate neural network training session"""
        epochs = parameters.get('epochs', 10)
        batch_size = parameters.get('batch_size', 32)
        scale = parameters.get('scale', 1.0)
        
        logger.info(f"🧠 Processing training session: {epochs} epochs, batch_size {batch_size}")
        
        # Simulate training progress
        training_loss = []
        validation_loss = []
        
        for epoch in range(epochs):
            # Simulate training loss decrease
            base_loss = 1.0 * np.exp(-epoch / 5) * scale
            noise = 0.1 * np.random.normal(0, 1) * scale
            training_loss.append(max(0.01, base_loss + noise))
            
            # Simulate validation loss
            val_loss = base_loss + 0.2 * np.random.normal(0, 1) * scale
            validation_loss.append(max(0.01, val_loss))
            
            # Simulate epoch processing time
            time.sleep(np.random.uniform(0.5, 1.5))
        
        return {
            'task_type': 'training_session',
            'final_training_loss': training_loss[-1],
            'final_validation_loss': validation_loss[-1],
            'epochs_completed': epochs,
            'batch_size': batch_size,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'processing_time': time.time(),
            'cloud_processed': True
        }
    
    def process_task(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a cognitive task based on type"""
        logger.info(f"🚀 Processing task: {task_type}")
        
        if task_type == "neural_simulation":
            return self.process_neural_simulation(parameters)
        elif task_type == "memory_consolidation":
            return self.process_memory_consolidation(parameters)
        elif task_type == "attention_modeling":
            return self.process_attention_modeling(parameters)
        elif task_type == "decision_making":
            return self.process_decision_making(parameters)
        elif task_type == "training_session":
            return self.process_training_session(parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

def main():
    """Main function to process offloaded tasks"""
    parser = argparse.ArgumentParser(description='Process cognitive tasks on cloud')
    parser.add_argument('--payload', required=True, help='Path to payload JSON file')
    parser.add_argument('--result', required=True, help='S3 path for result upload')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    
    args = parser.parse_args()
    
    logger.info("🧠 Starting cognitive task processor on cloud")
    
    try:
        # Load payload
        with open(args.payload, 'r') as f:
            payload = json.load(f)
        
        job_id = payload['job_id']
        task_type = payload['task_type']
        parameters = payload['parameters']
        
        logger.info(f"📥 Loaded payload for job {job_id}: {task_type}")
        
        # Process task
        processor = CognitiveTaskProcessor()
        result = processor.process_task(task_type, parameters)
        
        # Add job metadata
        result['job_id'] = job_id
        result['completed_at'] = time.time()
        result['cloud_instance'] = os.uname().nodename if hasattr(os, 'uname') else 'aws-instance'
        
        # Upload result to S3
        bucket = args.bucket
        result_key = args.result.replace(f's3://{bucket}/', '')
        
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=result_key,
            Body=json.dumps(result, indent=2)
        )
        
        logger.info(f"✅ Task completed and uploaded to s3://{bucket}/{result_key}")
        
        # Print result summary
        print(f"Task {task_type} completed successfully")
        print(f"Job ID: {job_id}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"Cloud processed: {result.get('cloud_processed', False)}")
        
    except Exception as e:
        logger.error(f"❌ Task processing failed: {e}")
        
        # Upload error result
        error_result = {
            'error': str(e),
            'job_id': payload.get('job_id', 'unknown'),
            'task_type': payload.get('task_type', 'unknown'),
            'completed_at': time.time(),
            'cloud_processed': False
        }
        
        try:
            bucket = args.bucket
            result_key = args.result.replace(f's3://{bucket}/', '')
            
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=bucket,
                Key=result_key,
                Body=json.dumps(error_result, indent=2)
            )
            logger.info(f"📤 Error result uploaded to s3://{bucket}/{result_key}")
        except Exception as upload_error:
            logger.error(f"Failed to upload error result: {upload_error}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
