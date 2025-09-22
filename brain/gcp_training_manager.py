# -*- coding: utf-8 -*-
"""
Manages the connection, job submission, and monitoring of training tasks on Google Cloud AI Platform.
"""

import json
from google.oauth2 import service_account
from google.cloud import aiplatform

class GCPTrainingManager:
    """
    A manager class to handle interactions with Google Cloud AI Platform for model training.
    """

    def __init__(self, credentials_path, project_id, location='australia-southeast1'):
        """
        Initializes the GCPTrainingManager.

        Args:
            credentials_path (str): The file path to the GCP service account credentials.
            project_id (str): The GCP project ID.
            location (str): The GCP region to run the jobs in.
        """
        self.credentials = self._load_credentials(credentials_path)
        self.project_id = project_id
        self.location = location
        self.api_endpoint = f"{location}-aiplatform.googleapis.com"
        
        # Initialize the AI Platform client
        aiplatform.init(
            project=self.project_id,
            location=self.location,
            credentials=self.credentials
        )

    def _load_credentials(self, credentials_path):
        """
        Loads GCP credentials from a JSON file.

        Args:
            credentials_path (str): Path to the service account key file.

        Returns:
            google.oauth2.service_account.Credentials: The loaded credentials.
        """
        try:
            return service_account.Credentials.from_service_account_file(credentials_path)
        except Exception as e:
            print(f"Error loading credentials from {credentials_path}: {e}")
            raise

    def submit_training_job(self, display_name, container_image_uri, command_args, machine_type='n1-standard-4', accelerator_type='NVIDIA_TESLA_T4', accelerator_count=1, tensorboard_log_dir=None):
        """
        Submits a custom training job to AI Platform and sets up TensorBoard.

        Args:
            display_name (str): The name of the training job.
            container_image_uri (str): The URI of the container image for the job.
            command_args (list): The arguments for the container's entry point.
            machine_type (str): The machine type for the job.
            accelerator_type (str): The type of GPU.
            accelerator_count (int): The number of GPUs.
            tensorboard_log_dir (str): GCS path for TensorBoard logs.

        Returns:
            aiplatform.CustomJob: The created training job.
        """
        # Define worker pool specs for the container-based job
        worker_pool_specs = [
            {
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": accelerator_type,
                    "accelerator_count": accelerator_count,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_image_uri,
                    "args": command_args,
                },
            }
        ]

        # Try to get TensorBoard instance, but don't fail if permissions are insufficient
        tensorboard_resource_name = None
        try:
            tensorboard_instance = self.get_or_create_tensorboard_instance()
            tensorboard_resource_name = tensorboard_instance.resource_name
            print(f"✅ TensorBoard instance ready: {tensorboard_resource_name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not set up TensorBoard (insufficient permissions): {e}")
            print("   Training will proceed without TensorBoard integration.")

        # Use CustomJob for container-based training
        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
            project=self.project_id,
            location=self.location,
            staging_bucket='gs://quark-brain-segmentation-bucket'  # Add staging bucket
        )

        print(f"Submitting training job '{display_name}'...")
        
        # Submit the job with the correct parameters
        run_params = {
            'service_account': self.credentials.service_account_email,
            'sync': True  # Wait for completion
        }
        
        # Only add TensorBoard if we successfully created it
        if tensorboard_resource_name:
            run_params['tensorboard'] = tensorboard_resource_name
        
        job.run(**run_params)
        
        print("Job submission complete.")
        return job

    def get_or_create_tensorboard_instance(self, instance_name="brain-segmentation-tensorboard"):
        """
        Retrieves an existing TensorBoard instance or creates a new one.
        Args:
            instance_name (str): The display name for the TensorBoard instance.
        Returns:
            aiplatform.Tensorboard: The TensorBoard instance.
        """
        tensorboards = aiplatform.Tensorboard.list(filter=f'display_name="{instance_name}"')
        if tensorboards:
            print(f"Found existing TensorBoard instance: {tensorboards[0].resource_name}")
            return tensorboards[0]
        else:
            print(f"Creating new TensorBoard instance: {instance_name}")
            return aiplatform.Tensorboard.create(display_name=instance_name)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials_file', type=str, required=True, help='Path to GCP service account credentials file.')
    args = parser.parse_args()

    # --- CONFIGURATION ---
    PROJECT_ID = 'quark-469612' # Make sure this is correct
    GCS_BUCKET = 'gs://quark-brain-segmentation-bucket'
    CONTAINER_IMAGE_URI = f'gcr.io/{PROJECT_ID}/brain-segmentation-trainer:latest'
    LOCATION = 'australia-southeast1' # Explicitly define location

    model_dir = f"{GCS_BUCKET}/models"
    log_dir = f"{GCS_BUCKET}/logs"

    # --- JOB SUBMISSION ---
    try:
        manager = GCPTrainingManager(args.credentials_file, PROJECT_ID, location=LOCATION)
        
        # Command-line arguments for the training script
        command_args = [
            '--epochs', '50',
            '--batch_size', '4',
            '--learning_rate', '0.0001',
            '--num_samples', '100',
            '--grid_size', '32',
            '--model_dir', model_dir,
            '--log_dir', log_dir,
        ]

        # Submit the job
        training_job = manager.submit_training_job(
            display_name='brain-segmentation-training-v2', # Incremented version
            container_image_uri=CONTAINER_IMAGE_URI,
            command_args=command_args,
            tensorboard_log_dir=log_dir
        )
        
        # Since job.run() is synchronous, we can get the results right after.
        print("\n--- JOB RESULTS ---")
        print(f"Job State: {training_job.state.name}")
        print(f"GCP Job Dashboard: {training_job.web_ui}")
        # Construct TensorBoard URL after job completion
        if training_job.experiment_resource_name:
             tensorboard_url = f"https://{LOCATION}.tensorboard.googleusercontent.com/experiment/{training_job.experiment_resource_name.split('/')[-1]}"
             print(f"TensorBoard URL: {tensorboard_url}")
        print(f"GCS Output Path: {training_job.gcs_output_directory}")

    except Exception as e:
        print(f"An error occurred: {e}")