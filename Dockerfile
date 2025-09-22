# Use a lean Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files for the training job
COPY brain/modules/brainstem_segmentation/requirements.txt .
COPY brain/ /app/brain/

# Install the minimal dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint for the training script
ENTRYPOINT ["python", "brain/modules/brainstem_segmentation/run_gcp_training.py"]