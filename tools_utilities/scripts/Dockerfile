# Quark SageMaker training image
# Base: AWS Deep Learning Container for PyTorch 2.1, CUDA 12.1, Python 3.10
# NOTE: Replace <REGION> if you build in a different region.
FROM 763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04

# Extra Python dependencies – leverage requirements.txt if present
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy Quark source into /app
COPY . /app
WORKDIR /app

# Environment expected by SageMaker training jobs
ENV SAGEMAKER_PROGRAM=tools_utilities/scripts/train_streaming.py

ENTRYPOINT ["/bin/bash", "-c", "python $SAGEMAKER_PROGRAM --config management/configurations/project/training_config.yaml --backend local"]
