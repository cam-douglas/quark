# Ray - Distributed ML/AI Computing

## Overview
Ray is the most powerful open-source distributed computing framework specifically designed for ML/AI workloads.

## Key Features
- **Ray Train**: Distributed model training
- **Ray Serve**: Scalable model serving
- **Ray Tune**: Hyperparameter tuning
- **Ray RLlib**: Reinforcement learning
- **Ray Data**: Distributed data processing
- **Ray Core**: General distributed computing

## Production Setup

### 1. Ray Cluster Configuration
```python
# ray-cluster-config.yaml
cluster_name: production-ray-cluster
max_workers: 100
min_workers: 10
initial_workers: 20
autoscaling_mode: default
target_utilization_fraction: 0.8
idle_timeout_minutes: 5

head_node:
  instance_type: m5.2xlarge
  image_id: ami-12345678

worker_nodes:
  instance_type: p3.2xlarge  # GPU instances for ML
  image_id: ami-12345678
  min_workers: 5
  max_workers: 50
```

### 2. Ray Cluster Deployment
```bash
# Deploy Ray cluster
ray up ray-cluster-config.yaml

# Connect to cluster
ray attach ray-cluster-config.yaml

# Submit job to cluster
ray submit ray-cluster-config.yaml train_model.py
```

## ML/AI Workloads

### 1. Distributed Model Training (Ray Train)
```python
# distributed_training.py
import ray
from ray import train
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn

def train_func(config):
    # Model definition
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Data loading
    train_dataset = torch.randn(10000, 784)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"]
    )
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["epochs"]):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Report metrics
            train.report({"loss": loss.item(), "epoch": epoch})

# Configure training
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"batch_size": 32, "epochs": 10},
    scaling_config={"num_workers": 4, "use_gpu": True}
)

# Start training
result = trainer.fit()
print(f"Final loss: {result.metrics['loss']}")
```

### 2. Model Serving (Ray Serve)
```python
# model_serving.py
import ray
from ray import serve
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)
    
    def forward(self, x):
        return self.linear(x)

# Create FastAPI app
app = FastAPI()

class PredictionRequest(BaseModel):
    data: list

@serve.deployment(num_replicas=3, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class ModelServer:
    def __init__(self):
        self.model = SimpleModel()
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()
    
    @app.post("/predict")
    async def predict(self, request: PredictionRequest):
        data = torch.tensor(request.data, dtype=torch.float32)
        with torch.no_grad():
            prediction = self.model(data)
        return {"prediction": prediction.tolist()}
    
    @app.get("/health")
    async def health(self):
        return {"status": "healthy"}

# Deploy model
serve.run(ModelServer.bind())
```

### 3. Hyperparameter Tuning (Ray Tune)
```python
# hyperparameter_tuning.py
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
import torch.nn as nn

def train_model(config):
    # Model with configurable hyperparameters
    model = nn.Sequential(
        nn.Linear(784, config["hidden_size"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["hidden_size"], 10)
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"]
    )
    
    # Training loop
    for epoch in range(10):
        # Training code here
        loss = torch.randn(1).item()  # Placeholder
        
        # Report metrics to Tune
        tune.report(loss=loss, accuracy=1-loss)

# Define search space
search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "hidden_size": tune.choice([256, 512, 1024]),
    "dropout": tune.uniform(0.1, 0.5)
}

# Configure scheduler and search algorithm
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

search_alg = OptunaSearch(metric="loss", mode="min")

# Run hyperparameter tuning
analysis = tune.run(
    train_model,
    config=search_space,
    scheduler=scheduler,
    search_alg=search_alg,
    num_samples=50,
    resources_per_trial={"cpu": 2, "gpu": 0.5}
)

print("Best config:", analysis.get_best_config("loss", "min"))
```

### 4. Reinforcement Learning (Ray RLlib)
```python
# reinforcement_learning.py
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
import gym

# Custom environment
class CustomEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
    
    def reset(self):
        return np.random.random(10)
    
    def step(self, action):
        reward = np.random.random()
        done = np.random.random() > 0.9
        obs = np.random.random(10)
        return obs, reward, done, {}

# Configure PPO algorithm
config = PPOConfig().environment(
    CustomEnv,
    env_config={"custom_param": 1.0}
).framework("torch").training(
    train_batch_size=4000,
    lr=0.0003,
    gamma=0.99,
    lambda_=0.95,
    model={"fcnet_hiddens": [64, 64]}
).rollouts(
    num_rollout_workers=4,
    rollout_fragment_length=200
).resources(
    num_gpus=1,
    num_cpus_per_worker=1
)

# Train agent
tune.run(
    "PPO",
    config=config,
    stop={"training_iteration": 100},
    checkpoint_freq=10
)
```

## Distributed Data Processing (Ray Data)

### 1. Large-scale Data Processing
```python
# distributed_data_processing.py
import ray
import ray.data as data
import pandas as pd
import numpy as np

# Create distributed dataset
ds = data.range(1000000)

# Transform data
def transform_batch(batch):
    return batch * 2 + 1

transformed_ds = ds.map_batches(transform_batch)

# Filter data
filtered_ds = transformed_ds.filter(lambda x: x > 1000)

# Aggregate data
result = filtered_ds.sum()
print(f"Sum: {result}")

# Write to storage
filtered_ds.write_parquet("s3://bucket/processed_data")
```

### 2. ML Data Pipeline
```python
# ml_data_pipeline.py
import ray
import ray.data as data
from ray.data.preprocessors import StandardScaler, LabelEncoder

# Load data
ds = data.read_csv("s3://bucket/raw_data.csv")

# Preprocess data
preprocessor = StandardScaler(columns=["feature1", "feature2"])
ds = preprocessor.fit_transform(ds)

# Encode labels
label_encoder = LabelEncoder(columns=["label"])
ds = label_encoder.fit_transform(ds)

# Split data
train_ds, test_ds = ds.train_test_split(test_size=0.2)

# Save processed data
train_ds.write_parquet("s3://bucket/train_data")
test_ds.write_parquet("s3://bucket/test_data")
```

## Monitoring & Observability

### 1. Ray Dashboard
```python
# Start Ray with dashboard
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# Access dashboard at http://localhost:8265
```

### 2. Custom Metrics
```python
# custom_metrics.py
import ray
from ray import train
import time

def training_with_metrics():
    for epoch in range(10):
        # Training code
        loss = torch.randn(1).item()
        accuracy = 1 - loss
        
        # Report custom metrics
        train.report({
            "loss": loss,
            "accuracy": accuracy,
            "epoch": epoch,
            "timestamp": time.time()
        })
```

## Best Practices

### 1. Resource Management
- **GPU allocation**: Use `num_gpus` parameter for GPU workloads
- **Memory management**: Monitor memory usage and garbage collection
- **CPU allocation**: Set appropriate `num_cpus` for CPU-intensive tasks
- **Object store**: Use `ray.put()` for large objects

### 2. Performance Optimization
- **Batch processing**: Process data in batches for efficiency
- **Object serialization**: Minimize object transfer overhead
- **Caching**: Use Ray object store for frequently accessed data
- **Load balancing**: Distribute work evenly across workers

### 3. Fault Tolerance
- **Checkpointing**: Regular checkpointing for long-running jobs
- **Error handling**: Proper exception handling in distributed code
- **Retry logic**: Implement retry mechanisms for failed tasks
- **Monitoring**: Continuous monitoring of cluster health

## Integration with Other Tools

### 1. Kubernetes Integration
```yaml
# ray-cluster-k8s.yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: ray-cluster
spec:
  headGroupSpec:
    serviceType: ClusterIP
    replicas: 1
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:latest
          ports:
          - containerPort: 6379
          - containerPort: 8265
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 4
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:latest
          resources:
            limits:
              nvidia.com/gpu: 1
```

### 2. MLflow Integration
```python
# ray_mlflow_integration.py
import mlflow
import ray
from ray import tune

def train_with_mlflow(config):
    mlflow.set_tracking_uri("http://localhost:5000")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Training code
        loss = train_model(config)
        
        # Log metrics
        mlflow.log_metric("loss", loss)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")

# Run with MLflow tracking
tune.run(
    train_with_mlflow,
    config=search_space,
    num_samples=10
)
```

## Cost Optimization

### 1. Spot Instances
```python
# spot_instance_config.py
cluster_config = {
    "head_node": {
        "instance_type": "m5.large",
        "spot_price": 0.05
    },
    "worker_nodes": {
        "instance_type": "p3.2xlarge",
        "spot_price": 0.50,
        "min_workers": 2,
        "max_workers": 20
    }
}
```

### 2. Auto-scaling
```python
# autoscaling_config.py
autoscaling_config = {
    "min_workers": 2,
    "max_workers": 50,
    "target_utilization_fraction": 0.8,
    "idle_timeout_minutes": 5
}
```
