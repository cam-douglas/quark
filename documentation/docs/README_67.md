# MLflow - ML Lifecycle Management

## Overview
MLflow is the most powerful open-source platform for managing the complete ML lifecycle, including experimentation, reproducibility, and deployment.

## Key Features
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Centralized model versioning and deployment
- **Model Serving**: Production-ready model deployment
- **Projects**: Reproducible ML workflows
- **Model Lineage**: Track model dependencies and versions

## Production Setup

### 1. MLflow Server Configuration
```yaml
# mlflow-server-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlflow-config
  namespace: mlflow
data:
  # Database configuration
  database_url: "postgresql://mlflow:password@postgres:5432/mlflow"
  
  # Artifact store
  artifact_root: "s3://mlflow-artifacts"
  
  # Model registry
  registry_uri: "s3://mlflow-models"
  
  # Authentication
  auth_type: "basic"
  
  # Monitoring
  prometheus_enabled: "true"
  grafana_enabled: "true"
```

### 2. Kubernetes Deployment
```yaml
# mlflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: mlflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:latest
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio:9000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: aws-access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: aws-secret-access-key
        - name: MLFLOW_S3_IGNORE_TLS
          value: "true"
        resources:
          limits:
            memory: "2Gi"
            cpu: "1"
          requests:
            memory: "1Gi"
            cpu: "0.5"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-server
  namespace: mlflow
spec:
  selector:
    app: mlflow-server
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

## ML/AI Workloads

### 1. Experiment Tracking
```python
# experiment_tracking.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("http://mlflow-server:5000")

# Load data
df = pd.read_csv("data.csv")
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start MLflow run
with mlflow.start_run(experiment_name="random_forest_experiment"):
    # Log parameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Log artifacts
    mlflow.log_artifact("data.csv", "data")
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv", "feature_importance")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
```

### 2. Model Registry
```python
# model_registry.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("http://mlflow-server:5000")

# Register model
model_name = "random_forest_classifier"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/random_forest_model",
    name=model_name
)

# Load model from registry
loaded_model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

# Transition model to staging
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

# Transition model to production
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

# Get model by stage
production_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
```

### 3. Model Serving
```python
# model_serving.py
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="MLflow Model Serving")

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

# Load model from registry
model_name = "random_forest_classifier"
model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0].max()
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": model_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. MLflow Projects
```yaml
# MLproject
name: random_forest_project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: string, default: "data.csv"}
      n_estimators: {type: integer, default: 100}
      max_depth: {type: integer, default: 10}
    command: "python train.py {data_path} {n_estimators} {max_depth}"
  
  evaluate:
    parameters:
      model_path: {type: string, default: "models/random_forest.pkl"}
      test_data_path: {type: string, default: "data/test.csv"}
    command: "python evaluate.py {model_path} {test_data_path}"
```

```python
# train.py
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # Get parameters
    data_path = sys.argv[1]
    n_estimators = int(sys.argv[2])
    max_depth = int(sys.argv[3])
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
```

### 5. Hyperparameter Tuning
```python
# hyperparameter_tuning.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("http://mlflow-server:5000")

# Load data
df = pd.read_csv("data.csv")
X = df.drop('target', axis=1)
y = df['target']

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Start MLflow run
with mlflow.start_run(experiment_name="hyperparameter_tuning"):
    # Fit grid search
    grid_search.fit(X, y)
    
    # Log best parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Log best score
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Log best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
    
    # Log all results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv("grid_search_results.csv", index=False)
    mlflow.log_artifact("grid_search_results.csv", "grid_search_results")
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
```

## Monitoring & Observability

### 1. MLflow UI Configuration
```yaml
# mlflow-ui-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlflow-ui-config
  namespace: mlflow
data:
  # UI configuration
  ui_config.yaml: |
    tracking_uri: http://mlflow-server:5000
    registry_uri: s3://mlflow-models
    artifact_root: s3://mlflow-artifacts
    
    # Authentication
    auth:
      type: basic
      users:
        admin: admin123
        user: user123
    
    # Monitoring
    monitoring:
      prometheus_enabled: true
      grafana_enabled: true
```

### 2. Custom Metrics Tracking
```python
# custom_metrics.py
import mlflow
import time
import psutil

def log_system_metrics():
    """Log system metrics during training."""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    mlflow.log_metric("cpu_usage", cpu_percent)
    mlflow.log_metric("memory_usage", memory_percent)
    mlflow.log_metric("timestamp", time.time())

# Use in training loop
with mlflow.start_run():
    for epoch in range(num_epochs):
        # Training code
        train_loss = train_epoch()
        
        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        log_system_metrics()
```

## Best Practices

### 1. Experiment Management
- **Naming Conventions**: Use consistent experiment names
- **Tagging**: Add tags for easy filtering and organization
- **Artifacts**: Log all relevant artifacts (data, models, plots)
- **Versioning**: Use semantic versioning for models

### 2. Model Registry
- **Staging Process**: Use staging environment before production
- **Model Validation**: Implement validation checks before promotion
- **Rollback Strategy**: Keep previous versions for rollback
- **Documentation**: Document model changes and improvements

### 3. Security
- **Authentication**: Implement proper authentication
- **Authorization**: Use role-based access control
- **Secrets Management**: Store sensitive data securely
- **Audit Logging**: Log all model registry operations

### 4. Performance
- **Artifact Storage**: Use efficient storage backends
- **Caching**: Cache frequently accessed models
- **Parallel Processing**: Use parallel training when possible
- **Resource Management**: Monitor and optimize resource usage

### 5. Monitoring
- **Model Performance**: Track model performance over time
- **Data Drift**: Monitor for data drift and model degradation
- **System Metrics**: Monitor system resources and performance
- **Alerting**: Set up alerts for critical issues
