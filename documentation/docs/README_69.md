# Kubeflow - ML Toolkit for Kubernetes

## Overview
Kubeflow is the most powerful open-source ML toolkit for Kubernetes, providing end-to-end ML workflows from training to serving.

## Key Features
- **ML Pipelines**: End-to-end workflow orchestration
- **Distributed Training**: Multi-framework training support
- **Model Serving**: Production-ready model deployment
- **Notebooks**: Jupyter notebook integration
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Feature Store**: ML feature management

## Production Setup

### 1. Kubeflow Installation
```bash
# Install Kubeflow using manifests
export KUBEFLOW_VERSION=1.8.0
export KUBEFLOW_NAMESPACE=kubeflow

# Download Kubeflow manifests
wget https://github.com/kubeflow/manifests/archive/refs/tags/v${KUBEFLOW_VERSION}.tar.gz
tar -xvf v${KUBEFLOW_VERSION}.tar.gz
cd manifests-${KUBEFLOW_VERSION}

# Install Kubeflow
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
```

### 2. Kubeflow Configuration
```yaml
# kubeflow-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kubeflow-config
  namespace: kubeflow
data:
  # MLflow tracking server
  mlflow_tracking_uri: "http://mlflow.kubeflow.svc.cluster.local:5000"
  
  # Model registry
  model_registry_uri: "s3://kubeflow-models"
  
  # Feature store
  feature_store_uri: "redis://feature-store.kubeflow.svc.cluster.local:6379"
  
  # Monitoring
  prometheus_url: "http://prometheus.kubeflow.svc.cluster.local:9090"
  grafana_url: "http://grafana.kubeflow.svc.cluster.local:3000"
```

## ML/AI Workloads

### 1. Distributed Training (PyTorch)
```yaml
# pytorch-training-job.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-distributed-training
  namespace: kubeflow
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0.0
            command:
            - python
            - /opt/pytorch-mnist/mnist.py
            - --backend=nccl
            env:
            - name: MASTER_ADDR
              value: pytorch-distributed-training-master-0
            - name: MASTER_PORT
              value: "23456"
            - name: WORLD_SIZE
              value: "3"
            - name: RANK
              value: "0"
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "8Gi"
                cpu: "4"
              requests:
                nvidia.com/gpu: 1
                memory: "4Gi"
                cpu: "2"
            volumeMounts:
            - name: training-data
              mountPath: /opt/pytorch-mnist/data
            - name: model-storage
              mountPath: /opt/pytorch-mnist/models
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: model-storage
            persistentVolumeClaim:
              claimName: model-storage-pvc
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0.0
            command:
            - python
            - /opt/pytorch-mnist/mnist.py
            - --backend=nccl
            env:
            - name: MASTER_ADDR
              value: pytorch-distributed-training-master-0
            - name: MASTER_PORT
              value: "23456"
            - name: WORLD_SIZE
              value: "3"
            - name: RANK
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['training.kubeflow.org/replica-index']
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "8Gi"
                cpu: "4"
              requests:
                nvidia.com/gpu: 1
                memory: "4Gi"
                cpu: "2"
            volumeMounts:
            - name: training-data
              mountPath: /opt/pytorch-mnist/data
            - name: model-storage
              mountPath: /opt/pytorch-mnist/models
          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: model-storage
            persistentVolumeClaim:
              claimName: model-storage-pvc
```

### 2. ML Pipeline
```python
# ml_pipeline.py
import kfp
from kfp import dsl
from kfp.components import create_component_from_func
import mlflow

@create_component_from_func
def data_preprocessing_op(
    input_data_path: str,
    output_data_path: str
) -> str:
    """Data preprocessing component."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    df = pd.read_csv(input_data_path)
    
    # Preprocess data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.drop('target', axis=1))
    
    # Save processed data
    pd.DataFrame(df_scaled).to_csv(output_data_path, index=False)
    
    return output_data_path

@create_component_from_func
def model_training_op(
    input_data_path: str,
    model_path: str,
    mlflow_tracking_uri: str
) -> str:
    """Model training component."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import mlflow
    import mlflow.sklearn
    import pickle
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Load data
    df = pd.read_csv(input_data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest")
        
        # Log metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path

@create_component_from_func
def model_evaluation_op(
    model_path: str,
    test_data_path: str,
    evaluation_results_path: str
) -> str:
    """Model evaluation component."""
    import pandas as pd
    import pickle
    from sklearn.metrics import classification_report, confusion_matrix
    import json
    
    # Load model and test data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Generate evaluation report
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    evaluation_results = {
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "accuracy": report["accuracy"]
    }
    
    # Save results
    with open(evaluation_results_path, 'w') as f:
        json.dump(evaluation_results, f)
    
    return evaluation_results_path

@dsl.pipeline(
    name="ML Training Pipeline",
    description="End-to-end ML training pipeline"
)
def ml_training_pipeline(
    input_data_path: str = "s3://bucket/raw_data.csv",
    mlflow_tracking_uri: str = "http://mlflow.kubeflow.svc.cluster.local:5000"
):
    """ML training pipeline."""
    
    # Data preprocessing
    preprocessed_data = data_preprocessing_op(
        input_data_path=input_data_path,
        output_data_path="s3://bucket/preprocessed_data.csv"
    )
    
    # Model training
    trained_model = model_training_op(
        input_data_path=preprocessed_data.output,
        model_path="s3://bucket/models/random_forest.pkl",
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    
    # Model evaluation
    evaluation_results = model_evaluation_op(
        model_path=trained_model.output,
        test_data_path="s3://bucket/test_data.csv",
        evaluation_results_path="s3://bucket/evaluation_results.json"
    )

# Compile and run pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        ml_training_pipeline,
        "ml_training_pipeline.yaml"
    )
```

### 3. Model Serving (KFServing)
```yaml
# model-serving.yaml
apiVersion: serving.kubeflow.org/v1beta1
kind: InferenceService
metadata:
  name: ml-model-service
  namespace: kubeflow
spec:
  predictor:
    sklearn:
      storageUri: s3://bucket/models/random_forest.pkl
      resources:
        limits:
          memory: "2Gi"
          cpu: "1"
        requests:
          memory: "1Gi"
          cpu: "0.5"
  transformer:
    custom:
      container:
        image: custom-transformer:latest
        env:
        - name: MODEL_NAME
          value: "random_forest"
        resources:
          limits:
            memory: "1Gi"
            cpu: "0.5"
          requests:
            memory: "512Mi"
            cpu: "0.25"
  explainer:
    alibi:
      type: KernelShap
      storageUri: s3://bucket/models/random_forest.pkl
      resources:
        limits:
          memory: "2Gi"
          cpu: "1"
        requests:
          memory: "1Gi"
          cpu: "0.5"
```

### 4. Hyperparameter Tuning (Katib)
```yaml
# hyperparameter-tuning.yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: random-forest-tuning
  namespace: kubeflow
spec:
  objective:
    type: maximize
    goal: 0.95
    objectiveMetricName: accuracy
    additionalMetricNames:
    - f1_score
  algorithm:
    algorithmName: random
  parallelTrialCount: 3
  maxTrialCount: 10
  maxFailedTrialCount: 3
  parameters:
  - name: n_estimators
    parameterType: int
    feasibleSpace:
      min: "50"
      max: "200"
  - name: max_depth
    parameterType: int
    feasibleSpace:
      min: "3"
      max: "15"
  - name: min_samples_split
    parameterType: int
    feasibleSpace:
      min: "2"
      max: "10"
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
    - name: learningRate
      description: Learning rate for the training model
      reference: n_estimators
    - name: numberLayers
      description: Number of layers in the training model
      reference: max_depth
    - name: optimizer
      description: Training optimizer
      reference: min_samples_split
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
            - name: training-container
              image: training-image:latest
              command:
              - python
              - train.py
              - --n_estimators=${trialParameters.learningRate}
              - --max_depth=${trialParameters.numberLayers}
              - --min_samples_split=${trialParameters.optimizer}
              resources:
                limits:
                  memory: "4Gi"
                  cpu: "2"
                requests:
                  memory: "2Gi"
                  cpu: "1"
            restartPolicy: Never
```

## Monitoring & Observability

### 1. Prometheus Configuration
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: kubeflow
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'kubeflow-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
```

### 2. Grafana Dashboards
```yaml
# grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kubeflow-dashboard
  namespace: kubeflow
data:
  kubeflow-dashboard.json: |
    {
      "dashboard": {
        "title": "Kubeflow ML Dashboard",
        "panels": [
          {
            "title": "Training Jobs",
            "type": "stat",
            "targets": [
              {
                "expr": "kubeflow_training_jobs_total",
                "legendFormat": "Total Jobs"
              }
            ]
          },
          {
            "title": "Model Serving Requests",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(kubeflow_serving_requests_total[5m])",
                "legendFormat": "Requests/sec"
              }
            ]
          },
          {
            "title": "GPU Utilization",
            "type": "graph",
            "targets": [
              {
                "expr": "nvidia_gpu_utilization",
                "legendFormat": "GPU {{gpu}}"
              }
            ]
          }
        ]
      }
    }
```

## Best Practices

### 1. Resource Management
- **GPU Scheduling**: Use GPU operator for efficient GPU allocation
- **Memory Limits**: Set appropriate memory limits for containers
- **CPU Requests**: Set CPU requests to ensure resource allocation
- **Storage**: Use persistent volumes for model storage

### 2. Security
- **RBAC**: Implement role-based access control
- **Network Policies**: Restrict pod-to-pod communication
- **Secrets Management**: Use Kubernetes secrets for sensitive data
- **Image Scanning**: Scan container images for vulnerabilities

### 3. Monitoring
- **Metrics Collection**: Collect custom metrics from training jobs
- **Logging**: Centralize logs using ELK stack
- **Alerting**: Set up alerts for job failures and resource usage
- **Tracing**: Use distributed tracing for pipeline debugging

### 4. Cost Optimization
- **Spot Instances**: Use spot instances for training jobs
- **Auto-scaling**: Scale clusters based on workload
- **Resource Quotas**: Set resource quotas to prevent over-allocation
- **Job Scheduling**: Use priority classes for job scheduling

### 5. Performance
- **Data Locality**: Keep data close to compute resources
- **Caching**: Cache frequently accessed data
- **Parallelization**: Use parallel training strategies
- **Optimization**: Optimize model serving for latency and throughput
