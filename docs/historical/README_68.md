# Kubernetes (K8s) - Enterprise Cloud Computing

## Overview
Kubernetes is the most powerful open-source container orchestration platform for production cloud computing.

## Key Features
- **Container Orchestration**: Manage thousands of containers across clusters
- **Auto-scaling**: Horizontal and vertical pod autoscaling
- **Load Balancing**: Built-in load balancing and service discovery
- **Self-healing**: Automatic restart, replacement, and rescheduling
- **Rolling Updates**: Zero-downtime deployments
- **Resource Management**: CPU, memory, and storage allocation

## Production Setup

### 1. Cluster Configuration
```yaml
# cluster-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-config
data:
  cluster-name: "production-cluster"
  environment: "production"
  region: "us-west-2"
```

### 2. High Availability Setup
- **Multi-zone deployment**: Spread across availability zones
- **Etcd clustering**: Distributed key-value store
- **Load balancers**: Multiple ingress controllers
- **Backup strategy**: Regular etcd snapshots

### 3. Security Configuration
- **RBAC**: Role-based access control
- **Network policies**: Pod-to-pod communication rules
- **Secrets management**: Kubernetes secrets or external vaults
- **Pod security policies**: Security context enforcement

## ML/AI Workloads

### 1. Model Serving
```yaml
# model-serving-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: model-server
        image: ml-model:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
```

### 2. Distributed Training
```yaml
# distributed-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: distributed-training
spec:
  parallelism: 4
  template:
    spec:
      containers:
      - name: training-worker
        image: training-worker:latest
        env:
        - name: WORLD_SIZE
          value: "4"
        - name: RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
      restartPolicy: Never
```

### 3. Auto-scaling
```yaml
# hpa-config.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-serving
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring & Observability

### 1. Prometheus Configuration
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### 2. Grafana Dashboards
- **Cluster Overview**: Node status, pod count, resource usage
- **Application Metrics**: Request rate, latency, error rates
- **ML Model Metrics**: Inference latency, throughput, accuracy
- **Resource Utilization**: CPU, memory, GPU usage

## Best Practices

### 1. Resource Management
- **Resource requests and limits**: Always set both
- **Quality of Service**: Guaranteed, Burstable, BestEffort
- **Node affinity**: Control pod placement
- **Taints and tolerations**: Node isolation

### 2. Security
- **Pod security standards**: Enforce security policies
- **Network policies**: Control traffic flow
- **Secrets rotation**: Regular credential updates
- **Image scanning**: Vulnerability detection

### 3. Performance
- **Horizontal pod autoscaling**: Based on metrics
- **Vertical pod autoscaling**: Resource optimization
- **Cluster autoscaling**: Node pool management
- **Load balancing**: Multiple ingress controllers

## Deployment Strategies

### 1. Rolling Update
```bash
kubectl set image deployment/ml-model-serving ml-model=ml-model:v2
```

### 2. Blue-Green Deployment
```yaml
# blue-green-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
    version: blue  # Switch to green for new version
  ports:
  - port: 80
    targetPort: 8080
```

### 3. Canary Deployment
```yaml
# canary-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-canary
spec:
  replicas: 1  # Small percentage of traffic
  template:
    spec:
      containers:
      - name: ml-model
        image: ml-model:v2
```

## Cost Optimization

### 1. Spot Instances
```yaml
# spot-instance-node-pool.yaml
apiVersion: v1
kind: Node
metadata:
  labels:
    cloud.google.com/gke-spot: "true"
spec:
  taints:
  - key: cloud.google.com/gke-spot
    value: "true"
    effect: NoSchedule
```

### 2. Resource Optimization
- **Right-sizing**: Match resources to actual usage
- **Pod disruption budgets**: Control eviction rates
- **Priority classes**: Manage scheduling priority
- **Resource quotas**: Limit resource consumption

## Integration with ML Tools

### 1. Kubeflow
- **Pipelines**: End-to-end ML workflows
- **Training**: Distributed model training
- **Serving**: Model deployment and serving
- **Notebooks**: Jupyter notebook integration

### 2. Ray
- **Ray Cluster**: Distributed computing
- **Ray Serve**: Model serving
- **Ray Tune**: Hyperparameter tuning
- **Ray RLlib**: Reinforcement learning

### 3. MLflow
- **Model Registry**: Model versioning
- **Tracking**: Experiment tracking
- **Projects**: Reproducible ML workflows
- **Models**: Model packaging and deployment
