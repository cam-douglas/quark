# Large Language Models (LLMs) - Production Ready

## Overview
Production-ready implementations of Large Language Models for cloud deployment.

## Supported Models

### 1. GPT Models
- **GPT-2**: 117M to 1.5B parameters
- **GPT-3**: 175B parameters (API access)
- **GPT-4**: Latest generation (API access)
- **Custom Fine-tuned Models**

### 2. BERT Variants
- **BERT-Base**: 110M parameters
- **BERT-Large**: 340M parameters
- **RoBERTa**: Optimized BERT
- **DistilBERT**: Distilled BERT (66M parameters)

### 3. T5 Models
- **T5-Base**: 220M parameters
- **T5-Large**: 770M parameters
- **T5-3B**: 3B parameters
- **mT5**: Multilingual T5

## Model Serving Templates

### 1. FastAPI Model Server
```python
# models/llm_models/fastapi_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Model Server", version="1.0.0")

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class TextGenerationResponse(BaseModel):
    generated_text: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    model_name: str

class LLMModelServer:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResponse:
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            input_tokens = inputs.input_ids.shape[1]
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=input_tokens + request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][input_tokens:], 
                skip_special_tokens=True
            )
            
            output_tokens = outputs.shape[1] - input_tokens
            generation_time = time.time() - start_time
            
            return TextGenerationResponse(
                generated_text=generated_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=generation_time,
                model_name=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize model server
model_server = LLMModelServer()

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text using the loaded LLM model."""
    return model_server.generate_text(request)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": model_server.model_name,
        "device": str(model_server.device)
    }

@app.get("/model-info")
async def model_info():
    """Get model information."""
    return {
        "model_name": model_server.model_name,
        "device": str(model_server.device),
        "parameters": sum(p.numel() for p in model_server.model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model_server.model.parameters() if p.requires_grad)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Ray Serve LLM Deployment
```python
# models/llm_models/ray_serve_llm.py
import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
from typing import List, Dict, Any

app = FastAPI()

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class TextGenerationResponse(BaseModel):
    generated_text: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    model_name: str

@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_gpus": 1, "num_cpus": 2}
)
@serve.ingress(app)
class LLMModelDeployment:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
    
    @app.post("/generate")
    async def generate_text(self, request: TextGenerationRequest):
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(
                request.prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            input_tokens = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=input_tokens + request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][input_tokens:], 
                skip_special_tokens=True
            )
            
            output_tokens = outputs.shape[1] - input_tokens
            generation_time = time.time() - start_time
            
            return TextGenerationResponse(
                generated_text=generated_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=generation_time,
                model_name=self.model_name
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check(self):
        return {
            "status": "healthy",
            "model": self.model_name,
            "device": str(self.device)
        }

# Deploy the model
if __name__ == "__main__":
    serve.run(LLMModelDeployment.bind("gpt2"))
```

### 3. Kubernetes Deployment
```yaml
# models/llm_models/k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-model-server
  labels:
    app: llm-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-model
  template:
    metadata:
      labels:
        app: llm-model
    spec:
      containers:
      - name: llm-server
        image: llm-model-server:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "gpt2"
        - name: MAX_LENGTH
          value: "100"
        - name: TEMPERATURE
          value: "0.7"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: llm-model-service
spec:
  selector:
    app: llm-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-model-server
  minReplicas: 2
  maxReplicas: 10
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

## Model Fine-tuning Templates

### 1. Custom Dataset Fine-tuning
```python
# models/llm_models/fine_tuning.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import logging

class LLMFineTuner:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def prepare_dataset(self, texts: List[str]) -> Dataset:
        """Prepare dataset for fine-tuning."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def fine_tune(
        self, 
        train_dataset: Dataset,
        output_dir: str = "./fine_tuned_model",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5
    ):
        """Fine-tune the model on custom dataset."""
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            push_to_hub=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return output_dir

# Example usage
if __name__ == "__main__":
    # Sample training data
    training_texts = [
        "This is a sample text for fine-tuning.",
        "Another example of training data.",
        "More training examples for the model."
    ]
    
    fine_tuner = LLMFineTuner("gpt2")
    dataset = fine_tuner.prepare_dataset(training_texts)
    fine_tuner.fine_tune(dataset, num_epochs=3)
```

## Model Monitoring

### 1. Performance Metrics
```python
# models/llm_models/monitoring.py
import time
import psutil
import torch
from typing import Dict, Any
import logging

class LLMMonitor:
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory": self._get_gpu_memory() if torch.cuda.is_available() else None,
            "timestamp": time.time()
        }
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage."""
        gpu_memory = {}
        for i in range(torch.cuda.device_count()):
            gpu_memory[f"gpu_{i}_allocated"] = torch.cuda.memory_allocated(i) / 1024**3
            gpu_memory[f"gpu_{i}_cached"] = torch.cuda.memory_reserved(i) / 1024**3
        return gpu_memory
    
    def log_inference_metrics(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        generation_time: float,
        model_name: str
    ):
        """Log inference performance metrics."""
        tokens_per_second = output_tokens / generation_time
        
        metrics = {
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "system_metrics": self.get_system_metrics()
        }
        
        self.logger.info(f"Inference metrics: {metrics}")
        return metrics
```

## Optimal Cloud Computing Resources

### 1. **AWS Resources for LLMs**

#### Compute Resources
- **EC2 P4d/P5 Instances**: Latest NVIDIA A100/H100 GPUs for training
  - P4d.24xlarge: 8x A100 GPUs, 400 Gbps networking
  - P5.48xlarge: 8x H100 GPUs, 400 Gbps networking
- **EC2 G5 Instances**: Cost-effective inference with A10G GPUs
  - g5.2xlarge: 1x A10G GPU, 8 vCPUs, 32 GB RAM
  - g5.12xlarge: 4x A10G GPUs, 48 vCPUs, 192 GB RAM
- **SageMaker**: Managed ML platform with auto-scaling
  - ml.p4d.24xlarge: 8x A100 GPUs for training
  - ml.g5.2xlarge: 1x A10G GPU for inference

#### Storage Resources
- **S3**: Object storage for model artifacts and datasets
  - Standard for frequently accessed data
  - Intelligent Tiering for cost optimization
  - Glacier for long-term archival
- **EBS**: Block storage for training volumes
  - gp3: General purpose SSD
  - io2: High-performance SSD for I/O intensive workloads
- **EFS**: Shared file system for distributed training

#### Networking Resources
- **Elastic Fabric Adapter (EFA)**: High-performance networking
  - 400 Gbps bandwidth for P4d/P5 instances
  - Low-latency communication for distributed training
- **VPC**: Network isolation and security
- **CloudFront**: Global content delivery for model serving

### 2. **Google Cloud Resources for LLMs**

#### Compute Resources
- **A3 Instances**: Latest H100 GPUs for training
  - a3-highgpu-8g: 8x H100 GPUs, 400 Gbps networking
- **G2 Instances**: L4 GPUs for inference
  - g2-standard-4: 1x L4 GPU, 4 vCPUs, 16 GB RAM
  - g2-standard-48: 8x L4 GPUs, 48 vCPUs, 192 GB RAM
- **Vertex AI**: Managed ML platform
  - Custom training with auto-scaling
  - Model serving with traffic splitting

#### Storage Resources
- **Cloud Storage**: Object storage with global edge locations
  - Standard for active data
  - Nearline for infrequent access
  - Coldline for archival
- **Persistent Disk**: Block storage for training
  - pd-standard: Balanced performance
  - pd-ssd: High-performance SSD
- **Filestore**: Managed NFS for shared storage

#### Networking Resources
- **Cloud Interconnect**: High-bandwidth connections
- **VPC**: Network isolation and firewall rules
- **Cloud CDN**: Global content delivery

### 3. **Azure Resources for LLMs**

#### Compute Resources
- **NC H100 v5**: Latest H100 GPUs for training
  - Standard_NC80ads_H100_v5: 8x H100 GPUs
- **NC A100 v4**: A100 GPUs for training and inference
  - Standard_NC48ads_H100_v5: 4x H100 GPUs
- **ND A100 v4**: A100 GPUs with InfiniBand
  - Standard_ND96asr_v4: 8x A100 GPUs, 400 Gbps InfiniBand
- **Azure ML**: Managed ML platform
  - Compute clusters with auto-scaling
  - Managed endpoints for model serving

#### Storage Resources
- **Blob Storage**: Object storage with tiered access
  - Hot tier for frequently accessed data
  - Cool tier for infrequent access
  - Archive tier for long-term storage
- **Managed Disks**: Block storage for compute instances
  - Premium SSD for high-performance workloads
  - Standard SSD for balanced performance
- **Azure Files**: Managed file shares

#### Networking Resources
- **ExpressRoute**: Private connections to Azure
- **Virtual Network**: Network isolation and security
- **Azure CDN**: Global content delivery

### 4. **Open-Source Cloud Computing Resources**

#### Kubernetes-Based Solutions
- **Kubernetes with GPU Operator**: Self-managed GPU clusters
  - NVIDIA GPU Operator for GPU management
  - Node auto-scaling with cluster autoscaler
  - Multi-GPU scheduling with device plugins
- **Kubeflow**: ML toolkit for Kubernetes
  - Distributed training with TFJob/PyTorchJob
  - Model serving with KFServing
  - Pipeline orchestration

#### Ray-Based Solutions
- **Ray Cluster**: Distributed computing framework
  - Ray Train for distributed training
  - Ray Serve for model serving
  - Ray Tune for hyperparameter optimization
- **Ray on Kubernetes**: Managed Ray clusters
  - Auto-scaling worker nodes
  - GPU resource management
  - Fault tolerance and recovery

#### Apache Spark-Based Solutions
- **Spark on Kubernetes**: Distributed data processing
  - Spark MLlib for ML algorithms
  - Spark Streaming for real-time processing
  - Delta Lake for ACID transactions

### 5. **Cost Optimization Strategies**

#### Instance Selection
- **Spot Instances**: 60-90% cost savings for batch workloads
  - Use for training jobs that can be interrupted
  - Implement checkpointing for fault tolerance
- **Reserved Instances**: 30-60% savings for predictable workloads
  - 1-year or 3-year commitments
  - All Upfront or Partial Upfront payment options
- **Savings Plans**: Flexible pricing for compute usage
  - 1-year or 3-year commitments
  - Automatic discount application

#### Storage Optimization
- **Lifecycle Policies**: Automatic tiering based on access patterns
- **Compression**: Reduce storage costs and transfer times
- **Deduplication**: Eliminate redundant data storage
- **Caching**: Reduce storage I/O costs

#### Network Optimization
- **Data Transfer**: Minimize cross-region data movement
- **CDN**: Cache frequently accessed models globally
- **Compression**: Reduce bandwidth costs
- **Batch Processing**: Optimize for throughput over latency

### 6. **Performance Optimization**

#### Training Optimization
- **Mixed Precision**: FP16 training for 2x speedup
- **Gradient Accumulation**: Simulate larger batch sizes
- **Model Parallelism**: Distribute large models across GPUs
- **Data Parallelism**: Distribute data across workers
- **Pipeline Parallelism**: Overlap computation and communication

#### Inference Optimization
- **Model Quantization**: INT8/FP16 for faster inference
- **Model Pruning**: Remove unnecessary weights
- **Knowledge Distillation**: Create smaller, faster models
- **Batch Inference**: Process multiple requests together
- **Model Caching**: Cache frequently used models

#### Infrastructure Optimization
- **Auto-scaling**: Scale based on demand
- **Load Balancing**: Distribute traffic evenly
- **Connection Pooling**: Reuse database connections
- **Caching**: Cache frequently accessed data
- **CDN**: Cache static content globally

## Best Practices

### 1. Model Optimization
- **Quantization**: Use INT8 or FP16 for faster inference
- **Model Pruning**: Remove unnecessary weights
- **Knowledge Distillation**: Create smaller, faster models
- **Caching**: Cache model outputs for repeated queries

### 2. Security
- **Input Validation**: Sanitize all inputs
- **Rate Limiting**: Prevent abuse
- **Authentication**: Secure API access
- **Content Filtering**: Filter inappropriate content

### 3. Performance
- **Batch Processing**: Process multiple requests together
- **Async Processing**: Use async/await for I/O operations
- **Load Balancing**: Distribute requests across replicas
- **Caching**: Cache frequently requested responses

### 4. Monitoring
- **Latency Tracking**: Monitor response times
- **Error Rate Monitoring**: Track failed requests
- **Resource Usage**: Monitor CPU, memory, GPU usage
- **Model Drift**: Detect performance degradation
