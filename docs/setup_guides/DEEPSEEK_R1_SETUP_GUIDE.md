# DeepSeek-R1 Training and Fine-tuning Guide

Complete guide for training and fine-tuning DeepSeek-R1 models with your brain simulation framework.

## 📋 Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies
pip install torch transformers datasets accelerate
pip install vllm sglang  # For serving (optional)

# For CUDA support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Model Selection

DeepSeek-R1 offers several distilled models based on your hardware:

| Model | Parameters | GPU Memory | Use Case |
|-------|------------|------------|----------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 4GB | Development/Testing |
| DeepSeek-R1-Distill-Qwen-7B | 7B | 16GB | Production/Research |
| DeepSeek-R1-Distill-Qwen-14B | 14B | 32GB | High Performance |
| DeepSeek-R1-Distill-Qwen-32B | 32B | 64GB | State-of-the-art |
| DeepSeek-R1-Distill-Llama-8B | 8B | 18GB | Llama Integration |
| DeepSeek-R1-Distill-Llama-70B | 70B | 140GB | Maximum Performance |

### 3. Basic Usage

```python
from src.core.deepseek_r1_trainer import DeepSeekR1Trainer

# Auto-select model based on your hardware
trainer = DeepSeekR1Trainer()

# Generate reasoning response
response = trainer.generate_reasoning_response(
    "How does consciousness emerge in neural networks?",
    max_length=1024,
    temperature=0.6
)
print(response)
```

## 🧠 Dataset Creation

### Brain Simulation Datasets

The trainer includes pre-built datasets focused on:
- Neural network processing
- Consciousness and complexity
- Brain simulation principles
- Cognitive architectures

### Custom Datasets

```python
# Create custom reasoning examples
custom_examples = [
    {
        "question": "Your question here",
        "reasoning": "<think>\nStep-by-step reasoning...\n</think>\nFinal answer",
        "answer": "Concise answer"
    }
]

# Prepare dataset
train_dataset = trainer.prepare_training_dataset(custom_examples)
```

## 🔧 Fine-tuning Process

### 1. Setup Training

```python
# Prepare training dataset
train_dataset = trainer.prepare_training_dataset()

# Setup fine-tuning configuration
hf_trainer, training_args = trainer.setup_fine_tuning(
    train_dataset, 
    output_dir="./my_fine_tuned_model"
)
```

### 2. Run Training

```python
# Execute fine-tuning
result = trainer.run_fine_tuning(hf_trainer)

if result["success"]:
    print(f"✅ Training completed! Model saved to: {result['model_path']}")
else:
    print(f"❌ Training failed: {result['error']}")
```

### 3. Evaluate Results

```python
# Test the fine-tuned model
evaluation = trainer.evaluate_model(
    model_path="./my_fine_tuned_model",
    test_prompts=[
        "How do neural networks learn?",
        "What is consciousness?",
        "Explain brain plasticity."
    ]
)
```

## 🚀 Deployment Options

### Option 1: vLLM (Recommended for High Performance)

```bash
# Install vLLM
pip install vllm

# Serve your fine-tuned model
vllm serve ./my_fine_tuned_model \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enforce-eager
```

### Option 2: SGLang (Alternative Serving)

```bash
# Install SGLang
pip install sglang

# Serve model
python3 -m sglang.launch_server \
    --model ./my_fine_tuned_model \
    --trust-remote-code \
    --tp 2
```

### Option 3: Direct Integration

```python
# Load and use directly in Python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./my_fine_tuned_model")
model = AutoModelForCausalLM.from_pretrained("./my_fine_tuned_model")

# Generate responses
inputs = tokenizer("Your prompt", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 🌐 API Usage

Once served, you can interact via HTTP API:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "./my_fine_tuned_model",
        "prompt": "<think>\nExplain neural plasticity\n\nPlease reason step by step.</think>",
        "max_tokens": 1024,
        "temperature": 0.6
    }
)

print(response.json()["choices"][0]["text"])
```

## 📊 Monitoring and Optimization

### Training Metrics

The system automatically tracks:
- Training loss progression
- Model convergence
- Memory usage
- Training duration
- Parameter counts

### Performance Optimization

1. **Memory Optimization**:
   - Use gradient checkpointing
   - Enable mixed precision (FP16)
   - Adjust batch sizes
   - Use gradient accumulation

2. **Speed Optimization**:
   - Multi-GPU training
   - Tensor parallelism
   - Optimized data loading
   - Efficient tokenization

3. **Quality Optimization**:
   - Early stopping
   - Learning rate scheduling
   - Data augmentation
   - Validation monitoring

## 🧪 Integration with Brain Simulation

### Real-time Analysis

```python
from src.core.deepseek_r1_trainer import BrainSimulationIntegration

# Initialize integration
integration = BrainSimulationIntegration(trainer)

# Analyze brain state
brain_state = {"neural_activity": 0.75, "consciousness_score": 0.68}
analysis = integration.analyze_brain_state(brain_state)

# Continuous monitoring
monitoring_results = integration.continuous_monitoring(duration_minutes=10)
```

### Insight Generation

```python
# Generate simulation insights
metrics = {
    'neural_activity': 0.75,
    'consciousness_score': 0.68,
    'memory_utilization': 0.82,
    'attention_focus': 0.91
}

insights = integration.generate_simulation_insights(metrics)
print(insights['insights'])
```

## ⚙️ Configuration Options

### Training Parameters

```python
training_args = TrainingArguments(
    output_dir="./my_model",
    num_train_epochs=3,                 # Training epochs
    per_device_train_batch_size=1,      # Batch size per device
    gradient_accumulation_steps=8,       # Gradient accumulation
    learning_rate=1e-5,                 # Learning rate
    weight_decay=0.01,                  # Weight decay
    warmup_steps=100,                   # Warmup steps
    fp16=True,                          # Mixed precision
    gradient_checkpointing=True,        # Memory optimization
)
```

### Generation Parameters

```python
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.6,          # 0.5-0.7 recommended
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model variant
   - Enable CPU offloading

2. **Slow Training**:
   - Increase batch size (if memory allows)
   - Use multiple GPUs
   - Enable mixed precision
   - Optimize data loading

3. **Poor Quality Responses**:
   - Increase training epochs
   - Improve dataset quality
   - Adjust learning rate
   - Add more diverse examples

4. **Model Loading Errors**:
   - Check model path
   - Verify HuggingFace cache
   - Ensure sufficient disk space
   - Update transformers library

### Hardware Requirements

| Model Size | Minimum RAM | Recommended GPU | Training Time* |
|------------|-------------|-----------------|----------------|
| 1.5B | 8GB | RTX 3080 (10GB) | 1-2 hours |
| 7B | 32GB | RTX 4090 (24GB) | 3-6 hours |
| 14B | 64GB | A100 (40GB) | 6-12 hours |
| 32B | 128GB | A100 (80GB) | 12-24 hours |
| 70B | 256GB | Multi A100 | 24-48 hours |

*Estimated for 1000 training examples, 3 epochs

## 📚 Additional Resources

### Documentation Links

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)
- [HuggingFace Model Hub](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://sgl-project.github.io/)

### Example Notebooks

- `notebooks/training/deepseek_r1_training.ipynb` - Complete training walkthrough
- `notebooks/training/interactive_model_training.ipynb` - Interactive training
- `notebooks/prototyping/getting_started_example.ipynb` - Basic usage

### Integration Examples

- Brain simulation analysis
- Consciousness metric interpretation
- Real-time monitoring
- Custom dataset creation

## 💡 Best Practices

1. **Start Small**: Begin with the 1.5B model for testing
2. **Quality Data**: Focus on high-quality reasoning examples
3. **Iterative Improvement**: Fine-tune incrementally
4. **Monitor Performance**: Track training metrics closely
5. **Validate Results**: Test on diverse examples
6. **Document Changes**: Keep track of modifications
7. **Regular Backups**: Save model checkpoints frequently

---

**Ready to start training? Run the notebook or use the Python API directly!**

For support, check the troubleshooting section or review the example implementations in the `notebooks/` directory.
