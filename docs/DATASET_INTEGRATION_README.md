# Dataset Integration for SmallMind

This module integrates high-quality open LLM datasets into your SmallMind model, providing enhanced natural language understanding while maintaining code capabilities.

## 🎯 Overview

The dataset integration system provides:

- **Web-scale pretraining data**: FineWeb, Dolma, RedPajama, RefinedWeb
- **Code-aware corpora**: The Stack v2, OpenCodeReasoning
- **Post-training datasets**: Tülu, UltraFeedback, OpenMathInstruct
- **Streaming support**: Memory-efficient data loading
- **Interleaving**: Intelligent mixing of datasets with configurable weights
- **Training pipeline**: Full integration with PyTorch/Transformers

## 🚀 Quick Start

### Installation

```bash
# Install core dependencies
pip install datasets transformers torch accelerate

# Or use the requirements file
pip install -r src/smallmind/requirements/requirements_dataset_integration.txt
```

### Basic Usage

```python
from smallmind.models import get_dataset_integrator, get_trainer

# Get a dataset integrator
integrator = get_dataset_integrator()

# Create a balanced training mixture
mixture = integrator.create_training_mixture("balanced")

# Sample some data
for example in mixture.take(5):
    print(example)
```

## 📊 Available Training Mixtures

### 1. Balanced Mixture (Default)
- **FineWeb**: 18% - High-quality web data
- **Dolma**: 18% - Web, papers, books, code
- **RedPajama**: 10% - Quality-annotated web data
- **RefinedWeb**: 7% - Clean web corpus
- **The Stack v2**: 10% - Code across 600+ languages
- **Tülu SFT**: 12% - Instruction following
- **OpenCodeReasoning**: 6% - Code reasoning
- **OpenMathInstruct**: 6% - Math instruction
- **UltraFeedback**: 6% - Preference learning
- **MADLAD**: 7% - Multilingual support

### 2. Code-Focused Mixture
- **The Stack v2**: 30% - Heavy code emphasis
- **OpenCodeReasoning**: 20% - Code reasoning
- **FineWeb**: 20% - General web data
- **Dolma**: 20% - Mixed content
- **Tülu SFT**: 10% - Instruction following

### 3. Reasoning-Focused Mixture
- **OpenMathInstruct**: 25% - Math reasoning
- **OpenCodeReasoning**: 20% - Code reasoning
- **FineWeb-Edu**: 25% - Educational content
- **Tülu SFT**: 20% - Instruction following
- **UltraFeedback**: 10% - Preference learning

## 🔧 Custom Mixtures

Create your own training mixtures with custom weights:

```python
# Create a custom mixture
custom_mixture = integrator.create_custom_mixture(
    name="my_special_mixture",
    datasets=["fineweb", "stack_v2", "opencode_reasoning"],
    weights=[0.4, 0.4, 0.2],
    seed=42
)

# Use it for training
mixture = integrator.create_training_mixture("my_special_mixture")
```

## 🏋️ Training Pipeline

### Basic Training

```python
from smallmind.models import get_trainer, TrainingConfig

# Get trainer
trainer = get_trainer()

# Configure training
config = TrainingConfig(
    model_name_or_path="./models/checkpoints/your-model",
    output_dir="./trained_models/output",
    mixture_name="balanced",
    max_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5
)

# Start training
results = trainer.train(config)
```

### Quick Training Function

```python
from smallmind.models import quick_train

# Quick training with sensible defaults
results = quick_train(
    model_path="./models/checkpoints/your-model",
    mixture_name="code_focused",
    output_dir="./trained_models/quick_run",
    max_steps=500
)
```

## 📁 Dataset Details

### FineWeb
- **Source**: Common Crawl (2013-2024)
- **Size**: ~15T tokens
- **Quality**: High-quality web text with filtering
- **Subsets**: CC-MAIN-2024-10 (recommended for recent data)

### The Stack v2
- **Source**: GitHub repositories
- **Size**: Billions of files
- **Languages**: 600+ programming languages
- **Features**: Code, documentation, notebooks

### OpenCodeReasoning
- **Source**: Competitive programming
- **Size**: ~735k problems
- **Features**: Step-by-step reasoning traces
- **Use case**: Code + reasoning skills

### Tülu 3
- **Source**: Instruction following data
- **Size**: ~939k SFT examples
- **Features**: Multi-domain, safety-focused
- **Use case**: Post-training refinement

## ⚙️ Configuration Options

### Dataset Configuration

```python
from smallmind.models import DatasetConfig

config = DatasetConfig(
    name="Custom Dataset",
    dataset_id="your/dataset",
    split="train",
    subset="specific_config",  # Optional
    weight=0.2,
    max_samples=1000000,  # Limit samples
    streaming=True,
    filters={"license": ["mit", "apache-2.0"]}  # Apply filters
)
```

### Training Configuration

```python
from smallmind.models import TrainingConfig

config = TrainingConfig(
    model_name_or_path="microsoft/DialoGPT-medium",
    output_dir="./output",
    mixture_name="balanced",
    max_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=3,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    push_to_hub=False,
    hub_model_id=None,
    hub_token=None
)
```

## 🔍 Data Exploration

### Explore Available Datasets

```python
# List all available mixtures
mixtures = integrator.list_available_mixtures()
print(f"Available mixtures: {mixtures}")

# Get info about a specific dataset
info = integrator.get_dataset_info("fineweb")
print(f"FineWeb info: {info}")

# Get mixture details
mixture_info = trainer.get_mixture_info("balanced")
print(f"Balanced mixture: {mixture_info}")
```

### Sample Data

```python
# Sample from a specific dataset
fineweb = integrator.load_dataset(integrator.dataset_configs["fineweb"])

for example in fineweb.take(3):
    # Find text content
    text = None
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 100:
            text = value
            break
    
    if text:
        print(f"Text: {text[:200]}...")
```

## 🚨 Error Handling

The system includes robust error handling:

- **Dataset loading failures**: Automatic fallback to minimal datasets
- **Network issues**: Graceful degradation with cached data
- **Memory constraints**: Streaming support for large datasets
- **Validation**: Config validation before training starts

## 📊 Performance Considerations

### Memory Efficiency
- **Streaming**: Datasets load on-demand, not all at once
- **Interleaving**: Efficient mixing without loading everything into memory
- **Batch processing**: Configurable batch sizes for your hardware

### Training Optimizations
- **Mixed precision**: Automatic FP16/BF16 when available
- **Gradient checkpointing**: Memory-efficient training
- **Parallel data loading**: Multi-worker dataloaders
- **Accelerate integration**: Distributed training support

## 🔧 Advanced Usage

### Custom Data Processing

```python
# Custom tokenization function
def custom_tokenize(examples, tokenizer, max_length=2048):
    # Your custom logic here
    return tokenized_examples

# Use in training
tokenized_dataset = trainer.create_dataset_loader(
    dataset, tokenizer, config, max_length=2048
)
```

### Save/Load Mixture Configs

```python
# Save a mixture configuration
trainer.save_mixture_config("balanced", "./configs/balanced_mixture.json")

# Load a saved configuration
mixture = trainer.load_mixture_config("./configs/balanced_mixture.json")
```

### Validation

```python
# Validate training configuration
issues = trainer.validate_training_config(config)
if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

## 📚 Examples

### Complete Training Example

```python
from smallmind.models import get_trainer, TrainingConfig

# Initialize
trainer = get_trainer()

# Configure
config = TrainingConfig(
    model_name_or_path="./models/checkpoints/deepseek-v2",
    output_dir="./trained_models/enhanced_model",
    mixture_name="code_focused",
    max_steps=2000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    warmup_steps=200
)

# Train
try:
    results = trainer.train(config)
    print(f"Training completed! Final loss: {results['final_loss']}")
    print(f"Training time: {results['training_time_hours']:.2f} hours")
except Exception as e:
    print(f"Training failed: {e}")
```

### Data Exploration Example

```python
from smallmind.models import get_dataset_integrator

integrator = get_dataset_integrator()

# Explore different mixtures
for mixture_name in ["balanced", "code_focused", "reasoning_focused"]:
    print(f"\n=== {mixture_name.upper()} ===")
    
    mixture = integrator.create_training_mixture(mixture_name)
    
    # Sample examples
    for i, example in enumerate(mixture.take(2)):
        text = next((v for v in example.values() if isinstance(v, str) and len(v) > 50), None)
        if text:
            print(f"Example {i+1}: {text[:100]}...")
```

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Install required dependencies
   ```bash
   pip install datasets transformers torch accelerate
   ```

2. **Memory Issues**: Reduce batch size or use streaming
   ```python
   config.per_device_train_batch_size = 1
   config.gradient_accumulation_steps = 8
   ```

3. **Dataset Loading Failures**: Check internet connection and dataset availability

4. **Training Failures**: Validate configuration and check model path

### Getting Help

- Check the demo script: `python src/smallmind/demos/dataset_integration_demo.py`
- Review error logs in the output directory
- Validate your training configuration
- Ensure all dependencies are installed

## 🔮 Future Enhancements

- **More datasets**: Additional high-quality open datasets
- **Advanced filtering**: ML-based quality filtering
- **Curriculum learning**: Dynamic dataset mixing based on training progress
- **Distributed training**: Multi-GPU and multi-node support
- **Evaluation**: Built-in evaluation on downstream tasks

## 📄 License

This module integrates with datasets under their respective licenses:
- FineWeb: ODC-By v1.0
- The Stack v2: Various open source licenses
- Tülu: Apache 2.0
- Other datasets: Check individual dataset licenses

## 🤝 Contributing

To add new datasets or improve the integration:

1. Add dataset configuration to `DatasetIntegrator._init_dataset_configs()`
2. Create appropriate training mixtures
3. Add tests for new functionality
4. Update documentation

---

**Happy training with high-quality open datasets! 🚀**
