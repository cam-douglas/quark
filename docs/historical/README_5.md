# SmallMind: Comprehensive Brain Development & Neuroscience Simulation Platform

[![CI/CD](https://github.com/smallmind/smallmind/workflows/SmallMind%20CI%2FCD/badge.svg)](https://github.com/smallmind/smallmind/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://smallmind.readthedocs.io/)

SmallMind is a comprehensive platform for simulating brain development, neural networks, and neuroscience research. It integrates multiple simulation engines, data sources, and AI models to provide a unified environment for computational neuroscience research.

## 🌟 Features

- **Multi-Engine Simulation**: Support for MuJoCo, NEST, CompuCell3D, and VISIT
- **Brain Development Modeling**: Fetal brain development simulation with morphogen dynamics
- **Neural Network Integration**: Connectomics, plasticity, and learning algorithms
- **Data Integration**: FlyWire, NeuroData, and open neuroscience datasets
- **AI/ML Integration**: BabyAGI, MoE models, and machine learning pipelines
- **Continuous Training**: Infinite loop training with high-quality open datasets
- **Cross-Platform**: Web, desktop, and mobile deployment support
- **Extensible Architecture**: Plugin system for custom simulations and models

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or higher** (3.11+ recommended)
- **Git** for cloning the repository
- **Virtual environment** (recommended)
- **Internet connection** for dependency downloads

### 🎯 Installation Options

#### **Option 1: Universal Installer (Recommended)**
```bash
# Clone the repository
git clone https://github.com/smallmind/smallmind.git
cd smallmind

# Run the universal installer - ONE COMMAND SETS UP EVERYTHING!
python install_universal.py
```

**What this does automatically:**
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Sets up Small-Mind package
- ✅ Configures terminal integration
- ✅ Creates desktop shortcuts
- ✅ Tests the installation

#### **Option 2: Manual Installation**
```bash
# Clone the repository
git clone https://github.com/smallmind/smallmind.git
cd smallmind

# Create virtual environment
python -m venv smallmind_env
source smallmind_env/bin/activate  # On Windows: smallmind_env\Scripts\activate

# Install working dependencies
pip install -r requirements_working.txt

# Install Small-Mind package
pip install -e .

# Setup Cursor integration
python cursor_smallmind_integration.py
```

#### **Option 3: Quick Fix for Issues**
```bash
# If you encounter dependency issues
python quick_fix.py

# This will fix common problems and install core dependencies
```

#### **Option 4: Docker Installation**
```bash
git clone https://github.com/smallmind/smallmind.git
cd smallmind
docker-compose up -d
```

### 🧪 Quick Demo

```bash
# After installation, test the system
python src/smallmind/demos/advanced_integration_demo.py

# Or use the CLI
smallmind --help
smallmind-optimize --help
smallmind-simulate --help
```

## 🎨 Icon Integration

The Small-Mind + Cursor AI package includes a custom lightbulb-brain fusion icon:

```bash
# Create the icon files
python create_icon_integration.py

# Convert to different formats
python convert_icons.py
```

**Icon Features:**
- 🧠 **Brain half**: Represents neuroscience simulation and intelligence
- 💡 **Lightbulb half**: Represents AI assistance and innovative ideas
- ✨ **Light rays**: Symbolize enlightenment and knowledge
- 🎯 **Professional branding**: Suitable for desktop shortcuts and installers

## 🖥️ Usage

### **Command Line Interface**

```bash
# Main Small-Mind CLI
smallmind --help

# Advanced optimization
smallmind-optimize --help
smallmind-optimize --model "meta-llama/Meta-Llama-3-8B-Instruct"

# Brain simulation
smallmind-simulate --help
smallmind-simulate --steps 2000 --physics pybullet

# Neural network optimization
smallmind-neural --help

# Continuous training (infinite loop)
python src/smallmind/scripts/start_continuous_training.py --model-path ./models/your-model

# Dataset integration and exploration
python src/smallmind/cli/dataset_cli.py --help
python src/smallmind/cli/dataset_cli.py list-mixtures
python src/smallmind/cli/dataset_cli.py explore-mixture --mixture balanced --samples 3
```

### **Unified Cursor + Small-Mind Interface**

```bash
# Launch unified CLI
cursor-smallmind --help

# Run Small-Mind commands
cursor-smallmind smallmind --help
cursor-smallmind simulate --steps 2000
cursor-smallmind optimize --model "meta-llama/Meta-Llama-3-8B-Instruct"

# Use Cursor AI
cursor-smallmind cursor --query "Explain brain development"
```

### **Python API**

```python
from smallmind.ml_optimization.advanced_optimizer import SmallMindAdvancedOptimizer
from smallmind.simulation.simulation_runner import BrainDevelopmentSimulation
from smallmind.models import get_dataset_integrator, get_trainer
from smallmind.models.continuous_trainer import train_forever

# Run optimization
optimizer = SmallMindAdvancedOptimizer()
results = optimizer.optimize_model("meta-llama/Meta-Llama-3-8B-Instruct")

# Run brain simulation
simulation = BrainDevelopmentSimulation()
results = simulation.run_simulation()

# Continuous training (infinite loop)
train_forever(
    model_path="./models/checkpoints/your-model",
    output_dir="./continuous_training"
)

# Dataset integration
integrator = get_dataset_integrator()
mixture = integrator.create_training_mixture("balanced")
```

## 📚 Documentation

- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Simulation Examples](docs/SIMULATION_EXAMPLES.md)
- [Development Guide](docs/DEVELOPMENT.md)

## 🏗️ Architecture

SmallMind follows a modular architecture with specialized components:

```
src/
├── baby_agi/           # BabyAGI integration
├── cli/                # Command-line interfaces
├── data/               # Data management and caching
├── demos/              # Example demonstrations
├── integration/        # Third-party integrations
├── ml_optimization/    # Machine learning optimization
├── models/             # AI models and checkpoints
├── neurodata/          # Neuroscience data interfaces
├── physics_simulation/ # Physics simulation engines
├── requirements/       # Dependency specifications
├── simulation/         # Simulation frameworks
├── tests/              # Test suite
├── tools/              # Utility tools
└── visualization/      # Visualization components
```

## 🚀 Continuous Training System

SmallMind includes a powerful continuous training system that runs in an infinite loop, continuously improving your models with high-quality open datasets until manually stopped.

### **Training Modes**

1. **🖥️ Local Training**: Train on your local machine
2. **🔄 Multi-Model Training**: Train all 3 models simultaneously locally
3. **☁️ Cloud Training**: Train on cloud platforms for maximum performance

### **Key Features**
- **🔄 Infinite Loop**: Training never stops unless manually interrupted
- **🔄 Dataset Rotation**: Automatically cycles through balanced, code-focused, and reasoning-focused mixtures
- **💾 Auto-Checkpointing**: Saves progress after each epoch
- **🛑 Graceful Shutdown**: Clean stops and automatic resume capability
- **📊 Progress Tracking**: Monitor improvement over time
- **🚀 Auto-Recovery**: Handles failures automatically

### **Available Training Mixtures**

1. **Balanced Mixture** (18% FineWeb, 18% Dolma, 10% The Stack v2, 12% Tülu, etc.)
2. **Code-Focused Mixture** (30% The Stack v2, 20% OpenCodeReasoning, 20% FineWeb)
3. **Reasoning-Focused Mixture** (25% OpenMathInstruct, 20% OpenCodeReasoning, 25% FineWeb-Edu)

### **Quick Start Commands**

```bash
# Start infinite training
python src/smallmind/scripts/start_continuous_training.py \
    --model-path ./models/checkpoints/your-model \
    --output-dir ./continuous_training

# Explore available datasets
python src/smallmind/cli/dataset_cli.py list-mixtures
python src/smallmind/cli/dataset_cli.py explore-mixture --mixture balanced --samples 3

# Run interactive demo
python src/smallmind/demos/continuous_training_demo.py
```

### **Python API for Continuous Training**

```python
from smallmind.models.continuous_trainer import train_forever

# Train forever until stopped
train_forever(
    model_path="./models/checkpoints/your-model",
    output_dir="./continuous_training"
)

# Or use the trainer class for more control
from smallmind.models.continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer("./models/your-model", "./output")
trainer.train_forever()  # Runs until manually stopped
```

## ☁️ **Cloud-Based Training System**

SmallMind's cloud integration enables you to train all models simultaneously on cloud platforms for maximum performance, scalability, and cost efficiency.

### **Supported Cloud Platforms**

| Platform | GPU Types | Instance Types | Cost/Hour | Setup |
|----------|-----------|----------------|-----------|-------|
| **AWS** | T4, V100, A100 | g4dn.xlarge, p3.2xlarge | $0.53-$32.77 | SSH Key + Security Group |
| **Google Cloud** | T4, V100, A100 | n1-standard-4/8/16 | $0.54-$3.69 | Project ID |
| **Azure** | T4, V100, A100 | Standard_NC4as_T4_v3 | Coming Soon | Subscription ID |

### **Cloud Training Benefits**

- **🚀 Maximum Performance**: Dedicated GPU instances
- **📈 Scalability**: Launch as many instances as needed
- **💰 Cost Control**: Pay only for what you use
- **🌐 Remote Access**: Train from anywhere
- **📊 Real-time Monitoring**: Track progress and costs
- **🛑 Auto-shutdown**: Stop instances when done

### **Quick Cloud Start**

```bash
# 1. Install cloud dependencies
pip install -r src/smallmind/requirements/requirements_cloud.txt

# 2. Configure cloud credentials
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# 3. Start AWS training
python src/smallmind/scripts/start_cloud_training.py \
    --platform aws \
    --region us-east-1 \
    --ssh-key your-key \
    --security-group sg-12345678 \
    --check-costs

# 4. Start GCP training
python src/smallmind/scripts/start_cloud_training.py \
    --platform gcp \
    --region us-central1 \
    --project-id your-project \
    --check-costs
```

### **Cost Optimization Tips**

- **🕐 Spot Instances**: Use AWS Spot or GCP Preemptible for 60-90% savings
- **⏰ Auto-shutdown**: Set up automatic instance termination
- **📊 Monitor Billing**: Use cloud cost monitoring tools
- **🎯 Right-sizing**: Choose appropriate instance types for your models

## 🎯 **Complete Command Reference & Help**

### **Available Models in Your System**

```bash
# DeepSeek V2 (64 experts, 32GB memory) - RECOMMENDED
src/smallmind/models/models/checkpoints/deepseek-v2

# Qwen 1.5 MoE (8 experts, 8GB memory) - LOWER MEMORY
src/smallmind/models/models/checkpoints/qwen1.5-moe

# MixTAO MoE (16 experts, 16GB memory) - BALANCED
src/smallmind/models/models/checkpoints/mix-tao-moe
```

### **Continuous Training Commands**

```bash
# Start infinite training with DeepSeek V2
python src/smallmind/scripts/start_continuous_training.py \
    --model-path src/smallmind/models/models/checkpoints/deepseek-v2 \
    --output-dir ./continuous_training_deepseek \
    --epochs 1000 \
    --batch-size 2 \
    --learning-rate 5e-5

# Start with Qwen 1.5 MoE (lower memory usage)
python src/smallmind/scripts/start_continuous_training.py \
    --model-path src/smallmind/models/models/checkpoints/qwen1.5-moe \
    --output-dir ./continuous_training_qwen \
    --epochs 1000 \
    --batch-size 4 \
    --learning-rate 5e-5

# Get help for continuous training script
python src/smallmind/scripts/start_continuous_training.py --help
```

### **Multi-Model Training Commands**

```bash
# Train ALL 3 models simultaneously! 🚀
python src/smallmind/scripts/start_multi_model_training.py \
    --output-dir ./multi_model_training

# Check available models before starting
python src/smallmind/scripts/start_multi_model_training.py \
    --check-models \
    --output-dir ./multi_model_training

# Get help for multi-model training
python src/smallmind/scripts/start_multi_model_training.py --help
```

### **☁️ Cloud-Based Training Commands**

```bash
# Train on AWS (GPU instances)
python src/smallmind/scripts/start_cloud_training.py \
    --platform aws \
    --region us-east-1 \
    --ssh-key your-key-name \
    --security-group sg-12345678 \
    --check-costs

# Train on Google Cloud Platform
python src/smallmind/scripts/start_cloud_training.py \
    --platform gcp \
    --region us-central1 \
    --project-id your-project-id \
    --check-costs

# Train on Azure (coming soon)
python src/smallmind/scripts/start_cloud_training.py \
    --platform azure \
    --region eastus

# Get help for cloud training
python src/smallmind/scripts/start_cloud_training.py --help
```

### **Dataset Integration Commands**

```bash
# List all available training mixtures
python src/smallmind/cli/dataset_cli.py list-mixtures

# Explore specific training mixture
python src/smallmind/cli/dataset_cli.py explore-mixture --mixture balanced --samples 3
python src/smallmind/cli/dataset_cli.py explore-mixture --mixture code_focused --samples 2
python src/smallmind/cli/dataset_cli.py explore-mixture --mixture reasoning_focused --samples 2

# Explore specific datasets
python src/smallmind/cli/dataset_cli.py explore-dataset --dataset fineweb --samples 2
python src/smallmind/cli/dataset_cli.py explore-dataset --dataset stack_v2 --samples 2
python src/smallmind/cli/dataset_cli.py explore-dataset --dataset opencode_reasoning --samples 2

# Create custom training mixture
python src/smallmind/cli/dataset_cli.py create-mixture \
    --name my_custom_mix \
    --datasets fineweb,stack_v2,opencode_reasoning \
    --weights 0.5,0.3,0.2

# Validate training configuration
python src/smallmind/cli/dataset_cli.py validate-training \
    --model-path src/smallmind/models/models/checkpoints/deepseek-v2 \
    --mixture balanced

# Get help for any dataset command
python src/smallmind/cli/dataset_cli.py --help
```

### **Demo & Interactive Commands**

```bash
# Run continuous training demo
python src/smallmind/demos/continuous_training_demo.py

# Run dataset integration demo
python src/smallmind/demos/dataset_integration_demo.py

# Run advanced integration demo
python src/smallmind/demos/advanced_integration_demo.py

# Run brain development demo
python src/smallmind/demos/brain_development_demo.py
```

### **Core SmallMind Commands**

```bash
# Main SmallMind CLI
smallmind --help

# Advanced optimization
smallmind-optimize --help
smallmind-optimize --model "meta-llama/Meta-Llama-3-8B-Instruct"

# Brain simulation
smallmind-simulate --help
smallmind-simulate --steps 2000 --physics pybullet

# Neural network optimization
smallmind-neural --help

# Unified Cursor + Small-Mind interface
cursor-smallmind --help
cursor-smallmind smallmind --help
cursor-smallmind simulate --steps 2000
cursor-smallmind optimize --model "meta-llama/Meta-Llama-3-8B-Instruct"
```

### **Python API Examples**

```python
# Continuous Training
from smallmind.models.continuous_trainer import train_forever, ContinuousTrainer

# Simple infinite training
train_forever(
    model_path="src/smallmind/models/models/checkpoints/deepseek-v2",
    output_dir="./my_continuous_training"
)

# Advanced control
trainer = ContinuousTrainer(
    "src/smallmind/models/models/checkpoints/deepseek-v2", 
    "./output"
)
trainer.train_forever()

# Multi-Model Training (ALL 3 models simultaneously!)
from smallmind.models.multi_model_trainer import start_multi_model_training

# Start training all models at once
trainer = start_multi_model_training("./multi_model_output")

# Or use the trainer class for more control
from smallmind.models.multi_model_trainer import MultiModelTrainer

multi_trainer = MultiModelTrainer("./output")
multi_trainer.start_all_training()  # Trains all models simultaneously

# ☁️ Cloud-Based Training (Maximum Performance!)
from smallmind.models.cloud_integration import create_aws_trainer, create_gcp_trainer

# AWS Cloud Training
aws_trainer = create_aws_trainer(
    region="us-east-1",
    ssh_key="your-key-name",
    security_group="sg-12345678"
)

# Launch instances for all models
instances = aws_trainer.launch_training_instances([
    "DeepSeek-V2",
    "Qwen1.5-MoE", 
    "MixTAO-MoE"
])

# Monitor training progress
while True:
    status = aws_trainer.monitor_training()
    print(f"Running: {status['running']}, Cost/Hour: ${status['cost_per_hour']:.3f}")
    time.sleep(30)

# GCP Cloud Training
gcp_trainer = create_gcp_trainer(
    region="us-central1",
    project_id="your-project-id"
)

# Launch and monitor
gcp_trainer.launch_training_instances(["DeepSeek-V2", "Qwen1.5-MoE", "MixTAO-MoE"])
```

# Dataset Integration
from smallmind.models import get_dataset_integrator, get_trainer

# Explore datasets
integrator = get_dataset_integrator()
mixture = integrator.create_training_mixture("balanced")

# Training pipeline
trainer = get_trainer()
config = TrainingConfig(
    model_name_or_path="src/smallmind/models/models/checkpoints/deepseek-v2",
    output_dir="./training_output",
    mixture_name="code_focused",
    max_steps=1000
)
results = trainer.train(config)

# Core SmallMind functionality
from smallmind.ml_optimization.advanced_optimizer import SmallMindAdvancedOptimizer
from smallmind.simulation.simulation_runner import BrainDevelopmentSimulation

optimizer = SmallMindAdvancedOptimizer()
results = optimizer.optimize_model("meta-llama/Meta-Llama-3-8B-Instruct")

simulation = BrainDevelopmentSimulation()
results = simulation.run_simulation()
```

### **Verification Commands**

```bash
# Check if everything is working
python src/smallmind/cli/dataset_cli.py list-mixtures

# Test dataset loading
python src/smallmind/cli/dataset_cli.py explore-mixture --mixture balanced --samples 1

# Verify model availability
ls -la src/smallmind/models/models/checkpoints/

# Check dependencies
python -c "import datasets, transformers, torch, accelerate; print('✅ All dependencies available')"
```

### **Troubleshooting Commands**

```bash
# If you get import errors
pip install datasets transformers torch accelerate

# If you get model path errors
find . -name "*.bin" -type f | head -10

# If you get memory errors, use smaller model
python src/smallmind/scripts/start_continuous_training.py \
    --model-path src/smallmind/models/models/checkpoints/qwen1.5-moe \
    --batch-size 1 \
    --epochs 500
```

## 🏗️ Build & Packaging

### **Building the Complete Package**

```bash
# Build Small-Mind + Cursor AI unified package
python build_and_package.py
```

**What this creates:**
- 📦 **Unified package archive** with all components
- 🔧 **Self-contained installer** for easy distribution
- 🎨 **Icon integration** for professional appearance
- 📋 **Build summary** with distribution instructions

### **Creating Distributable Packages**

```bash
# Step 1: Fix any installation issues
python quick_fix.py

# Step 2: Test the system
python src/smallmind/demos/advanced_integration_demo.py

# Step 3: Build the package
python build_and_package.py

# Step 4: Check the dist/ directory for generated packages
ls dist/
```

**Generated Files:**
- `cursor_smallmind_unified_YYYYMMDD_HHMMSS.zip` - Complete package
- `cursor_smallmind_installer_YYYYMMDD_HHMMSS.zip` - Self-contained installer
- `build_summary.json` - Build details and instructions

### **Distribution Instructions**

1. **Share the installer archive** with users
2. **Users run**: `python install_self_contained.py`
3. **Everything installs automatically** - no manual setup required
4. **Professional icon** appears in desktop shortcuts

### **Troubleshooting Build Issues**

```bash
# If build fails due to dependencies
python quick_fix.py

# If specific modules fail to import
python -c "import smallmind; print('✅ Import successful')"

# Clean and rebuild
rm -rf build/ dist/ *.egg-info/
python setup.py clean --all
python build_and_package.py
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/smallmind/smallmind.git
cd smallmind

# Install development dependencies
make install-dev

# Set up pre-commit hooks
make setup-dev

# Run tests
make test

# Format code
make format

# Run linting
make lint
```

### Available Make Commands

```bash
make help              # Show all available commands
make install           # Install in development mode
make install-dev       # Install with development dependencies
make install-docs      # Install with documentation dependencies
make build             # Build the package
make clean             # Clean build artifacts
make test              # Run tests
make test-cov          # Run tests with coverage
make lint              # Run linting checks
make format            # Format code
make docs              # Build documentation
make serve             # Serve documentation locally
```

### Running Specific Demos

```bash
# Brain development simulation
make brain-demo

# MuJoCo physics simulation
make sim-demo

# FlyWire integration
make flywire-demo

# Quick start
make quick-start
```

## 🧪 Testing

```bash
# Run all tests
pytest src/tests/

# Run specific test categories
pytest src/tests/ -m neuroscience
pytest src/tests/ -m simulation

# Run with coverage
pytest src/tests/ --cov=src --cov-report=html

# Run integration tests
pytest src/tests/ -m integration
```

## 📦 Building and Packaging

```bash
# Build the package
python -m build

# Install from built package
pip install dist/smallmind-*.whl

# Build Docker image
docker build -t smallmind .

# Run with Docker Compose
docker-compose up -d
```

## 🌐 Web Interface

SmallMind includes a web-based dashboard for visualization and control:

```bash
# Start the web server
python src/visual_server.py

# Access at http://localhost:8000
```

## 🔌 Extending SmallMind

### Creating Custom Simulations

```python
from smallmind.simulation import BaseSimulation

class CustomBrainSimulation(BaseSimulation):
    def __init__(self, config):
        super().__init__(config)
    
    def step(self):
        # Custom simulation logic
        pass
    
    def visualize(self):
        # Custom visualization
        pass
```

### Adding New Data Sources

```python
from smallmind.neurodata import BaseDataSource

class CustomDataSource(BaseDataSource):
    def __init__(self, config):
        super().__init__(config)
    
    def fetch_data(self, query):
        # Custom data fetching logic
        pass
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`make test`)
6. Format your code (`make format`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CompuCell3D**: Cellular Potts modeling framework
- **MuJoCo**: Physics simulation engine
- **NEST**: Neural simulation technology
- **FlyWire**: Connectomics data platform
- **NeuroData**: Open neuroscience data
- **BabyAGI**: Autonomous AI agent framework

## 🚨 Troubleshooting

### **Common Installation Issues**

#### **Dependency Installation Failures**
```bash
# If pip install fails with complex packages
python quick_fix.py

# Install only working dependencies
pip install -r requirements_working.txt

# Try individual packages
pip install numpy scipy pandas torch transformers
```

#### **Import Errors**
```bash
# Test basic imports
python -c "import numpy; print('✅ NumPy OK')"
python -c "import torch; print('✅ PyTorch OK')"
python -c "import smallmind; print('✅ Small-Mind OK')"

# Fix import paths
python quick_fix.py
```

#### **Build Failures**
```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info/

# Reinstall in development mode
pip install -e .

# Try building again
python build_and_package.py
```

### **Runtime Issues**

#### **Simulation Failures**
```bash
# Check if physics engines are available
python -c "import pybullet; print('✅ PyBullet OK')"

# Run with mock physics (if PyBullet unavailable)
python src/smallmind/demos/advanced_integration_demo.py
```

#### **Memory Issues**
```bash
# Reduce batch sizes in configuration
# Use smaller models for testing
# Enable gradient checkpointing
```

### **Getting Help**

1. **Check the logs** for specific error messages
2. **Run quick_fix.py** to resolve common issues
3. **Test with demo scripts** to isolate problems
4. **Check system requirements** (Python version, disk space)
5. **Review this troubleshooting section**

## 📞 Support

- **Documentation**: [https://smallmind.readthedocs.io/](https://smallmind.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/smallmind/smallmind/issues)
- **Quick Fix**: Run `python quick_fix.py` for common issues
- **Discussions**: [GitHub Discussions](https://github.com/smallmind/smallmind/discussions)
- **Email**: team@smallmind.ai

## 🔬 Research Applications

SmallMind is designed for:

- **Developmental Neuroscience**: Fetal brain development modeling
- **Computational Psychiatry**: Neural circuit dysfunction analysis
- **Brain-Computer Interfaces**: Neural signal processing and decoding
- **Drug Discovery**: Neural target identification and validation
- **Education**: Interactive neuroscience learning platforms

## 🚀 Roadmap

- [ ] Enhanced GPU acceleration support
- [ ] Real-time collaboration features
- [ ] Mobile app development
- [ ] Cloud deployment automation
- [ ] Advanced visualization tools
- [ ] Machine learning model integration
- [ ] Multi-species brain modeling
- [ ] Clinical validation tools

---

**SmallMind**: Advancing computational neuroscience through integrated simulation and AI.
