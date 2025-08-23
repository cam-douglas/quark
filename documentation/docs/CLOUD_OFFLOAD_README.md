# Cloud Offload System for Conscious Agent

## Overview
The cloud offload system automatically detects when your conscious agent is experiencing heavy cognitive load and offloads computationally intensive tasks to AWS via SkyPilot. This keeps your local agent responsive while leveraging cloud computing power for heavy neural simulations, memory consolidation, and attention modeling.

## Components

### 1. SkyOffloader (`cloud_offload.py`)
- **Purpose**: Manages cloud task submission and result retrieval
- **Features**: 
  - Automatic S3 bucket creation and management
  - SkyPilot job orchestration
  - Result polling and integration
  - Error handling and cleanup

### 2. Offload Worker (`offload_worker.py`)
- **Purpose**: Runs on AWS to process cognitive tasks
- **Features**:
  - Neural simulation processing
  - Memory consolidation modeling
  - Attention focus dynamics
  - Decision making confidence analysis
  - Training session simulation

### 3. Simple Integration (`simple_cloud_integration.py`)
- **Purpose**: Easy integration with existing conscious agents
- **Features**:
  - Automatic load detection
  - Cooldown management
  - Result integration
  - Manual offload triggers

## Quick Start

### 1. Test the System
```bash
cd cloud_computing
python test_cloud_offload.py
```

### 2. Add to Your Agent
```python
from cloud_computing.simple_cloud_integration import add_cloud_integration_to_agent

# Add cloud integration to your agent
add_cloud_integration_to_agent(your_agent_instance)

# Cloud offload will now trigger automatically when:
# - consciousness_level > 0.7
# - cognitive_load > 0.8
```

### 3. Manual Offload
```python
# Manually trigger a cloud offload
result = your_agent_instance.manual_cloud_offload(
    "neural_simulation",
    {"duration": 3000, "num_neurons": 100, "scale": 0.8}
)
```

## How It Works

### Automatic Detection
The system monitors your agent's consciousness level and cognitive load:

```python
# In your agent's main loop
consciousness_level = agent.unified_state['consciousness_level']
cognitive_load = agent.unified_state.get('cognitive_load', 0.0)

# Check if offload is needed
if agent.check_cloud_offload(consciousness_level, cognitive_load):
    print("🧠 Heavy cognitive load detected, offloading to cloud...")
```

### Task Types

#### 1. Neural Simulation
- **Trigger**: High consciousness level (>0.8)
- **Parameters**: duration, num_neurons, scale
- **Output**: Activity level, spike patterns, neural metrics

#### 2. Memory Consolidation
- **Trigger**: High cognitive load (>0.8)
- **Parameters**: duration, scale
- **Output**: Consolidation level, memory patterns

#### 3. Attention Modeling
- **Trigger**: Moderate load (0.7-0.8)
- **Parameters**: duration, scale
- **Output**: Focus level, attention dynamics

### Cloud Processing Flow
1. **Detection**: Agent detects high cognitive load
2. **Payload Creation**: Task parameters packaged into JSON
3. **S3 Upload**: Payload uploaded to S3 bucket
4. **SkyPilot Launch**: Small EC2 instance launched with worker
5. **Processing**: Worker downloads payload, processes task
6. **Result Upload**: Results uploaded back to S3
7. **Integration**: Results integrated into agent state
8. **Cleanup**: Temporary files and instance cleaned up

## Configuration

### AWS Setup
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check SkyPilot
sky check

# Test S3 access
aws s3 ls
```

### Customization
```python
# Customize offload thresholds
cloud_integration.offload_cooldown = 60  # seconds between offloads

# Customize task parameters
parameters = {
    'duration': 5000,      # milliseconds
    'num_neurons': 300,    # for neural simulation
    'scale': 1.0,          # intensity multiplier
    'epochs': 20           # for training sessions
}
```

## Monitoring

### Status Check
```python
status = agent.get_cloud_status()
print(f"Cloud offloader available: {status['cloud_offloader_available']}")
print(f"Last offload: {status['time_since_last_offload']:.1f}s ago")
```

### Metrics
```python
metrics = agent.unified_state['cloud_offload_metrics']
print(f"Tasks offloaded: {metrics['tasks_offloaded']}")
print(f"Success rate: {metrics['offload_success_rate']:.2f}")
```

## Troubleshooting

### Common Issues

#### 1. AWS Credentials
```bash
# Check credentials
aws sts get-caller-identity

# If using named profile
export AWS_PROFILE=your-profile
```

#### 2. SkyPilot Issues
```bash
# Check SkyPilot setup
sky check

# Reinstall if needed
pip install -U skypilot
```

#### 3. S3 Permissions
```bash
# Test S3 access
aws s3 ls s3://quark-offload-us-west-2/

# Create bucket manually if needed
aws s3 mb s3://quark-offload-us-west-2 --region us-west-2
```

#### 4. VCPU Limits
If you see "VcpuLimitExceeded":
- Request limit increase in AWS Console
- Use smaller instance types
- Switch to different region

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
from cloud_offload import SkyOffloader
offloader = SkyOffloader()
status = offloader.get_status()
print(status)
```

## Cost Optimization

### Instance Types
- **CPU Tasks**: Use `cpus: 2` for cost efficiency
- **GPU Tasks**: Use `accelerators: T4:1` for neural simulations
- **Memory**: Use `memory: 8GB` for most tasks

### Cooldown Management
```python
# Increase cooldown to reduce costs
cloud_integration.offload_cooldown = 120  # 2 minutes

# Disable automatic offload
cloud_integration.cloud_offloader = None
```

### Spot Instances
SkyPilot automatically uses spot instances when available for cost savings.

## Integration Examples

### Basic Integration
```python
from cloud_computing.simple_cloud_integration import add_cloud_integration_to_agent

class MyConsciousAgent:
    def __init__(self):
        self.unified_state = {
            'consciousness_level': 0.0,
            'cognitive_load': 0.0,
            'cloud_offload_metrics': {}
        }
        
        # Add cloud integration
        add_cloud_integration_to_agent(self)
    
    def run(self):
        while True:
            # Your agent logic here
            self.unified_state['consciousness_level'] = 0.8  # High load
            
            # Cloud offload will trigger automatically
            time.sleep(1)
```

### Advanced Integration
```python
class AdvancedConsciousAgent:
    def __init__(self):
        # ... existing initialization ...
        add_cloud_integration_to_agent(self)
    
    def process_heavy_task(self):
        # Manual offload for specific heavy tasks
        result = self.manual_cloud_offload(
            "training_session",
            {"epochs": 50, "batch_size": 64, "scale": 1.0}
        )
        
        if result:
            print(f"Training completed: loss={result['final_training_loss']:.4f}")
```

## Performance Tips

1. **Batch Tasks**: Group multiple small tasks into larger offloads
2. **Cache Results**: Store and reuse similar computation results
3. **Monitor Costs**: Use AWS Cost Explorer to track spending
4. **Optimize Parameters**: Tune task parameters for your specific needs
5. **Use Cooldowns**: Prevent excessive offloading with appropriate cooldowns

## Security

- All communication uses AWS IAM roles and policies
- S3 buckets are private by default
- Temporary credentials are used for SkyPilot
- No sensitive data is stored in cloud (only computation parameters)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run `python test_cloud_offload.py` to verify setup
3. Check AWS CloudWatch logs for detailed error information
4. Verify SkyPilot configuration with `sky check`
