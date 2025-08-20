# 🚀 AWS Cloud Computing Performance Optimization Guide

This directory contains comprehensive tools and configurations to optimize AWS cloud computing performance for your brain development simulation project.

## 📁 Files Overview

- **`aws_performance_optimization.py`** - Main Python script for AWS optimization
- **`aws_optimization_config.yaml`** - Configuration file with all optimization settings
- **`requirements_aws_optimization.txt`** - Python dependencies
- **`run_aws_optimization.sh`** - Easy launcher script
- **`README_AWS_OPTIMIZATION.md`** - This documentation file

## 🚀 Quick Start

### Option 1: Use the Launcher Script (Recommended)
```bash
cd scripts
./run_aws_optimization.sh
```

### Option 2: Manual Setup
```bash
cd scripts
python3 -m venv env
source env/bin/activate
pip install -r requirements_aws_optimization.txt
python3 aws_performance_optimization.py
```

## ⚙️ Prerequisites

1. **Python 3.8+** installed
2. **AWS CLI** configured with appropriate credentials
3. **IAM permissions** for EC2, CloudWatch, Auto Scaling, and S3

## 🔧 Configuration

### 1. AWS Credentials Setup
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
# Enter your output format (json)
```

### 2. IAM Permissions Required
Your AWS user/role needs these permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:*",
                "cloudwatch:*",
                "autoscaling:*",
                "s3:*",
                "iam:GetRole",
                "iam:PassRole"
            ],
            "Resource": "*"
        }
    ]
}
```

## 🎯 What Gets Optimized

### 1. **Instance Performance**
- Kernel parameter tuning (swappiness, memory management)
- GPU optimization for ML workloads
- Memory management (huge pages, transparent huge pages)

### 2. **Storage Performance**
- EBS volume optimization (GP3, IO2 Block Express)
- S3 transfer acceleration and intelligent tiering
- Local NVMe RAID configuration

### 3. **Network Performance**
- TCP congestion control (BBR algorithm)
- Network buffer sizing
- Placement group configuration for low-latency communication

### 4. **Auto Scaling**
- ML-optimized scaling policies
- GPU utilization-based scaling
- Cost-aware scaling decisions

### 5. **Cost Optimization**
- Spot instance management
- Reserved instance recommendations
- Savings plans optimization

### 6. **Monitoring & Alerting**
- Custom CloudWatch metrics for ML workloads
- Performance-based alarms
- Cost monitoring and alerts

## 📊 Usage Examples

### Basic Instance Optimization
```python
from aws_performance_optimization import AWSPerformanceOptimizer

# Initialize optimizer
optimizer = AWSPerformanceOptimizer(region='us-east-1')

# Optimize a single instance
result = optimizer.optimize_instance_performance('i-1234567890abcdef0')
print(result)
```

### Full Optimization Pipeline
```python
# Optimize multiple instances
instance_ids = ['i-1234567890abcdef0', 'i-0987654321fedcba0']
results = optimizer.run_full_optimization(instance_ids)
print(results['summary'])
```

### Custom Configuration
```python
# Load custom configuration
import yaml
with open('aws_optimization_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Apply custom settings
optimizer = AWSPerformanceOptimizer(region=config['aws_region'])
```

## 🔍 Monitoring Results

After running optimization, check:

1. **`aws_optimization_results.json`** - Detailed optimization results
2. **CloudWatch Dashboard** - Real-time performance metrics
3. **AWS Cost Explorer** - Cost optimization results

## 🚨 Important Notes

### Security Considerations
- The script requires elevated IAM permissions
- Review all changes before applying to production
- Test in a development environment first

### Cost Impact
- Some optimizations may increase costs initially
- Monitor usage and adjust as needed
- Use cost alerts to prevent unexpected charges

### Performance Impact
- Some optimizations require instance restart
- Test performance improvements in staging
- Monitor for any negative impacts

## 🛠️ Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Check IAM permissions
   aws sts get-caller-identity
   aws iam get-user
   ```

2. **Instance Not Found**
   ```bash
   # Verify instance exists and is accessible
   aws ec2 describe-instances --instance-ids i-1234567890abcdef0
   ```

3. **GPU Optimization Failed**
   ```bash
   # Check if instance has GPU
   aws ec2 describe-instance-types --instance-types p4d.24xlarge
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

optimizer = AWSPerformanceOptimizer(region='us-east-1')
# Run with detailed logging
```

## 📈 Performance Benchmarks

### Expected Improvements

| Optimization | Expected Gain | Notes |
|--------------|---------------|-------|
| Kernel Tuning | 5-15% | CPU and memory performance |
| GPU Optimization | 10-25% | ML training and inference |
| Network Tuning | 20-40% | Inter-instance communication |
| Storage Optimization | 30-60% | I/O intensive workloads |
| Auto Scaling | 40-80% | Resource utilization |

### Benchmarking Commands
```bash
# CPU Performance
sysbench cpu --cpu-max-prime=20000 run

# Memory Performance
sysbench memory --memory-block-size=1K --memory-total-size=100G run

# Disk I/O
sysbench fileio --file-test-mode=seqwr run

# Network Performance
iperf3 -c <target-ip> -t 30
```

## 🔄 Continuous Optimization

### Automated Optimization
```bash
# Set up cron job for regular optimization
0 2 * * * /path/to/scripts/run_aws_optimization.sh

# Or use AWS Systems Manager for scheduled optimization
aws ssm create-maintenance-window \
    --name "WeeklyOptimization" \
    --schedule "cron(0 2 ? * SUN *)" \
    --duration 2 \
    --cutoff 1
```

### Performance Tracking
- Monitor CloudWatch metrics over time
- Track cost vs. performance ratios
- Document optimization results
- Share learnings with team

## 📚 Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)
- [EBS Volume Types](https://aws.amazon.com/ebs/volume-types/)
- [Auto Scaling Best Practices](https://docs.aws.amazon.com/autoscaling/ec2/userguide/best-practices.html)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review AWS CloudTrail logs
3. Check CloudWatch metrics and alarms
4. Contact your AWS support team

---

**Happy Optimizing! 🚀**
