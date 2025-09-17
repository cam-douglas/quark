# 🚀 Quick SmallMind AWS Deployment

## Prerequisites
- AWS CLI installed and configured (`aws configure`)
- EC2 key pair created in your target region
- Python 3.7+ with pip

## Quick Start

### 1. Check AWS Setup
```bash
cd scripts
python3 aws_config.py
```

### 2. Deploy SmallMind
```bash
# Option 1: Python script
python3 aws_deploy.py --key-name YOUR_KEY_NAME

# Option 2: Shell script (easier)
./quick_deploy.sh YOUR_KEY_NAME

# Option 3: Custom region/instance
./quick_deploy.sh YOUR_KEY_NAME us-west-2 g4dn.2xlarge
```

## What Gets Deployed

- **GPU Instance**: g4dn.xlarge (1x T4 GPU, 4 vCPU, 16 GB RAM)
- **Deep Learning AMI**: Pre-configured with CUDA, Docker, NVIDIA drivers
- **SmallMind**: Cloned from GitHub and built in Docker container
- **Ports**: 8888 (Jupyter), 5000 (API)

## Access Your Instance

After deployment (5-10 minutes):
- **Jupyter Lab**: `http://YOUR_IP:8888`
- **API**: `http://YOUR_IP:5000`
- **SSH**: `ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@YOUR_IP`

## Cost Estimate
- **g4dn.xlarge**: ~$0.53/hour (~$12.50/day)
- **g4dn.2xlarge**: ~$1.05/hour (~$25/day)

## Cleanup
```bash
# Find your instance ID from the deployment output
aws ec2 terminate-instances --instance-ids i-xxxxxxxxx
```

## Troubleshooting

### Common Issues
1. **Key pair not found**: Create key pair in target region
2. **Security group issues**: Use default security group or create custom one
3. **AMI not found**: Change region or use different AMI ID

### Check Instance Status
```bash
aws ec2 describe-instances --filters "Name=tag:Name,Values=smallmind-gpu"
```

### View Logs
```bash
# SSH into instance and check Docker logs
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@YOUR_IP
sudo docker logs smallmind
```
