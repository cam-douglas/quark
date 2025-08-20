#!/usr/bin/env python3
"""Quick AWS deployment for SmallMind using existing keys"""

import boto3
import time
import argparse

def deploy_smallmind(region='us-east-1', instance_type='g4dn.xlarge', key_name=None):
    """Deploy SmallMind on AWS GPU instance"""
    
    print(f"🚀 Deploying SmallMind on {instance_type} in {region}")
    
    # Setup clients
    ec2 = boto3.client('ec2', region_name=region)
    
    # Deep Learning AMI
    ami_id = 'ami-0c02fb55956c7d316'  # Deep Learning AMI
    
    # User data script
    user_data = """#!/bin/bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu
sudo systemctl start docker
cd HOME/ubuntu
git clone https://github.com/cam-douglas/small-mind.git
cd small-mind
sudo docker build -t smallmind:latest .
sudo docker run -d --name smallmind --gpus all -p 8888:8888 -p 5000:5000 -v $(pwd):/workspace smallmind:latest
echo "SmallMind deployed at $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8888"
"""
    
    # Launch instance
    print("📦 Launching GPU instance...")
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroups=['default'],
        UserData=user_data,
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': 'smallmind-gpu'}]
        }]
    )
    
    instance_id = response['Instances'][0]['InstanceId']
    print(f"✅ Instance launched: {instance_id}")
    
    # Wait for running
    print("⏳ Waiting for instance to start...")
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    
    # Get public IP
    instance = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = instance['Reservations'][0]['Instances'][0]['PublicIpAddress']
    
    print(f"🎯 Instance running at: {public_ip}")
    print(f"🔗 Jupyter: http://{public_ip}:8888")
    print(f"🔌 API: http://{public_ip}:5000")
    if key_name:
        print(f"💻 SSH: ssh -i ~/.ssh/{key_name}.pem ubuntu@{public_ip}")
    
    return instance_id, public_ip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--instance-type', default='g4dn.xlarge')
    parser.add_argument('--key-name')
    args = parser.parse_args()
    
    deploy_smallmind(args.region, args.instance_type, args.key_name)
