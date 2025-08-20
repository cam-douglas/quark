#!/usr/bin/env python3
"""
AWS Deployment Setup Script for SmallMind
Uses existing AWS keys to configure and deploy the brain development simulation
"""

import os
import json
import subprocess
import boto3
from pathlib import Path
import argparse
import time

class SmallMindAWSDeployer:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.ec2_client = None
        self.ec2_resource = None
        self.vpc_id = None
        self.subnet_id = None
        self.security_group_id = None
        self.instance_id = None
        
    def setup_aws_session(self):
        """Setup AWS session using existing credentials"""
        try:
            # Check if credentials are configured
            session = boto3.Session(region_name=self.region)
            sts = session.client('sts')
            sts.get_caller_identity()
            print("✅ AWS credentials verified successfully")
            
            self.ec2_client = session.client('ec2')
            self.ec2_resource = session.resource('ec2')
            return True
            
        except Exception as e:
            print(f"❌ AWS credentials error: {e}")
            print("Please ensure your AWS credentials are properly configured")
            return False
    
    def create_vpc_and_networking(self):
        """Create VPC, subnet, and security group"""
        try:
            print("🏗️  Creating VPC and networking infrastructure...")
            
            # Create VPC
            vpc_response = self.ec2_client.create_vpc(
                CidrBlock='10.0.0.0/16',
                TagSpecifications=[{
                    'ResourceType': 'vpc',
                    'Tags': [{'Key': 'Name', 'Value': 'smallmind-vpc'}]
                }]
            )
            self.vpc_id = vpc_response['Vpc']['VpcId']
            print(f"✅ Created VPC: {self.vpc_id}")
            
            # Wait for VPC to be available
            self.ec2_client.get_waiter('vpc_available').wait(VpcIds=[self.vpc_id])
            
            # Create subnet
            subnet_response = self.ec2_client.create_subnet(
                VpcId=self.vpc_id,
                CidrBlock='10.0.1.0/24',
                AvailabilityZone=f'{self.region}a',
                TagSpecifications=[{
                    'ResourceType': 'subnet',
                    'Tags': [{'Key': 'Name', 'Value': 'smallmind-subnet'}]
                }]
            )
            self.subnet_id = subnet_response['Subnet']['SubnetId']
            print(f"✅ Created subnet: {self.subnet_id}")
            
            # Create security group
            sg_response = self.ec2_client.create_security_group(
                GroupName='smallmind-sg',
                Description='SmallMind brain development simulation security group',
                VpcId=self.vpc_id,
                TagSpecifications=[{
                    'ResourceType': 'security-group',
                    'Tags': [{'Key': 'Name', 'Value': 'smallmind-sg'}]
                }]
            )
            self.security_group_id = sg_response['GroupId']
            print(f"✅ Created security group: {self.security_group_id}")
            
            # Add security group rules
            self.ec2_client.authorize_security_group_ingress(
                GroupId=self.security_group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 8888,
                        'ToPort': 8888,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 5000,
                        'ToPort': 5000,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            print("✅ Security group rules configured")
            
            return True
            
        except Exception as e:
            print(f"❌ Networking setup failed: {e}")
            return False
    
    def create_iam_role(self):
        """Create IAM role for EC2 instances"""
        try:
            print("🔐 Creating IAM role...")
            
            iam = boto3.client('iam')
            
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            role_response = iam.create_role(
                RoleName='SmallMindEC2Role',
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Role for SmallMind EC2 instances'
            )
            print("✅ Created IAM role: SmallMindEC2Role")
            
            # Attach policies
            iam.attach_role_policy(
                RoleName='SmallMindEC2Role',
                PolicyArn='arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'
            )
            
            iam.attach_role_policy(
                RoleName='SmallMindEC2Role',
                PolicyArn='arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy'
            )
            
            # Create instance profile
            iam.create_instance_profile(
                InstanceProfileName='SmallMindEC2Profile'
            )
            
            iam.add_role_to_instance_profile(
                InstanceProfileName='SmallMindEC2Profile',
                RoleName='SmallMindEC2Role'
            )
            
            print("✅ IAM role and instance profile configured")
            return True
            
        except Exception as e:
            print(f"❌ IAM setup failed: {e}")
            return False
    
    def launch_gpu_instance(self, instance_type='g4dn.xlarge', key_name=None):
        """Launch GPU instance with SmallMind"""
        try:
            print(f"🚀 Launching {instance_type} GPU instance...")
            
            # Deep Learning AMI (adjust for your region)
            ami_id = self._get_deep_learning_ami()
            
            # Launch configuration
            launch_config = {
                'ImageId': ami_id,
                'InstanceType': instance_type,
                'SecurityGroupIds': [self.security_group_id],
                'SubnetId': self.subnet_id,
                'IamInstanceProfile': {'Name': 'SmallMindEC2Profile'},
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [{'Key': 'Name', 'Value': 'smallmind-gpu'}]
                }],
                'UserData': self._get_user_data_script(),
                'MinCount': 1,
                'MaxCount': 1
            }
            
            if key_name:
                launch_config['KeyName'] = key_name
            
            # Launch instance
            response = self.ec2_client.run_instances(**launch_config)
            self.instance_id = response['Instances'][0]['InstanceId']
            
            print(f"✅ Instance launched: {self.instance_id}")
            print("⏳ Waiting for instance to be running...")
            
            # Wait for running state
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[self.instance_id])
            
            # Get public IP
            instance = self.ec2_resource.Instance(self.instance_id)
            instance.reload()
            public_ip = instance.public_ip_address
            
            print(f"🎯 Instance is running at: {public_ip}")
            print(f"🔗 SSH: ssh -i ~/.ssh/{key_name}.pem ubuntu@{public_ip}")
            print(f"🌐 Jupyter: http://{public_ip}:8888")
            print(f"🔌 API: http://{public_ip}:5000")
            
            return public_ip
            
        except Exception as e:
            print(f"❌ Instance launch failed: {e}")
            return None
    
    def _get_deep_learning_ami(self):
        """Get Deep Learning AMI for the region"""
        try:
            # Search for Deep Learning AMI
            response = self.ec2_client.describe_images(
                Owners=['amazon'],
                Filters=[
                    {
                        'Name': 'name',
                        'Values': ['*Deep Learning AMI GPU PyTorch*']
                    },
                    {
                        'Name': 'state',
                        'Values': ['available']
                    }
                ]
            )
            
            if response['Images']:
                # Get the most recent one
                latest_ami = sorted(response['Images'], 
                                  key=lambda x: x['CreationDate'], 
                                  reverse=True)[0]
                return latest_ami['ImageId']
            else:
                # Fallback to a known AMI
                fallback_amis = {
                    'us-east-1': 'ami-0c02fb55956c7d316',
                    'us-west-2': 'ami-0c02fb55956c7d316',
                    'eu-west-1': 'ami-0c02fb55956c7d316'
                }
                return fallback_amis.get(self.region, 'ami-0c02fb55956c7d316')
                
        except Exception as e:
            print(f"⚠️  Could not find Deep Learning AMI: {e}")
            return 'ami-0c02fb55956c7d316'  # Fallback
    
    def _get_user_data_script(self):
        """Get user data script for instance initialization"""
        script = """#!/bin/bash
# SmallMind GPU Instance Setup

echo "🚀 Starting SmallMind GPU instance setup..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone SmallMind
cd HOME/ubuntu
git clone https://github.com/cam-douglas/small-mind.git
cd small-mind

# Build Docker image
sudo docker build -t smallmind:latest .

# Start container
sudo docker run -d \
  --name smallmind \
  --gpus all \
  -p 8888:8888 \
  -p 5000:5000 \
  -v $(pwd):/workspace \
  smallmind:latest

echo "✅ SmallMind deployed successfully!"
echo "🌐 Jupyter available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8888"
echo "🔌 API available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
"""
        return script
    
    def cleanup_resources(self):
        """Clean up created resources"""
        try:
            print("🧹 Cleaning up resources...")
            
            if self.instance_id:
                self.ec2_client.terminate_instances(InstanceIds=[self.instance_id])
                print(f"✅ Terminated instance: {self.instance_id}")
            
            if self.security_group_id:
                self.ec2_client.delete_security_group(GroupId=self.security_group_id)
                print(f"✅ Deleted security group: {self.security_group_id}")
            
            if self.subnet_id:
                self.ec2_client.delete_subnet(SubnetId=self.subnet_id)
                print(f"✅ Deleted subnet: {self.subnet_id}")
            
            if self.vpc_id:
                self.ec2_client.delete_vpc(VpcId=self.vpc_id)
                print(f"✅ Deleted VPC: {self.vpc_id}")
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

def main():
    parser = argparse.ArgumentParser(description='Deploy SmallMind on AWS')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--instance-type', default='g4dn.xlarge', help='EC2 instance type')
    parser.add_argument('--key-name', help='EC2 key pair name')
    parser.add_argument('--cleanup', action='store_true', help='Clean up resources')
    
    args = parser.parse_args()
    
    deployer = SmallMindAWSDeployer(region=args.region)
    
    if args.cleanup:
        deployer.cleanup_resources()
        return
    
    print("🧠 SmallMind AWS Deployment")
    print("=" * 40)
    
    # Setup AWS session
    if not deployer.setup_aws_session():
        return
    
    # Create infrastructure
    if not deployer.create_vpc_and_networking():
        return
    
    # Create IAM role
    if not deployer.create_iam_role():
        return
    
    # Launch instance
    public_ip = deployer.launch_gpu_instance(
        instance_type=args.instance_type,
        key_name=args.key_name
    )
    
    if public_ip:
        print("\n🎉 Deployment successful!")
        print(f"Your SmallMind instance is running at: {public_ip}")
        print("\nNext steps:")
        print("1. Wait 5-10 minutes for setup to complete")
        print("2. Access Jupyter: http://" + public_ip + ":8888")
        print("3. Access API: http://" + public_ip + ":5000")
        print("4. SSH: ssh -i ~/.ssh/" + (args.key_name or "your-key") + ".pem ubuntu@" + public_ip)
    else:
        print("❌ Deployment failed")

if __name__ == "__main__":
    main()
