#!/usr/bin/env python3
"""AWS Configuration Helper for SmallMind"""

import os
import boto3
import json

def check_aws_setup():
    """Check if AWS is properly configured"""
    try:
        session = boto3.Session()
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        
        print("✅ AWS Configuration Verified")
        print(f"   Account: {identity['Account']}")
        print(f"   User: {identity['Arn']}")
        print(f"   Region: {session.region_name}")
        
        return True
    except Exception as e:
        print(f"❌ AWS Configuration Error: {e}")
        return False

def list_regions():
    """List available AWS regions"""
    ec2 = boto3.client('ec2')
    regions = ec2.describe_regions()
    
    print("\n🌍 Available Regions:")
    for region in regions['Regions']:
        print(f"   {region['RegionName']} - {region['Description']}")

def list_key_pairs():
    """List available EC2 key pairs"""
    try:
        ec2 = boto3.client('ec2')
        keys = ec2.describe_key_pairs()
        
        print("\n🔑 Available Key Pairs:")
        for key in keys['KeyPairs']:
            print(f"   {key['KeyName']} - {key['KeyType']}")
            
    except Exception as e:
        print(f"⚠️  Could not list key pairs: {e}")

def get_instance_types():
    """Show available GPU instance types"""
    print("\n🖥️  Recommended GPU Instance Types:")
    instances = [
        ("g4dn.xlarge", "1x T4 GPU", "4 vCPU", "16 GB RAM", "$0.526/hour"),
        ("g4dn.2xlarge", "1x T4 GPU", "8 vCPU", "32 GB RAM", "$1.052/hour"),
        ("g4dn.4xlarge", "1x T4 GPU", "16 vCPU", "64 GB RAM", "$2.104/hour"),
        ("p3.2xlarge", "1x V100 GPU", "8 vCPU", "61 GB RAM", "$3.06/hour")
    ]
    
    for instance, gpu, cpu, ram, cost in instances:
        print(f"   {instance:<15} | {gpu:<12} | {cpu:<8} | {ram:<10} | {cost}")

if __name__ == "__main__":
    print("🔧 SmallMind AWS Configuration Check")
    print("=" * 40)
    
    check_aws_setup()
    list_regions()
    list_key_pairs()
    get_instance_types()
    
    print("\n📋 Next Steps:")
    print("1. Ensure your key pair exists in the target region")
    print("2. Run: python3 aws_deploy.py --key-name YOUR_KEY_NAME")
    print("3. Or use: ./quick_deploy.sh YOUR_KEY_NAME")
