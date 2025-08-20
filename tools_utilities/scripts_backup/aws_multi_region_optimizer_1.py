#!/usr/bin/env python3
"""
🚀 AWS Multi-Region Instance Optimizer
Discovers and optimizes instances across all regions
"""

import boto3
import json

def discover_instances_in_region(region):
    """Find all instances in a specific region"""
    try:
        ec2 = boto3.client('ec2', region_name=region)
        response = ec2.describe_instances()
        instances = []
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append({
                    'InstanceId': instance['InstanceId'],
                    'InstanceType': instance['InstanceType'],
                    'State': instance['State']['Name'],
                    'Region': region,
                    'PublicIp': instance.get('PublicIpAddress', 'N/A')
                })
        
        return instances
    except Exception as e:
        print(f"Error in {region}: {e}")
        return []

def main():
    print("🚀 AWS Multi-Region Instance Discovery")
    print("=" * 50)
    
    # Get all regions
    ec2_global = boto3.client('ec2')
    regions = [r['RegionName'] for r in ec2_global.describe_regions()['Regions']]
    
    all_instances = []
    
    # Check each region for instances
    for region in regions:
        print(f"🔍 Checking {region}...")
        instances = discover_instances_in_region(region)
        if instances:
            all_instances.extend(instances)
            print(f"   Found {len(instances)} instances")
        else:
            print(f"   No instances found")
    
    print(f"\n📊 Total instances found: {len(all_instances)}")
    
    if all_instances:
        print("\n🖥️ Instance Details:")
        for instance in all_instances:
            print(f"  - {instance['InstanceId']} ({instance['InstanceType']}) - {instance['State']} in {instance['Region']}")
        
        # Save results
        with open('discovered_instances.json', 'w') as f:
            json.dump(all_instances, f, indent=2)
        print(f"\n💾 Results saved to: discovered_instances.json")
        
        # Show optimization commands
        print(f"\n🚀 To optimize these instances, run:")
        for instance in all_instances:
            if instance['State'] == 'running':
                print(f"  python3 aws_performance_optimization.py")
                print(f"  # Enter: {instance['InstanceId']}")
                break
    else:
        print("\n❌ No instances found. Create some EC2 instances first!")

if __name__ == "__main__":
    main()
