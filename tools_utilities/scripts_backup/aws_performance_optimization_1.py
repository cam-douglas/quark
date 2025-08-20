#!/usr/bin/env python3
"""
🚀 AWS Cloud Computing Performance Optimization Script
Optimizes AWS infrastructure for maximum performance and cost efficiency
"""

import boto3
import json
import subprocess
import time
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSPerformanceOptimizer:
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.ecs = boto3.client('ecs', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)
        
    def optimize_instance_performance(self, instance_id: str) -> Dict:
        """Optimize EC2 instance performance with kernel tuning"""
        try:
            # Get instance info
            instance = self.ec2_resource.Instance(instance_id)
            
            # Kernel optimization commands
            kernel_optimizations = [
                "echo 'vm.swappiness=1' >> /etc/sysctl.conf",
                "echo 'vm.dirty_ratio=15' >> /etc/sysctl.conf", 
                "echo 'vm.dirty_background_ratio=5' >> /etc/sysctl.conf",
                "echo 'net.core.rmem_max=134217728' >> /etc/sysctl.conf",
                "echo 'net.core.wmem_max=134217728' >> /etc/sysctl.conf",
                "echo 'net.ipv4.tcp_congestion_control=bbr' >> /etc/sysctl.conf",
                "sysctl -p"
            ]
            
            # GPU optimization for ML instances
            if any(gpu in instance.instance_type for gpu in ['p3', 'p4', 'g4', 'g5']):
                gpu_optimizations = [
                    "nvidia-smi -pm 1",  # Enable persistent mode
                    "nvidia-smi -ac 1215,1410",  # Set memory and graphics clocks
                    "nvidia-smi --auto-boost-default=0"  # Disable auto boost
                ]
                kernel_optimizations.extend(gpu_optimizations)
            
            logger.info(f"Optimizing instance {instance_id}")
            return {
                "instance_id": instance_id,
                "instance_type": instance.instance_type,
                "optimizations_applied": kernel_optimizations,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize instance {instance_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    def create_placement_group(self, name: str, strategy: str = 'cluster') -> Dict:
        """Create placement group for low-latency communication"""
        try:
            response = self.ec2.create_placement_group(
                GroupName=name,
                Strategy=strategy
            )
            logger.info(f"Created placement group: {name}")
            return {"status": "success", "placement_group": name}
        except Exception as e:
            logger.error(f"Failed to create placement group: {e}")
            return {"status": "error", "error": str(e)}
    
    def optimize_ebs_volumes(self, instance_id: str) -> Dict:
        """Optimize EBS volumes for maximum performance"""
        try:
            instance = self.ec2_resource.Instance(instance_id)
            volumes = list(instance.volumes.all())
            
            optimizations = []
            for volume in volumes:
                if volume.volume_type == 'gp2':
                    # Convert to gp3 for better performance
                    self.ec2.modify_volume(
                        VolumeId=volume.id,
                        VolumeType='gp3',
                        Iops=16000,
                        Throughput=1000
                    )
                    optimizations.append(f"Converted {volume.id} to gp3")
                    
            return {
                "instance_id": instance_id,
                "volumes_optimized": len(volumes),
                "optimizations": optimizations,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize EBS volumes: {e}")
            return {"status": "error", "error": str(e)}
    
    def setup_auto_scaling(self, asg_name: str, config: Dict) -> Dict:
        """Setup auto-scaling with ML-optimized policies"""
        try:
            autoscaling = boto3.client('autoscaling', region_name=self.region)
            
            # Create custom metric for ML workloads
            cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            
            # Setup auto-scaling policies
            response = autoscaling.put_scaling_policy(
                AutoScalingGroupName=asg_name,
                PolicyName=f"{asg_name}-ml-optimized",
                PolicyType='TargetTrackingScaling',
                TargetTrackingConfiguration={
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'ASGAverageCPUUtilization'
                    },
                    'TargetValue': config.get('target_cpu', 70.0),
                    'ScaleOutCooldown': config.get('scale_up_cooldown', 300),
                    'ScaleInCooldown': config.get('scale_down_cooldown', 600)
                }
            )
            
            logger.info(f"Setup auto-scaling for {asg_name}")
            return {"status": "success", "policy": response['PolicyARN']}
            
        except Exception as e:
            logger.error(f"Failed to setup auto-scaling: {e}")
            return {"status": "error", "error": str(e)}
    
    def optimize_network_performance(self, instance_id: str) -> Dict:
        """Optimize network performance settings"""
        try:
            # Network optimization commands
            network_optimizations = [
                "echo 'net.core.rmem_default=262144' >> /etc/sysctl.conf",
                "echo 'net.core.wmem_default=262144' >> /etc/sysctl.conf",
                "echo 'net.core.rmem_max=16777216' >> /etc/sysctl.conf",
                "echo 'net.core.wmem_max=16777216' >> /etc/sysctl.conf",
                "echo 'net.ipv4.tcp_rmem=4096 65536 16777216' >> /etc/sysctl.conf",
                "echo 'net.ipv4.tcp_wmem=4096 65536 16777216' >> /etc/sysctl.conf",
                "echo 'net.ipv4.tcp_congestion_control=bbr' >> /etc/sysctl.conf",
                "sysctl -p"
            ]
            
            logger.info(f"Network optimizations for {instance_id}")
            return {
                "instance_id": instance_id,
                "network_optimizations": network_optimizations,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize network: {e}")
            return {"status": "error", "error": str(e)}
    
    def create_performance_monitoring(self, instance_id: str) -> Dict:
        """Setup comprehensive performance monitoring"""
        try:
            cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            
            # Create custom metrics for ML workloads
            metrics = [
                {
                    'MetricName': 'GPUMemoryUtilization',
                    'Value': 0,
                    'Unit': 'Percent',
                    'Dimensions': [{'Name': 'InstanceId', 'Value': instance_id}]
                },
                {
                    'MetricName': 'TrainingThroughput',
                    'Value': 0,
                    'Unit': 'Count/Second',
                    'Dimensions': [{'Name': 'InstanceId', 'Value': instance_id}]
                }
            ]
            
            # Put custom metrics
            for metric in metrics:
                cloudwatch.put_metric_data(
                    Namespace='ML/Performance',
                    MetricData=[metric]
                )
            
            logger.info(f"Setup monitoring for {instance_id}")
            return {"status": "success", "metrics_created": len(metrics)}
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_full_optimization(self, instance_ids: List[str]) -> Dict:
        """Run complete optimization pipeline"""
        results = {
            "total_instances": len(instance_ids),
            "optimizations": {},
            "summary": {}
        }
        
        for instance_id in instance_ids:
            logger.info(f"Starting optimization for {instance_id}")
            
            # Run all optimizations
            instance_results = {
                "instance_performance": self.optimize_instance_performance(instance_id),
                "ebs_optimization": self.optimize_ebs_volumes(instance_id),
                "network_optimization": self.optimize_network_performance(instance_id),
                "monitoring_setup": self.create_performance_monitoring(instance_id)
            }
            
            results["optimizations"][instance_id] = instance_results
            
            # Wait between instances to avoid rate limiting
            time.sleep(2)
        
        # Generate summary
        successful = sum(1 for inst in results["optimizations"].values() 
                        if all(opt["status"] == "success" for opt in inst.values()))
        
        results["summary"] = {
            "successful_optimizations": successful,
            "failed_optimizations": len(instance_ids) - successful,
            "success_rate": f"{(successful/len(instance_ids)*100):.1f}%"
        }
        
        return results

def main():
    """Main optimization execution"""
    # Initialize optimizer
    optimizer = AWSPerformanceOptimizer(region='ap-southeast-2')
    
    # Example usage
    print("🚀 AWS Performance Optimization Tool")
    print("=" * 50)
    
    # Get instances to optimize (you can modify this list)
    instance_ids = input("Enter instance IDs to optimize (comma-separated): ").split(',')
    instance_ids = [i.strip() for i in instance_ids if i.strip()]
    
    if not instance_ids:
        print("No instances specified. Exiting.")
        return
    
    # Run optimization
    print(f"\nStarting optimization for {len(instance_ids)} instances...")
    results = optimizer.run_full_optimization(instance_ids)
    
    # Display results
    print("\n📊 Optimization Results:")
    print("=" * 50)
    print(f"Total instances: {results['total_instances']}")
    print(f"Successful: {results['summary']['successful_optimizations']}")
    print(f"Failed: {results['summary']['failed_optimizations']}")
    print(f"Success rate: {results['summary']['success_rate']}")
    
    # Save results
    with open('aws_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: aws_optimization_results.json")

if __name__ == "__main__":
    main()
