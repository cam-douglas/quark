#!/usr/bin/env python3
"""
Start Cloud-Based Multi-Model Training Script

Launches SmallMind training on cloud platforms (AWS, GCP, Azure)
for maximum performance and scalability.
"""

import os, sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Start cloud-based multi-model training for SmallMind")
    parser.add_argument("--platform", choices=["aws", "gcp", "azure"], required=True, 
                       help="Cloud platform to use")
    parser.add_argument("--region", default="us-east-1", help="Cloud region")
    parser.add_argument("--output-dir", default="./cloud_training", help="Local output directory")
    parser.add_argument("--instance-type", help="Instance type (overrides default)")
    parser.add_argument("--max-instances", type=int, default=3, help="Maximum instances to launch")
    parser.add_argument("--ssh-key", help="SSH key name (AWS)")
    parser.add_argument("--security-group", help="Security group ID (AWS)")
    parser.add_argument("--project-id", help="Project ID (GCP)")
    parser.add_argument("--check-costs", action="store_true", help="Show cost estimates before starting")
    
    args = parser.parse_args()
    
    print("☁️  SmallMind Cloud-Based Multi-Model Training")
    print("=" * 60)
    print(f"Platform: {args.platform.upper()}")
    print(f"Region: {args.region}")
    print(f"Max Instances: {args.max_instances}")
    print(f"Output Directory: {args.output_dir}")
    print()
    print("🚀 This will launch cloud instances for ALL available models!")
    print("⚠️  Training will run FOREVER until manually stopped")
    print("💰 Cloud costs will be incurred - monitor your billing!")
    print("=" * 60)
    
    try:
        # Import after adding src to path
        from smallmind.models.cloud_integration import CloudTrainer, CloudConfig, create_aws_trainer, create_gcp_trainer
        
        # Create cloud trainer based on platform
        if args.platform == "aws":
            if not args.ssh_key:
                print("❌ SSH key required for AWS. Use --ssh-key")
                return 1
            
            if not args.security_group:
                print("❌ Security group required for AWS. Use --security-group")
                return 1
            
            trainer = create_aws_trainer(
                region=args.region,
                ssh_key=args.ssh_key,
                security_group=args.security_group
            )
            
        elif args.platform == "gcp":
            if not args.project_id:
                print("❌ Project ID required for GCP. Use --project-id")
                return 1
            
            trainer = create_gcp_trainer(
                region=args.region,
                project_id=args.project_id
            )
            
        elif args.platform == "azure":
            print("❌ Azure not yet implemented")
            return 1
        
        # Override instance type if specified
        if args.instance_type:
            trainer.cloud_config.instance_type = args.instance_type
        
        # Override max instances
        trainer.cloud_config.max_instances = args.max_instances
        
        # Check available models
        print("\n🔍 Checking available models...")
        models = [
            "DeepSeek-V2",
            "Qwen1.5-MoE", 
            "Mix-Tao-MoE"  # Fixed: use exact directory name
        ]
        
        available_models = []
        for model in models:
            # Convert model names to directory names (preserve hyphens and dots)
            model_dir = model.lower()
            model_path = f"src/smallmind/models/models/checkpoints/{model_dir}"
            if os.path.exists(model_path):
                print(f"✅ {model}: {model_path}")
                available_models.append(model)
            else:
                print(f"❌ {model}: {model_path} (not found)")
        
        if not available_models:
            print("\n❌ No models found! Please check your model paths.")
            return 1
        
        print(f"\n📊 Found {len(available_models)} models: {', '.join(available_models)}")
        
        # Show cost estimates
        if args.check_costs:
            print("\n💰 Cost Estimates:")
            print("-" * 40)
            
            # Calculate costs for available models
            total_cost_per_hour = 0
            for model in available_models:
                if args.platform == "aws":
                    cost = trainer._get_aws_instance_cost(trainer.cloud_config.instance_type)
                elif args.platform == "gcp":
                    cost = trainer._get_gcp_instance_cost(trainer.cloud_config.instance_type, "nvidia-tesla-t4")
                else:
                    cost = 1.0
                
                total_cost_per_hour += cost
                print(f"  {model}: ${cost:.3f}/hour")
            
            print(f"\n  Total: ${total_cost_per_hour:.3f}/hour")
            print(f"  Per Day: ${total_cost_per_hour * 24:.2f}")
            print(f"  Per Week: ${total_cost_per_hour * 24 * 7:.2f}")
            print(f"  Per Month: ${total_cost_per_hour * 24 * 30:.2f}")
        
        # Confirm before starting
        print(f"\n⚠️  WARNING: This will launch {len(available_models)} cloud instances!")
        print("Cloud costs will be incurred. Monitor your billing dashboard.")
        
        confirm = input("\nStart cloud-based training? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Cloud training cancelled.")
            return 0
        
        print(f"\n🚀 Launching cloud instances for {len(available_models)} models...")
        print("=" * 60)
        
        # Launch instances
        launched_instances = trainer.launch_training_instances(available_models)
        
        if not launched_instances:
            print("❌ Failed to launch any instances")
            return 1
        
        print(f"\n✅ Successfully launched {len(launched_instances)} instances!")
        
        # Monitor training
        print("\n📊 Monitoring cloud training...")
        print("Press Ctrl+C to stop all instances")
        print("=" * 60)
        
        try:
            while True:
                status = trainer.monitor_training()
                
                print(f"\n📊 Cloud Training Status - {status['total_instances']} instances")
                print("-" * 50)
                print(f"🔄 Running: {status['running']}")
                print(f"✅ Completed: {status['completed']}")
                print(f"❌ Failed: {status['failed']}")
                print(f"💰 Cost/Hour: ${status['cost_per_hour']:.3f}")
                
                for model_name, instance_info in status['instances'].items():
                    status_emoji = {
                        "running": "🔄",
                        "terminated": "✅",
                        "stopped": "❌",
                        "stopping": "⏳"
                    }.get(instance_info['status'], "❓")
                    
                    print(f"{status_emoji} {model_name}: {instance_info['status']}")
                    if instance_info['status'] == "running":
                        print(f"   🌐 IP: {instance_info['public_ip']}")
                        print(f"   🧠 GPU: {instance_info['gpu_info']}")
                        print(f"   💰 Cost: ${instance_info['cost_per_hour']:.3f}/hour")
                
                print("-" * 50)
                
                # Wait before next status check
                import time
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping all cloud instances...")
            trainer.stop_all_instances()
            
            # Final cost summary
            cost_estimate = trainer.get_cost_estimate()
            print(f"\n💰 Final Cost Summary:")
            print(f"  Total Cost/Hour: ${cost_estimate['cost_per_hour']:.3f}")
            print(f"  Estimated Total: ${cost_estimate['estimated_total']:.2f}")
            
            print("\n🎉 Cloud training stopped. Check your cloud console for final status.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nInstall cloud dependencies:")
        if args.platform == "aws":
            print("pip install boto3")
        elif args.platform == "gcp":
            print("pip install google-cloud-compute google-cloud-storage")
        elif args.platform == "azure":
            print("pip install azure-mgmt-compute azure-identity")
        return 1
    except Exception as e:
        print(f"❌ Cloud training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
