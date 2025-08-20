#!/usr/bin/env python3
"""
Start Multi-Cloud Training Script

Launches SmallMind training simultaneously on AWS and Google Cloud
for better resource availability and redundancy.
"""

import os, sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Start multi-cloud training for SmallMind")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS region")
    parser.add_argument("--gcp-region", default="us-central1", help="Google Cloud region")
    parser.add_argument("--output-dir", default="./multi_cloud_training", help="Output directory")
    parser.add_argument("--max-instances-per-platform", type=int, default=2, help="Max instances per platform")
    parser.add_argument("--check-costs", action="store_true", help="Show cost estimates before starting")
    parser.add_argument("--aws-only", action="store_true", help="Use only AWS")
    parser.add_argument("--gcp-only", action="store_true", help="Use only Google Cloud")
    
    args = parser.parse_args()
    
    print("☁️  SmallMind Multi-Cloud Training")
    print("=" * 60)
    print(f"AWS Region: {args.aws_region}")
    print(f"GCP Region: {args.gcp_region}")
    print(f"Max Instances per Platform: {args.max_instances_per_platform}")
    print(f"Output Directory: {args.output_dir}")
    print()
    print("🚀 This will launch instances across multiple cloud platforms!")
    print("⚠️  Training will run FOREVER until manually stopped")
    print("💰 Cloud costs will be incurred - monitor your billing!")
    print("=" * 60)
    
    try:
        from models.multi_cloud_trainer import create_multi_cloud_trainer
        
        # Determine which platforms to use
        aws_enabled = not args.gcp_only
        gcp_enabled = not args.aws_only
        
        if not aws_enabled and not gcp_enabled:
            print("❌ Error: At least one cloud platform must be enabled")
            return 1
        
        # Check available models
        print("\n🔍 Checking available models...")
        models = [
            "DeepSeek-V2",
            "Qwen1.5-MoE", 
            "Mix-Tao-MoE"
        ]
        
        available_models = []
        for model in models:
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
        
        # Create multi-cloud trainer
        trainer = create_multi_cloud_trainer(
            aws_enabled=aws_enabled,
            gcp_enabled=gcp_enabled,
            aws_region=args.aws_region,
            gcp_region=args.gcp_region,
            max_instances_per_platform=args.max_instances_per_platform
        )
        
        # Show cost estimates
        if args.check_costs:
            print("\n💰 Multi-Cloud Cost Estimates:")
            print("-" * 40)
            
            try:
                cost_estimate = trainer.get_multi_cloud_cost_estimate()
                print(f"  Total Cost/Hour: ${cost_estimate['total_cost_per_hour']:.3f}")
                print(f"  Per Day: ${cost_estimate['total_cost_per_day']:.2f}")
                print(f"  Per Week: ${cost_estimate['total_cost_per_week']:.2f}")
                print(f"  Per Month: ${cost_estimate['total_cost_per_month']:.2f}")
                
                if 'aws' in cost_estimate['platform_costs']:
                    aws_cost = cost_estimate['platform_costs']['aws']
                    print(f"\n  AWS Cost/Hour: ${aws_cost['cost_per_hour']:.3f}")
                
                if 'gcp' in cost_estimate['platform_costs']:
                    gcp_cost = cost_estimate['platform_costs']['gcp']
                    print(f"  GCP Cost/Hour: ${gcp_cost['cost_per_hour']:.3f}")
                    
            except Exception as e:
                print(f"  Could not calculate costs: {e}")
        
        # Confirm before starting
        print(f"\n⚠️  WARNING: This will launch instances across multiple cloud platforms!")
        print("Cloud costs will be incurred. Monitor your billing dashboard.")
        
        confirm = input("\nStart multi-cloud training? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Multi-cloud training cancelled.")
            return 0
        
        print(f"\n🚀 Launching multi-cloud instances for {len(available_models)} models...")
        print("=" * 60)
        
        # Launch multi-cloud training
        launched_instances = trainer.launch_multi_cloud_training(available_models)
        
        if not launched_instances:
            print("❌ Failed to launch any instances")
            return 1
        
        print(f"\n✅ Successfully launched multi-cloud training!")
        
        # Monitor training
        print("\n📊 Monitoring multi-cloud training...")
        print("Press Ctrl+C to stop all instances")
        print("=" * 60)
        
        try:
            while True:
                status = trainer.monitor_multi_cloud_training()
                
                print(f"\n📊 Multi-Cloud Training Status - {status['total_instances']} instances")
                print("-" * 50)
                print(f"🔄 Running: {status['overall_status']['running']}")
                print(f"✅ Completed: {status['overall_status']['completed']}")
                print(f"❌ Failed: {status['overall_status']['failed']}")
                print(f"💰 Total Cost/Hour: ${status['overall_status']['cost_per_hour']:.3f}")
                
                # Show platform-specific status
                for platform, platform_status in status['platforms'].items():
                    print(f"\n☁️  {platform.upper()}:")
                    for model_name, instance_info in platform_status['instances'].items():
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
            print("\n\n🛑 Stopping all multi-cloud instances...")
            trainer.stop_all_instances()
            
            # Final cost summary
            try:
                cost_estimate = trainer.get_multi_cloud_cost_estimate()
                print(f"\n💰 Final Multi-Cloud Cost Summary:")
                print(f"  Total Cost/Hour: ${cost_estimate['total_cost_per_hour']:.3f}")
                print(f"  Estimated Total: ${cost_estimate['estimated_total']:.2f}")
            except:
                print("\n💰 Cost summary unavailable")
            
            print("\n🎉 Multi-cloud training stopped. Check your cloud consoles for final status.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nInstall multi-cloud dependencies:")
        print("pip install boto3 google-cloud-compute")
        return 1
    except Exception as e:
        print(f"❌ Multi-cloud training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
