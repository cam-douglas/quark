#!/usr/bin/env python3
"""
AWS Credentials Setup Helper
Helps you configure AWS credentials using existing keys
"""

import os
import subprocess
import sys

def setup_aws_credentials():
    """Interactive AWS credentials setup"""
    
    print("🔑 AWS Credentials Setup")
    print("=" * 30)
    print("This will help you configure AWS CLI with your existing keys")
    print()
    
    # Get credentials from user
    access_key = input("Enter your AWS Access Key ID: ").strip()
    secret_key = input("Enter your AWS Secret Access Key: ").strip()
    region = input("Enter your preferred region (default: us-east-1): ").strip() or "us-east-1"
    
    if not access_key or not secret_key:
        print("❌ Access Key and Secret Key are required!")
        return False
    
    # Configure AWS CLI
    try:
        print("\n⚙️  Configuring AWS CLI...")
        
        # Set access key
        subprocess.run([
            'aws', 'configure', 'set', 'aws_access_key_id', access_key
        ], check=True, capture_output=True)
        
        # Set secret key
        subprocess.run([
            'aws', 'configure', 'set', 'aws_secret_access_key', secret_key
        ], check=True, capture_output=True)
        
        # Set region
        subprocess.run([
            'aws', 'configure', 'set', 'region', region
        ], check=True, capture_output=True)
        
        # Set output format
        subprocess.run([
            'aws', 'configure', 'set', 'output', 'json'
        ], check=True, capture_output=True)
        
        print("✅ AWS credentials configured successfully!")
        
        # Test configuration
        print("\n🧪 Testing configuration...")
        result = subprocess.run([
            'aws', 'sts', 'get-caller-identity'
        ], check=True, capture_output=True, text=True)
        
        print("✅ Configuration test passed!")
        print(f"Account info: {result.stdout.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error configuring AWS: {e}")
        print(f"Error output: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_key_pair():
    """Create EC2 key pair if needed"""
    print("\n🔐 EC2 Key Pair Setup")
    print("=" * 25)
    
    create_key = input("Do you need to create an EC2 key pair? (y/n): ").strip().lower()
    
    if create_key == 'y':
        key_name = input("Enter key pair name (e.g., smallmind-key): ").strip()
        region = input("Enter region for key pair (default: us-east-1): ").strip() or "us-east-1"
        
        if key_name:
            try:
                print(f"🔑 Creating key pair '{key_name}' in {region}...")
                
                result = subprocess.run([
                    'aws', 'ec2', 'create-key-pair',
                    '--key-name', key_name,
                    '--region', region,
                    '--query', 'KeyMaterial',
                    '--output', 'text'
                ], check=True, capture_output=True, text=True)
                
                # Save private key to file
                key_file = f"~/.ssh/{key_name}.pem"
                key_path = os.path.expanduser(key_file)
                
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                
                with open(key_path, 'w') as f:
                    f.write(result.stdout)
                
                # Set proper permissions
                os.chmod(key_path, 0o600)
                
                print(f"✅ Key pair created successfully!")
                print(f"📁 Private key saved to: {key_path}")
                print(f"🔑 Key name: {key_name}")
                print(f"🌍 Region: {region}")
                
                return key_name
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Error creating key pair: {e}")
                return None
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return None
    
    return None

if __name__ == "__main__":
    print("🚀 SmallMind AWS Setup Helper")
    print("=" * 35)
    
    # Setup credentials
    if setup_aws_credentials():
        print("\n🎉 AWS setup completed successfully!")
        
        # Offer to create key pair
        key_name = create_key_pair()
        
        if key_name:
            print(f"\n📋 Next steps:")
            print(f"1. Your AWS credentials are configured")
            print(f"2. EC2 key pair '{key_name}' created")
            print(f"3. Run: python3 aws_config.py (to verify)")
            print(f"4. Deploy: python3 aws_deploy.py --key-name {key_name}")
        else:
            print(f"\n📋 Next steps:")
            print(f"1. Your AWS credentials are configured")
            print(f"2. Run: python3 aws_config.py (to verify)")
            print(f"3. Deploy: python3 aws_deploy.py --key-name YOUR_KEY_NAME")
    else:
        print("\n❌ AWS setup failed. Please check your credentials and try again.")
        sys.exit(1)
