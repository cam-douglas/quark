#!/bin/bash

# SmallMind Cloud Credentials Setup Script
# This script helps you set up cloud credentials for training

set -e

echo "☁️  SmallMind Cloud Credentials Setup"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "src/smallmind/scripts/start_cloud_training.py" ]; then
    echo "❌ Error: Please run this script from the small-mind project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: /path/to/small-mind/"
    exit 1
fi

echo "✅ Found SmallMind project"
echo ""

# Function to setup AWS
setup_aws() {
    echo "🔑 Setting up AWS credentials..."
    echo ""
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        echo "📦 Installing AWS CLI..."
        pip install boto3
    fi
    
    # Get AWS credentials
    echo "Please enter your AWS credentials:"
    read -p "AWS Access Key ID: " aws_access_key
    read -s -p "AWS Secret Access Key: " aws_secret_key
    echo ""
    read -p "AWS Region (default: us-east-1): " aws_region
    aws_region=${aws_region:-us-east-1}
    
    # Get other AWS settings
    read -p "SSH Key Pair Name: " ssh_key_name
    read -p "Security Group ID (optional): " security_group_id
    
    # Create .env file
    cat > .env.aws << EOF
# AWS Credentials for SmallMind Cloud Training
AWS_ACCESS_KEY_ID=$aws_access_key
AWS_SECRET_ACCESS_KEY=$aws_secret_key
AWS_DEFAULT_REGION=$aws_region
AWS_SSH_KEY_NAME=$ssh_key_name
AWS_SECURITY_GROUP_ID=$security_group_id
EOF
    
    echo ""
    echo "✅ AWS credentials saved to .env.aws"
    echo "📝 Add these to your ~/.zshrc or ~/.bashrc:"
    echo "   export AWS_ACCESS_KEY_ID=\"$aws_access_key\""
    echo "   export AWS_SECRET_ACCESS_KEY=\"$aws_secret_key\""
    echo "   export AWS_DEFAULT_REGION=\"$aws_region\""
    echo ""
    
    # Test AWS credentials
    echo "🧪 Testing AWS credentials..."
    if python -c "import boto3; client = boto3.client('ec2', region_name='$aws_region'); print('✅ AWS credentials working!')" 2>/dev/null; then
        echo "✅ AWS setup complete!"
    else
        echo "❌ AWS credentials test failed. Please check your keys."
    fi
}

# Function to setup Google Cloud
setup_gcp() {
    echo "🌐 Setting up Google Cloud credentials..."
    echo ""
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo "📦 Installing Google Cloud libraries..."
        pip install google-cloud-compute google-cloud-storage
    fi
    
    # Get GCP settings
    read -p "Google Cloud Project ID: " project_id
    read -p "Service Account Key JSON path: " service_account_path
    
    # Validate service account file
    if [ ! -f "$service_account_path" ]; then
        echo "❌ Service account file not found: $service_account_path"
        echo "   Please download it from Google Cloud Console first"
        return 1
    fi
    
    # Create .env file
    cat > .env.gcp << EOF
# Google Cloud Credentials for SmallMind Cloud Training
GOOGLE_APPLICATION_CREDENTIALS=$service_account_path
GOOGLE_CLOUD_PROJECT=$project_id
EOF
    
    echo ""
    echo "✅ Google Cloud credentials saved to .env.gcp"
    echo "📝 Add these to your ~/.zshrc or ~/.bashrc:"
    echo "   export GOOGLE_APPLICATION_CREDENTIALS=\"$service_account_path\""
    echo "   export GOOGLE_CLOUD_PROJECT=\"$project_id\""
    echo ""
    
    # Test GCP credentials
    echo "🧪 Testing Google Cloud credentials..."
    if python -c "from google.cloud import compute_v1; client = compute_v1.InstancesClient(); print('✅ Google Cloud credentials working!')" 2>/dev/null; then
        echo "✅ Google Cloud setup complete!"
    else
        echo "❌ Google Cloud credentials test failed. Please check your service account key."
    fi
}

# Function to setup shell profile
setup_shell_profile() {
    echo "🐚 Setting up shell profile..."
    
    # Determine shell profile
    if [ -n "$ZSH_VERSION" ]; then
        profile_file="$HOME/.zshrc"
        shell_name="zsh"
    elif [ -n "$BASH_VERSION" ]; then
        profile_file="$HOME/.bashrc"
        shell_name="bash"
    else
        echo "⚠️  Unknown shell. Please manually add exports to your shell profile."
        return
    fi
    
    echo "📝 Adding exports to $profile_file..."
    
    # Check if exports already exist
    if grep -q "AWS_ACCESS_KEY_ID" "$profile_file" 2>/dev/null; then
        echo "⚠️  AWS exports already exist in $profile_file"
    else
        if [ -f ".env.aws" ]; then
            echo "" >> "$profile_file"
            echo "# SmallMind AWS Cloud Training" >> "$profile_file"
            cat .env.aws | grep -v "^#" | sed 's/^/export /' >> "$profile_file"
            echo "✅ Added AWS exports to $profile_file"
        fi
    fi
    
    if grep -q "GOOGLE_APPLICATION_CREDENTIALS" "$profile_file" 2>/dev/null; then
        echo "⚠️  Google Cloud exports already exist in $profile_file"
    else
        if [ -f ".env.gcp" ]; then
            echo "" >> "$profile_file"
            echo "# SmallMind Google Cloud Training" >> "$profile_file"
            cat .env.gcp | grep -v "^#" | sed 's/^/export /' >> "$profile_file"
            echo "✅ Added Google Cloud exports to $profile_file"
        fi
    fi
    
    echo ""
    echo "🔄 To activate changes, run: source $profile_file"
}

# Function to test cloud training
test_cloud_training() {
    echo "🧪 Testing cloud training setup..."
    echo ""
    
    if [ -f ".env.aws" ]; then
        echo "🔑 Testing AWS cloud training..."
        source .env.aws
        if python src/smallmind/scripts/start_cloud_training.py --platform aws --region $AWS_DEFAULT_REGION --ssh-key $AWS_SSH_KEY_NAME --check-costs 2>/dev/null | head -20; then
            echo "✅ AWS cloud training test successful!"
        else
            echo "❌ AWS cloud training test failed"
        fi
        echo ""
    fi
    
    if [ -f ".env.gcp" ]; then
        echo "🌐 Testing Google Cloud training..."
        source .env.gcp
        if python src/smallmind/scripts/start_cloud_training.py --platform gcp --region us-central1 --project-id $GOOGLE_CLOUD_PROJECT --check-costs 2>/dev/null | head -20; then
            echo "✅ Google Cloud training test successful!"
        else
            echo "❌ Google Cloud training test failed"
        fi
        echo ""
    fi
}

# Main menu
echo "Choose your cloud platform:"
echo "1) AWS (recommended for GPU selection)"
echo "2) Google Cloud (good pricing, easy setup)"
echo "3) Both platforms"
echo "4) Exit"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        setup_aws
        setup_shell_profile
        ;;
    2)
        setup_gcp
        setup_shell_profile
        ;;
    3)
        setup_aws
        setup_gcp
        setup_shell_profile
        ;;
    4)
        echo "👋 Setup cancelled"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Cloud credentials setup complete!"
echo ""
echo "📚 Next steps:"
echo "1. Read the full setup guide: docs/CLOUD_CREDENTIALS_SETUP.md"
echo "2. Test your setup: source ~/.zshrc (or ~/.bashrc)"
echo "3. Start cloud training: python src/smallmind/scripts/start_cloud_training.py --help"
echo ""
echo "🚀 Happy cloud training!"
