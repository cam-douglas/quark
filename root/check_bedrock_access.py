#!/usr/bin/env python3
"""
Quick Bedrock Access Checker
Run this periodically to check when model access is approved
"""

import boto3
import json
from datetime import datetime

def quick_access_test():
    """Quick test to check if any models are accessible"""
    
    test_models = [
        ("amazon.nova-micro-v1:0", "Nova Micro"),
        ("amazon.titan-embed-text-v1", "Titan Embeddings"), 
        ("anthropic.claude-3-haiku-20240307-v1:0", "Claude Haiku"),
        ("amazon.titan-text-lite-v1", "Titan Text Lite")
    ]
    
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    print(f"🕐 Checking Bedrock Access - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    
    accessible_models = []
    
    for model_id, model_name in test_models:
        try:
            # Try a minimal test request
            if "nova" in model_id:
                body = {
                    "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
                    "max_tokens": 10
                }
            elif "claude" in model_id:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            elif "embed" in model_id:
                body = {"inputText": "test"}
            else:
                body = {
                    "inputText": "Hi",
                    "textGenerationConfig": {"maxTokenCount": 10}
                }
            
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            print(f"✅ {model_name}: ACCESS GRANTED!")
            accessible_models.append(model_name)
            
        except Exception as e:
            if "AccessDeniedException" in str(e):
                print(f"⏳ {model_name}: Still pending approval")
            else:
                print(f"❌ {model_name}: {str(e)[:50]}...")
    
    print(f"\n📊 Accessible models: {len(accessible_models)}/{len(test_models)}")
    
    if accessible_models:
        print(f"🎉 READY! Models available: {', '.join(accessible_models)}")
        return True
    else:
        print("⏳ Still waiting for model access approval...")
        return False

if __name__ == "__main__":
    quick_access_test()
