#!/usr/bin/env python3
"""
Amazon Bedrock Integration Test for Quark Brain Simulation
"""

import boto3
import json
import os
from typing import Dict, Any, List
from datetime import datetime

def test_bedrock_access():
    """Test basic Bedrock access and list available models"""
    
    print("🧠 Testing Amazon Bedrock Integration")
    print("=" * 50)
    
    try:
        # Create Bedrock client
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        
        # List available models
        response = bedrock.list_foundation_models()
        models = response.get('modelSummaries', [])
        
        # Filter active models
        active_models = [m for m in models if m.get('modelLifecycle', {}).get('status') == 'ACTIVE']
        
        print(f"✅ Connected to Amazon Bedrock")
        print(f"📊 Total models available: {len(models)}")
        print(f"🟢 Active models: {len(active_models)}")
        
        # Show key models by provider
        providers = {}
        for model in active_models:
            provider = model['providerName']
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model['modelName'])
        
        print(f"\n🏢 Available Providers:")
        for provider, model_list in providers.items():
            print(f"   {provider}: {len(model_list)} models")
        
        return True, active_models
        
    except Exception as e:
        print(f"❌ Bedrock connection failed: {e}")
        return False, []

def test_claude_model():
    """Test Claude model for text generation"""
    
    print(f"\n🤖 Testing Claude Model")
    print("-" * 30)
    
    try:
        # Create Bedrock Runtime client
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Test with Claude model
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        # Prepare prompt for brain simulation
        prompt = """You are a quantum-enhanced brain simulation AI. Explain in 2 sentences how quantum computing could improve neural network modeling."""
        
        # Request body for Claude
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Invoke model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'content' in response_body and response_body['content']:
            generated_text = response_body['content'][0]['text']
            print(f"✅ Claude Response:")
            print(f"   {generated_text}")
            return True, generated_text
        else:
            print(f"❌ Unexpected response format: {response_body}")
            return False, None
            
    except Exception as e:
        print(f"❌ Claude test failed: {e}")
        return False, None

def test_amazon_nova_model():
    """Test Amazon Nova model"""
    
    print(f"\n🚀 Testing Amazon Nova Model")
    print("-" * 30)
    
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Test with Nova Micro
        model_id = "amazon.nova-micro-v1:0"
        
        prompt = "Explain quantum entanglement in brain simulation in one sentence."
        
        # Request body for Nova
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "top_p": 0.9,
            "temperature": 0.7
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        
        if 'output' in response_body and 'message' in response_body['output']:
            generated_text = response_body['output']['message']['content'][0]['text']
            print(f"✅ Nova Response:")
            print(f"   {generated_text}")
            return True, generated_text
        else:
            print(f"❌ Unexpected Nova response: {response_body}")
            return False, None
            
    except Exception as e:
        print(f"❌ Nova test failed: {e}")
        return False, None

def test_bedrock_embedding():
    """Test text embedding capabilities"""
    
    print(f"\n🧮 Testing Text Embeddings")
    print("-" * 30)
    
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Test with Titan Text Embeddings
        model_id = "amazon.titan-embed-text-v1"
        
        text = "quantum brain simulation neural networks"
        
        body = {
            "inputText": text
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        
        if 'embedding' in response_body:
            embedding = response_body['embedding']
            print(f"✅ Embedding generated:")
            print(f"   Dimensions: {len(embedding)}")
            print(f"   Sample values: {embedding[:5]}...")
            return True, embedding
        else:
            print(f"❌ Unexpected embedding response: {response_body}")
            return False, None
            
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False, None

def create_bedrock_brain_integration():
    """Create integration with Quark brain simulation"""
    
    print(f"\n🧠 Creating Bedrock-Brain Integration")
    print("-" * 40)
    
    integration_config = {
        "bedrock_config": {
            "region": "us-east-1",
            "preferred_models": {
                "text_generation": "anthropic.claude-3-haiku-20240307-v1:0",
                "embeddings": "amazon.titan-embed-text-v1",
                "reasoning": "amazon.nova-micro-v1:0"
            }
        },
        "brain_regions": {
            "language_cortex": "claude",
            "memory_hippocampus": "embeddings", 
            "reasoning_prefrontal": "nova"
        },
        "capabilities": [
            "Natural language processing",
            "Semantic understanding",
            "Reasoning and inference",
            "Memory encoding/retrieval"
        ],
        "integration_timestamp": datetime.now().isoformat()
    }
    
    # Save configuration
    with open("bedrock_brain_config.json", "w") as f:
        json.dump(integration_config, f, indent=2)
    
    print(f"✅ Integration config created:")
    print(f"   Text generation: Claude")
    print(f"   Embeddings: Titan")
    print(f"   Reasoning: Nova")
    print(f"   Config saved: bedrock_brain_config.json")
    
    return integration_config

def main():
    """Run comprehensive Bedrock integration test"""
    
    # Check for bearer token
    bearer_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
    if bearer_token:
        print(f"🔑 Bearer token detected: {bearer_token[:20]}...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: Basic Bedrock access
    success, models = test_bedrock_access()
    results["tests"]["bedrock_access"] = {
        "success": success,
        "model_count": len(models) if models else 0
    }
    
    if not success:
        print(f"\n❌ Bedrock access failed. Check your credentials and region.")
        return results
    
    # Test 2: Claude model
    success, response = test_claude_model()
    results["tests"]["claude_model"] = {
        "success": success,
        "response_length": len(response) if response else 0
    }
    
    # Test 3: Nova model  
    success, response = test_amazon_nova_model()
    results["tests"]["nova_model"] = {
        "success": success,
        "response_length": len(response) if response else 0
    }
    
    # Test 4: Embeddings
    success, embedding = test_bedrock_embedding()
    results["tests"]["embeddings"] = {
        "success": success,
        "embedding_dimensions": len(embedding) if embedding else 0
    }
    
    # Test 5: Create integration
    config = create_bedrock_brain_integration()
    results["tests"]["integration_setup"] = {
        "success": True,
        "capabilities": len(config["capabilities"])
    }
    
    # Summary
    successful_tests = sum(1 for test in results["tests"].values() if test["success"])
    total_tests = len(results["tests"])
    
    print(f"\n📊 Test Summary:")
    print(f"   Successful tests: {successful_tests}/{total_tests}")
    print(f"   Success rate: {successful_tests/total_tests:.1%}")
    
    # Save results
    with open("bedrock_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"   Results saved: bedrock_test_results.json")
    
    return results

if __name__ == "__main__":
    results = main()
