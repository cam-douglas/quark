#!/usr/bin/env python3
"""
Test Trained Wikipedia Model
============================

Test script to validate the trained Wikipedia model and demonstrate
its knowledge capabilities for brain simulation integration.

Author: Quark Brain Simulation Team
Date: 2025-01-20
"""

import os, sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def load_trained_model(model_path: str):
    """Load the trained Wikipedia model."""
    print(f"🔍 Loading trained model from: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Set to evaluation mode
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def test_wikipedia_knowledge(tokenizer, model, test_questions: list):
    """Test the model's Wikipedia knowledge with various questions."""
    print(f"\n🧠 Testing Wikipedia Knowledge")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        # Prepare input
        input_text = f"Question: {question}\nAnswer:"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        try:
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.replace(input_text, "").strip()
            
            print(f"🤖 Answer: {answer}")
            
        except Exception as e:
            print(f"❌ Generation error: {e}")

def test_model_capabilities(tokenizer, model):
    """Test various model capabilities."""
    print(f"\n🔬 Testing Model Capabilities")
    print("=" * 50)
    
    # Test 1: Basic text generation
    print(f"\n📝 Test 1: Basic Text Generation")
    input_text = "The human brain is"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=100, truncation=True)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {input_text}")
        print(f"Output: {response}")
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
    
    # Test 2: Knowledge completion
    print(f"\n📚 Test 2: Knowledge Completion")
    input_text = "Albert Einstein was a physicist who"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=100, truncation=True)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {input_text}")
        print(f"Output: {response}")
        
    except Exception as e:
        print(f"❌ Generation error: {e}")

def main():
    """Main function to test the trained model."""
    print("🧠 Wikipedia Model Testing for Brain Simulation")
    print("=" * 60)
    
    # Check for trained models
    model_path = Path("scaled_wikipedia_trained")
    if not model_path.exists():
        print("❌ No training outputs found. Please run training first.")
        return
    
    print(f"📁 Using training output: {model_path}")
    
    # Load the trained model
    tokenizer, model = load_trained_model(str(model_path))
    if tokenizer is None or model is None:
        return
    
    # Test Wikipedia knowledge
    test_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical symbol for gold?",
        "What year did World War II end?",
        "What is the largest planet in our solar system?"
    ]
    
    test_wikipedia_knowledge(tokenizer, model, test_questions)
    
    # Test model capabilities
    test_model_capabilities(tokenizer, model)
    
    print(f"\n🎉 Model testing completed!")
    print(f"📊 The model is ready for brain simulation integration.")

if __name__ == "__main__":
    main()
