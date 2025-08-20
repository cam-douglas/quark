#!/usr/bin/env python3
"""
Simple test using fallback model
"""

import torch
import torch.nn as nn
from super_mind import SuperMindConfig

def test_fallback_model():
    """Test with fallback model to avoid HuggingFace issues."""
    print("🧪 Testing with fallback model...")
    
    # Create config that will trigger fallback
    config = SuperMindConfig(
        base_model="nonexistent_model",  # This will trigger fallback
        hidden_size=64,
        num_layers=2,
        num_experts=2,
        max_steps=10
    )
    
    try:
        # This should create a fallback model
        from super_mind import UnifiedSuperMind
        model = UnifiedSuperMind(config)
        print(f"✅ Fallback model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        print("✅ Forward pass successful")
        print(f"   • Logits shape: {outputs['logits'].shape}")
        print(f"   • Hidden states shape: {outputs['hidden_states'].shape}")
        
        # Test generation
        response = model.generate_response("Hello", max_length=10)
        print(f"✅ Generation successful: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧠 Simple Fallback Model Test")
    print("=" * 40)
    
    if test_fallback_model():
        print("\n🎉 Test passed!")
    else:
        print("\n❌ Test failed!")
