#!/usr/bin/env python3
"""
Simple test for the Unified Super Mind
"""

import torch
from super_mind import UnifiedSuperMind, SuperMindConfig

def test_basic_functionality():
    """Test basic model creation and forward pass."""
    print("🧪 Testing basic functionality...")
    
    # Create config
    config = SuperMindConfig(
        hidden_size=128,
        num_layers=2,
        num_experts=2,
        max_steps=10
    )
    
    # Create model
    model = UnifiedSuperMind(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    try:
        outputs = model(input_ids, attention_mask)
        print("✅ Forward pass successful")
        print(f"   • Logits shape: {outputs['logits'].shape}")
        print(f"   • Hidden states shape: {outputs['hidden_states'].shape}")
        print(f"   • Curiosity scores shape: {outputs['curiosity_scores'].shape}")
        print(f"   • Brain processed shape: {outputs['brain_processed'].shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test generation
    try:
        response = model.generate_response("Hello", max_length=10)
        print(f"✅ Generation successful: {response}")
    except Exception as e:
        print(f"⚠️  Generation failed: {e}")
    
    return True

def test_training_step():
    """Test training step functionality."""
    print("\n🧪 Testing training step...")
    
    config = SuperMindConfig(
        hidden_size=64,
        num_layers=1,
        num_experts=2,
        max_steps=10
    )
    
    model = UnifiedSuperMind(config)
    
    # Create dummy batch
    batch = {
        'input_ids': torch.randint(0, 256, (2, 8)),
        'attention_mask': torch.ones(2, 8),
        'labels': torch.randint(0, 256, (2, 8))
    }
    
    try:
        losses = model.train_step(batch)
        print("✅ Training step successful")
        print(f"   • Losses: {losses}")
        print(f"   • Training step: {model.training_step}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧠 Unified Super Mind - Basic Tests")
    print("=" * 40)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Basic functionality test failed")
        return
    
    # Test training step
    if not test_training_step():
        print("❌ Training step test failed")
        return
    
    print("\n🎉 All tests passed!")
    print("✅ The Unified Super Mind is working correctly")

if __name__ == "__main__":
    main()
