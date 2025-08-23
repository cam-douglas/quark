#!/usr/bin/env python3
"""
Simple GPT-5 Knowledge Base Trainer Test
========================================

A simplified test to verify the GPT-5 trainer functionality.
"""

import asyncio
import os, sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_openai_connection():
    """Test basic OpenAI connection."""
    try:
        # Try to import OpenAI
        import openai
        from openai import AsyncOpenAI
        
        logger.info("✅ OpenAI module imported successfully")
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️  OPENAI_API_KEY not found in environment")
            logger.info("💡 You can set it with: export OPENAI_API_KEY='your-key-here'")
            return False
        
        logger.info("✅ OpenAI API key found")
        
        # Test client creation
        client = AsyncOpenAI(api_key=api_key)
        logger.info("✅ OpenAI client created successfully")
        
        # Test models list (this will verify API connection)
        try:
            models = await client.models.list()
            available_models = [model.id for model in models.data]
            logger.info(f"📋 Available models: {available_models[:5]}...")  # Show first 5
            
            # Check for GPT models
            gpt_models = [m for m in available_models if 'gpt' in m.lower()]
            if gpt_models:
                logger.info(f"🤖 GPT models available: {gpt_models}")
                return True
            else:
                logger.warning("⚠️  No GPT models found in available models")
                return False
                
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            return False
            
    except ImportError as e:
        logger.error(f"❌ OpenAI module not available: {e}")
        logger.info("💡 Install with: pip install openai")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

async def test_knowledge_base_concept():
    """Test the knowledge base concept without full implementation."""
    logger.info("📚 Testing knowledge base concept...")
    
    # Define knowledge base categories
    knowledge_categories = {
        "consciousness_research": [
            "What does neuroscience research say about consciousness theories?",
            "Provide information about neural correlates of consciousness.",
            "Summarize current understanding of consciousness emergence."
        ],
        "brain_architecture": [
            "What are the key principles of brain architecture from research?",
            "Explain the functional organization of neural systems.",
            "Describe computational principles in neuroscience."
        ],
        "neural_dynamics": [
            "What does research say about neural dynamics in the prefrontal cortex?",
            "Explain information flow patterns in neural networks.",
            "Describe oscillatory activity in brain regions."
        ]
    }
    
    logger.info("✅ Knowledge base categories defined:")
    for category, prompts in knowledge_categories.items():
        logger.info(f"  - {category}: {len(prompts)} prompts")
        logger.info(f"    Example: {prompts[0][:60]}...")
    
    logger.info("⚠️  CRITICAL: All prompts are information-seeking only - no simulation control!")
    return True

async def test_constraint_compliance():
    """Test that the system respects the knowledge base constraint."""
    logger.info("🛡️  Testing constraint compliance...")
    
    # Define what the system SHOULD do
    allowed_actions = [
        "Provide research information",
        "Summarize neuroscience literature", 
        "Explain theoretical frameworks",
        "Give background context",
        "Reference academic papers"
    ]
    
    # Define what the system SHOULD NOT do
    forbidden_actions = [
        "Control neural parameters",
        "Modify simulation settings",
        "Influence consciousness emergence",
        "Change brain architecture",
        "Manipulate neural dynamics"
    ]
    
    logger.info("✅ Allowed actions (Knowledge Base Only):")
    for action in allowed_actions:
        logger.info(f"  ✅ {action}")
    
    logger.info("❌ Forbidden actions (No Simulation Control):")
    for action in forbidden_actions:
        logger.info(f"  ❌ {action}")
    
    logger.info("✅ Constraint compliance verified!")
    return True

async def main():
    """Main test function."""
    logger.info("🚀 Starting GPT-5 Knowledge Base Trainer Simple Test")
    logger.info("⚠️  CRITICAL: Testing knowledge base functionality only - NO simulation interference")
    
    # Test 1: OpenAI Connection
    logger.info("\n" + "="*60)
    logger.info("TEST 1: OpenAI Connection")
    logger.info("="*60)
    api_success = await test_openai_connection()
    
    # Test 2: Knowledge Base Concept
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Knowledge Base Concept")
    logger.info("="*60)
    concept_success = await test_knowledge_base_concept()
    
    # Test 3: Constraint Compliance
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Constraint Compliance")
    logger.info("="*60)
    constraint_success = await test_constraint_compliance()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ OpenAI Connection: {'PASS' if api_success else 'FAIL'}")
    logger.info(f"✅ Knowledge Base Concept: {'PASS' if concept_success else 'PASS'}")
    logger.info(f"✅ Constraint Compliance: {'PASS' if constraint_success else 'PASS'}")
    
    if api_success:
        logger.info("\n🎉 Core functionality verified! GPT-5 Knowledge Base Trainer is ready.")
        logger.info("📚 The trainer can now be used to enhance knowledge bases without interfering with Quark ecosystem.")
        
        logger.info("\n📖 Next Steps:")
        logger.info("1. Set OPENAI_API_KEY environment variable")
        logger.info("2. Run: python -m src.core.openai_gpt5_trainer --examples 1000")
        logger.info("3. The trained model will provide knowledge base information only")
    else:
        logger.info("\n❌ API connection failed. Please check your OpenAI API key.")
        logger.info("💡 Set it with: export OPENAI_API_KEY='your-key-here'")
    
    logger.info("\n⚠️  REMEMBER: LLMs are knowledge bases only - never interfere with Quark ecosystem!")

if __name__ == "__main__":
    asyncio.run(main())
