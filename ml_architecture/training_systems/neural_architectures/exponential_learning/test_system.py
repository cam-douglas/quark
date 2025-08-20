#!/usr/bin/env python3
"""
Simple test script for the Exponential Learning System
Tests individual components without full system startup
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_components():
    """Test individual components"""
    logger.info("🧪 Testing Exponential Learning System Components...")
    
    try:
        # Test 1: Exponential Learning System
        logger.info("1️⃣ Testing Exponential Learning System...")
        from exponential_learning_system import ExponentialLearningSystem
        learning_system = ExponentialLearningSystem()
        logger.info(f"✅ Learning system initialized: {learning_system.learning_cycles} cycles")
        
        # Test 2: Knowledge Synthesizer
        logger.info("2️⃣ Testing Knowledge Synthesizer...")
        from knowledge_synthesizer import KnowledgeSynthesizer
        synthesizer = KnowledgeSynthesizer()
        logger.info("✅ Knowledge synthesizer initialized")
        
        # Test 3: Knowledge Validation System
        logger.info("3️⃣ Testing Knowledge Validation System...")
        from knowledge_validation_system import KnowledgeValidationSystem
        validator = KnowledgeValidationSystem()
        logger.info("✅ Knowledge validation system initialized")
        
        # Test 4: Research Agents (basic)
        logger.info("4️⃣ Testing Research Agents...")
        from research_agents import ResearchAgentHub
        research_hub = ResearchAgentHub()
        logger.info("✅ Research agent hub initialized")
        
        # Test 5: Cloud Training Orchestrator
        logger.info("5️⃣ Testing Cloud Training Orchestrator...")
        from cloud_training_orchestrator import CloudTrainingOrchestrator
        cloud_orch = CloudTrainingOrchestrator()
        logger.info("✅ Cloud training orchestrator initialized")
        
        # Test 6: Neuro Agent Enhancer
        logger.info("6️⃣ Testing Neuro Agent Enhancer...")
        from neuro_agent_enhancer import NeuroAgentEnhancer
        neuro_agent = NeuroAgentEnhancer()
        logger.info("✅ Neuro agent enhancer initialized")
        
        # Test 7: Main Orchestrator
        logger.info("7️⃣ Testing Main Orchestrator...")
        from main_orchestrator import ExponentialLearningOrchestrator
        main_orch = ExponentialLearningOrchestrator()
        logger.info("✅ Main orchestrator initialized")
        
        logger.info("🎉 All components tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        return False

async def test_basic_functionality():
    """Test basic functionality without full initialization"""
    logger.info("🔧 Testing Basic Functionality...")
    
    try:
        # Test learning system growth
        from exponential_learning_system import ExponentialLearningSystem
        ls = ExponentialLearningSystem()
        
        # Run a few learning cycles
        for i in range(3):
            ls.learning_cycles += 1
            ls.grow_learning_capacity()
            logger.info(f"Cycle {i+1}: Rate={ls.learning_rate:.2f}, Exploration={ls.exploration_factor:.2f}")
        
        # Test knowledge synthesis
        from knowledge_synthesizer import KnowledgeSynthesizer
        ks = KnowledgeSynthesizer()
        
        # Mock research results
        mock_results = {
            "wikipedia": [type('MockResult', (), {'content': 'Test content', 'connections': [], 'metadata': {}})()],
            "arxiv": [type('MockResult', (), {'content': 'Test paper', 'connections': [], 'metadata': {}})()]
        }
        
        # Test synthesis
        synthesized = await ks.synthesize_research_findings(mock_results)
        logger.info(f"✅ Knowledge synthesis test: {len(synthesized.core_concepts)} concepts")
        
        logger.info("🎯 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Exponential Learning System Tests...")
    
    # Test 1: Component initialization
    components_ok = await test_components()
    
    # Test 2: Basic functionality
    functionality_ok = await test_basic_functionality()
    
    # Summary
    if components_ok and functionality_ok:
        logger.info("🎉 ALL TESTS PASSED! System is ready to run.")
        logger.info("🚀 You can now run: python run_exponential_learning.py")
    else:
        logger.error("❌ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
