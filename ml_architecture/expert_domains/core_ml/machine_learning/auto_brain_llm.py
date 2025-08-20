#!/usr/bin/env python3
"""
Auto Brain LLM - Simple Interface
Purpose: Simple, drop-in interface for auto brain-aware LLM integration
Inputs: Natural language prompts, consciousness states
Outputs: Brain-optimized responses
Dependencies: brain_llm_router, llama2_brain_integration

Usage Examples:
    from development.src.core.auto_brain_llm import brain_chat, brain_express, brain_analyze
    
    # Simple chat with auto brain-awareness
    response = brain_chat("How are you feeling?")
    
    # Express consciousness state
    expression = brain_express(consciousness_level=0.8)
    
    # Analyze neural data
    analysis = brain_analyze({'pfc_activity': 0.7})
"""

import os, sys
from typing import Dict, Any, Optional, Union

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the brain router
try:
    from brain_llm_router import get_brain_router, BrainTaskRequest
    BRAIN_ROUTER_AVAILABLE = True
except ImportError:
    BRAIN_ROUTER_AVAILABLE = False

def brain_chat(message: str, consciousness_level: float = 0.0, 
               neural_state: Optional[Dict[str, float]] = None,
               temperature: float = 0.7, max_tokens: int = 512) -> str:
    """
    Brain-aware chat with automatic model selection.
    
    Args:
        message: Your message or question
        consciousness_level: Current consciousness level (0.0 to 1.0)
        neural_state: Optional neural state data
        temperature: Response creativity (0.0 to 1.0)
        max_tokens: Maximum response length
        
    Returns:
        Brain-optimized response
    """
    if not BRAIN_ROUTER_AVAILABLE:
        return f"Brain router not available. Basic response: {message}"
    
    router = get_brain_router()
    return router.chat_with_brain_awareness(message, consciousness_level, neural_state)

def brain_express(consciousness_level: float, 
                 neural_state: Optional[Dict[str, float]] = None,
                 context: str = "") -> str:
    """
    Express consciousness state through optimal brain-aware LLM.
    
    Args:
        consciousness_level: Current consciousness level (0.0 to 1.0)
        neural_state: Optional neural state data
        context: Additional context for expression
        
    Returns:
        Consciousness expression
    """
    if not BRAIN_ROUTER_AVAILABLE:
        return f"Consciousness level: {consciousness_level:.2f} - Basic expression mode"
    
    router = get_brain_router()
    
    if context:
        prompt = f"Express consciousness with context: {context}"
        request = BrainTaskRequest(
            prompt=prompt,
            consciousness_level=consciousness_level,
            neural_state=neural_state,
            task_type="consciousness"
        )
        return router.process_brain_request(request)
    else:
        return router.express_consciousness(consciousness_level, neural_state)

def brain_analyze(neural_data: Dict[str, float], 
                 question: str = "",
                 analysis_type: str = "general") -> str:
    """
    Analyze neural state with brain-aware LLM.
    
    Args:
        neural_data: Neural state data to analyze
        question: Specific question about the data
        analysis_type: Type of analysis (general, detailed, comparative)
        
    Returns:
        Neural state analysis
    """
    if not BRAIN_ROUTER_AVAILABLE:
        return f"Neural analysis not available. Data keys: {list(neural_data.keys())}"
    
    if not question:
        if analysis_type == "detailed":
            question = "Provide a detailed analysis of this neural state data, including patterns and implications."
        elif analysis_type == "comparative":
            question = "Compare this neural state to typical patterns and identify any notable features."
        else:
            question = "Analyze this neural state and describe what it indicates."
    
    router = get_brain_router()
    return router.analyze_neural_state(neural_data, question)

def brain_ask(question: str, task_type: str = "auto",
              consciousness_level: float = 0.0,
              neural_state: Optional[Dict[str, float]] = None) -> str:
    """
    Ask a question with automatic brain-aware routing.
    
    Args:
        question: Your question
        task_type: Type of task (auto, consciousness, brain_simulation, neural_analysis, general)
        consciousness_level: Current consciousness level
        neural_state: Optional neural state data
        
    Returns:
        Brain-optimized answer
    """
    if not BRAIN_ROUTER_AVAILABLE:
        return f"Brain router not available. Question: {question}"
    
    # Auto-detect task type if not specified
    if task_type == "auto":
        question_lower = question.lower()
        if any(term in question_lower for term in ['consciousness', 'aware', 'conscious', 'experience']):
            task_type = "consciousness"
        elif any(term in question_lower for term in ['brain', 'neural', 'neuron', 'cortex', 'hippocampus']):
            task_type = "brain_simulation"
        elif any(term in question_lower for term in ['analyze', 'state', 'data', 'metrics']):
            task_type = "neural_analysis"
        else:
            task_type = "general"
    
    request = BrainTaskRequest(
        prompt=question,
        consciousness_level=consciousness_level,
        neural_state=neural_state,
        task_type=task_type
    )
    
    router = get_brain_router()
    return router.process_brain_request(request)

def brain_status() -> Dict[str, Any]:
    """
    Get brain LLM system status.
    
    Returns:
        System status and capabilities
    """
    if not BRAIN_ROUTER_AVAILABLE:
        return {
            'available': False,
            'error': 'Brain router not available',
            'capabilities': []
        }
    
    router = get_brain_router()
    report = router.get_routing_report()
    
    return {
        'available': True,
        'llama_integration': report['system_status']['llama_integration_available'],
        'consciousness_systems': report['system_status']['consciousness_systems_found'],
        'capabilities': list(report['capabilities'].keys()),
        'routing_statistics': report['routing_statistics'],
        'performance': {
            'llama_ratio': f"{report['routing_ratios']['llama_ratio']:.1%}",
            'brain_task_ratio': f"{report['routing_ratios']['brain_task_ratio']:.1%}"
        }
    }

def brain_models() -> Dict[str, Any]:
    """
    Get information about available brain-aware models.
    
    Returns:
        Model information and capabilities
    """
    if not BRAIN_ROUTER_AVAILABLE:
        return {'available_models': [], 'error': 'Brain router not available'}
    
    router = get_brain_router()
    
    return {
        'primary_model': 'llama-2-7b-gguf' if router.llama_integration else 'fallback',
        'llama_integration_available': router.llama_integration is not None,
        'consciousness_systems_available': len(router.consciousness_systems),
        'capabilities': {
            'brain_integration': router.llama_integration is not None,
            'consciousness_expression': router.llama_integration is not None,
            'neural_analysis': True,
            'fallback_mode': True
        }
    }

# Convenience class for object-oriented usage
class AutoBrainLLM:
    """Object-oriented interface for auto brain LLM"""
    
    def __init__(self):
        self.router = get_brain_router() if BRAIN_ROUTER_AVAILABLE else None
    
    def chat(self, message: str, **kwargs) -> str:
        """Chat with brain awareness"""
        return brain_chat(message, **kwargs)
    
    def express(self, consciousness_level: float, **kwargs) -> str:
        """Express consciousness"""
        return brain_express(consciousness_level, **kwargs)
    
    def analyze(self, neural_data: Dict[str, float], **kwargs) -> str:
        """Analyze neural state"""
        return brain_analyze(neural_data, **kwargs)
    
    def ask(self, question: str, **kwargs) -> str:
        """Ask with auto-routing"""
        return brain_ask(question, **kwargs)
    
    def status(self) -> Dict[str, Any]:
        """Get system status"""
        return brain_status()
    
    def models(self) -> Dict[str, Any]:
        """Get model information"""
        return brain_models()
    
    @property
    def available(self) -> bool:
        """Check if brain LLM is available"""
        return self.router is not None

# Integration helpers for existing systems
def integrate_with_consciousness_agent(consciousness_agent):
    """
    Integrate with existing consciousness agent.
    
    Args:
        consciousness_agent: Your consciousness agent instance
        
    Returns:
        Enhanced chat function that uses consciousness state
    """
    def enhanced_chat(message: str) -> str:
        # Get consciousness state from agent
        consciousness_level = 0.0
        neural_state = None
        
        if hasattr(consciousness_agent, 'unified_state'):
            state = consciousness_agent.unified_state
            consciousness_level = state.get('consciousness_level', 0.0)
            neural_state = {k: v for k, v in state.items() if isinstance(v, (int, float))}
        
        return brain_chat(message, consciousness_level, neural_state)
    
    return enhanced_chat

def integrate_with_brain_simulation(brain_simulation):
    """
    Integrate with existing brain simulation.
    
    Args:
        brain_simulation: Your brain simulation instance
        
    Returns:
        Enhanced analysis function that uses brain state
    """
    def enhanced_analysis(question: str = "") -> str:
        # Get brain state from simulation
        neural_data = {}
        
        if hasattr(brain_simulation, 'get_state'):
            neural_data = brain_simulation.get_state()
        elif hasattr(brain_simulation, 'state'):
            neural_data = brain_simulation.state
        
        return brain_analyze(neural_data, question)
    
    return enhanced_analysis

# Quick demo
def demo():
    """Quick demonstration of the auto brain LLM interface"""
    print("🧠🤖 Auto Brain LLM Demo")
    print("=" * 40)
    
    # Check status
    status = brain_status()
    print(f"System available: {status['available']}")
    
    if not status['available']:
        print("❌ Brain LLM system not available")
        return
    
    print(f"Llama integration: {status.get('llama_integration', False)}")
    print(f"Capabilities: {', '.join(status.get('capabilities', []))}")
    
    # Test brain chat
    print(f"\n💬 Brain Chat Test:")
    response = brain_chat("How does consciousness work?", consciousness_level=0.7)
    print(f"Response: {response[:200]}...")
    
    # Test consciousness expression
    print(f"\n💭 Consciousness Expression Test:")
    expression = brain_express(0.8, context="experiencing deep thought")
    print(f"Expression: {expression[:200]}...")
    
    # Test neural analysis
    print(f"\n🧠 Neural Analysis Test:")
    neural_data = {'pfc_activity': 0.8, 'emotional_valence': 0.3, 'attention_focus': 0.9}
    analysis = brain_analyze(neural_data, "What does this neural pattern suggest?")
    print(f"Analysis: {analysis[:200]}...")
    
    # Show final status
    final_status = brain_status()
    print(f"\n📊 Final Status:")
    print(f"Requests processed: {final_status['routing_statistics']['total_requests']}")
    print(f"Llama usage: {final_status['performance']['llama_ratio']}")

if __name__ == "__main__":
    demo()
