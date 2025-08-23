#!/usr/bin/env python3
"""
LLM Brain Auto Selector
Purpose: Automatically select and integrate the best LLM for brain simulation tasks
Inputs: Task prompts, brain states, consciousness levels
Outputs: Optimal LLM integration with brain systems
Seeds: Model selection criteria, brain integration parameters
Dependencies: openai_gpt5_trainer, llama2_brain_integration

Key Features:
- Automatic model selection based on task requirements
- Brain-specific model optimization
- Consciousness-aware routing
- Seamless integration with existing auto LLM system
- Performance monitoring and adaptation
"""

import os, sys
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BrainLLMRequest:
    """Request for brain-aware LLM selection"""
    prompt: str
    consciousness_level: float = 0.0
    neural_state: Dict[str, float] = None
    brain_metrics: Dict[str, float] = None
    task_type: str = "general"  # general, consciousness, brain_simulation, neural_analysis
    privacy_required: bool = True  # Default to privacy for brain data
    max_tokens: int = 512
    temperature: float = 0.7

class LLMBrainAutoSelector:
    """Automatically select and integrate LLMs for brain simulation tasks"""
    
    def __init__(self):
        self.model_selector = None
        self.llama_integration = None
        self.active_model = None
        self.performance_metrics = {
            'total_requests': 0,
            'llama_selections': 0,
            'other_selections': 0,
            'brain_task_accuracy': 0.0,
            'average_response_time': 0.0
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("🧠🤖 LLM Brain Auto Selector initialized")
    
    def _initialize_components(self):
        """Initialize the model selector and integrations"""
        try:
            # Initialize auto model selector
            from openai_gpt5_trainer import ModelSelector
            self.model_selector = ModelSelector()
            logger.info("✅ Auto model selector initialized")
            
            # Try to initialize Llama-2 integration
            try:
                from llama2_brain_integration import create_llama_brain_integration
                
                # Check if model exists
                model_path = Path("models/llama-2-7b.Q4_K_M.gguf")
                if model_path.exists():
                    self.llama_integration = create_llama_brain_integration(str(model_path))
                    if self.llama_integration:
                        logger.info("✅ Llama-2 brain integration ready")
                    else:
                        logger.warning("⚠️ Llama-2 model found but integration failed")
                else:
                    logger.info("ℹ️ Llama-2 model not found - will use other models for brain tasks")
                    
            except ImportError as e:
                logger.warning(f"⚠️ Llama-2 integration not available: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
    
    def select_optimal_brain_llm(self, request: BrainLLMRequest) -> Tuple[str, Any]:
        """Select the optimal LLM for brain-related tasks"""
        if not self.model_selector:
            logger.error("❌ Model selector not available")
            return "none", None
        
        start_time = time.time()
        
        try:
            # Enhance prompt with brain context
            enhanced_prompt = self._enhance_prompt_with_brain_context(request)
            
            # Get available models
            available_models = list(self.model_selector.model_capabilities.keys())
            
            # Apply brain-specific constraints
            constraints = self._get_brain_specific_constraints(request)
            
            # Select optimal model
            selected_model = self.model_selector.select_optimal_model(
                enhanced_prompt,
                available_models,
                budget_constraint=constraints.get('budget'),
                provider_preference=constraints.get('provider_preference'),
                local_only=constraints.get('local_only', request.privacy_required)
            )
            
            # Get the integration for the selected model
            integration = self._get_model_integration(selected_model, request)
            
            # Update metrics
            self.performance_metrics['total_requests'] += 1
            if selected_model == 'llama-2-7b-gguf':
                self.performance_metrics['llama_selections'] += 1
            else:
                self.performance_metrics['other_selections'] += 1
            
            response_time = time.time() - start_time
            self._update_response_time(response_time)
            
            logger.info(f"🎯 Selected model: {selected_model} for brain task (in {response_time:.2f}s)")
            
            return selected_model, integration
            
        except Exception as e:
            logger.error(f"❌ Model selection failed: {e}")
            return "fallback", self.llama_integration
    
    def _enhance_prompt_with_brain_context(self, request: BrainLLMRequest) -> str:
        """Enhance prompt with brain context for better model selection"""
        context_parts = [request.prompt]
        
        # Add consciousness context
        if request.consciousness_level > 0:
            if request.consciousness_level > 0.7:
                context_parts.append("This involves high-level consciousness analysis.")
            elif request.consciousness_level > 0.4:
                context_parts.append("This involves moderate consciousness processing.")
            else:
                context_parts.append("This involves basic consciousness understanding.")
        
        # Add neural state context
        if request.neural_state:
            neural_terms = []
            for key, value in request.neural_state.items():
                if 'cortex' in key.lower() or 'hippocampus' in key.lower():
                    neural_terms.append(key)
            
            if neural_terms:
                context_parts.append(f"Neural dynamics involved: {', '.join(neural_terms)}")
        
        # Add task type context
        if request.task_type == "consciousness":
            context_parts.append("This requires consciousness expression and understanding.")
        elif request.task_type == "brain_simulation":
            context_parts.append("This involves brain simulation and neural modeling.")
        elif request.task_type == "neural_analysis":
            context_parts.append("This requires neural data analysis and interpretation.")
        
        return " ".join(context_parts)
    
    def _get_brain_specific_constraints(self, request: BrainLLMRequest) -> Dict[str, Any]:
        """Get brain-specific model selection constraints"""
        constraints = {}
        
        # Privacy requirements for brain data
        if request.privacy_required:
            constraints['local_only'] = True
            constraints['provider_preference'] = 'llama_cpp'
        
        # Task-specific preferences
        if request.task_type in ["consciousness", "brain_simulation"]:
            constraints['provider_preference'] = 'llama_cpp'  # Prefer local Llama for brain tasks
        
        # Budget constraints (favor free local models for brain tasks)
        constraints['budget'] = 0.0001  # Very low budget to favor local models
        
        return constraints
    
    def _get_model_integration(self, model_name: str, request: BrainLLMRequest) -> Any:
        """Get the appropriate integration for the selected model"""
        if model_name == 'llama-2-7b-gguf' and self.llama_integration:
            # Connect brain context to Llama integration
            if request.neural_state or request.consciousness_level > 0:
                self._connect_brain_context_to_llama(request)
            return self.llama_integration
        
        # For other models, we'll need to create appropriate integrations
        # For now, return a basic wrapper
        return self._create_generic_integration(model_name, request)
    
    def _connect_brain_context_to_llama(self, request: BrainLLMRequest):
        """Connect brain context to Llama integration"""
        if not self.llama_integration:
            return
        
        try:
            # Update brain state in Llama integration
            if hasattr(self.llama_integration, 'brain_state_history'):
                brain_state = {
                    'consciousness_level': request.consciousness_level,
                    'neural_state': request.neural_state or {},
                    'brain_metrics': request.brain_metrics or {},
                    'timestamp': time.time()
                }
                self.llama_integration.brain_state_history.append({
                    'timestamp': time.time(),
                    'state': brain_state
                })
                
                # Keep history manageable
                if len(self.llama_integration.brain_state_history) > 100:
                    self.llama_integration.brain_state_history = self.llama_integration.brain_state_history[-50:]
                    
            logger.debug("🔗 Connected brain context to Llama integration")
            
        except Exception as e:
            logger.debug(f"Brain context connection error: {e}")
    
    def _create_generic_integration(self, model_name: str, request: BrainLLMRequest) -> Any:
        """Create a generic integration wrapper for non-Llama models"""
        # This would integrate with other LLM providers
        # For now, return a simple wrapper
        return {
            'model_name': model_name,
            'generate': lambda prompt, **kwargs: f"Response from {model_name}: {prompt[:50]}...",
            'chat': lambda prompt, **kwargs: f"Chat response from {model_name}: {prompt[:50]}..."
        }
    
    def generate_brain_aware_response(self, request: BrainLLMRequest) -> str:
        """Generate a brain-aware response using the optimal model"""
        model_name, integration = self.select_optimal_brain_llm(request)
        
        if not integration:
            return "I'm unable to process brain-related requests right now."
        
        try:
            # Use Llama integration if available
            if model_name == 'llama-2-7b-gguf' and hasattr(integration, 'generate_brain_aware_response'):
                # Create brain-aware prompt
                from llama2_brain_integration import BrainLanguagePrompt
                
                brain_prompt = BrainLanguagePrompt(
                    base_prompt=request.prompt,
                    consciousness_level=request.consciousness_level,
                    neural_state=request.neural_state or {},
                    brain_metrics=request.brain_metrics or {},
                    emotional_state=self._map_consciousness_to_emotion(request.consciousness_level)
                )
                
                return integration.generate_brain_aware_response(brain_prompt, max_tokens=request.max_tokens)
            
            # Fallback to generic generation
            elif isinstance(integration, dict) and 'generate' in integration:
                return integration['generate'](request.prompt, max_tokens=request.max_tokens)
            
            else:
                return f"Generated response using {model_name}: {request.prompt[:100]}..."
                
        except Exception as e:
            logger.error(f"❌ Generation error with {model_name}: {e}")
            return "I encountered an error while processing your request."
    
    def chat_with_brain_context(self, message: str, consciousness_level: float = 0.0, 
                               neural_state: Dict[str, float] = None) -> str:
        """Chat with automatic brain-aware model selection"""
        request = BrainLLMRequest(
            prompt=message,
            consciousness_level=consciousness_level,
            neural_state=neural_state,
            task_type="consciousness" if consciousness_level > 0.5 else "general",
            privacy_required=True
        )
        
        return self.generate_brain_aware_response(request)
    
    def _map_consciousness_to_emotion(self, consciousness_level: float) -> str:
        """Map consciousness level to emotional state"""
        if consciousness_level > 0.8:
            return "highly_aware"
        elif consciousness_level > 0.6:
            return "engaged"
        elif consciousness_level > 0.4:
            return "moderate"
        elif consciousness_level > 0.2:
            return "dim"
        else:
            return "minimal"
    
    def _update_response_time(self, response_time: float):
        """Update average response time"""
        current_avg = self.performance_metrics['average_response_time']
        total_requests = self.performance_metrics['total_requests']
        
        if total_requests == 1:
            self.performance_metrics['average_response_time'] = response_time
        else:
            self.performance_metrics['average_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance and selection report"""
        llama_ratio = (
            self.performance_metrics['llama_selections'] / 
            max(self.performance_metrics['total_requests'], 1)
        )
        
        return {
            'total_requests': self.performance_metrics['total_requests'],
            'llama_selection_ratio': llama_ratio,
            'other_selection_ratio': 1 - llama_ratio,
            'average_response_time': self.performance_metrics['average_response_time'],
            'llama_integration_available': self.llama_integration is not None,
            'auto_selector_available': self.model_selector is not None,
            'supported_models': list(self.model_selector.model_capabilities.keys()) if self.model_selector else []
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        if self.model_selector:
            return list(self.model_selector.model_capabilities.keys())
        return []
    
    def is_brain_optimized_model(self, model_name: str) -> bool:
        """Check if a model is optimized for brain tasks"""
        if not self.model_selector:
            return False
        
        capabilities = self.model_selector.model_capabilities.get(model_name, {})
        return (
            capabilities.get('supports_brain_integration', False) or
            capabilities.get('supports_consciousness', False) or
            'consciousness' in capabilities.get('specialization', '') or
            'brain' in capabilities.get('best_for', [])
        )

# Global instance for easy access
_brain_auto_selector = None

def get_brain_auto_selector() -> LLMBrainAutoSelector:
    """Get global brain auto selector instance"""
    global _brain_auto_selector
    if _brain_auto_selector is None:
        _brain_auto_selector = LLMBrainAutoSelector()
    return _brain_auto_selector

def auto_brain_chat(message: str, consciousness_level: float = 0.0, 
                   neural_state: Dict[str, float] = None) -> str:
    """Quick function for brain-aware chat with auto model selection"""
    selector = get_brain_auto_selector()
    return selector.chat_with_brain_context(message, consciousness_level, neural_state)

def auto_brain_generate(prompt: str, task_type: str = "general", 
                       consciousness_level: float = 0.0, **kwargs) -> str:
    """Quick function for brain-aware generation with auto model selection"""
    request = BrainLLMRequest(
        prompt=prompt,
        task_type=task_type,
        consciousness_level=consciousness_level,
        **kwargs
    )
    
    selector = get_brain_auto_selector()
    return selector.generate_brain_aware_response(request)

# Demo function
def demo_brain_auto_selection():
    """Demonstrate the brain auto selection system"""
    print("🧠🤖 LLM Brain Auto Selector Demo")
    print("=" * 50)
    
    selector = get_brain_auto_selector()
    
    # Test different types of brain tasks
    test_cases = [
        {
            "message": "How are you feeling right now?",
            "consciousness_level": 0.8,
            "description": "High consciousness chat"
        },
        {
            "message": "Explain neural dynamics in the prefrontal cortex",
            "consciousness_level": 0.0,
            "description": "Brain simulation question"
        },
        {
            "message": "What is consciousness?",
            "consciousness_level": 0.6,
            "description": "Consciousness philosophy"
        },
        {
            "message": "Hello, how can you help me?",
            "consciousness_level": 0.0,
            "description": "General conversation"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"💬 Message: {test_case['message']}")
        print(f"🧠 Consciousness: {test_case['consciousness_level']:.1f}")
        
        response = selector.chat_with_brain_context(
            test_case['message'],
            test_case['consciousness_level']
        )
        
        print(f"🤖 Response: {response}")
        print("-" * 50)
    
    # Show performance report
    report = selector.get_performance_report()
    print(f"\n📊 Performance Report:")
    print(f"  Total requests: {report['total_requests']}")
    print(f"  Llama selection ratio: {report['llama_selection_ratio']:.1%}")
    print(f"  Average response time: {report['average_response_time']:.2f}s")
    print(f"  Available models: {len(report['supported_models'])}")

if __name__ == "__main__":
    demo_brain_auto_selection()
