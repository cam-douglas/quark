#!/usr/bin/env python3
"""
Brain LLM Router
Purpose: Simple router for brain-aware LLM selection and integration
Inputs: Task prompts, brain states, consciousness levels
Outputs: Optimal brain-aware responses
Seeds: Model routing logic, brain integration parameters
Dependencies: llama2_brain_integration, existing consciousness systems

Key Features:
- Brain-task aware routing
- Consciousness-driven model selection
- Seamless Llama-2 integration
- Fallback to other models
- Simple, reliable architecture
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
class BrainTaskRequest:
    """Request for brain-aware task processing"""
    prompt: str
    consciousness_level: float = 0.0
    neural_state: Dict[str, float] = None
    brain_metrics: Dict[str, float] = None
    task_type: str = "general"  # general, consciousness, brain_simulation, neural_analysis
    max_tokens: int = 512
    temperature: float = 0.7

class BrainLLMRouter:
    """Smart router for brain-aware LLM selection and processing"""
    
    def __init__(self):
        self.llama_integration = None
        self.consciousness_systems = {}
        self.routing_stats = {
            'total_requests': 0,
            'llama_routed': 0,
            'fallback_routed': 0,
            'brain_tasks': 0,
            'consciousness_tasks': 0,
            'general_tasks': 0
        }
        
        # Initialize available integrations
        self._initialize_integrations()
        
        logger.info("🧠🔀 Brain LLM Router initialized")
    
    def _initialize_integrations(self):
        """Initialize available LLM integrations"""
        # Initialize Llama-2 brain integration
        self._init_llama_integration()
        
        # Initialize consciousness systems
        self._init_consciousness_systems()
    
    def _init_llama_integration(self):
        """Initialize Llama-2 brain integration"""
        try:
            from llama2_brain_integration import create_llama_brain_integration
            
            # Check if model exists
            model_path = Path("models/llama-2-7b.Q4_K_M.gguf")
            if model_path.exists():
                logger.info(f"🦙 Found Llama-2 model: {model_path}")
                self.llama_integration = create_llama_brain_integration(str(model_path))
                
                if self.llama_integration:
                    logger.info("✅ Llama-2 brain integration ready")
                    return True
                else:
                    logger.warning("⚠️ Llama-2 model found but integration failed")
            else:
                logger.info("ℹ️ Llama-2 model not found - using fallback responses")
                
        except ImportError as e:
            logger.warning(f"⚠️ Llama-2 integration not available: {e}")
        except Exception as e:
            logger.error(f"❌ Error initializing Llama-2: {e}")
        
        return False
    
    def _init_consciousness_systems(self):
        """Initialize consciousness system integrations"""
        consciousness_modules = [
            ("consciousness_agent.cloud_integrated_consciousness", "CloudIntegratedConsciousness"),
            ("consciousness_agent.enhanced_consciousness_simulator", "EnhancedConsciousnessSimulator"),
            ("consciousness_agent.llama2_consciousness_bridge", "Llama2ConsciousnessBridge")
        ]
        
        for module_name, class_name in consciousness_modules:
            try:
                # Add consciousness agent path
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / "database"))
                
                module = __import__(module_name, fromlist=[class_name])
                consciousness_class = getattr(module, class_name)
                
                self.consciousness_systems[class_name] = consciousness_class
                logger.info(f"🔍 Found consciousness system: {class_name}")
                
            except ImportError:
                logger.debug(f"⚠️ {class_name} not available")
            except Exception as e:
                logger.debug(f"Error loading {class_name}: {e}")
    
    def route_brain_task(self, request: BrainTaskRequest) -> Tuple[str, Any]:
        """Route brain task to optimal LLM integration"""
        self.routing_stats['total_requests'] += 1
        
        # Analyze task requirements
        task_analysis = self._analyze_task_requirements(request)
        
        # Determine optimal routing
        routing_decision = self._make_routing_decision(task_analysis, request)
        
        # Update statistics
        self._update_routing_stats(routing_decision, request)
        
        # Get integration
        integration = self._get_integration(routing_decision)
        
        logger.info(f"🔀 Routed '{request.task_type}' task to: {routing_decision}")
        
        return routing_decision, integration
    
    def _analyze_task_requirements(self, request: BrainTaskRequest) -> Dict[str, Any]:
        """Analyze task requirements for optimal routing"""
        prompt_lower = request.prompt.lower()
        
        analysis = {
            'is_brain_related': any(term in prompt_lower for term in [
                'brain', 'neural', 'neuron', 'cortex', 'hippocampus', 'synapse',
                'prefrontal', 'amygdala', 'thalamus', 'cerebellum', 'dendrite'
            ]),
            'is_consciousness_related': any(term in prompt_lower for term in [
                'consciousness', 'awareness', 'conscious', 'experience', 'subjective',
                'qualia', 'introspection', 'self-aware', 'phenomenal'
            ]),
            'requires_brain_integration': request.consciousness_level > 0.3 or bool(request.neural_state),
            'complexity_level': self._estimate_complexity(request.prompt),
            'privacy_sensitive': True,  # Default to privacy for brain tasks
            'token_count': len(request.prompt.split()) * 1.3  # Rough estimate
        }
        
        return analysis
    
    def _estimate_complexity(self, prompt: str) -> str:
        """Estimate prompt complexity"""
        complex_indicators = [
            'analyze', 'explain', 'compare', 'evaluate', 'synthesize',
            'theory', 'mechanism', 'dynamics', 'interaction', 'integration'
        ]
        
        complexity_score = sum(1 for indicator in complex_indicators if indicator in prompt.lower())
        
        if complexity_score >= 3:
            return 'high'
        elif complexity_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _make_routing_decision(self, analysis: Dict[str, Any], request: BrainTaskRequest) -> str:
        """Make routing decision based on analysis"""
        # Prefer Llama-2 for brain and consciousness tasks
        if self.llama_integration and (
            analysis['is_brain_related'] or 
            analysis['is_consciousness_related'] or 
            analysis['requires_brain_integration'] or
            request.task_type in ['consciousness', 'brain_simulation', 'neural_analysis']
        ):
            return 'llama-2-brain'
        
        # Fallback routing
        if self.llama_integration:
            return 'llama-2-general'
        else:
            return 'fallback'
    
    def _get_integration(self, routing_decision: str) -> Any:
        """Get integration based on routing decision"""
        if routing_decision in ['llama-2-brain', 'llama-2-general']:
            return self.llama_integration
        else:
            return self._create_fallback_integration()
    
    def _create_fallback_integration(self) -> Dict[str, Any]:
        """Create fallback integration when Llama-2 is not available"""
        return {
            'type': 'fallback',
            'generate': self._fallback_generate,
            'chat': self._fallback_chat,
            'supports_brain': False
        }
    
    def _fallback_generate(self, prompt: str, **kwargs) -> str:
        """Fallback generation method"""
        # Simple rule-based responses for common brain/consciousness queries
        prompt_lower = prompt.lower()
        
        if 'consciousness' in prompt_lower:
            return "Consciousness is a complex phenomenon involving awareness, subjective experience, and the ability to process and integrate information about both internal states and the external environment."
        
        elif 'brain' in prompt_lower or 'neural' in prompt_lower:
            return "The brain is a complex network of interconnected neurons that process information through electrical and chemical signals, enabling cognition, perception, and behavior."
        
        elif 'feeling' in prompt_lower or 'emotion' in prompt_lower:
            return "As an AI, I process information and can simulate understanding of emotions, but my 'feelings' are computational processes rather than subjective experiences."
        
        else:
            return f"I understand you're asking about: {prompt[:100]}... However, I have limited capabilities without the full brain integration system."
    
    def _fallback_chat(self, message: str, **kwargs) -> str:
        """Fallback chat method"""
        return self._fallback_generate(message, **kwargs)
    
    def _update_routing_stats(self, routing_decision: str, request: BrainTaskRequest):
        """Update routing statistics"""
        if routing_decision.startswith('llama-2'):
            self.routing_stats['llama_routed'] += 1
        else:
            self.routing_stats['fallback_routed'] += 1
        
        if request.task_type == 'brain_simulation':
            self.routing_stats['brain_tasks'] += 1
        elif request.task_type == 'consciousness':
            self.routing_stats['consciousness_tasks'] += 1
        else:
            self.routing_stats['general_tasks'] += 1
    
    def process_brain_request(self, request: BrainTaskRequest) -> str:
        """Process brain request with optimal routing"""
        routing_decision, integration = self.route_brain_task(request)
        
        try:
            # Use Llama-2 brain integration if available
            if routing_decision == 'llama-2-brain' and hasattr(integration, 'generate_brain_aware_response'):
                
                # Create brain-aware prompt if we have the integration
                try:
                    from llama2_brain_integration import BrainLanguagePrompt
                    
                    brain_prompt = BrainLanguagePrompt(
                        base_prompt=request.prompt,
                        consciousness_level=request.consciousness_level,
                        neural_state=request.neural_state or {},
                        brain_metrics=request.brain_metrics or {}
                    )
                    
                    return integration.generate_brain_aware_response(
                        brain_prompt, 
                        max_tokens=request.max_tokens
                    )
                
                except ImportError:
                    # Fallback to basic generation
                    return integration.generate_brain_aware_response(
                        request.prompt,
                        max_tokens=request.max_tokens
                    )
            
            # Use Llama-2 general integration
            elif routing_decision == 'llama-2-general' and hasattr(integration, 'chat_with_brain_context'):
                return integration.chat_with_brain_context(request.prompt)
            
            # Use fallback integration
            elif isinstance(integration, dict) and 'generate' in integration:
                return integration['generate'](request.prompt, max_tokens=request.max_tokens)
            
            else:
                return f"Processing with {routing_decision}: {request.prompt[:50]}..."
                
        except Exception as e:
            logger.error(f"❌ Processing error with {routing_decision}: {e}")
            return "I encountered an error while processing your request. Please try again."
    
    def chat_with_brain_awareness(self, message: str, consciousness_level: float = 0.0,
                                 neural_state: Dict[str, float] = None) -> str:
        """Chat with brain awareness and auto-routing"""
        request = BrainTaskRequest(
            prompt=message,
            consciousness_level=consciousness_level,
            neural_state=neural_state,
            task_type="consciousness" if consciousness_level > 0.5 else "general"
        )
        
        return self.process_brain_request(request)
    
    def analyze_neural_state(self, neural_data: Dict[str, float], question: str = "") -> str:
        """Analyze neural state with appropriate model"""
        if not question:
            question = "Analyze the current neural state and describe what it indicates."
        
        request = BrainTaskRequest(
            prompt=question,
            neural_state=neural_data,
            task_type="neural_analysis",
            consciousness_level=neural_data.get('consciousness_level', 0.0)
        )
        
        return self.process_brain_request(request)
    
    def express_consciousness(self, consciousness_level: float, 
                            neural_state: Dict[str, float] = None) -> str:
        """Express consciousness state through optimal model"""
        prompt = f"Express your current consciousness state with level {consciousness_level:.2f}"
        
        request = BrainTaskRequest(
            prompt=prompt,
            consciousness_level=consciousness_level,
            neural_state=neural_state,
            task_type="consciousness"
        )
        
        return self.process_brain_request(request)
    
    def get_routing_report(self) -> Dict[str, Any]:
        """Get routing statistics and status report"""
        total = max(self.routing_stats['total_requests'], 1)
        
        return {
            'routing_statistics': self.routing_stats.copy(),
            'routing_ratios': {
                'llama_ratio': self.routing_stats['llama_routed'] / total,
                'fallback_ratio': self.routing_stats['fallback_routed'] / total,
                'brain_task_ratio': self.routing_stats['brain_tasks'] / total,
                'consciousness_task_ratio': self.routing_stats['consciousness_tasks'] / total
            },
            'system_status': {
                'llama_integration_available': self.llama_integration is not None,
                'consciousness_systems_found': len(self.consciousness_systems),
                'routing_active': True
            },
            'capabilities': {
                'supports_brain_integration': self.llama_integration is not None,
                'supports_consciousness_expression': self.llama_integration is not None,
                'supports_neural_analysis': True,
                'fallback_available': True
            }
        }

# Global router instance
_brain_router = None

def get_brain_router() -> BrainLLMRouter:
    """Get global brain router instance"""
    global _brain_router
    if _brain_router is None:
        _brain_router = BrainLLMRouter()
    return _brain_router

def brain_chat(message: str, consciousness_level: float = 0.0, 
               neural_state: Dict[str, float] = None) -> str:
    """Quick brain-aware chat function"""
    router = get_brain_router()
    return router.chat_with_brain_awareness(message, consciousness_level, neural_state)

def brain_express_consciousness(consciousness_level: float, 
                               neural_state: Dict[str, float] = None) -> str:
    """Quick consciousness expression function"""
    router = get_brain_router()
    return router.express_consciousness(consciousness_level, neural_state)

def brain_analyze_neural_state(neural_data: Dict[str, float], question: str = "") -> str:
    """Quick neural state analysis function"""
    router = get_brain_router()
    return router.analyze_neural_state(neural_data, question)

# Demo function
def demo_brain_router():
    """Demonstrate the brain router system"""
    print("🧠🔀 Brain LLM Router Demo")
    print("=" * 50)
    
    router = get_brain_router()
    
    # Test different types of requests
    test_cases = [
        {
            "message": "How do you feel right now?",
            "consciousness_level": 0.8,
            "description": "High consciousness query"
        },
        {
            "message": "Explain the role of the hippocampus in memory formation",
            "consciousness_level": 0.0,
            "description": "Brain science question"
        },
        {
            "message": "What is the nature of consciousness?",
            "consciousness_level": 0.6,
            "description": "Consciousness philosophy"
        },
        {
            "message": "Hello, can you help me with something?",
            "consciousness_level": 0.0,
            "description": "General conversation"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"💬 Message: {test_case['message']}")
        print(f"🧠 Consciousness: {test_case['consciousness_level']:.1f}")
        
        response = router.chat_with_brain_awareness(
            test_case['message'],
            test_case['consciousness_level']
        )
        
        print(f"🤖 Response: {response}")
        print("-" * 50)
    
    # Test consciousness expression
    print(f"\n💭 Consciousness Expression Test:")
    expression = router.express_consciousness(0.75, {'pfc_activity': 0.8, 'emotional_valence': 0.3})
    print(f"🧠 Expression: {expression}")
    
    # Show routing report
    report = router.get_routing_report()
    print(f"\n📊 Routing Report:")
    print(f"  Total requests: {report['routing_statistics']['total_requests']}")
    print(f"  Llama routing ratio: {report['routing_ratios']['llama_ratio']:.1%}")
    print(f"  Brain task ratio: {report['routing_ratios']['brain_task_ratio']:.1%}")
    print(f"  System available: {report['system_status']['llama_integration_available']}")

if __name__ == "__main__":
    demo_brain_router()
