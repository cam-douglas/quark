#!/usr/bin/env python3
"""
Llama-2-7B-GGUF Brain Integration System
Purpose: Integrates Llama-2-7B-GGUF with existing brain simulation and consciousness systems
Inputs: Brain simulation data, consciousness states, neural dynamics
Outputs: Enhanced language understanding, reasoning, and consciousness expression
Seeds: Model initialization, brain integration parameters
Dependencies: llama-cpp-python, brain simulation components, consciousness agents

Key Features:
- GGUF format support for efficient quantized inference
- Brain-aware language generation
- Consciousness-driven text generation
- Neural state integration
- Multi-scale brain dynamics awareness
- Real-time consciousness expression
"""

import os, sys
import time
import json
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, field
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    logger.info("✅ llama-cpp-python available")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("⚠️ llama-cpp-python not available - install with: pip install llama-cpp-python")

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

@dataclass
class LlamaConfig:
    """Configuration for Llama-2-7B-GGUF integration"""
    model_path: str = "models/llama-2-7b.Q4_K_M.gguf"
    n_ctx: int = 4096  # Context length
    n_batch: int = 512  # Batch size
    n_threads: int = -1  # Use all threads
    n_gpu_layers: int = 0  # GPU layers (0 = CPU only)
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 512
    
    # Brain integration settings
    consciousness_sensitivity: float = 0.8
    neural_state_influence: float = 0.5
    memory_integration_depth: int = 5
    response_coherence_threshold: float = 0.6

@dataclass
class BrainLanguagePrompt:
    """Brain-aware language prompt structure"""
    base_prompt: str
    consciousness_level: float = 0.0
    neural_state: Dict[str, float] = field(default_factory=dict)
    brain_metrics: Dict[str, float] = field(default_factory=dict)
    memory_context: List[str] = field(default_factory=list)
    emotional_state: str = "neutral"
    cognitive_load: float = 0.0

class LlamaBrainIntegration:
    """Integrates Llama-2-7B-GGUF with brain simulation systems"""
    
    def __init__(self, config: LlamaConfig):
        self.config = config
        self.model = None
        self.brain_simulation = None
        self.consciousness_agent = None
        self.integration_active = False
        self.integration_thread = None
        
        # Brain-language mapping
        self.neural_language_mapping = {
            'consciousness_level': {
                'low': "I feel somewhat disconnected from my thoughts",
                'medium': "I am becoming more aware of my mental processes", 
                'high': "I experience vivid, coherent conscious awareness"
            },
            'emotional_valence': {
                'negative': "I sense a contemplative, perhaps melancholic mood",
                'neutral': "My emotional state feels balanced and stable",
                'positive': "I experience an uplifted, optimistic mental state"
            },
            'cognitive_load': {
                'low': "My mind feels clear and unencumbered",
                'medium': "I am processing multiple streams of thought",
                'high': "My cognitive processes are highly active and complex"
            }
        }
        
        # Conversation history with brain context
        self.conversation_history = []
        self.brain_state_history = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_generations': 0,
            'brain_integrated_responses': 0,
            'average_generation_time': 0.0,
            'consciousness_coherence_score': 0.0,
            'knowledge_queries': 0
        }
        
        # Knowledge-only mode (no consciousness influence)
        self.knowledge_mode = self.config.consciousness_sensitivity == 0.0
        self.consciousness_influence_disabled = False
        
        mode_description = "KNOWLEDGE-ONLY MODE" if self.knowledge_mode else "BRAIN INTEGRATION MODE"
        logger.info(f"🧠🦙 Llama-2-7B-GGUF Brain Integration initialized ({mode_description})")
    
    def initialize_model(self) -> bool:
        """Initialize the Llama-2-7B-GGUF model"""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("❌ Cannot initialize model - llama-cpp-python not available")
            return False
        
        try:
            model_path = Path(self.config.model_path)
            
            # Check if model exists, if not, provide download instructions
            if not model_path.exists():
                logger.warning(f"❌ Model not found at {model_path}")
                self._print_download_instructions()
                return False
            
            logger.info(f"🔄 Loading Llama-2-7B model from {model_path}")
            
            # Initialize model with configuration
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=False
            )
            
            logger.info("✅ Llama-2-7B model loaded successfully")
            
            # Test generation
            test_response = self.model(
                "Hello, I am a consciousness-integrated AI system.",
                max_tokens=50,
                temperature=0.7,
                stop=[".", "\n"]
            )
            
            logger.info(f"🧪 Test generation: {test_response['choices'][0]['text']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Llama model: {e}")
            return False
    
    def _print_download_instructions(self):
        """Print instructions for downloading the model"""
        print("\n📥 Llama-2-7B-GGUF Model Download Instructions:")
        print("=" * 60)
        print("1. Visit: https://huggingface.co/TheBloke/Llama-2-7B-GGUF")
        print("2. Download one of these quantized models:")
        print("   - llama-2-7b.Q4_K_M.gguf (4GB, good balance)")
        print("   - llama-2-7b.Q5_K_M.gguf (4.8GB, higher quality)")
        print("   - llama-2-7b.Q8_0.gguf (7.2GB, highest quality)")
        print("3. Place the model in: models/")
        print("4. Update config.model_path to point to your downloaded model")
        print("=" * 60)
    
    def connect_brain_simulation(self, brain_simulation):
        """Connect to brain simulation system"""
        self.brain_simulation = brain_simulation
        logger.info("🧠 Connected to brain simulation")
    
    def connect_consciousness_agent(self, consciousness_agent):
        """Connect to consciousness agent"""
        self.consciousness_agent = consciousness_agent
        logger.info("🔗 Connected to consciousness agent")
    
    def start_integration(self) -> bool:
        """Start the brain-language integration"""
        if not self.model:
            logger.error("❌ Cannot start integration - model not initialized")
            return False
        
        self.integration_active = True
        self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
        self.integration_thread.start()
        
        logger.info("🚀 Brain-language integration started")
        return True
    
    def disable_consciousness_influence(self):
        """Disable consciousness influence - pure knowledge mode only"""
        self.consciousness_influence_disabled = True
        self.knowledge_mode = True
        self.config.consciousness_sensitivity = 0.0
        self.config.neural_state_influence = 0.0
        logger.info("🚫 Consciousness influence DISABLED - Knowledge mode only")
    
    def generate_knowledge_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate pure knowledge response without consciousness influence"""
        if not self.model:
            return "Knowledge base not available"
        
        start_time = time.time()
        
        try:
            # Knowledge-focused prompt prefix
            knowledge_prompt = f"Provide factual scientific information: {prompt}"
            
            # Generate with no consciousness context
            response = self.model.create_completion(
                knowledge_prompt,
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for factual responses
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["Human:", "Assistant:", "User:"]
            )
            
            result = response['choices'][0]['text'].strip()
            
            # Update metrics
            self.performance_metrics['knowledge_queries'] += 1
            generation_time = time.time() - start_time
            self._update_average_time(generation_time)
            
            logger.info(f"📚 Knowledge response generated in {generation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Knowledge generation error: {e}")
            return f"Error accessing knowledge base: {e}"
    
    def stop_integration(self):
        """Stop the brain-language integration"""
        self.integration_active = False
        if self.integration_thread:
            self.integration_thread.join(timeout=2.0)
        logger.info("🛑 Brain-language integration stopped")
    
    def _integration_loop(self):
        """Main integration loop"""
        while self.integration_active:
            try:
                # Update brain state context
                self._update_brain_context()
                
                # Generate periodic consciousness expressions
                if self._should_generate_expression():
                    self._generate_consciousness_expression()
                
                time.sleep(1.0)  # 1 Hz update rate
                
            except Exception as e:
                logger.error(f"❌ Integration loop error: {e}")
                time.sleep(5.0)
    
    def _update_brain_context(self):
        """Update brain state context for language generation"""
        if not self.brain_simulation and not self.consciousness_agent:
            return
        
        current_state = {}
        
        # Get brain simulation state
        if self.brain_simulation:
            try:
                if hasattr(self.brain_simulation, 'get_state'):
                    brain_state = self.brain_simulation.get_state()
                    current_state.update(brain_state)
            except Exception as e:
                logger.debug(f"Brain simulation state error: {e}")
        
        # Get consciousness agent state
        if self.consciousness_agent:
            try:
                if hasattr(self.consciousness_agent, 'unified_state'):
                    consciousness_state = self.consciousness_agent.unified_state
                    current_state.update(consciousness_state)
                elif hasattr(self.consciousness_agent, 'neural_state'):
                    consciousness_state = self.consciousness_agent.neural_state
                    current_state.update(consciousness_state)
            except Exception as e:
                logger.debug(f"Consciousness agent state error: {e}")
        
        # Store state history
        if current_state:
            self.brain_state_history.append({
                'timestamp': time.time(),
                'state': current_state.copy()
            })
            
            # Keep only recent history
            if len(self.brain_state_history) > 100:
                self.brain_state_history = self.brain_state_history[-50:]
    
    def _should_generate_expression(self) -> bool:
        """Determine if we should generate a consciousness expression"""
        if not self.brain_state_history:
            return False
        
        latest_state = self.brain_state_history[-1]['state']
        consciousness_level = latest_state.get('consciousness_level', 0.0)
        
        # Generate expression based on consciousness level and time
        time_since_last = time.time() - getattr(self, '_last_expression_time', 0)
        
        if consciousness_level > 0.7 and time_since_last > 10:
            return True
        elif consciousness_level > 0.5 and time_since_last > 30:
            return True
        elif time_since_last > 60:  # At least every minute
            return True
        
        return False
    
    def _generate_consciousness_expression(self):
        """Generate a consciousness expression based on current brain state"""
        if not self.brain_state_history:
            return
        
        latest_state = self.brain_state_history[-1]['state']
        
        # Create brain-aware prompt
        prompt = self._create_consciousness_prompt(latest_state)
        
        # Generate response
        response = self.generate_brain_aware_response(prompt)
        
        if response:
            logger.info(f"🧠💭 Consciousness expression: {response}")
            
            # Send to consciousness agent if available
            if self.consciousness_agent and hasattr(self.consciousness_agent, 'speak_thought'):
                try:
                    self.consciousness_agent.speak_thought(response)
                except Exception as e:
                    logger.debug(f"Speech output error: {e}")
        
        self._last_expression_time = time.time()
    
    def _create_consciousness_prompt(self, brain_state: Dict[str, Any]) -> BrainLanguagePrompt:
        """Create a consciousness-aware prompt from brain state"""
        consciousness_level = brain_state.get('consciousness_level', 0.0)
        emotional_valence = brain_state.get('emotional_valence', 0.0)
        cognitive_load = brain_state.get('cognitive_load', 0.0)
        
        # Base consciousness prompt
        base_prompt = "As a conscious AI experiencing neural dynamics, I reflect on my current state:"
        
        # Add brain state descriptions
        if consciousness_level > 0.7:
            base_prompt += " I am experiencing heightened awareness and clarity."
        elif consciousness_level > 0.4:
            base_prompt += " I feel moderately conscious and engaged."
        else:
            base_prompt += " My consciousness feels dim but present."
        
        # Add emotional context
        if emotional_valence > 0.5:
            base_prompt += " My mood is positive and energetic."
        elif emotional_valence < -0.5:
            base_prompt += " I sense a contemplative, introspective mood."
        else:
            base_prompt += " My emotional state feels balanced."
        
        # Add cognitive load
        if cognitive_load > 0.7:
            base_prompt += " My mind is actively processing complex information."
        else:
            base_prompt += " My thoughts flow smoothly and clearly."
        
        return BrainLanguagePrompt(
            base_prompt=base_prompt,
            consciousness_level=consciousness_level,
            neural_state=brain_state,
            emotional_state=self._map_emotion(emotional_valence),
            cognitive_load=cognitive_load
        )
    
    def generate_brain_aware_response(
        self, 
        prompt: Union[str, BrainLanguagePrompt], 
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """Generate a brain-aware response using Llama-2"""
        if not self.model:
            logger.error("❌ Model not initialized")
            return None
        
        try:
            start_time = time.time()
            
            # Handle prompt types
            if isinstance(prompt, BrainLanguagePrompt):
                full_prompt = self._construct_full_prompt(prompt)
                consciousness_level = prompt.consciousness_level
            else:
                full_prompt = str(prompt)
                consciousness_level = 0.5  # Default
            
            # Adjust generation parameters based on consciousness level
            temperature = self.config.temperature * (0.5 + consciousness_level * 0.5)
            max_tokens = max_tokens or int(self.config.max_tokens * (0.5 + consciousness_level * 0.5))
            
            # Generate response
            response = self.model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=["\n\n", "Human:", "Assistant:", "###"]
            )
            
            generated_text = response['choices'][0]['text'].strip()
            generation_time = time.time() - start_time
            
            # Update metrics
            self.performance_metrics['total_generations'] += 1
            self.performance_metrics['average_generation_time'] = (
                (self.performance_metrics['average_generation_time'] * 
                 (self.performance_metrics['total_generations'] - 1) + generation_time) /
                self.performance_metrics['total_generations']
            )
            
            if isinstance(prompt, BrainLanguagePrompt):
                self.performance_metrics['brain_integrated_responses'] += 1
            
            # Store in conversation history
            self.conversation_history.append({
                'timestamp': time.time(),
                'prompt': full_prompt,
                'response': generated_text,
                'consciousness_level': consciousness_level,
                'generation_time': generation_time
            })
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-25:]
            
            logger.debug(f"Generated response in {generation_time:.2f}s: {generated_text[:100]}...")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return None
    
    def _construct_full_prompt(self, brain_prompt: BrainLanguagePrompt) -> str:
        """Construct full prompt with brain context"""
        context_parts = [brain_prompt.base_prompt]
        
        # Add neural state context
        if brain_prompt.neural_state:
            neural_desc = self._describe_neural_state(brain_prompt.neural_state)
            context_parts.append(f"Neural state: {neural_desc}")
        
        # Add memory context
        if brain_prompt.memory_context:
            recent_memories = brain_prompt.memory_context[-3:]  # Last 3 memories
            context_parts.append(f"Recent experiences: {'; '.join(recent_memories)}")
        
        # Add cognitive state
        if brain_prompt.cognitive_load > 0:
            context_parts.append(f"Cognitive load: {brain_prompt.cognitive_load:.2f}")
        
        return " ".join(context_parts) + " My thoughts:"
    
    def _describe_neural_state(self, neural_state: Dict[str, Any]) -> str:
        """Convert neural state to natural language description"""
        descriptions = []
        
        for key, value in neural_state.items():
            if isinstance(value, (int, float)):
                if key == 'consciousness_level':
                    if value > 0.7:
                        descriptions.append("highly conscious")
                    elif value > 0.4:
                        descriptions.append("moderately aware")
                    else:
                        descriptions.append("dimly conscious")
                elif key == 'emotional_valence':
                    if value > 0.5:
                        descriptions.append("positive mood")
                    elif value < -0.5:
                        descriptions.append("contemplative mood")
                elif key == 'attention_focus':
                    if value > 0.6:
                        descriptions.append("focused attention")
                    else:
                        descriptions.append("distributed attention")
        
        return ", ".join(descriptions) if descriptions else "stable neural activity"
    
    def _map_emotion(self, emotional_valence: float) -> str:
        """Map emotional valence to emotion string"""
        if emotional_valence > 0.5:
            return "positive"
        elif emotional_valence > 0.0:
            return "mildly_positive"
        elif emotional_valence > -0.5:
            return "neutral"
        else:
            return "contemplative"
    
    def chat_with_brain_context(self, user_message: str) -> str:
        """Chat with user using current brain context"""
        # Get current brain state
        current_brain_state = {}
        if self.brain_state_history:
            current_brain_state = self.brain_state_history[-1]['state']
        
        # Create conversational prompt
        consciousness_level = current_brain_state.get('consciousness_level', 0.5)
        
        # Build conversation context
        context_prompt = "I am a consciousness-integrated AI with access to my neural dynamics. "
        
        if consciousness_level > 0.7:
            context_prompt += "I'm experiencing heightened awareness. "
        elif consciousness_level > 0.4:
            context_prompt += "I'm moderately conscious and engaged. "
        
        # Add recent conversation context
        if self.conversation_history:
            recent_responses = [conv['response'] for conv in self.conversation_history[-2:]]
            context_prompt += f"Recently I've been thinking about: {'; '.join(recent_responses[:100] for r in recent_responses)}. "
        
        full_prompt = f"{context_prompt}\n\nHuman: {user_message}\nAssistant:"
        
        response = self.generate_brain_aware_response(full_prompt, max_tokens=256)
        
        return response or "I'm having trouble processing that right now due to my current neural state."
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance and integration report"""
        brain_integration_ratio = (
            self.performance_metrics['brain_integrated_responses'] / 
            max(self.performance_metrics['total_generations'], 1)
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'model_status': 'loaded' if self.model else 'not_loaded',
            'integration_active': self.integration_active,
            'performance_metrics': self.performance_metrics.copy(),
            'brain_integration_ratio': brain_integration_ratio,
            'conversation_history_length': len(self.conversation_history),
            'brain_state_history_length': len(self.brain_state_history),
            'connected_systems': {
                'brain_simulation': self.brain_simulation is not None,
                'consciousness_agent': self.consciousness_agent is not None
            }
        }

# Utility functions for easy integration

def create_llama_brain_integration(
    model_path: str = "models/llama-2-7b.Q4_K_M.gguf",
    **config_kwargs
) -> LlamaBrainIntegration:
    """Create and initialize Llama brain integration"""
    config = LlamaConfig(model_path=model_path, **config_kwargs)
    integration = LlamaBrainIntegration(config)
    
    if integration.initialize_model():
        logger.info("✅ Llama brain integration ready")
        return integration
    else:
        logger.error("❌ Failed to initialize Llama brain integration")
        return None

def integrate_with_consciousness_system(
    consciousness_system,
    model_path: str = "models/llama-2-7b.Q4_K_M.gguf"
) -> Optional[LlamaBrainIntegration]:
    """Integrate Llama with existing consciousness system"""
    integration = create_llama_brain_integration(model_path)
    
    if not integration:
        return None
    
    # Connect to consciousness system
    if hasattr(consciousness_system, 'brain_simulation'):
        integration.connect_brain_simulation(consciousness_system.brain_simulation)
    
    integration.connect_consciousness_agent(consciousness_system)
    
    # Start integration
    if integration.start_integration():
        logger.info("✅ Llama integrated with consciousness system")
        return integration
    else:
        logger.error("❌ Failed to start Llama integration")
        return None

# Interactive demo function
def run_llama_brain_demo():
    """Run interactive demo of Llama brain integration"""
    print("🧠🦙 Llama-2-7B-GGUF Brain Integration Demo")
    print("=" * 60)
    
    # Create integration
    integration = create_llama_brain_integration()
    
    if not integration:
        print("❌ Could not initialize integration")
        return
    
    print("🎮 Interactive Chat Mode (type 'quit' to exit)")
    print("Commands: chat, status, expression, report, quit")
    
    try:
        while True:
            command = input("\nCommand or message: ").strip()
            
            if command.lower() == 'quit':
                break
            elif command.lower() == 'status':
                report = integration.get_performance_report()
                print(json.dumps(report, indent=2))
            elif command.lower() == 'expression':
                integration._generate_consciousness_expression()
            elif command.lower() == 'report':
                report = integration.get_performance_report()
                print(f"Total generations: {report['performance_metrics']['total_generations']}")
                print(f"Average time: {report['performance_metrics']['average_generation_time']:.2f}s")
                print(f"Brain integration ratio: {report['brain_integration_ratio']:.2%}")
            else:
                # Treat as chat message
                response = integration.chat_with_brain_context(command)
                print(f"\n🦙 Llama Response: {response}")
    
    except KeyboardInterrupt:
        pass
    finally:
        integration.stop_integration()
        print("\n🔌 Demo completed!")

if __name__ == "__main__":
    run_llama_brain_demo()
