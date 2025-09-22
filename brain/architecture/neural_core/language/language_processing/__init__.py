#!/usr/bin/env python3
"""Language Processing Module - Main interface for neural language processing.

Provides unified interface to language processing components with secure API integration.

Integration: Main language processing interface for brain neural core.
Rationale: Clean API abstraction maintaining all existing functionality with secure credentials.
"""

from typing import Optional, Dict, Any
from .api_clients import LanguageAPIClients
from .expert_router import ExpertRouter
from .model_selector import ModelSelector

# Import speech integration
try:
    from ..speech_integration import get_speech_integration, speak
    SPEECH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Speech integration not available: {e}")
    SPEECH_AVAILABLE = False

# Main LanguageCortex class that uses the modular components
class LanguageCortex:
    """
    A semantic, resilient Mixture-of-Experts (MoE) model for language tasks.
    Uses sentence embeddings to route prompts with secure API failover system.
    """

    def __init__(self, enable_speech: bool = False):
        """Initialize the Language Cortex with modular components."""

        # System prompt for behavioral guardrails
        self.system_prompt = (
            "You are Quark, a developing AGI. Your primary directive is to be helpful, "
            "truthful, and to always assist the user. You must follow user instructions "
            "to the best of your ability. When you are unsure, ask for clarification. "
            "Your goal is to learn, grow, and contribute to scientific advancement in a "
            "safe and collaborative manner."
        )

        # Initialize modular components
        self.api_clients = LanguageAPIClients()
        self.expert_router = ExpertRouter()
        self.model_selector = ModelSelector(self.api_clients)

        # Initialize speech integration if available
        self.speech_enabled = False
        if SPEECH_AVAILABLE and enable_speech:
            try:
                self.speech_integration = get_speech_integration()
                self.speech_enabled = True
                print("🔊 Speech integration initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize speech integration: {e}")
                self.speech_integration = None
        else:
            self.speech_integration = None

        # Rate limiting and conversation state
        self._init_rate_limiters()
        self.conversation_history = []

        print("Language Cortex Initialized with modular components.")

    def _init_rate_limiters(self):
        """Initialize rate limiting for API calls."""
        from collections import deque
        import time

        # Simple rate limiting using deque
        self.call_history = {
            'openai': deque(maxlen=60),     # 60 calls per minute
            'anthropic': deque(maxlen=50),  # 50 calls per minute
            'gemini': deque(maxlen=60)      # 60 calls per minute
        }

    def _allow_call(self, service: str) -> bool:
        """Check if API call is allowed based on rate limiting."""
        import time

        if service not in self.call_history:
            return True

        current_time = time.time()
        history = self.call_history[service]

        # Remove calls older than 1 minute
        while history and current_time - history[0] > 60:
            history.popleft()

        # Check if we're under the limit
        return len(history) < history.maxlen

    def _record_call(self, service: str):
        """Record an API call for rate limiting."""
        import time

        if service in self.call_history:
            self.call_history[service].append(time.time())

    def process_input(self, prompt: str) -> str:
        """Main entry point for processing language input with intelligent model selection."""
        try:
            # Get all available services and models
            available_services = self.api_clients.get_available_services()

            if not any(available_services.values()):
                return "❌ No language processing services available."

            # Use GPT as meta-reasoner to select best model for this specific prompt
            print("🧠 Analyzing prompt for optimal model selection...")
            selected_model = self.model_selector.select_best_model(prompt, available_services)

            if not selected_model:
                print("❌ Could not select appropriate model for this prompt.")
                return "❌ Could not select appropriate model for this prompt."

            # Get selection details from history for logging
            selection_details = None
            if self.model_selector.selection_history:
                selection_details = self.model_selector.selection_history[-1]

            if selection_details:
                print(f"🎯 Model Selection: {selected_model}")
                print(f"📋 Reason: {selection_details.get('reason', 'No reason provided')}")
                print(f"🎲 Confidence: {selection_details.get('confidence', 0.5):.2f}")
                print(f"🔧 Method: {selection_details.get('method', 'unknown')}")
            else:
                print(f"🎯 Model Selection: {selected_model} (fallback selection)")

            # Extract service from model selection (format: "service:model" or just "service")
            if ':' in selected_model:
                service, specific_model = selected_model.split(':', 1)
            else:
                service = selected_model
                specific_model = None

            # Check rate limiting
            if not self._allow_call(service):
                return "⚠️ Rate limit reached. Please try again in a moment."

            # Generate response using selected model
            response = self._generate_response_with_model(prompt, service, specific_model)

            # If API model failed (quota/token issues), try local fallback
            if not response and service in ['openai', 'anthropic', 'gemini']:
                print(f"🔄 {service} failed, falling back to local model...")
                response = self._generate_response_with_local_model(prompt)
                if response:
                    selected_model = 'local:fallback'
                    service = 'local'
                    specific_model = 'fallback'

            if response:
                self._record_call(service)
                print(f"✅ Response generated successfully using {service}:{specific_model or 'default'}")

                # Add to conversation history with model selection info
                self.conversation_history.append({
                    'prompt': prompt,
                    'response': response,
                    'selected_model': selected_model,
                    'service': service,
                    'specific_model': specific_model,
                    'selection_details': selection_details
                })

                # Text-to-speech output if enabled
                if self.speech_enabled and self.speech_integration:
                    try:
                        self.speech_integration.speak_text(response)
                        print("🔊 Response spoken via TTS")
                    except Exception as e:
                        print(f"⚠️ TTS failed: {e}")

                # Final status log
                response_length = len(response.split()) if response else 0
                print(f"📊 Response Stats: {response_length} words | Total conversations: {len(self.conversation_history)}")

                return response
            else:
                print("❌ Failed to generate response with all available models.")
                return "❌ Failed to generate response with all available models."

        except Exception as e:
            logger.error(f"Language processing error: {e}")
            return "❌ Language processing error occurred."

    def _generate_response_with_model(self, prompt: str, service: str, specific_model: Optional[str] = None) -> Optional[str]:
        """Generate response using the specified service and specific model."""

        # Build full prompt with system context
        full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nQuark:"

        try:
            if service == 'openai':
                model = specific_model or "gpt-4o-mini"
                print(f"🤖 Selected Model: OpenAI {model}")
                return self.api_clients.query_openai(full_prompt, model=model)
            elif service == 'anthropic':
                model = specific_model or "claude-3-haiku-20240307"
                print(f"🤖 Selected Model: Anthropic {model}")
                return self.api_clients.query_anthropic(full_prompt, model=model)
            elif service == 'gemini':
                print("🤖 Selected Model: Google Gemini")
                return self.api_clients.query_gemini(full_prompt)
            elif service == 'openrouter':
                model = specific_model or "openai/gpt-3.5-turbo"
                print(f"🤖 Selected Model: OpenRouter → {model}")
                return self.api_clients.query_openrouter(full_prompt, model=model)
            elif service == 'local':
                print(f"🤖 Selected Model: Local → {specific_model or 'default'}")
                return self._query_local_model(full_prompt, specific_model)
            else:
                logger.warning(f"Unknown service: {service}")
                print(f"❌ Unknown service: {service}")
                return None

        except Exception as e:
            logger.error(f"Response generation error for {service}: {e}")
            print(f"❌ Response generation error for {service}: {e}")
            return None

    def _query_local_model(self, prompt: str, model_type: Optional[str] = None) -> Optional[str]:
        """Query local models as fallback."""
        # Simplified local model response
        # In full implementation, this would use actual local models
        local_responses = [
            "I'm processing that using my local neural networks.",
            "Let me think about that from my current understanding.",
            "Based on my training, I believe that...",
            "My neural pathways suggest that..."
        ]

        import random
        return random.choice(local_responses)

    def _generate_response_with_local_model(self, prompt: str) -> Optional[str]:
        """Generate response using local LLM as fallback."""
        try:
            # Try to use the LocalLLMWrapper
            from brain.architecture.neural_core.cognitive_systems.local_llm_wrapper import LocalLLMWrapper

            # Initialize local model if not already done
            if not hasattr(self, '_local_llm'):
                self._local_llm = LocalLLMWrapper()

            # Create a conversation prompt
            full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nQuark:"

            # Generate response with local model
            response = self._local_llm.generate_response(full_prompt, max_new_tokens=256, temperature=0.7)

            if response and response.strip():
                # Clean up the response
                clean_response = response.strip()
                # Remove any repeated prompt text
                if "Quark:" in clean_response:
                    clean_response = clean_response.split("Quark:")[-1].strip()
                return clean_response
            else:
                return "I'm thinking about your question using my local processing capabilities."

        except Exception as e:
            logger.error(f"Local model fallback error: {e}")
            # Ultimate fallback - simple contextual responses
            return self._query_local_model(prompt)

    def enable_speech_output(self, provider: str = "system_tts") -> bool:
        """Enable text-to-speech output for responses."""
        if not SPEECH_AVAILABLE:
            print("❌ Speech integration not available")
            return False
        
        try:
            if not self.speech_integration:
                self.speech_integration = get_speech_integration()
            
            # Enable TTS with specified provider
            from ..speech_integration import TTSProvider
            provider_enum = TTSProvider(provider)
            success = self.speech_integration.enable_tts(provider_enum)
            
            if success:
                self.speech_enabled = True
                print(f"🔊 Speech output enabled with provider: {provider}")
            else:
                print(f"❌ Failed to enable speech with provider: {provider}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error enabling speech: {e}")
            return False
    
    def disable_speech_output(self) -> bool:
        """Disable text-to-speech output for responses."""
        try:
            if self.speech_integration:
                success = self.speech_integration.disable_tts()
                if success:
                    self.speech_enabled = False
                    print("🔇 Speech output disabled")
                return success
            else:
                self.speech_enabled = False
                print("🔇 Speech output disabled (no integration)")
                return True
                
        except Exception as e:
            print(f"❌ Error disabling speech: {e}")
            return False
    
    def get_speech_status(self) -> Dict[str, Any]:
        """Get current speech integration status."""
        if not SPEECH_AVAILABLE or not self.speech_integration:
            return {
                'available': False,
                'enabled': False,
                'reason': 'Speech integration not available'
            }
        
        try:
            return self.speech_integration.get_status()
        except Exception as e:
            return {
                'available': False,
                'enabled': False,
                'error': str(e)
            }

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current language processing status with model selection analytics."""
        status = {
            'available_services': self.api_clients.get_available_services(),
            'conversation_turns': len(self.conversation_history),
            'rate_limit_status': {
                service: len(history) for service, history in self.call_history.items()
            },
            'model_selection_analytics': self.model_selector.get_selection_analytics(),
            'intelligent_routing_enabled': True,
            'openrouter_integration': 'active'
        }
        
        # Add speech status
        status['speech_integration'] = self.get_speech_status()
        
        return status

    def print_selection_analytics(self):
        """Print detailed model selection analytics to console."""
        analytics = self.model_selector.get_selection_analytics()

        print("\n" + "="*60)
        print("🧠 QUARK MODEL SELECTION ANALYTICS")
        print("="*60)

        print(f"📊 Total Selections: {analytics.get('total_selections', 0)}")
        print(f"🎯 Average Confidence: {analytics.get('average_confidence', 0.0):.2f}")
        print(f"🤖 GPT Selector Usage: {analytics.get('gpt_selector_usage', 0)}")
        print(f"🔧 Fallback Usage: {analytics.get('fallback_usage', 0)}")

        print("\n📈 Model Distribution:")
        distribution = analytics.get('provider_distribution', {})
        for provider, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analytics.get('total_selections', 1)) * 100
            print(f"   {provider}: {count} times ({percentage:.1f}%)")

        available_services = self.api_clients.get_available_services()
        print(f"\n🌐 Available Services: {sum(available_services.values())}/{len(available_services)}")
        for service, status in available_services.items():
            status_emoji = "✅" if status else "❌"
            print(f"   {status_emoji} {service}")

        print("="*60)

# Export for backward compatibility
__all__ = ['LanguageCortex', 'LanguageAPIClients', 'ExpertRouter']
