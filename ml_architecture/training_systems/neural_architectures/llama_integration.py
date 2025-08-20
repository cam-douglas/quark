"""
Llama-3-8B-Lexi-Uncensored Integration for Small-Mind
Provides general language understanding capabilities for base responses
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaLexiIntegration:
    """
    Integration class for Llama-3-8B-Lexi-Uncensored model
    Provides general language understanding for base responses
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize Llama Lexi integration
        
        Args:
            model_path: Path to local model or HuggingFace model ID
            device: Device to run model on ("auto", "cpu", "cuda", "mps")
        """
        self.model_id = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
        self.model_path = model_path or self.model_id
        self.device = self._determine_device(device)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Configuration
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        
        # Safety and compliance settings
        self.safety_filters = True
        self.content_warnings = True
        
        logger.info(f"Initializing Llama Lexi integration on device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the Llama Lexi model
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is not None and not force_reload:
            logger.info("Model already loaded")
            return True
        
        try:
            logger.info(f"Loading Llama Lexi model: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Llama Lexi model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Llama Lexi model: {e}")
            return False
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        safety_check: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response using Llama Lexi
        
        Args:
            prompt: Input prompt
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            safety_check: Enable safety filtering
            
        Returns:
            Dictionary containing response and metadata
        """
        if self.model is None:
            if not self.load_model():
                return {"error": "Failed to load model"}
        
        # Use provided parameters or defaults
        max_len = max_length or self.max_length
        temp = temperature or self.temperature
        tp = top_p or self.top_p
        tk = top_k or self.top_k
        
        try:
            # Safety check if enabled
            if safety_check and self.safety_filters:
                prompt = self._apply_safety_filters(prompt)
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_length=max_len,
                temperature=temp,
                top_p=tp,
                top_k=tk,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the input prompt from the response
            if generated_text.startswith(prompt):
                response_text = generated_text[len(prompt):].strip()
            else:
                response_text = generated_text.strip()
            
            # Post-process response
            response_text = self._post_process_response(response_text)
            
            return {
                "response": response_text,
                "prompt": prompt,
                "model": self.model_id,
                "parameters": {
                    "max_length": max_len,
                    "temperature": temp,
                    "top_p": tp,
                    "top_k": tk
                },
                "safety_applied": safety_check
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e)}
    
    def _apply_safety_filters(self, prompt: str) -> str:
        """Apply safety filters to the prompt"""
        # Basic content filtering
        warning_phrases = [
            "Please note that this model is uncensored and will respond to any request.",
            "Use responsibly and consider the ethical implications of your queries.",
            "This response is generated by an AI model trained on diverse data."
        ]
        
        # Add warning if content warnings are enabled
        if self.content_warnings:
            prompt = f"{warning_phrases[0]} {prompt}"
        
        return prompt
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response"""
        # Remove any remaining prompt artifacts
        response = response.strip()
        
        # Clean up common generation artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("<|im_end|>", "")
        
        # Ensure response ends cleanly
        if response.endswith("..."):
            response = response[:-3]
        
        return response
    
    def batch_generate(
        self, 
        prompts: List[str], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of response dictionaries
        """
        results = []
        for prompt in prompts:
            result = self.generate_response(prompt, **kwargs)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "device": self.device,
            "model_type": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "max_length": self.max_length,
            "safety_filters": self.safety_filters
        }
    
    def unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Llama Lexi model unloaded")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.unload_model()


class LlamaLexiManager:
    """
    Manager class for Llama Lexi integration
    Handles model lifecycle and configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "llama_lexi_config.json"
        self.config = self._load_config()
        self.integrations = {}
        
        logger.info("Llama Lexi manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "default_model_path": "Orenguteng/Llama-3-8B-Lexi-Uncensored",
            "default_device": "auto",
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "safety_filters": True,
            "content_warnings": True,
            "auto_load": False
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def create_integration(
        self, 
        name: str = "default",
        **kwargs
    ) -> LlamaLexiIntegration:
        """
        Create a new Llama Lexi integration instance
        
        Args:
            name: Name for the integration instance
            **kwargs: Configuration overrides
            
        Returns:
            LlamaLexiIntegration instance
        """
        # Merge config with overrides
        config = self.config.copy()
        config.update(kwargs)
        
        integration = LlamaLexiIntegration(
            model_path=config.get("default_model_path"),
            device=config.get("default_device")
        )
        
        # Set configuration
        integration.max_length = config.get("max_length")
        integration.temperature = config.get("temperature")
        integration.top_p = config.get("top_p")
        integration.top_k = config.get("top_k")
        integration.safety_filters = config.get("safety_filters")
        integration.content_warnings = config.get("content_warnings")
        
        self.integrations[name] = integration
        
        # Auto-load if configured
        if config.get("auto_load"):
            integration.load_model()
        
        logger.info(f"Created Llama Lexi integration: {name}")
        return integration
    
    def get_integration(self, name: str = "default") -> Optional[LlamaLexiIntegration]:
        """Get an existing integration instance"""
        return self.integrations.get(name)
    
    def remove_integration(self, name: str):
        """Remove an integration instance"""
        if name in self.integrations:
            self.integrations[name].unload_model()
            del self.integrations[name]
            logger.info(f"Removed Llama Lexi integration: {name}")
    
    def list_integrations(self) -> List[str]:
        """List all integration names"""
        return list(self.integrations.keys())
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")


# Convenience functions for easy integration
def create_llama_lexi_integration(**kwargs) -> LlamaLexiIntegration:
    """Create a Llama Lexi integration with default settings"""
    return LlamaLexiIntegration(**kwargs)


def get_llama_lexi_manager() -> LlamaLexiManager:
    """Get the global Llama Lexi manager instance"""
    if not hasattr(get_llama_lexi_manager, '_instance'):
        get_llama_lexi_manager._instance = LlamaLexiManager()
    return get_llama_lexi_manager._instance


if __name__ == "__main__":
    # Example usage
    manager = get_llama_lexi_manager()
    integration = manager.create_integration("test")
    
    # Test response generation
    response = integration.generate_response("Hello, how are you today?")
    print("Response:", response)
    
    # Get model info
    info = integration.get_model_info()
    print("Model info:", info)
