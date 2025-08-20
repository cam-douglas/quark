"""
Unified Llama Lexi Runner for Small-Mind
Integrates Llama-3-8B-Lexi-Uncensored with existing Small-Mind capabilities
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Import Small-Mind components
try:
    from ................................................llama_integration import LlamaLexiIntegration, LlamaLexiManager
    from ................................................model_manager import MoEModelManager
    from ................................................moe_manager import MoEManager
except ImportError:
    # Fallback for direct execution
    from llama_integration import LlamaLexiIntegration, LlamaLexiManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedLlamaRunner:
    """
    Unified runner that combines Llama Lexi with Small-Mind capabilities
    Provides general language understanding + specialized neuroscience expertise
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified runner
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "unified_llama_config.json"
        self.config = self._load_config()
        
        # Initialize managers
        self.llama_manager = LlamaLexiManager()
        self.moe_manager = None  # Will be initialized if available
        
        # Active integrations
        self.llama_integration = None
        self.active_models = {}
        
        # Response routing
        self.response_routing = {
            "general_language": "llama_lexi",
            "neuroscience": "smallmind_moe",
            "physics": "smallmind_moe",
            "ml_optimization": "smallmind_moe",
            "visualization": "smallmind_moe"
        }
        
        logger.info("Unified Llama Runner initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load unified configuration"""
        default_config = {
            "llama_lexi": {
                "enabled": True,
                "auto_load": False,
                "priority": "high"
            },
            "smallmind_moe": {
                "enabled": True,
                "auto_load": False,
                "priority": "medium"
            },
            "routing": {
                "smart_routing": True,
                "fallback_to_llama": True,
                "confidence_threshold": 0.7
            },
            "performance": {
                "max_concurrent_models": 2,
                "memory_optimization": True,
                "gpu_priority": True
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # Deep merge configuration
                    self._deep_merge(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def initialize_llama_lexi(self, **kwargs) -> bool:
        """
        Initialize Llama Lexi integration
        
        Args:
            **kwargs: Configuration overrides
            
        Returns:
            True if successful
        """
        try:
            if not self.config["llama_lexi"]["enabled"]:
                logger.info("Llama Lexi integration disabled in config")
                return False
            
            # Create integration
            self.llama_integration = self.llama_manager.create_integration(
                "unified_runner",
                **kwargs
            )
            
            # Auto-load if configured
            if self.config["llama_lexi"]["auto_load"]:
                success = self.llama_integration.load_model()
                if success:
                    logger.info("Llama Lexi model loaded successfully")
                else:
                    logger.warning("Failed to auto-load Llama Lexi model")
            
            logger.info("Llama Lexi integration initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Llama Lexi: {e}")
            return False
    
    def initialize_smallmind_moe(self) -> bool:
        """Initialize Small-Mind MoE models if available"""
        try:
            if not self.config["smallmind_moe"]["enabled"]:
                logger.info("Small-Mind MoE integration disabled in config")
                return False
            
            # Try to initialize MoE manager
            try:
                from ................................................moe_manager import MoEManager
                self.moe_manager = MoEManager()
                logger.info("Small-Mind MoE manager initialized")
                return True
            except ImportError:
                logger.warning("Small-Mind MoE components not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Small-Mind MoE: {e}")
            return False
    
    def route_query(self, query: str) -> str:
        """
        Route a query to the appropriate model
        
        Args:
            query: User query
            
        Returns:
            Model type to use
        """
        if not self.config["routing"]["smart_routing"]:
            return "llama_lexi"
        
        query_lower = query.lower()
        
        # Check for specialized domains
        if any(term in query_lower for term in [
            "brain", "neural", "neuron", "synapse", "cognitive", "neuroscience",
            "development", "plasticity", "neurotransmitter"
        ]):
            return "smallmind_moe"
        
        if any(term in query_lower for term in [
            "physics", "simulation", "mujoco", "dynamics", "force", "collision"
        ]):
            return "smallmind_moe"
        
        if any(term in query_lower for term in [
            "optimization", "ml", "machine learning", "training", "hyperparameter"
        ]):
            return "smallmind_moe"
        
        if any(term in query_lower for term in [
            "visualization", "plot", "graph", "3d", "render"
        ]):
            return "smallmind_moe"
        
        # Default to Llama Lexi for general language
        return "llama_lexi"
    
    def generate_response(
        self, 
        query: str, 
        force_model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using the appropriate model
        
        Args:
            query: User query
            force_model: Force use of specific model
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        try:
            # Determine which model to use
            if force_model:
                model_type = force_model
            else:
                model_type = self.route_query(query)
            
            logger.info(f"Routing query to: {model_type}")
            
            # Generate response based on model type
            if model_type == "llama_lexi":
                return self._generate_llama_response(query, **kwargs)
            elif model_type == "smallmind_moe":
                return self._generate_smallmind_response(query, **kwargs)
            else:
                # Fallback to Llama Lexi
                logger.warning(f"Unknown model type: {model_type}, falling back to Llama Lexi")
                return self._generate_llama_response(query, **kwargs)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "error": str(e),
                "model": "unified_runner",
                "fallback": True
            }
    
    def _generate_llama_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Llama Lexi"""
        if self.llama_integration is None:
            if not self.initialize_llama_lexi():
                return {"error": "Llama Lexi not available"}
        
        # Ensure model is loaded
        if self.llama_integration.model is None:
            if not self.llama_integration.load_model():
                return {"error": "Failed to load Llama Lexi model"}
        
        # Generate response
        response = self.llama_integration.generate_response(query, **kwargs)
        
        # Add metadata
        response["model_type"] = "llama_lexi"
        response["routing_method"] = "smart_routing"
        response["unified_runner"] = True
        
        return response
    
    def _generate_smallmind_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Small-Mind MoE models"""
        if self.moe_manager is None:
            if not self.initialize_smallmind_moe():
                # Fallback to Llama Lexi
                logger.info("Small-Mind MoE not available, falling back to Llama Lexi")
                return self._generate_llama_response(query, **kwargs)
        
        try:
            # Use Small-Mind capabilities
            # This would integrate with your existing MoE system
            response = {
                "response": f"Small-Mind MoE response to: {query}",
                "model_type": "smallmind_moe",
                "routing_method": "smart_routing",
                "unified_runner": True,
                "note": "This is a placeholder - integrate with your actual MoE system"
            }
            return response
            
        except Exception as e:
            logger.error(f"Small-Mind MoE error: {e}")
            # Fallback to Llama Lexi
            if self.config["routing"]["fallback_to_llama"]:
                logger.info("Falling back to Llama Lexi")
                return self._generate_llama_response(query, **kwargs)
            else:
                return {"error": f"Small-Mind MoE error: {e}"}
    
    def batch_generate(
        self, 
        queries: List[str], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple queries
        
        Args:
            queries: List of queries
            **kwargs: Additional parameters
            
        Returns:
            List of response dictionaries
        """
        results = []
        for query in queries:
            result = self.generate_response(query, **kwargs)
            results.append(result)
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        status = {
            "unified_runner": {
                "status": "active",
                "version": "1.0.0"
            },
            "llama_lexi": {
                "status": "initialized" if self.llama_integration else "not_initialized",
                "model_loaded": self.llama_integration.model is not None if self.llama_integration else False
            },
            "smallmind_moe": {
                "status": "initialized" if self.moe_manager else "not_initialized",
                "available": self.moe_manager is not None
            },
            "routing": {
                "smart_routing": self.config["routing"]["smart_routing"],
                "fallback_enabled": self.config["routing"]["fallback_to_llama"]
            }
        }
        
        # Add Llama Lexi model info if available
        if self.llama_integration:
            status["llama_lexi"]["model_info"] = self.llama_integration.get_model_info()
        
        return status
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if self.config["performance"]["memory_optimization"]:
            # Unload models if not actively used
            if self.llama_integration and self.llama_integration.model:
                # Keep Llama Lexi loaded if it's the primary model
                pass
            
            # Clear any cached data
            import gc
            gc.collect()
            
            logger.info("Memory optimization completed")
    
    def shutdown(self):
        """Shutdown the unified runner"""
        try:
            # Unload Llama Lexi
            if self.llama_integration:
                self.llama_integration.unload_model()
            
            # Clear managers
            self.llama_integration = None
            self.moe_manager = None
            
            # Optimize memory
            self.optimize_memory()
            
            logger.info("Unified Llama Runner shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global instance
_unified_runner = None

def get_unified_runner() -> UnifiedLlamaRunner:
    """Get the global unified runner instance"""
    global _unified_runner
    if _unified_runner is None:
        _unified_runner = UnifiedLlamaRunner()
    return _unified_runner


def generate_unified_response(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to generate a response"""
    runner = get_unified_runner()
    return runner.generate_response(query, **kwargs)


if __name__ == "__main__":
    # Example usage
    runner = get_unified_runner()
    
    # Initialize Llama Lexi
    runner.initialize_llama_lexi()
    
    # Test queries
    test_queries = [
        "Hello, how are you today?",
        "Explain neural plasticity in the brain",
        "What's the weather like?",
        "How do I optimize machine learning models?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = runner.generate_response(query)
        print(f"Response: {response}")
    
    # Get system status
    status = runner.get_system_status()
    print(f"\nSystem Status: {json.dumps(status, indent=2)}")
    
    # Cleanup
    runner.shutdown()
