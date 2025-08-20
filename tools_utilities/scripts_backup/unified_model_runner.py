#!/usr/bin/env python3
"""
Unified Model Runner for Small-Mind
Ensures consistent model execution across all interfaces (Cursor, terminal, web browser)
"""

import os, sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelCapability:
    """Defines what a model can do"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    max_input_length: int
    max_output_length: int
    specialized_for: List[str]

@dataclass
class ModelInstance:
    """Represents a loaded model instance"""
    model_id: str
    model_name: str
    model_type: str
    is_loaded: bool
    memory_usage_gb: float
    capabilities: List[ModelCapability]
    status: str

class UnifiedModelRunner:
    """
    Unified model runner that ensures consistent execution across all interfaces.
    This class manages model loading, execution, and provides a consistent API
    regardless of how small-mind is accessed.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Auto-detect project root
            current_file = Path(__file__)
            self.project_root = current_file.parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        # Ensure project root is in Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        # Initialize components
        self.model_manager = None
        self.moe_router = None
        self.neuroscience_experts = None
        self.loaded_models: Dict[str, ModelInstance] = {}
        
        # Model registry path
        self.registry_path = self.project_root / "src" / "smallmind" / "models" / "models" / "configs" / "model_registry.json"
        
        # Initialize the system
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all model components."""
        try:
            logger.info("Initializing Small-Mind Unified Model Runner...")
            
            # Initialize model manager
            self._init_model_manager()
            
            # Initialize MoE router
            self._init_moe_router()
            
            # Initialize neuroscience experts
            self._init_neuroscience_experts()
            
            # Load available models
            self._load_available_models()
            
            logger.info("Unified Model Runner initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Unified Model Runner: {e}")
            raise
    
    def _init_model_manager(self):
        """Initialize the model manager."""
        try:
            from src.smallmind.models.model_manager import MoEModelManager
            self.model_manager = MoEModelManager()
            logger.info("Model Manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Model Manager: {e}")
            self.model_manager = None
    
    def _init_moe_router(self):
        """Initialize the MoE router."""
        try:
            from src.smallmind.models.moe_router import MoERouter
            self.moe_router = MoERouter()
            logger.info("MoE Router initialized")
        except Exception as e:
            logger.warning(f"Could not initialize MoE Router: {e}")
            self.moe_router = None
    
    def _init_neuroscience_experts(self):
        """Initialize neuroscience experts."""
        try:
            from src.smallmind.models.neuroscience_experts import NeuroscienceExperts
            self.neuroscience_experts = NeuroscienceExperts()
            logger.info("Neuroscience Experts initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Neuroscience Experts: {e}")
            self.neuroscience_experts = None
    
    def _load_available_models(self):
        """Load information about available models."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for model_id, model_info in registry_data.items():
                    # Create model instance
                    model_instance = ModelInstance(
                        model_id=model_id,
                        model_name=model_info.get('model_name', model_id),
                        model_type=self._determine_model_type(model_id),
                        is_loaded=model_info.get('is_loaded', False),
                        memory_usage_gb=model_info.get('model_size_gb', 0.0),
                        capabilities=self._get_model_capabilities(model_id),
                        status="Available"
                    )
                    
                    self.loaded_models[model_id] = model_instance
                
                logger.info(f"Loaded {len(self.loaded_models)} available models")
            else:
                logger.warning("Model registry not found")
                
        except Exception as e:
            logger.error(f"Failed to load available models: {e}")
    
    def _determine_model_type(self, model_id: str) -> str:
        """Determine the type of a model based on its ID."""
        if "moe" in model_id.lower():
            return "MoE (Mixture of Experts)"
        elif "deepseek" in model_id.lower():
            return "Large Language Model"
        elif "qwen" in model_id.lower():
            return "Large Language Model"
        else:
            return "Unknown"
    
    def _get_model_capabilities(self, model_id: str) -> List[ModelCapability]:
        """Get the capabilities of a specific model."""
        capabilities = []
        
        # Base language model capabilities
        base_capabilities = ModelCapability(
            name="Text Generation",
            description="Generate human-like text responses",
            input_types=["text", "prompt"],
            output_types=["text", "response"],
            max_input_length=8192,
            max_output_length=4096,
            specialized_for=["general conversation", "text completion"]
        )
        capabilities.append(base_capabilities)
        
        # Model-specific capabilities
        if "moe" in model_id.lower():
            moe_capabilities = ModelCapability(
                name="Expert Routing",
                description="Route queries to specialized expert models",
                input_types=["text", "query"],
                output_types=["text", "expert_response"],
                max_input_length=8192,
                max_output_length=4096,
                specialized_for=["specialized tasks", "expert knowledge"]
            )
            capabilities.append(moe_capabilities)
        
        if "deepseek" in model_id.lower():
            deepseek_capabilities = ModelCapability(
                name="Code Generation",
                description="Generate and understand programming code",
                input_types=["text", "code", "prompt"],
                output_types=["text", "code", "explanation"],
                max_input_length=16384,
                max_output_length=8192,
                specialized_for=["programming", "code analysis", "software development"]
            )
            capabilities.append(deepseek_capabilities)
        
        return capabilities
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available models with their information."""
        models_info = []
        for model_id, model_instance in self.loaded_models.items():
            model_info = asdict(model_instance)
            model_info['capabilities'] = [asdict(cap) for cap in model_instance.capabilities]
            models_info.append(model_info)
        
        return models_info
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        total_models = len(self.loaded_models)
        loaded_models = sum(1 for m in self.loaded_models.values() if m.is_loaded)
        total_memory = sum(m.memory_usage_gb for m in self.loaded_models.values())
        
        return {
            "total_models": total_models,
            "loaded_models": loaded_models,
            "available_models": total_models - loaded_models,
            "total_memory_gb": total_memory,
            "system_status": "Ready" if self.model_manager else "Not Ready",
            "components": {
                "model_manager": self.model_manager is not None,
                "moe_router": self.moe_router is not None,
                "neuroscience_experts": self.neuroscience_experts is not None
            }
        }
    
    def execute_query(self, query: str, model_id: Optional[str] = None, 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a query using the appropriate model.
        This is the main entry point for all interfaces.
        """
        try:
            # Determine which model to use
            if model_id is None:
                model_id = self._select_best_model(query)
            
            # Validate model availability
            if model_id not in self.loaded_models:
                return {
                    "success": False,
                    "error": f"Model {model_id} not available",
                    "available_models": list(self.loaded_models.keys())
                }
            
            # Execute the query
            result = self._execute_with_model(model_id, query, context)
            
            return {
                "success": True,
                "model_id": model_id,
                "query": query,
                "response": result,
                "model_info": asdict(self.loaded_models[model_id])
            }
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _select_best_model(self, query: str) -> str:
        """Select the best model for a given query."""
        # Simple heuristic-based model selection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["code", "program", "function", "class", "algorithm"]):
            return "deepseek-v2"  # Best for code
        elif any(word in query_lower for word in ["brain", "neuron", "simulation", "physics"]):
            return "qwen1.5-moe"  # Good for scientific tasks
        else:
            return "qwen1.5-moe"  # Default choice
    
    def _execute_with_model(self, model_id: str, query: str, 
                           context: Optional[Dict[str, Any]] = None) -> str:
        """Execute a query with a specific model."""
        try:
            # For now, return a placeholder response
            # This would integrate with your actual model execution logic
            model_info = self.loaded_models[model_id]
            
            response = f"Query processed by {model_info.model_name} ({model_info.model_type})\n"
            response += f"Query: {query}\n"
            response += f"Model capabilities: {', '.join([cap.name for cap in model_info.capabilities])}\n"
            
            if context:
                response += f"Context: {json.dumps(context, indent=2)}\n"
            
            response += "\n(Actual model execution to be implemented)"
            
            return response
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model into memory."""
        try:
            if model_id not in self.loaded_models:
                logger.error(f"Model {model_id} not found")
                return False
            
            # Update model status
            self.loaded_models[model_id].is_loaded = True
            self.loaded_models[model_id].status = "Loaded"
            
            logger.info(f"Model {model_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model from memory."""
        try:
            if model_id not in self.loaded_models:
                logger.error(f"Model {model_id} not found")
                return False
            
            # Update model status
            self.loaded_models[model_id].is_loaded = False
            self.loaded_models[model_id].status = "Available"
            
            logger.info(f"Model {model_id} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False

# Global instance for consistent access
_global_runner: Optional[UnifiedModelRunner] = None

def get_global_runner() -> UnifiedModelRunner:
    """Get the global model runner instance."""
    global _global_runner
    if _global_runner is None:
        _global_runner = UnifiedModelRunner()
    return _global_runner

def execute_query(query: str, model_id: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Global function to execute queries - used by all interfaces."""
    runner = get_global_runner()
    return runner.execute_query(query, model_id, context)

def get_available_models() -> List[Dict[str, Any]]:
    """Global function to get available models."""
    runner = get_global_runner()
    return runner.get_available_models()

def get_model_status() -> Dict[str, Any]:
    """Global function to get system status."""
    runner = get_global_runner()
    return runner.get_model_status()

if __name__ == "__main__":
    # Test the unified model runner
    runner = UnifiedModelRunner()
    print("Available models:")
    for model in runner.get_available_models():
        print(f"  - {model['model_name']} ({model['model_type']})")
    
    print("\nSystem status:")
    status = runner.get_model_status()
    print(json.dumps(status, indent=2))
    
    print("\nTesting query execution:")
    result = runner.execute_query("What is the meaning of life?")
    print(json.dumps(result, indent=2))
