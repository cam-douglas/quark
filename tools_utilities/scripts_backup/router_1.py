import random
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class Router:
    """Intelligent router for selecting the best model based on needs and availability."""
    
    def __init__(self, registry, routing_config: List[Dict[str, Any]]):
        self.registry = registry
        self.routing_config = routing_config
        self._load_balancing_state = {}  # Track model usage for load balancing
        
    def choose_model(self, needs: Dict[str, Any], registry) -> str:
        """
        Choose the best model based on needs, capabilities, and current load.
        
        Args:
            needs: Inferred needs from planner
            registry: Model registry instance
            
        Returns:
            Model ID string
        """
        needset = set(needs.get("need", []))
        primary_need = needs.get("primary_need", "chat")
        complexity = needs.get("complexity", "low")
        
        logger.info(f"Routing request: needs={needset}, primary={primary_need}, complexity={complexity}")
        
        # First, try exact capability matches
        exact_matches = self._find_exact_capability_matches(needset, registry)
        if exact_matches:
            selected = self._select_best_match(exact_matches, needs, complexity)
            logger.info(f"Selected model via exact capability match: {selected}")
            return selected
        
        # Try routing rules
        rule_match = self._apply_routing_rules(needs, registry)
        if rule_match:
            logger.info(f"Selected model via routing rule: {rule_match}")
            return rule_match
        
        # Fallback to capability-based selection
        fallback = self._fallback_selection(needs, registry)
        logger.info(f"Selected model via fallback: {fallback}")
        return fallback
    
    def _find_exact_capability_matches(self, needset: set, registry) -> List[Dict[str, Any]]:
        """Find models that exactly match the required capabilities."""
        matches = []
        
        for model in registry.list():
            model_caps = set(model.get("capabilities", []))
            
            # Check if model has all required capabilities
            if needset.issubset(model_caps):
                # Calculate match score based on capability overlap
                overlap = len(needset.intersection(model_caps))
                total_needed = len(needset)
                match_score = overlap / total_needed if total_needed > 0 else 0
                
                matches.append({
                    "model": model,
                    "score": match_score,
                    "capability_overlap": overlap,
                    "total_capabilities": len(model_caps)
                })
        
        # Sort by match score, then by total capabilities (prefer more capable models)
        matches.sort(key=lambda x: (x["score"], x["total_capabilities"]), reverse=True)
        return matches
    
    def _select_best_match(self, matches: List[Dict[str, Any]], needs: Dict[str, Any], complexity: str) -> str:
        """Select the best model from capability matches considering load and complexity."""
        if not matches:
            return None
            
        # Filter by complexity requirements
        complexity_filtered = []
        for match in matches:
            model = match["model"]
            model_complexity = model.get("complexity", "medium")
            
            # Complexity hierarchy: low < medium < high
            complexity_levels = {"low": 1, "medium": 2, "high": 3}
            req_level = complexity_levels.get(complexity, 2)
            model_level = complexity_levels.get(model_complexity, 2)
            
            if model_level >= req_level:
                complexity_filtered.append(match)
        
        if not complexity_filtered:
            complexity_filtered = matches  # Fallback to all matches
        
        # Apply load balancing
        best_match = self._apply_load_balancing(complexity_filtered)
        return best_match["model"]["id"]
    
    def _apply_load_balancing(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply load balancing to select the best available model."""
        if not matches:
            return None
            
        # Simple round-robin load balancing
        for match in matches:
            model_id = match["model"]["id"]
            if model_id not in self._load_balancing_state:
                self._load_balancing_state[model_id] = 0
            
            # Check concurrency limits
            max_concurrency = match["model"].get("concurrency", 1)
            if self._load_balancing_state[model_id] < max_concurrency:
                self._load_balancing_state[model_id] += 1
                return match
        
        # If all models are at capacity, return the least loaded
        least_loaded = min(matches, key=lambda x: self._load_balancing_state.get(x["model"]["id"], 0))
        return least_loaded
    
    def _apply_routing_rules(self, needs: Dict[str, Any], registry) -> Optional[str]:
        """Apply explicit routing rules from configuration."""
        needset = set(needs.get("need", []))
        
        for rule in self.routing_config:
            if "if" in rule and "then" in rule:
                condition = rule["if"]
                
                # Check if condition matches
                if self._condition_matches(condition, needset, needs):
                    target_model = rule["then"]
                    
                    # Validate that the target model exists
                    try:
                        registry.get(target_model)
                        return target_model
                    except KeyError:
                        logger.warning(f"Routing rule targets non-existent model: {target_model}")
                        continue
        
        return None
    
    def _condition_matches(self, condition: Dict[str, Any], needset: set, needs: Dict[str, Any]) -> bool:
        """Check if a routing condition matches the current needs."""
        for key, value in condition.items():
            if key == "need":
                if isinstance(value, list):
                    if not needset.intersection(set(value)):
                        return False
                else:
                    if value not in needset:
                        return False
            elif key == "complexity":
                if needs.get("complexity") != value:
                    return False
            elif key == "confidence":
                # Check if any need meets the confidence threshold
                confidences = needs.get("confidence", {})
                if not any(conf >= value for conf in confidences.values()):
                    return False
        
        return True
    
    def _fallback_selection(self, needs: Dict[str, Any], registry) -> str:
        """Fallback selection when no specific routing applies."""
        # Try to find a model with the primary need
        primary_need = needs.get("primary_need", "chat")
        
        # Look for models with the primary capability
        candidates = []
        for model in registry.list():
            if primary_need in model.get("capabilities", []):
                candidates.append(model)
        
        if candidates:
            # Select the first available candidate
            return candidates[0]["id"]
        
        # Last resort: return the first available model
        available_models = registry.list()
        if available_models:
            return available_models[0]["id"]
        
        # Emergency fallback
        logger.error("No models available for routing")
        return "default"

def choose_model(needs: Dict[str, Any], routing_cfg, registry):
    """
    Legacy function for backward compatibility.
    Creates a Router instance and delegates to it.
    """
    router = Router(registry, routing_cfg)
    return router.choose_model(needs, registry)

def get_router(registry, routing_config: List[Dict[str, Any]]) -> Router:
    """Factory function to create a router instance."""
    return Router(registry, routing_config)
