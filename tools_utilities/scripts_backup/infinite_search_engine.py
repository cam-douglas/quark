#!/usr/bin/env python3
"""
Infinite Search Engine for Neuroscience Data with Self-Improvement

This engine provides endless search capabilities with infinite parameter combinations,
recursive searches, evolutionary algorithms, and fractal patterns. It also learns
from its search results to improve its own search strategies safely.
"""

import sys
import json
import random
import itertools
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from neurodata import NeurodataManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SelfImprovingSearchEngine:
    """Self-improving infinite search engine with endless parameters and capabilities"""
    
    def __init__(self, learning_enabled: bool = True, model_path: str = "search_engine_model.pkl"):
        self.manager = NeurodataManager()
        self.search_history = []
        self.learning_enabled = learning_enabled
        self.model_path = Path(model_path)
        
        # Learning components
        self.query_patterns = defaultdict(int)
        self.successful_combinations = defaultdict(float)
        self.parameter_effectiveness = defaultdict(lambda: defaultdict(float))
        self.search_strategies = defaultdict(float)
        self.result_quality_metrics = defaultdict(list)
        
        # Load existing learned model if available
        self._load_learned_model()
        
        # Endless parameter combinations
        self.parameter_space = {
            "species": ["human", "mouse", "rat", "monkey", "ferret", "zebrafish", "drosophila", "c.elegans"],
            "brain_regions": ["cortex", "hippocampus", "cerebellum", "thalamus", "basal_ganglia", "amygdala", 
                             "visual_cortex", "auditory_cortex", "somatosensory_cortex", "motor_cortex", "prefrontal_cortex"],
            "data_types": ["electrophysiology", "fMRI", "MEG", "EEG", "morphology", "transcriptomics", "connectomics", "behavior"],
            "modalities": ["patch_clamp", "extracellular", "calcium_imaging", "optogenetics", "dti", "resting_state", "task_based"],
            "techniques": ["whole_cell", "cell_attached", "loose_patch", "sharp_electrode", "multi_electrode", "single_unit"],
            "experimental_conditions": ["awake", "anesthetized", "sleeping", "learning", "memory", "attention", "decision_making"],
            "diseases": ["alzheimer", "parkinson", "autism", "schizophrenia", "epilepsy", "stroke", "trauma"],
            "developmental_stages": ["embryonic", "neonatal", "juvenile", "adult", "aging"],
            "temporal_resolution": ["millisecond", "second", "minute", "hour", "day", "week", "month", "year"],
            "spatial_resolution": ["nanometer", "micrometer", "millimeter", "centimeter", "meter"],
            "data_quality": ["high", "medium", "low", "validated", "peer_reviewed", "preliminary"],
            "licenses": ["CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "MIT", "GPL", "proprietary"],
            "file_formats": ["NWB", "BIDS", "HDF5", "MAT", "CSV", "JSON", "XML", "WARC", "ARC"],
            "computational_models": ["biophysical", "simplified", "network", "population", "mean_field", "spiking", "rate_based"],
            "analysis_methods": ["spike_sorting", "pca", "ica", "granger_causality", "graph_theory", "machine_learning", "deep_learning"],
            "hardware_platforms": ["intan", "blackrock", "plexon", "tucker_davis", "neuralynx", "custom"],
            "software_tools": ["neuron", "genesis", "brian", "nest", "arbor", "custom_python", "matlab", "julia"],
            "publication_years": list(range(1990, 2025)),
            "participant_counts": list(range(1, 10001)),
            "recording_durations": ["seconds", "minutes", "hours", "days", "weeks", "months"],
            "stimulus_types": ["visual", "auditory", "tactile", "olfactory", "gustatory", "multimodal", "cognitive"],
            "task_paradigms": ["working_memory", "decision_making", "attention", "learning", "perception", "motor_control"],
            "environmental_conditions": ["temperature", "humidity", "lighting", "noise", "social", "isolated"],
            "genetic_modifications": ["wild_type", "knockout", "knockin", "transgenic", "optogenetic", "chemogenetic"],
            "pharmacological_agents": ["control", "drug", "toxin", "neurotransmitter", "receptor_agonist", "receptor_antagonist"],
            "electrode_types": ["glass", "metal", "silicon", "carbon_fiber", "optrode", "tetrode", "hexatrode"],
            "imaging_methods": ["confocal", "two_photon", "light_sheet", "electron_microscopy", "x_ray", "mri", "pet"],
            "data_curation": ["raw", "processed", "analyzed", "validated", "published", "archived"],
            "access_levels": ["public", "restricted", "embargoed", "collaborative", "commercial"],
            "update_frequencies": ["real_time", "daily", "weekly", "monthly", "quarterly", "annually", "on_demand"]
        }
        
        # Adaptive search strategies learned from experience
        self.adaptive_strategies = {
            "high_yield_combinations": [],
            "query_expansion_patterns": [],
            "parameter_optimization": {},
            "search_depth_adjustment": 3,
            "result_quality_thresholds": {}
        }
    
    def _load_learned_model(self):
        """Load previously learned model safely"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    learned_data = pickle.load(f)
                
                # Safely restore learned components
                if isinstance(learned_data, dict):
                    self.query_patterns = learned_data.get('query_patterns', defaultdict(int))
                    self.successful_combinations = learned_data.get('successful_combinations', defaultdict(float))
                    self.parameter_effectiveness = learned_data.get('parameter_effectiveness', defaultdict(lambda: defaultdict(float)))
                    self.search_strategies = learned_data.get('search_strategies', defaultdict(float))
                    self.result_quality_metrics = learned_data.get('result_quality_metrics', defaultdict(list))
                    
                    # Handle adaptive strategies with backward compatibility
                    saved_adaptive = learned_data.get('adaptive_strategies', {})
                    if isinstance(saved_adaptive, dict):
                        # Update only existing keys to avoid attribute errors
                        for key in self.adaptive_strategies:
                            if key in saved_adaptive:
                                self.adaptive_strategies[key] = saved_adaptive[key]
                
                print(f"🧠 Loaded learned model with {len(self.query_patterns)} query patterns")
            else:
                print("🧠 Starting with fresh learning model")
        except Exception as e:
            print(f"⚠️  Could not load learned model: {e}. Starting fresh.")
            # Ensure clean state
            self.query_patterns.clear()
            self.successful_combinations.clear()
            self.parameter_effectiveness.clear()
            self.search_strategies.clear()
            self.result_quality_metrics.clear()
            # Reset adaptive strategies to defaults
            self.adaptive_strategies = {
                "high_yield_combinations": [],
                "query_expansion_patterns": [],
                "parameter_optimization": {},
                "search_depth_adjustment": 3,
                "result_quality_thresholds": {}
            }
    
    def _save_learned_model(self):
        """Save learned model safely"""
        try:
            learned_data = {
                'query_patterns': dict(self.query_patterns),
                'successful_combinations': dict(self.successful_combinations),
                'parameter_effectiveness': {k: dict(v) for k, v in self.parameter_effectiveness.items()},
                'search_strategies': dict(self.search_strategies),
                'result_quality_metrics': dict(self.result_quality_metrics),
                'adaptive_strategies': self.adaptive_strategies,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(learned_data, f)
            
            print(f"💾 Saved learned model with {len(self.query_patterns)} patterns")
        except Exception as e:
            print(f"⚠️  Could not save learned model: {e}")
    
    def _learn_from_search_result(self, query: str, params: Dict, results: Dict, search_type: str):
        """Learn from search results to improve future searches"""
        if not self.learning_enabled:
            return
        
        try:
            # Calculate result quality metrics
            total_results = sum(len(r) for r in results.values()) if results else 0
            source_diversity = len([r for r in results.values() if r]) if results else 0
            result_quality = self._calculate_result_quality(results)
            
            # Learn query patterns
            query_key = query.lower().strip()
            self.query_patterns[query_key] += 1
            
            # Learn successful parameter combinations
            param_key = self._create_parameter_key(params)
            if result_quality > 0.5:  # Only learn from successful searches
                self.successful_combinations[param_key] = result_quality
            
            # Learn parameter effectiveness
            for param_type, param_value in params.items():
                if param_value:
                    self.parameter_effectiveness[param_type][str(param_value)] += result_quality
            
            # Learn search strategy effectiveness
            self.search_strategies[search_type] += result_quality
            
            # Store quality metrics for analysis
            self.result_quality_metrics[search_type].append({
                'query': query,
                'params': params,
                'quality': result_quality,
                'total_results': total_results,
                'source_diversity': source_diversity,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update adaptive strategies
            self._update_adaptive_strategies(query, params, result_quality)
            
            # Periodically save learned model
            if len(self.search_history) % 10 == 0:
                self._save_learned_model()
                
        except Exception as e:
            print(f"⚠️  Learning error: {e}")
    
    def _calculate_result_quality(self, results: Dict) -> float:
        """Calculate quality score for search results"""
        if not results:
            return 0.0
        
        try:
            total_results = sum(len(r) for r in results.values())
            source_diversity = len([r for r in results.values() if r])
            
            # Quality factors
            result_abundance = min(total_results / 20.0, 1.0)  # Normalize to 0-1
            diversity_score = min(source_diversity / 3.0, 1.0)  # Normalize to 0-1
            
            # Weighted quality score
            quality = (result_abundance * 0.6) + (diversity_score * 0.4)
            return max(0.0, min(1.0, quality))  # Clamp to 0-1
            
        except Exception:
            return 0.0
    
    def _create_parameter_key(self, params: Dict) -> str:
        """Create a string key for parameter combination"""
        if not params:
            return "default"
        
        sorted_params = sorted(params.items())
        return "_".join([f"{k}:{v}" for k, v in sorted_params])
    
    def _update_adaptive_strategies(self, query: str, params: Dict, quality: float):
        """Update adaptive search strategies based on learning"""
        try:
            # Update high-yield combinations
            if quality > 0.7:  # High quality results
                param_key = self._create_parameter_key(params)
                if param_key not in self.adaptive_strategies["high_yield_combinations"]:
                    self.adaptive_strategies["high_yield_combinations"].append(param_key)
                    # Keep only top 20 combinations
                    if len(self.adaptive_strategies["high_yield_combinations"]) > 20:
                        self.adaptive_strategies["high_yield_combinations"] = \
                            self.adaptive_strategies["high_yield_combinations"][-20:]
            
            # Update search depth based on result quality
            if quality > 0.8:
                self.adaptive_strategies["search_depth_adjustment"] = min(
                    self.adaptive_strategies["search_depth_adjustment"] + 1, 5
                )
            elif quality < 0.3:
                self.adaptive_strategies["search_depth_adjustment"] = max(
                    self.adaptive_strategies["search_depth_adjustment"] - 1, 1
                )
            
            # Update quality thresholds - use a default search type if none provided
            search_type = "general"  # Default search type
            if not self.adaptive_strategies["result_quality_thresholds"].get(search_type):
                self.adaptive_strategies["result_quality_thresholds"][search_type] = 0.5
            
            # Adaptive threshold adjustment
            current_threshold = self.adaptive_strategies["result_quality_thresholds"][search_type]
            if quality > current_threshold:
                # Increase threshold slightly for better results
                new_threshold = current_threshold + (quality - current_threshold) * 0.1
                self.adaptive_strategies["result_quality_thresholds"][search_type] = min(new_threshold, 0.9)
            
        except Exception as e:
            print(f"⚠️  Strategy update error: {e}")
    
    def _apply_learned_knowledge(self, base_query: str, search_type: str = "general") -> Dict:
        """Apply learned knowledge to improve search parameters"""
        improved_params = {}
        
        try:
            # Apply successful parameter combinations
            if self.adaptive_strategies["high_yield_combinations"]:
                # Use a successful combination as base
                best_combo = random.choice(self.adaptive_strategies["high_yield_combinations"])
                improved_params = self._parse_parameter_key(best_combo)
            
            # Apply learned parameter effectiveness
            for param_type in ["species", "brain_regions", "data_types"]:
                if param_type not in improved_params and param_type in self.parameter_effectiveness:
                    # Select parameter value based on learned effectiveness
                    effectiveness = self.parameter_effectiveness[param_type]
                    if effectiveness:
                        # Weighted random selection based on effectiveness
                        values = list(effectiveness.keys())
                        weights = list(effectiveness.values())
                        if values and weights:
                            try:
                                selected_value = random.choices(values, weights=weights, k=1)[0]
                                improved_params[param_type] = selected_value
                            except Exception:
                                # Fallback to random selection
                                improved_params[param_type] = random.choice(self.parameter_space.get(param_type, []))
            
            # Apply learned search depth
            if search_type in self.search_strategies:
                improved_params["search_depth"] = self.adaptive_strategies["search_depth_adjustment"]
            
        except Exception as e:
            print(f"⚠️  Knowledge application error: {e}")
        
        return improved_params
    
    def _parse_parameter_key(self, param_key: str) -> Dict:
        """Parse parameter key back to dictionary"""
        params = {}
        try:
            if param_key != "default":
                parts = param_key.split("_")
                for part in parts:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        params[key] = value
        except Exception:
            pass
        return params
    
    def intelligent_search(self, base_query: str, search_type: str = "general", **kwargs) -> Dict[str, Any]:
        """
        Perform intelligent search using learned knowledge
        
        Args:
            base_query: Base search query
            search_type: Type of search to perform
            **kwargs: Additional search parameters
            
        Returns:
            Intelligent search results with learning metadata
        """
        print(f"🧠 INTELLIGENT SEARCH: {base_query} (using learned knowledge)")
        print("=" * 80)
        
        # Apply learned knowledge to improve search
        learned_params = self._apply_learned_knowledge(base_query, search_type)
        print(f"📚 Applied learned parameters: {learned_params}")
        
        # Combine learned params with user params
        combined_params = {**learned_params, **kwargs}
        
        # Perform search with improved parameters
        results = self._search_with_parameters(base_query, combined_params)
        
        # Learn from this search
        self._learn_from_search_result(base_query, combined_params, results, search_type)
        
        # Add learning metadata
        if results:
            results["learning_metadata"] = {
                "applied_learned_params": learned_params,
                "combined_params": combined_params,
                "search_type": search_type,
                "learning_enabled": self.learning_enabled,
                "model_patterns": len(self.query_patterns),
                "successful_combinations": len(self.adaptive_strategies["high_yield_combinations"])
            }
        
        return results
    
    def infinite_search(self, base_query: str, **kwargs) -> Dict[str, Any]:
        """Perform infinite search with endless parameter combinations and learning"""
        print(f"🚀 LAUNCHING INFINITE SEARCH: {base_query}")
        print("=" * 80)
        
        # Generate all possible parameter combinations
        all_combinations = self._generate_parameter_combinations(kwargs)
        print(f"🔍 Generated {len(all_combinations)} parameter combinations")
        
        results = {}
        total_results = 0
        learning_progress = []
        
        for i, params in enumerate(all_combinations):
            if i >= 100:  # Limit to prevent infinite loops
                print("⚠️  Reached search limit to prevent infinite loops")
                break
                
            print(f"\n🔬 Search {i+1}/{min(len(all_combinations), 100)}: {params}")
            
            # Perform search with current parameters
            search_result = self._search_with_parameters(base_query, params)
            
            if search_result:
                # Learn from this search
                self._learn_from_search_result(base_query, params, search_result, "infinite")
                
                results[f"combination_{i+1}"] = {
                    "parameters": params,
                    "results": search_result,
                    "result_count": sum(len(r) for r in search_result.values()),
                    "timestamp": datetime.now().isoformat(),
                    "learning_applied": True
                }
                total_results += results[f"combination_{i+1}"]["result_count"]
                
                # Track learning progress
                learning_progress.append({
                    "search_number": i + 1,
                    "quality": self._calculate_result_quality(search_result),
                    "patterns_learned": len(self.query_patterns),
                    "successful_combinations": len(self.adaptive_strategies["high_yield_combinations"])
                })
        
        # Add infinite search metadata with learning information
        results["metadata"] = {
            "total_combinations": len(all_combinations),
            "searches_performed": min(len(all_combinations), 100),
            "total_results": total_results,
            "search_type": "infinite_with_learning",
            "parameter_space_size": self._calculate_parameter_space_size(),
            "completion_time": datetime.now().isoformat(),
            "learning_progress": learning_progress,
            "final_learning_state": {
                "query_patterns": len(self.query_patterns),
                "successful_combinations": len(self.adaptive_strategies["high_yield_combinations"]),
                "search_strategies": len(self.search_strategies),
                "adaptive_strategies": self.adaptive_strategies
            }
        }
        
        # Save learned model after extensive search
        self._save_learned_model()
        
        return results
    
    def _generate_parameter_combinations(self, base_params: Dict) -> List[Dict]:
        """Generate all possible parameter combinations with learning influence"""
        combinations = []
        
        # Start with base parameters
        if base_params:
            combinations.append(base_params)
        
        # Use learned successful combinations
        if self.adaptive_strategies["high_yield_combinations"]:
            for combo_key in self.adaptive_strategies["high_yield_combinations"][:10]:  # Top 10
                combo_params = self._parse_parameter_key(combo_key)
                if combo_params:
                    combinations.append(combo_params)
        
        # Generate random combinations with learning bias
        for _ in range(40):  # Reduced from 50 to make room for learned combinations
            combo = {}
            for param_type, values in self.parameter_space.items():
                if random.random() < 0.3:  # 30% chance to include each parameter
                    # Bias towards more effective parameter values
                    if param_type in self.parameter_effectiveness and self.parameter_effectiveness[param_type]:
                        # Weighted selection based on learned effectiveness
                        effective_values = list(self.parameter_effectiveness[param_type].keys())
                        weights = list(self.parameter_effectiveness[param_type].values())
                        if effective_values and weights:
                            try:
                                combo[param_type] = random.choices(effective_values, weights=weights, k=1)[0]
                            except Exception:
                                combo[param_type] = random.choice(values)
                        else:
                            combo[param_type] = random.choice(values)
                    else:
                        combo[param_type] = random.choice(values)
            if combo:
                combinations.append(combo)
        
        # Generate systematic combinations for key parameters
        key_params = ["species", "brain_regions", "data_types"]
        for species in self.parameter_space["species"][:3]:  # First 3 species
            for brain_region in self.parameter_space["brain_regions"][:3]:  # First 3 regions
                for data_type in self.parameter_space["data_types"][:3]:  # First 3 data types
                    combinations.append({
                        "species": species,
                        "brain_regions": brain_region,
                        "data_types": data_type
                    })
        
        return combinations
    
    def _search_with_parameters(self, query: str, params: Dict) -> Optional[Dict]:
        """Perform search with specific parameters"""
        try:
            # Extract search parameters
            species = params.get("species")
            brain_regions = [params.get("brain_regions")] if params.get("brain_regions") else None
            data_types = [params.get("data_types")] if params.get("data_types") else None
            
            # Perform search
            results = self.manager.search_across_sources(
                query=query,
                data_types=data_types,
                species=species,
                brain_regions=brain_regions
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Search failed with params {params}: {e}")
            return None
    
    def _calculate_parameter_space_size(self) -> int:
        """Calculate total size of parameter space"""
        total = 1
        for values in self.parameter_space.values():
            total *= len(values)
        return total
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        return {
            "query_patterns_learned": len(self.query_patterns),
            "successful_combinations": len(self.adaptive_strategies["high_yield_combinations"]),
            "search_strategies": len(self.search_strategies),
            "parameter_effectiveness": {k: len(v) for k, v in self.parameter_effectiveness.items()},
            "result_quality_history": {k: len(v) for k, v in self.result_quality_metrics.items()},
            "adaptive_strategies": self.adaptive_strategies,
            "model_file": str(self.model_path),
            "learning_enabled": self.learning_enabled
        }
    
    def reset_learning(self):
        """Reset all learned knowledge (use with caution)"""
        print("⚠️  Resetting all learned knowledge...")
        self.query_patterns.clear()
        self.successful_combinations.clear()
        self.parameter_effectiveness.clear()
        self.search_strategies.clear()
        self.result_quality_metrics.clear()
        self.adaptive_strategies = {
            "high_yield_combinations": [],
            "query_expansion_patterns": [],
            "parameter_optimization": {},
            "search_depth_adjustment": 3,
            "result_quality_thresholds": {}
        }
        
        # Remove model file
        if self.model_path.exists():
            self.model_path.unlink()
        
        print("✅ Learning model reset complete")


def main():
    """Test the self-improving infinite search engine"""
    print("🧠 SELF-IMPROVING INFINITE SEARCH ENGINE TEST")
    print("=" * 60)
    
    # Create engine with learning enabled
    engine = SelfImprovingSearchEngine(learning_enabled=True)
    
    # Show initial learning state
    print("\n📊 Initial Learning State:")
    stats = engine.get_learning_stats()
    for key, value in stats.items():
        if key != "adaptive_strategies":
            print(f"  {key}: {value}")
    
    # Test intelligent search
    print("\n🧠 Testing intelligent search...")
    results = engine.intelligent_search("cortex", "intelligent")
    print(f"✅ Intelligent search completed")
    
    # Test infinite search with learning
    print("\n🚀 Testing infinite search with learning...")
    results = engine.infinite_search("neuron")
    print(f"✅ Infinite search with learning completed")
    
    # Show final learning state
    print("\n📊 Final Learning State:")
    stats = engine.get_learning_stats()
    for key, value in stats.items():
        if key != "adaptive_strategies":
            print(f"  {key}: {value}")
    
    # Show adaptive strategies
    print("\n🔧 Adaptive Strategies:")
    for key, value in stats["adaptive_strategies"].items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
