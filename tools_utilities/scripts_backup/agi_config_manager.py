#!/usr/bin/env python3
"""
AGI Configuration Manager
Centralized configuration management for AGI-enhanced brain simulation
Handles optimization, robustness, and capability domain configurations

Features:
- Dynamic configuration loading and validation
- Optimization level management
- Robustness configuration
- AGI capability domain toggling
- Performance tuning parameters
- Real-time configuration updates

Usage:
    from config.agi_config_manager import AGIConfigManager
    
    config_manager = AGIConfigManager("config/agi_enhanced_connectome.yaml")
    config = config_manager.get_optimized_config("high", "high")
    config_manager.validate_configuration(config)
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Optimization level enumeration"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RobustnessLevel(Enum):
    """Robustness level enumeration"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class AGICapabilityConfig:
    """Configuration for AGI capability domain"""
    name: str
    enabled: bool = True
    optimization_level: str = "medium"
    robustness_level: str = "medium"
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    performance_targets: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModuleConfig:
    """Configuration for individual brain module"""
    name: str
    module_type: str
    agi_capabilities: List[str] = field(default_factory=list)
    optimization_level: str = "medium"
    robustness_level: str = "medium"
    connections: Dict[str, List[str]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)

@dataclass
class SystemConfig:
    """Overall system configuration"""
    name: str
    version: str = "1.0"
    optimization_level: str = "medium"
    robustness_level: str = "medium"
    biological_fidelity: float = 0.9
    agi_domains: Dict[str, bool] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)

class AGIConfigManager:
    """Central manager for AGI configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.base_config = {}
        self.current_config = {}
        self.capability_registry = {}
        self.module_registry = {}
        self.optimization_profiles = {}
        self.robustness_profiles = {}
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        # Load configuration if path provided
        if self.config_path and self.config_path.exists():
            self.load_configuration(self.config_path)
        else:
            logger.warning("No configuration file provided, using defaults")
            self.current_config = self._get_default_config()
    
    def _initialize_default_configs(self):
        """Initialize default configuration profiles"""
        # Optimization profiles
        self.optimization_profiles = {
            OptimizationLevel.NONE.value: {
                "message_batching": False,
                "parallel_processing": False,
                "memory_management": False,
                "adaptive_parameters": False,
                "cache_size": 100,
                "batch_size": 1,
                "thread_pool_size": 1
            },
            OptimizationLevel.LOW.value: {
                "message_batching": True,
                "parallel_processing": False,
                "memory_management": True,
                "adaptive_parameters": False,
                "cache_size": 500,
                "batch_size": 10,
                "thread_pool_size": 2
            },
            OptimizationLevel.MEDIUM.value: {
                "message_batching": True,
                "parallel_processing": True,
                "memory_management": True,
                "adaptive_parameters": True,
                "cache_size": 1000,
                "batch_size": 50,
                "thread_pool_size": 4
            },
            OptimizationLevel.HIGH.value: {
                "message_batching": True,
                "parallel_processing": True,
                "memory_management": True,
                "adaptive_parameters": True,
                "cache_size": 5000,
                "batch_size": 100,
                "thread_pool_size": 8,
                "advanced_caching": True,
                "prediction_prefetch": True,
                "compression": True
            }
        }
        
        # Robustness profiles
        self.robustness_profiles = {
            RobustnessLevel.NONE.value: {
                "fault_tolerance": False,
                "error_recovery": False,
                "consistency_checking": False,
                "safety_constraints": False,
                "redundancy_factor": 1,
                "validation_frequency": 0
            },
            RobustnessLevel.LOW.value: {
                "fault_tolerance": True,
                "error_recovery": True,
                "consistency_checking": False,
                "safety_constraints": True,
                "redundancy_factor": 1,
                "validation_frequency": 100
            },
            RobustnessLevel.MEDIUM.value: {
                "fault_tolerance": True,
                "error_recovery": True,
                "consistency_checking": True,
                "safety_constraints": True,
                "redundancy_factor": 2,
                "validation_frequency": 50
            },
            RobustnessLevel.HIGH.value: {
                "fault_tolerance": True,
                "error_recovery": True,
                "consistency_checking": True,
                "safety_constraints": True,
                "redundancy_factor": 3,
                "validation_frequency": 10,
                "advanced_monitoring": True,
                "predictive_fault_detection": True,
                "automatic_recovery": True
            }
        }
    
    def load_configuration(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            self.base_config = config
            self.current_config = copy.deepcopy(config)
            self._parse_configuration(config)
            
            logger.info(f"Configuration loaded from: {config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _parse_configuration(self, config: Dict[str, Any]):
        """Parse and register configuration components"""
        # Parse system configuration
        if "system_config" in config:
            system_cfg = config["system_config"]
            self.system_config = SystemConfig(
                name=system_cfg.get("name", "AGI Brain"),
                version=system_cfg.get("version", "1.0"),
                optimization_level=system_cfg.get("optimization_level", "medium"),
                robustness_level=system_cfg.get("robustness_level", "medium"),
                biological_fidelity=system_cfg.get("biological_fidelity", 0.9),
                agi_domains=system_cfg.get("agi_domains", {}),
                performance_targets=system_cfg.get("performance_targets", {})
            )
        
        # Parse module configurations
        if "modules" in config:
            for module_name, module_cfg in config["modules"].items():
                self.module_registry[module_name] = ModuleConfig(
                    name=module_name,
                    module_type=module_cfg.get("type", "AGIEnhancedModule"),
                    agi_capabilities=module_cfg.get("agi_capabilities", []),
                    optimization_level=module_cfg.get("optimization_level", "medium"),
                    robustness_level=module_cfg.get("robustness_level", "medium"),
                    connections=module_cfg.get("connections", {}),
                    parameters=module_cfg.get("parameters", {}),
                    performance_targets=module_cfg.get("performance_targets", {})
                )
        
        # Parse AGI capability configurations
        agi_domains = config.get("system_config", {}).get("agi_domains", {})
        for domain, enabled in agi_domains.items():
            self.capability_registry[domain] = AGICapabilityConfig(
                name=domain,
                enabled=enabled,
                optimization_level=self.system_config.optimization_level,
                robustness_level=self.system_config.robustness_level
            )
    
    def get_optimized_config(self, optimization_level: str, robustness_level: str) -> Dict[str, Any]:
        """Get configuration optimized for specific levels"""
        # Start with base configuration
        optimized_config = copy.deepcopy(self.current_config)
        
        # Apply optimization profile
        opt_profile = self.optimization_profiles.get(optimization_level, {})
        rob_profile = self.robustness_profiles.get(robustness_level, {})
        
        # Update system configuration
        if "system_config" not in optimized_config:
            optimized_config["system_config"] = {}
        
        optimized_config["system_config"]["optimization_level"] = optimization_level
        optimized_config["system_config"]["robustness_level"] = robustness_level
        
        # Apply optimization settings
        if "optimization" not in optimized_config:
            optimized_config["optimization"] = {}
        
        optimized_config["optimization"].update(opt_profile)
        
        # Apply robustness settings
        if "robustness" not in optimized_config:
            optimized_config["robustness"] = {}
        
        optimized_config["robustness"].update(rob_profile)
        
        # Update module configurations
        if "modules" in optimized_config:
            for module_name, module_config in optimized_config["modules"].items():
                module_config["optimization_level"] = optimization_level
                module_config["robustness_level"] = robustness_level
                
                # Apply performance tuning
                self._apply_performance_tuning(module_config, optimization_level, robustness_level)
        
        return optimized_config
    
    def _apply_performance_tuning(self, module_config: Dict[str, Any], opt_level: str, rob_level: str):
        """Apply performance tuning to module configuration"""
        # Adjust parameters based on optimization level
        if "parameters" not in module_config:
            module_config["parameters"] = {}
        
        params = module_config["parameters"]
        
        if opt_level == OptimizationLevel.HIGH.value:
            # High optimization settings
            params.setdefault("cache_size", 5000)
            params.setdefault("batch_processing", True)
            params.setdefault("prefetch_enabled", True)
            params.setdefault("compression_enabled", True)
        elif opt_level == OptimizationLevel.MEDIUM.value:
            # Medium optimization settings
            params.setdefault("cache_size", 1000)
            params.setdefault("batch_processing", True)
            params.setdefault("prefetch_enabled", False)
        elif opt_level == OptimizationLevel.LOW.value:
            # Low optimization settings
            params.setdefault("cache_size", 500)
            params.setdefault("batch_processing", False)
        
        # Adjust for robustness level
        if rob_level == RobustnessLevel.HIGH.value:
            params.setdefault("validation_enabled", True)
            params.setdefault("redundancy_checks", True)
            params.setdefault("error_recovery_enabled", True)
            params.setdefault("safety_bounds_checking", True)
        elif rob_level == RobustnessLevel.MEDIUM.value:
            params.setdefault("validation_enabled", True)
            params.setdefault("error_recovery_enabled", True)
        
        # Set performance targets
        if "performance_targets" not in module_config:
            module_config["performance_targets"] = {}
        
        targets = module_config["performance_targets"]
        
        if opt_level == OptimizationLevel.HIGH.value:
            targets.setdefault("target_latency", 5)  # ms
            targets.setdefault("throughput_target", 2000)  # messages/sec
        elif opt_level == OptimizationLevel.MEDIUM.value:
            targets.setdefault("target_latency", 10)  # ms
            targets.setdefault("throughput_target", 1000)  # messages/sec
        else:
            targets.setdefault("target_latency", 50)  # ms
            targets.setdefault("throughput_target", 200)  # messages/sec
        
        if rob_level == RobustnessLevel.HIGH.value:
            targets.setdefault("reliability_target", 0.999)
            targets.setdefault("error_rate_target", 0.001)
        elif rob_level == RobustnessLevel.MEDIUM.value:
            targets.setdefault("reliability_target", 0.99)
            targets.setdefault("error_rate_target", 0.01)
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration for consistency and completeness"""
        errors = []
        
        try:
            # Validate system configuration
            if "system_config" not in config:
                errors.append("Missing system_config section")
            else:
                sys_cfg = config["system_config"]
                
                # Check required fields
                if "name" not in sys_cfg:
                    errors.append("Missing system name in system_config")
                
                # Validate optimization level
                opt_level = sys_cfg.get("optimization_level", "medium")
                if opt_level not in [e.value for e in OptimizationLevel]:
                    errors.append(f"Invalid optimization level: {opt_level}")
                
                # Validate robustness level
                rob_level = sys_cfg.get("robustness_level", "medium")
                if rob_level not in [e.value for e in RobustnessLevel]:
                    errors.append(f"Invalid robustness level: {rob_level}")
                
                # Validate biological fidelity
                bio_fidelity = sys_cfg.get("biological_fidelity", 0.9)
                if not isinstance(bio_fidelity, (int, float)) or not 0 <= bio_fidelity <= 1:
                    errors.append("Biological fidelity must be between 0 and 1")
            
            # Validate modules
            if "modules" not in config:
                errors.append("Missing modules section")
            else:
                modules = config["modules"]
                if not modules:
                    errors.append("No modules defined")
                
                for module_name, module_cfg in modules.items():
                    # Check module type
                    if "type" not in module_cfg:
                        errors.append(f"Module {module_name}: missing type")
                    
                    # Validate AGI capabilities
                    agi_caps = module_cfg.get("agi_capabilities", [])
                    if not isinstance(agi_caps, list):
                        errors.append(f"Module {module_name}: agi_capabilities must be a list")
                    
                    # Validate connections
                    connections = module_cfg.get("connections", {})
                    if connections and not isinstance(connections, dict):
                        errors.append(f"Module {module_name}: connections must be a dictionary")
            
            # Validate neuromodulation
            if "neuromodulation" in config:
                neuro_cfg = config["neuromodulation"]
                for neurotransmitter in ["dopamine", "norepinephrine", "serotonin", "acetylcholine"]:
                    if neurotransmitter in neuro_cfg:
                        nt_cfg = neuro_cfg[neurotransmitter]
                        baseline = nt_cfg.get("baseline", 0.5)
                        if not isinstance(baseline, (int, float)) or not 0 <= baseline <= 1:
                            errors.append(f"Neuromodulation {neurotransmitter}: baseline must be between 0 and 1")
            
            # Validate energy budget
            if "energy_budget" in config:
                budget_cfg = config["energy_budget"]
                total_budget = budget_cfg.get("total_budget", 100.0)
                if not isinstance(total_budget, (int, float)) or total_budget <= 0:
                    errors.append("Energy budget total_budget must be positive")
                
                allocation = budget_cfg.get("allocation", {})
                if allocation:
                    total_allocation = sum(allocation.values())
                    if abs(total_allocation - total_budget) > 0.1:
                        errors.append(f"Energy allocation ({total_allocation}) doesn't match total budget ({total_budget})")
            
            # Validate stages
            if "stages" in config:
                stages_cfg = config["stages"]
                for stage_name, stage_cfg in stages_cfg.items():
                    duration = stage_cfg.get("duration", 100)
                    if not isinstance(duration, int) or duration <= 0:
                        errors.append(f"Stage {stage_name}: duration must be positive integer")
        
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_capability_config(self, capability_name: str) -> Optional[AGICapabilityConfig]:
        """Get configuration for specific AGI capability"""
        return self.capability_registry.get(capability_name)
    
    def get_module_config(self, module_name: str) -> Optional[ModuleConfig]:
        """Get configuration for specific module"""
        return self.module_registry.get(module_name)
    
    def update_capability(self, capability_name: str, enabled: bool, 
                         optimization_level: Optional[str] = None,
                         robustness_level: Optional[str] = None):
        """Update AGI capability configuration"""
        if capability_name in self.capability_registry:
            cap_config = self.capability_registry[capability_name]
            cap_config.enabled = enabled
            if optimization_level:
                cap_config.optimization_level = optimization_level
            if robustness_level:
                cap_config.robustness_level = robustness_level
        else:
            # Create new capability config
            self.capability_registry[capability_name] = AGICapabilityConfig(
                name=capability_name,
                enabled=enabled,
                optimization_level=optimization_level or "medium",
                robustness_level=robustness_level or "medium"
            )
        
        # Update current configuration
        self._update_current_config()
    
    def _update_current_config(self):
        """Update current configuration based on registry changes"""
        # Update AGI domains in system config
        if "system_config" not in self.current_config:
            self.current_config["system_config"] = {}
        if "agi_domains" not in self.current_config["system_config"]:
            self.current_config["system_config"]["agi_domains"] = {}
        
        for cap_name, cap_config in self.capability_registry.items():
            self.current_config["system_config"]["agi_domains"][cap_name] = cap_config.enabled
    
    def export_configuration(self, output_path: Union[str, Path], format: str = "yaml") -> bool:
        """Export current configuration to file"""
        output_path = Path(output_path)
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(self.current_config, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(self.current_config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Configuration exported to: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def get_performance_profile(self, optimization_level: str, robustness_level: str) -> Dict[str, Any]:
        """Get performance profile for given optimization and robustness levels"""
        opt_profile = self.optimization_profiles.get(optimization_level, {})
        rob_profile = self.robustness_profiles.get(robustness_level, {})
        
        # Combine profiles
        performance_profile = {
            "optimization": opt_profile,
            "robustness": rob_profile,
            "combined_score": self._calculate_performance_score(optimization_level, robustness_level)
        }
        
        return performance_profile
    
    def _calculate_performance_score(self, optimization_level: str, robustness_level: str) -> float:
        """Calculate combined performance score"""
        opt_scores = {
            OptimizationLevel.NONE.value: 0.0,
            OptimizationLevel.LOW.value: 0.3,
            OptimizationLevel.MEDIUM.value: 0.6,
            OptimizationLevel.HIGH.value: 1.0
        }
        
        rob_scores = {
            RobustnessLevel.NONE.value: 0.0,
            RobustnessLevel.LOW.value: 0.3,
            RobustnessLevel.MEDIUM.value: 0.6,
            RobustnessLevel.HIGH.value: 1.0
        }
        
        opt_score = opt_scores.get(optimization_level, 0.5)
        rob_score = rob_scores.get(robustness_level, 0.5)
        
        # Weighted combination (optimization slightly more weighted for performance)
        combined_score = (opt_score * 0.6) + (rob_score * 0.4)
        return combined_score
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default AGI configuration"""
        return {
            "system_config": {
                "name": "AGI-Enhanced Cognitive Brain",
                "version": "1.0",
                "optimization_level": "medium",
                "robustness_level": "medium",
                "biological_fidelity": 0.9,
                "agi_domains": {
                    "cognitive": True,
                    "perception": True,
                    "action": True,
                    "communication": True,
                    "social": True,
                    "metacognition": True,
                    "knowledge": True,
                    "robustness": True,
                    "creativity": True,
                    "implementation": True
                }
            },
            "modules": {
                "Architecture_Agent": {
                    "type": "AGIArchitectureAgent",
                    "agi_capabilities": ["all"],
                    "optimization_level": "medium",
                    "robustness_level": "medium"
                }
            },
            "optimization": self.optimization_profiles[OptimizationLevel.MEDIUM.value],
            "robustness": self.robustness_profiles[RobustnessLevel.MEDIUM.value]
        }
    
    def get_recommendations(self, target_performance: Dict[str, float]) -> Dict[str, str]:
        """Get configuration recommendations based on target performance"""
        recommendations = {}
        
        # Analyze target performance requirements
        target_latency = target_performance.get("latency", 50.0)  # ms
        target_throughput = target_performance.get("throughput", 500.0)  # messages/sec
        target_reliability = target_performance.get("reliability", 0.95)
        
        # Recommend optimization level
        if target_latency <= 10 and target_throughput >= 1000:
            recommendations["optimization_level"] = OptimizationLevel.HIGH.value
        elif target_latency <= 25 and target_throughput >= 500:
            recommendations["optimization_level"] = OptimizationLevel.MEDIUM.value
        elif target_latency <= 100:
            recommendations["optimization_level"] = OptimizationLevel.LOW.value
        else:
            recommendations["optimization_level"] = OptimizationLevel.NONE.value
        
        # Recommend robustness level
        if target_reliability >= 0.999:
            recommendations["robustness_level"] = RobustnessLevel.HIGH.value
        elif target_reliability >= 0.99:
            recommendations["robustness_level"] = RobustnessLevel.MEDIUM.value
        elif target_reliability >= 0.95:
            recommendations["robustness_level"] = RobustnessLevel.LOW.value
        else:
            recommendations["robustness_level"] = RobustnessLevel.NONE.value
        
        # Additional recommendations
        if target_latency <= 5:
            recommendations["notes"] = "Consider enabling advanced caching and prediction prefetch"
        if target_reliability >= 0.999:
            recommendations["notes"] = "High reliability requires redundancy and extensive validation"
        
        return recommendations

# Utility functions
def load_agi_config(config_path: str) -> AGIConfigManager:
    """Convenience function to load AGI configuration"""
    return AGIConfigManager(config_path)

def create_optimized_config(optimization_level: str = "medium", 
                          robustness_level: str = "medium") -> Dict[str, Any]:
    """Create optimized configuration with specified levels"""
    config_manager = AGIConfigManager()
    return config_manager.get_optimized_config(optimization_level, robustness_level)

def validate_agi_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate AGI configuration"""
    config_manager = AGIConfigManager()
    return config_manager.validate_configuration(config)

if __name__ == "__main__":
    # Example usage
    config_manager = AGIConfigManager("agi_enhanced_connectome.yaml")
    
    # Get high-performance configuration
    high_perf_config = config_manager.get_optimized_config("high", "high")
    
    # Validate configuration
    is_valid, errors = config_manager.validate_configuration(high_perf_config)
    
    if is_valid:
        print("Configuration is valid!")
    else:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Export optimized configuration
    config_manager.export_configuration("optimized_agi_config.yaml")
