#!/usr/bin/env python3
"""AlphaGenome Integration Configuration
Central configuration for all biological development components

Integration: This module participates in biological workflows via BiologicalSimulator and related analyses.
Rationale: Biological modules used via BiologicalSimulator and downstream analyses.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class AlphaGenomeConfig:
    """Main configuration for AlphaGenome integration"""

    # Repository settings
    repository_path: str = "/Users/camdouglas/quark/data/external/alphagenome"
    use_local_fallback: bool = True

    # API settings
    api_key: Optional[str] = None
    api_endpoint: str = "https://api.alphagenome.deepmind.com"  # Hypothetical
    max_requests_per_hour: int = 1000

    # Biological settings
    max_sequence_length: int = 1000000  # 1M base pairs
    default_resolution: str = "single_bp"
    conservation_threshold: float = 0.6
    regulatory_threshold: float = 0.5

    # Neural development focus
    neural_gene_sets_enabled: bool = True
    developmental_stages_enabled: bool = True
    morphogen_modeling_enabled: bool = True
    spatial_modeling_enabled: bool = True

    # Output settings
    save_predictions: bool = True
    cache_results: bool = True
    export_visualizations: bool = True
    detailed_logging: bool = True

    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0

    # File paths
    cache_directory: str = "/Users/camdouglas/quark/data_knowledge/models_artifacts/alphagenome_cache"
    export_directory: str = "/Users/camdouglas/quark/data_knowledge/models_artifacts/alphagenome_exports"
    log_directory: str = "/Users/camdouglas/quark/logs/alphagenome"

@dataclass
class BiologicalRulesConfig:
    """Configuration for biological development rules"""

    # Cell type transitions
    enforce_valid_transitions: bool = True
    allow_dedifferentiation: bool = False
    stem_cell_renewal_enabled: bool = True

    # Developmental timing
    enforce_temporal_constraints: bool = True
    developmental_stage_duration_hours: Dict[str, float] = None

    # Spatial constraints
    enforce_spatial_organization: bool = True
    max_cell_density: float = 1000.0  # cells per cubic mm
    tissue_boundary_enforcement: bool = True

    # Molecular constraints
    gene_expression_bounds: tuple = (0.0, 1.0)
    morphogen_concentration_bounds: tuple = (0.0, 10.0)
    conservation_weight_factor: float = 1.5

    # Validation settings
    validate_biological_accuracy: bool = True
    require_literature_support: bool = True
    error_on_violations: bool = False  # Warning instead of error

    def __post_init__(self):
        if self.developmental_stage_duration_hours is None:
            self.developmental_stage_duration_hours = {
                "neural_induction": 8.0,
                "neural_plate": 4.0,
                "neural_tube_closure": 16.0,
                "neural_proliferation": 24.0,
                "neuronal_migration": 48.0,
                "differentiation": 72.0,
                "synaptogenesis": 120.0,
                "circuit_refinement": 240.0
            }

@dataclass
class SimulationConfig:
    """Configuration for biological simulation"""

    # Time settings
    default_time_step: float = 0.1  # hours
    max_simulation_time: float = 168.0  # 1 week
    save_frequency: float = 1.0  # hours

    # Spatial settings
    spatial_resolution: float = 10.0  # micrometers
    spatial_dimensions: tuple = (1000.0, 1000.0, 1000.0)  # micrometers

    # Process toggles
    morphogen_diffusion_enabled: bool = True
    cell_migration_enabled: bool = True
    cell_division_enabled: bool = True
    apoptosis_enabled: bool = True

    # Visualization settings
    real_time_visualization: bool = False
    save_visualization_frames: bool = True
    visualization_resolution: int = 300  # DPI

    # Performance settings
    multithreading_enabled: bool = False  # STABLE: Disabled to prevent mutex lock issues
    gpu_acceleration: bool = False  # Not implemented yet
    memory_efficient_mode: bool = True

class ConfigurationManager:
    """Manages all AlphaGenome integration configurations"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "/Users/camdouglas/quark/brain_modules/alphagenome_integration/alphagenome_config.json"

        # Default configurations
        self.alphagenome_config = AlphaGenomeConfig()
        self.biological_rules_config = BiologicalRulesConfig()
        self.simulation_config = SimulationConfig()

        # Load API key from main config file if not set
        self._load_main_config_api_key()

        # Load from file if exists
        self.load_configuration()

        # Ensure directories exist
        self._ensure_directories()

    def _load_main_config_api_key(self):
        """Load AlphaGenome API key from main config file."""
        try:
            import configparser
            config = configparser.ConfigParser()
            main_config_path = "/Users/camdouglas/quark/brain/architecture/config/config.ini"

            if os.path.exists(main_config_path):
                config.read(main_config_path)
                api_key = config.get('API_KEYS', 'alphagenome_api_key', fallback=None)
                if api_key and 'YOUR_ALPHAGENOME_API_KEY' not in api_key:
                    self.alphagenome_config.api_key = api_key
                    os.environ['ALPHAGENOME_API_KEY'] = api_key
        except Exception:
            pass  # Silently fail if main config not available

    def load_configuration(self) -> bool:
        """Load configuration from JSON file"""

        if not os.path.exists(self.config_file):
            return False

        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)

            # Update configurations
            if "alphagenome" in config_data:
                for key, value in config_data["alphagenome"].items():
                    if hasattr(self.alphagenome_config, key):
                        setattr(self.alphagenome_config, key, value)

            if "biological_rules" in config_data:
                for key, value in config_data["biological_rules"].items():
                    if hasattr(self.biological_rules_config, key):
                        setattr(self.biological_rules_config, key, value)

            if "simulation" in config_data:
                for key, value in config_data["simulation"].items():
                    if hasattr(self.simulation_config, key):
                        setattr(self.simulation_config, key, value)

            return True

        except Exception as e:
            print(f"Warning: Failed to load configuration: {e}")
            return False

    def save_configuration(self) -> bool:
        """Save configuration to JSON file"""

        try:
            config_data = {
                "alphagenome": asdict(self.alphagenome_config),
                "biological_rules": asdict(self.biological_rules_config),
                "simulation": asdict(self.simulation_config),
                "last_updated": datetime.now().isoformat()
            }

            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)

            return True

        except Exception as e:
            print(f"Error: Failed to save configuration: {e}")
            return False

    def _ensure_directories(self):
        """Ensure all required directories exist"""

        directories = [
            self.alphagenome_config.cache_directory,
            self.alphagenome_config.export_directory,
            self.alphagenome_config.log_directory
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_dna_controller_config(self) -> Dict[str, Any]:
        """Get configuration for DNA controller"""

        return {
            "api_key": self.alphagenome_config.api_key,
            "sequence_length_limit": self.alphagenome_config.max_sequence_length,
            "resolution": self.alphagenome_config.default_resolution,
            "cache_directory": self.alphagenome_config.cache_directory,
            "save_predictions": self.alphagenome_config.save_predictions,
            "conservation_threshold": self.alphagenome_config.conservation_threshold
        }

    def get_cell_constructor_config(self) -> Dict[str, Any]:
        """Get configuration for cell constructor"""

        return {
            "enforce_valid_transitions": self.biological_rules_config.enforce_valid_transitions,
            "allow_dedifferentiation": self.biological_rules_config.allow_dedifferentiation,
            "validate_biology": self.biological_rules_config.validate_biological_accuracy,
            "max_cell_density": self.biological_rules_config.max_cell_density,
            "export_directory": self.alphagenome_config.export_directory
        }

    def get_genome_analyzer_config(self) -> Dict[str, Any]:
        """Get configuration for genome analyzer"""

        return {
            "neural_gene_sets_enabled": self.alphagenome_config.neural_gene_sets_enabled,
            "conservation_threshold": self.alphagenome_config.conservation_threshold,
            "regulatory_threshold": self.alphagenome_config.regulatory_threshold,
            "cache_results": self.alphagenome_config.cache_results,
            "parallel_processing": self.alphagenome_config.parallel_processing,
            "max_workers": self.alphagenome_config.max_workers
        }

    def get_biological_simulator_config(self) -> Dict[str, Any]:
        """Get configuration for biological simulator"""

        return {
            "time_step": self.simulation_config.default_time_step,
            "max_time": self.simulation_config.max_simulation_time,
            "spatial_resolution": self.simulation_config.spatial_resolution,
            "spatial_dimensions": self.simulation_config.spatial_dimensions,
            "morphogen_diffusion": self.simulation_config.morphogen_diffusion_enabled,
            "cell_migration": self.simulation_config.cell_migration_enabled,
            "cell_division": self.simulation_config.cell_division_enabled,
            "visualization_enabled": self.alphagenome_config.export_visualizations,
            "save_frequency": self.simulation_config.save_frequency
        }

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate all configuration settings"""

        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }

        # Check AlphaGenome repository
        if not os.path.exists(self.alphagenome_config.repository_path):
            validation_results["errors"].append(
                f"AlphaGenome repository not found: {self.alphagenome_config.repository_path}"
            )
            validation_results["valid"] = False

        # Check API key
        if not self.alphagenome_config.api_key:
            validation_results["warnings"].append(
                "No AlphaGenome API key provided - will use simulation mode"
            )

        # Check memory limits
        if self.alphagenome_config.memory_limit_gb < 4.0:
            validation_results["warnings"].append(
                "Memory limit < 4GB may cause performance issues"
            )

        # Check sequence length
        if self.alphagenome_config.max_sequence_length > 1000000:
            validation_results["warnings"].append(
                "Sequence length > 1M bp may exceed AlphaGenome limits"
            )

        # Check spatial dimensions
        spatial_vol = (
            self.simulation_config.spatial_dimensions[0] *
            self.simulation_config.spatial_dimensions[1] *
            self.simulation_config.spatial_dimensions[2]
        )

        if spatial_vol > 1e12:  # > 1 cubic mm
            validation_results["warnings"].append(
                "Large spatial volume may require significant computational resources"
            )

        # Recommendations
        if self.alphagenome_config.parallel_processing and self.alphagenome_config.max_workers == 1:
            validation_results["recommendations"].append(
                "Consider increasing max_workers for better parallel performance"
            )

        if not self.alphagenome_config.cache_results:
            validation_results["recommendations"].append(
                "Enable result caching to improve performance"
            )

        return validation_results

    def get_system_requirements(self) -> Dict[str, Any]:
        """Get system requirements for current configuration"""

        # Estimate memory requirements
        spatial_cells = (
            self.simulation_config.spatial_dimensions[0] *
            self.simulation_config.spatial_dimensions[1] *
            self.simulation_config.spatial_dimensions[2] /
            (self.simulation_config.spatial_resolution ** 3)
        )

        estimated_memory_gb = (
            spatial_cells * 0.001 +  # Cell data
            len(self.biological_rules_config.developmental_stage_duration_hours) * 0.1 +  # Stage data
            4.0  # Base requirements
        )

        # Estimate disk space
        simulation_duration = self.simulation_config.max_simulation_time
        save_frequency = self.simulation_config.save_frequency
        save_points = simulation_duration / save_frequency

        estimated_disk_gb = save_points * 0.01  # ~10MB per save point

        return {
            "minimum_memory_gb": 4.0,
            "recommended_memory_gb": max(8.0, estimated_memory_gb),
            "estimated_disk_space_gb": estimated_disk_gb,
            "cpu_cores_recommended": self.alphagenome_config.max_workers,
            "python_version": ">=3.8",
            "required_packages": [
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "matplotlib>=3.4.0",
                "scipy>=1.7.0"
            ]
        }

# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def configure_alphagenome_integration(
    api_key: Optional[str] = None,
    repository_path: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> bool:
    """Configure AlphaGenome integration with custom settings"""

    config_manager = get_config_manager()

    # Update API key
    if api_key:
        config_manager.alphagenome_config.api_key = api_key

    # Update repository path
    if repository_path:
        config_manager.alphagenome_config.repository_path = repository_path

    # Apply custom configuration
    if custom_config:
        for section, settings in custom_config.items():
            if section == "alphagenome":
                for key, value in settings.items():
                    if hasattr(config_manager.alphagenome_config, key):
                        setattr(config_manager.alphagenome_config, key, value)
            elif section == "biological_rules":
                for key, value in settings.items():
                    if hasattr(config_manager.biological_rules_config, key):
                        setattr(config_manager.biological_rules_config, key, value)
            elif section == "simulation":
                for key, value in settings.items():
                    if hasattr(config_manager.simulation_config, key):
                        setattr(config_manager.simulation_config, key, value)

    # Save configuration
    success = config_manager.save_configuration()

    if success:
        print("✅ AlphaGenome integration configured successfully")
    else:
        print("❌ Failed to save AlphaGenome configuration")

    return success

def validate_system_setup() -> Dict[str, Any]:
    """Validate complete system setup"""

    config_manager = get_config_manager()

    # Configuration validation
    config_validation = config_manager.validate_configuration()

    # System requirements check
    requirements = config_manager.get_system_requirements()

    # AlphaGenome availability check
    alphagenome_available = os.path.exists(
        os.path.join(config_manager.alphagenome_config.repository_path, "src", "alphagenome")
    )

    # Directory permissions check
    directories_writable = all([
        os.access(config_manager.alphagenome_config.cache_directory, os.W_OK),
        os.access(config_manager.alphagenome_config.export_directory, os.W_OK),
        os.access(config_manager.alphagenome_config.log_directory, os.W_OK)
    ])

    return {
        "configuration_valid": config_validation["valid"],
        "alphagenome_available": alphagenome_available,
        "directories_writable": directories_writable,
        "system_requirements": requirements,
        "configuration_issues": config_validation,
        "ready_for_use": (
            config_validation["valid"] and
            directories_writable and
            len(config_validation["errors"]) == 0
        )
    }

if __name__ == "__main__":
    print("🔧 AlphaGenome Integration Configuration")
    print("=" * 50)

    # Create configuration manager
    config_manager = get_config_manager()

    # Display current configuration
    print("\n📋 Current Configuration:")
    print(f"Repository Path: {config_manager.alphagenome_config.repository_path}")
    print(f"API Key: {'Set' if config_manager.alphagenome_config.api_key else 'Not set'}")
    print(f"Cache Directory: {config_manager.alphagenome_config.cache_directory}")
    print(f"Export Directory: {config_manager.alphagenome_config.export_directory}")
    print(f"Max Sequence Length: {config_manager.alphagenome_config.max_sequence_length:,} bp")

    # Validate configuration
    print("\n🔍 Configuration Validation:")
    validation = config_manager.validate_configuration()

    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  ❌ {error}")

    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️ {warning}")

    if validation['recommendations']:
        print("Recommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")

    # System requirements
    print("\n💻 System Requirements:")
    requirements = config_manager.get_system_requirements()

    print(f"Minimum Memory: {requirements['minimum_memory_gb']} GB")
    print(f"Recommended Memory: {requirements['recommended_memory_gb']} GB")
    print(f"Estimated Disk Space: {requirements['estimated_disk_space_gb']:.1f} GB")
    print(f"CPU Cores: {requirements['cpu_cores_recommended']}")

    # Validate complete system
    print("\n✅ Complete System Validation:")
    system_validation = validate_system_setup()

    print(f"Configuration Valid: {system_validation['configuration_valid']}")
    print(f"AlphaGenome Available: {system_validation['alphagenome_available']}")
    print(f"Directories Writable: {system_validation['directories_writable']}")
    print(f"Ready for Use: {system_validation['ready_for_use']}")

    # Save configuration
    print("\n💾 Saving Configuration...")
    success = config_manager.save_configuration()
    print(f"Configuration saved: {success}")

    print("\n🎉 AlphaGenome configuration setup complete!")
