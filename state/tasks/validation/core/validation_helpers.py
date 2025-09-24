#!/usr/bin/env python3
"""
Validation Helpers Module
=========================
Helper functions for manual validation guidance.
"""

import json
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class ValidationHelpers:
    """Helper functions for validation tasks."""
    
    @staticmethod
    def guide_measurement_recording(current_sprint: Dict) -> None:
        """Guide user through recording their manual measurements."""
        print("\n📝 Recording Your Measurements")
        print("-" * 30)
        
        measurements = {}
        
        print("\nEnter your KPI measurements (or 'skip' to skip a measurement):")
        
        # Get common KPIs based on scope
        kpis = ValidationHelpers.get_expected_kpis(current_sprint.get("scope", ""))
        
        for kpi_name, target in kpis:
            value = input(f"\n{kpi_name} (target: {target}): ").strip()
            if value.lower() != 'skip' and value:
                try:
                    # Try to parse as number
                    measurements[kpi_name] = float(value)
                    # Check against target
                    if ValidationHelpers.check_against_target(measurements[kpi_name], target):
                        print(f"   ✅ Meets target")
                    else:
                        print(f"   ⚠️ Does not meet target")
                except ValueError:
                    measurements[kpi_name] = value
                    print(f"   📝 Recorded as: {value}")
        
        if measurements:
            current_sprint["measurements"] = measurements
            print("\n✅ Measurements recorded")
            print("\n📊 Your measurements:")
            for kpi, value in measurements.items():
                print(f"   • {kpi}: {value}")
    
    @staticmethod
    def create_evidence_templates(run_dir: Path, current_sprint: Dict) -> None:
        """Create template evidence files with instructions."""
        # metrics.json template
        metrics_template = {
            "checklist": current_sprint.get("scope", "UNKNOWN"),
            "timestamp": datetime.now().isoformat(),
            "kpis": current_sprint.get("measurements", {}),
            "_instructions": "Add your KPI measurements here"
        }
        
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_template, f, indent=2)
        
        # config.yaml template
        with open(run_dir / "config.yaml", "w") as f:
            f.write("# Configuration used for validation\n")
            f.write("# Add your configuration parameters here\n\n")
            f.write("model_config:\n")
            f.write("  # Add model parameters\n\n")
            f.write("training_config:\n")
            f.write("  # Add training parameters\n\n")
            f.write("validation_config:\n")
            f.write("  # Add validation parameters\n")
        
        # seeds.txt template
        with open(run_dir / "seeds.txt", "w") as f:
            f.write("# Random seeds used for reproducibility\n")
            f.write("# Format: component=seed\n\n")
            f.write("numpy=42\n")
            f.write("torch=42\n")
            f.write("random=42\n")
        
        # environment.txt template
        with open(run_dir / "environment.txt", "w") as f:
            f.write("# System and software environment\n")
            f.write("# Record versions of all relevant software\n\n")
            f.write(f"OS: {platform.system()} {platform.release()}\n")
            f.write(f"Python: {platform.python_version()}\n")
            f.write("\n# Add package versions:\n")
            f.write("# pip freeze > environment.txt\n")
        
        # logs.txt template
        with open(run_dir / "logs.txt", "w") as f:
            f.write("# Validation execution logs\n")
            f.write("# Paste your validation run output here\n\n")
            f.write(f"Validation run: {run_dir.name}\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write("\n# Add your validation logs below:\n")
        
        # dataset_hashes.txt template
        with open(run_dir / "dataset_hashes.txt", "w") as f:
            f.write("# Dataset integrity hashes\n")
            f.write("# Format: hash  filename\n\n")
            f.write("# Generate with: sha256sum your_data_files\n")
    
    @staticmethod
    def get_expected_kpis(scope: str) -> List[Tuple[str, str]]:
        """Get expected KPIs based on scope."""
        # Common KPIs by stage/domain
        kpi_map = {
            "STAGE1": [
                ("segmentation_dice", "≥ 0.80"),
                ("neuron_count_error_pct", "≤ 10%"),
                ("gradient_smoothness", "≥ 0.85"),
            ],
            "STAGE2": [
                ("laminar_accuracy", "≥ 0.85"),
                ("migration_completion", "≥ 95%"),
                ("neuron_count_error_pct", "≤ 10%"),
            ],
            "STAGE3": [
                ("synapse_density_ratio", "0.8-1.2"),
                ("ocular_dominance_dprime", "≥ 1.5"),
                ("critical_period_timing", "correct"),
            ],
            "STAGE4": [
                ("conduction_velocity", "≥ 50 m/s"),
                ("small_world_sigma", "≥ 1.5"),
            ],
            "STAGE5": [
                ("pruning_completion_pct", "≥ 80%"),
                ("stroop_accuracy", "≥ 0.85"),
            ],
            "STAGE6": [
                ("agi_domain_score_avg", "≥ 0.80"),
                ("uptime_pct", "≥ 99.9%"),
            ],
            "INTEGRATIONS": [
                ("reasoning_accuracy", "≥ 0.75"),
                ("ECE", "≤ 0.02"),
                ("response_time_ms", "≤ 100"),
            ],
            "SYSTEM": [
                ("latency_p99", "≤ 100ms"),
                ("throughput_rps", "≥ 1000"),
                ("memory_usage_gb", "≤ 16"),
            ]
        }
        
        # Find matching KPIs
        for key, kpis in kpi_map.items():
            if key in scope.upper():
                return kpis
        
        # Default KPIs
        return [
            ("accuracy", "≥ 0.80"),
            ("calibration_ECE", "≤ 0.02"),
            ("latency_ms", "≤ 100"),
        ]
    
    @staticmethod
    def check_against_target(value: float, target: str) -> bool:
        """Check if value meets target specification."""
        try:
            import re
            # Parse target like "≥ 0.80" or "≤ 10%"
            match = re.match(r"([≥≤<>=]+)\s*([\d.]+)", target)
            if match:
                op, threshold = match.groups()
                threshold = float(threshold)
                
                if op in ["≥", ">="]:
                    return value >= threshold
                elif op in ["≤", "<="]:
                    return value <= threshold
                elif op == ">":
                    return value > threshold
                elif op == "<":
                    return value < threshold
                elif op in ["=", "=="]:
                    return abs(value - threshold) < 0.001
            
            # Handle range targets like "0.8-1.2"
            range_match = re.match(r"([\d.]+)-([\d.]+)", target)
            if range_match:
                low, high = map(float, range_match.groups())
                return low <= value <= high
        except:
            pass
        return True  # Default to pass if can't parse
    
    @staticmethod
    def show_manual_validation_steps() -> None:
        """Show manual validation steps."""
        print("\n🔧 MANUAL VALIDATION - Perform Your Measurements")
        print("-" * 40)
        
        print("\n📝 Manual Validation Steps:")
        print("\n1. RUN YOUR IMPLEMENTATION")
        print("   • Execute your code/model")
        print("   • Run your test suite")
        print("   • Perform experiments")
        
        print("\n2. MEASURE EACH KPI")
        print("   • Use appropriate measurement tools")
        print("   • Record exact values")
        print("   • Note measurement conditions")
        
        print("\n3. COMPARE AGAINST TARGETS")
        print("   • Check if each KPI meets its threshold")
        print("   • Document any failures")
        print("   • Note areas needing improvement")
        
        print("\n4. DOCUMENT RESULTS")
        print("   • Record all measurements")
        print("   • Save configuration used")
        print("   • Note random seeds")
        print("   • Capture system environment")
        
        print("\n" + "=" * 50)
        print("⏸️  PAUSE HERE TO PERFORM MANUAL VALIDATION")
        print("=" * 50)
