#!/usr/bin/env python3
"""
Brain Simulation Stable Configuration Validator

Validates that all critical configuration settings are maintained
to ensure stable brain simulation execution.
"""

import sys
from pathlib import Path

def validate_alphagenome_config():
    """Validate AlphaGenome threading configuration."""
    config_path = Path("brain/modules/alphagenome_integration/configuration/config_core.py")

    if not config_path.exists():
        return False, "AlphaGenome config file not found"

    with open(config_path, 'r') as f:
        content = f.read()

    if "multithreading_enabled: bool = False" not in content:
        return False, "AlphaGenome multithreading not disabled"

    return True, "AlphaGenome threading properly disabled"

def validate_brain_main_defaults():
    """Validate brain_main.py default arguments."""
    brain_main_path = Path("brain/brain_main.py")

    if not brain_main_path.exists():
        return False, "brain_main.py not found"

    with open(brain_main_path, 'r') as f:
        content = f.read()

    if 'default=float(\'inf\')' not in content:
        return False, "Infinite steps not set as default"

    if 'default=True' not in content or '--viewer' not in content:
        return False, "MuJoCo viewer not enabled by default"

    return True, "brain_main.py defaults properly configured"

def validate_step_safety_checks():
    """Validate safety checks in step_part1.py."""
    step_path = Path("brain/core/step_part1.py")

    if not step_path.exists():
        return False, "step_part1.py not found"

    with open(step_path, 'r') as f:
        content = f.read()

    required_checks = [
        "hasattr(self, 'auditory_cortex')",
        "hasattr(self, 'somatosensory_cortex')",
        "hasattr(self, 'motor_cortex')",
        "hasattr(self, 'oculomotor_cortex')",
        "hasattr(self, 'cerebellum')"
    ]

    for check in required_checks:
        if check not in content:
            return False, f"Missing safety check: {check}"

    return True, "All module safety checks present"

def validate_quark_rules():
    """Validate QuarkDriver rules configuration."""
    driver_path = Path("state/quark_state_system/quark_driver.py")

    if not driver_path.exists():
        return False, "quark_driver.py not found"

    with open(driver_path, 'r') as f:
        content = f.read()

    if '".quark", "rules"' not in content:
        return False, "QuarkDriver not using .quark/rules/ directory"

    return True, "QuarkDriver properly configured for .quark/rules/"

def main():
    """Run all validation checks."""
    print("🔍 Validating Brain Simulation Stable Configuration...")
    print("=" * 60)

    validators = [
        ("AlphaGenome Threading", validate_alphagenome_config),
        ("Brain Main Defaults", validate_brain_main_defaults),
        ("Module Safety Checks", validate_step_safety_checks),
        ("Quark Rules Config", validate_quark_rules),
    ]

    all_passed = True

    for name, validator in validators:
        try:
            passed, message = validator()
            status = "✅" if passed else "❌"
            print(f"{status} {name}: {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: Error during validation - {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 All stability configurations validated successfully!")
        print("🧠 Brain simulation is ready for stable execution.")
        return 0
    else:
        print("⚠️  Some stability configurations failed validation.")
        print("🔧 Please review and fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
