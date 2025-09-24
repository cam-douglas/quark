#!/usr/bin/env python3
"""
KPI Runner Module
=================
Executes KPI measurements and benchmarks.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class KPIRunner:
    """Execute and measure KPIs from checklist specifications."""
    
    def __init__(self, validation_root: Path):
        self.validation_root = validation_root
        self.checklists_dir = validation_root / "checklists"
        self.evidence_dir = validation_root / "evidence"
        
        # KPI measurement scripts mapping
        self.kpi_scripts = {
            "segmentation_dice": "measure_segmentation.py",
            "neuron_count_error_pct": "measure_neuron_count.py",
            "synapse_density_ratio": "measure_synapse_density.py",
            "reasoning_accuracy": "run_reasoning_benchmark.py",
            "ECE": "measure_calibration.py"
        }
    
    def parse_kpi_specifications(self, checklist_path: Path) -> List[Dict[str, Any]]:
        """Parse KPI specifications from checklist."""
        kpis = []
        
        if not checklist_path.exists():
            return kpis
        
        with open(checklist_path) as f:
            lines = f.readlines()
            
        in_kpi_section = False
        current_kpi = {}
        
        for line in lines:
            if "### KPI Specifications" in line:
                in_kpi_section = True
            elif in_kpi_section:
                if "**KPI:**" in line:
                    if current_kpi:
                        kpis.append(current_kpi)
                    current_kpi = {"name": line.split("**KPI:**")[1].strip()}
                elif "**Target:**" in line:
                    current_kpi["target"] = line.split("**Target:**")[1].strip()
                elif "**Benchmark:**" in line:
                    current_kpi["benchmark"] = line.split("**Benchmark:**")[1].strip()
                elif "**Measurement:**" in line:
                    current_kpi["measurement"] = line.split("**Measurement:**")[1].strip()
                elif line.startswith("##") and "KPI" not in line:
                    break
        
        if current_kpi:
            kpis.append(current_kpi)
        
        return kpis
    
    def run_kpi_measurement(self, kpi: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single KPI measurement."""
        result = {
            "kpi": kpi["name"],
            "target": kpi.get("target", "N/A"),
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Check if we have a measurement script
        script_name = self.kpi_scripts.get(kpi["name"])
        
        if script_name:
            script_path = Path(__file__).parent.parent.parent.parent / "scripts" / script_name
            
            if script_path.exists():
                try:
                    # Run measurement script
                    start_time = time.time()
                    proc = subprocess.run(
                        ["python", str(script_path)],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    result["execution_time"] = time.time() - start_time
                    
                    if proc.returncode == 0:
                        # Parse output for metric value
                        try:
                            # Assume script outputs JSON
                            output_data = json.loads(proc.stdout)
                            result["value"] = output_data.get("value")
                            result["status"] = "success"
                        except json.JSONDecodeError:
                            # Try to extract numeric value
                            for line in proc.stdout.split("\n"):
                                if "result" in line.lower() or "value" in line.lower():
                                    # Extract number from line
                                    import re
                                    numbers = re.findall(r"[-+]?\d*\.?\d+", line)
                                    if numbers:
                                        result["value"] = float(numbers[0])
                                        result["status"] = "success"
                                        break
                    else:
                        result["status"] = "error"
                        result["error"] = proc.stderr
                
                except subprocess.TimeoutExpired:
                    result["status"] = "timeout"
                except Exception as e:
                    result["status"] = "error"
                    result["error"] = str(e)
            else:
                result["status"] = "script_not_found"
        else:
            result["status"] = "no_script_defined"
            result["value"] = "MANUAL_MEASUREMENT_REQUIRED"
        
        return result
    
    def run_all_kpis(self, checklist_name: str) -> Dict[str, Any]:
        """Run all KPIs for a checklist."""
        checklist_path = self.checklists_dir / f"{checklist_name}.md"
        kpis = self.parse_kpi_specifications(checklist_path)
        
        results = {
            "checklist": checklist_name,
            "timestamp": datetime.now().isoformat(),
            "kpis": {}
        }
        
        print(f"\n🎯 Running {len(kpis)} KPIs for {checklist_name}")
        
        for kpi in kpis:
            print(f"  → Measuring {kpi['name']}...")
            result = self.run_kpi_measurement(kpi)
            results["kpis"][kpi["name"]] = result
            
            # Check against target
            if "value" in result and "target" in kpi:
                if self._check_target(result["value"], kpi["target"]):
                    print(f"    ✅ PASS: {result['value']} {kpi['target']}")
                else:
                    print(f"    ❌ FAIL: {result['value']} (target: {kpi['target']})")
        
        return results
    
    def _check_target(self, value: Any, target: str) -> bool:
        """Check if value meets target specification."""
        try:
            # Parse target (e.g., "≥ 0.80", "< 0.02")
            import re
            match = re.match(r"([<>=≤≥]+)\s*([\d.]+)", target)
            if match:
                op, threshold = match.groups()
                threshold = float(threshold)
                value = float(value)
                
                if op in ["≥", ">="]:
                    return value >= threshold
                elif op in [">"]:
                    return value > threshold
                elif op in ["≤", "<="]:
                    return value <= threshold
                elif op in ["<"]:
                    return value < threshold
                elif op in ["=", "=="]:
                    return abs(value - threshold) < 0.0001
        except (ValueError, TypeError):
            pass
        
        return False
    
    def generate_measurement_script(self, kpi_name: str) -> str:
        """Generate a template measurement script for a KPI."""
        template = f'''#!/usr/bin/env python3
"""
Measurement script for {kpi_name}
Auto-generated by Quark Validation System
"""

import json
import sys
from pathlib import Path

def measure_{kpi_name.lower().replace(" ", "_")}():
    """Measure {kpi_name}."""
    # TODO: Implement measurement logic
    value = 0.0
    
    return {{
        "kpi": "{kpi_name}",
        "value": value,
        "unit": "ratio",
        "status": "success"
    }}

if __name__ == "__main__":
    result = measure_{kpi_name.lower().replace(" ", "_")}()
    print(json.dumps(result))
'''
        return template
