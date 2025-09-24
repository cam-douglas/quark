"""
Simulation Handler Module
=========================
Handles brain simulation commands.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


class SimulationHandler:
    """Handles brain simulation operations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.simulation_script = self.project_root / 'run_brain_simulation.py'
        
    def route_command(self, action: str, params: Dict) -> int:
        """Route simulation commands."""
        if action == 'simulate':
            return self.simulate_component(params.get('component', 'full'))
        elif action == 'run':
            return self.run_full_simulation(params)
        elif action == 'status':
            return self.check_simulation_status()
        elif action == 'analyze':
            return self.analyze_results()
        else:
            return self.show_help()
    
    def simulate_component(self, component: str) -> int:
        """Simulate specific brain component."""
        # Delegate to brain handler for brain components
        from .brain_handler import BrainHandler
        brain = BrainHandler(self.project_root)
        
        # Check if it's a brain component
        brain_components = brain._get_all_components()
        
        if component in brain_components or component == 'full':
            return brain.simulate_component({'component': component})
        else:
            # Non-brain simulations
            print(f"\n🔬 Simulating: {component}")
            print("=" * 50)
            
            if self.simulation_script.exists():
                cmd = [sys.executable, str(self.simulation_script)]
                cmd.extend(['--component', component])
                return subprocess.run(cmd).returncode
            else:
                print("⚠️ Simulation script not found")
                return 1
    
    def run_full_simulation(self, params: Dict) -> int:
        """Run full brain simulation."""
        print("\n🧠 Running Full Brain Simulation")
        print("=" * 50)
        # This will now be handled by brain_handler's simulate_component('full')
        from .brain_handler_wrapper import BrainHandler
        brain = BrainHandler(self.project_root)
        return brain.simulate_component({'component': 'full', **params})
    
    def check_status(self) -> int:
        """Check simulation status."""
        print("\n📊 Simulation Status")
        print("=" * 50)
        
        # Check for running processes
        try:
            result = subprocess.run(['pgrep', '-f', 'brain_simulation'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Simulation is running")
                print(f"   PID: {result.stdout.strip()}")
            else:
                print("⚠️ No simulation currently running")
        except:
            print("⚠️ Could not check simulation status")
        
        # Check for recent results
        results_dir = self.project_root / 'data' / 'experiments'
        if results_dir.exists():
            recent_files = sorted(results_dir.glob('**/simulation_*.json'))[-3:]
            if recent_files:
                print("\n📁 Recent simulation results:")
                for f in recent_files:
                    print(f"   • {f.name}")
        
        return 0
    
    def analyze_results(self) -> int:
        """Analyze simulation results."""
        print("\n📈 Analyzing Simulation Results")
        print("=" * 50)
        
        # Look for analysis scripts
        analysis_script = self.project_root / 'brain' / 'tools' / 'analyze_simulation.py'
        if analysis_script.exists():
            cmd = [sys.executable, str(analysis_script)]
            return subprocess.run(cmd).returncode
        else:
            print("📊 Summary of recent simulations:")
            print("   • Check data/experiments/ for results")
            print("   • Use 'todo dashboard' for visualization")
        
        return 0
    
    def show_help(self) -> int:
        """Show simulation help."""
        print("""
🧠 Brain Simulation Commands:
  todo simulate cerebellum     → Simulate cerebellum component
  todo simulate cortex         → Simulate cortex component  
  todo simulate brainstem      → Simulate brainstem component
  todo run brain simulation    → Run full brain simulation
  todo simulation status       → Check simulation status
  todo analyze simulation      → Analyze simulation results
""")
        return 0
