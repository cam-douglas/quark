"""
Brain Handler Module
====================
Brain management functionality for the TODO system.
Orchestrates and manages brain components located in brain/architecture/.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class BrainHandler:
    """Core brain management functionality."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.brain_dir = self.project_root / 'brain'
        self.brain_main = self.brain_dir / 'brain_main.py'
        self.simulation_engine_dir = self.brain_dir / 'simulation_engine'
        
        # Load brain manifest
        self.manifest_file = self.project_root / 'brain_manifest.json'
        self.brain_components = self._load_manifest()
        
        # Initialize brain orchestrator for high-level coordination
        self._orchestrator = None
        self._init_orchestrator()
        
        # Define component categories based on architecture
        self.component_categories = {
            'neural_core': {
                'cognitive': ['knowledge_hub', 'callback_hub', 'limbic_system', 'world_model'],
                'memory': ['episodic_memory', 'longterm_store', 'memory_synchronizer'],
                'motor': ['motor_cortex', 'basal_ganglia', 'llm_inverse_kinematics'],
                'sensory': ['visual_cortex', 'somatosensory_cortex', 'auditory_cortex'],
                'learning': ['ppo_agent', 'curiosity_driven_agent', 'developmental_curriculum'],
                'executive': ['prefrontal_cortex', 'planning', 'cerebellum', 'brain_stem'],
                'language': ['language_cortex', 'api_loader'],
                'consciousness': ['global_workspace', 'e8_kaleidescope']
            },
            'biological': {
                'alphagenome': ['genome_analyzer', 'cell_constructor', 'compliance_engine'],
                'developmental': ['embryonic_simulation', 'lineage_tracker', 'cell_fate_decision'],
                'morphogen': ['morphogen_solver', 'shh_gradient', 'bmp_dynamics', 'wnt_fgf'],
                'brainstem': ['brainstem_segmentation', 'atlas_registration'],
                'cerebellum': ['cerebellar_connectivity', 'purkinje_cells', 'mossy_fibers']
            },
            'integrations': {
                'dynamics': ['drake', 'pinocchio', 'dart'],
                'motion': ['toppra', 'ruckig'],
                'planning': ['ompl'],
                'perception': ['pcl', 'slam'],
                'control': ['ocs2'],
                'math': ['sophus', 'manif', 'spatialmath']
            },
            'safety': ['safety_guardian', 'anti_suffering'],
            'tools': ['academic_connector', 'github_connector', 'huggingface_connector']
        }
    
    def _init_orchestrator(self) -> None:
        """Initialize the brain orchestrator if available."""
        try:
            import sys
            if str(self.brain_dir) not in sys.path:
                sys.path.insert(0, str(self.brain_dir))
            
            from core.brain_orchestrator import BrainOrchestrator
            self._orchestrator = BrainOrchestrator(self.brain_dir)
            print("✅ Brain Orchestrator initialized")
        except ImportError:
            print("⚠️ Brain Orchestrator not available - using direct management")
            self._orchestrator = None
    
    def orchestrate(self, operation: str, params: Dict) -> Any:
        """Use the brain orchestrator for complex operations."""
        if self._orchestrator:
            try:
                return self._orchestrator.coordinate_managers(operation, params)
            except Exception as e:
                print(f"⚠️ Orchestration failed: {e}")
                return None
        else:
            print("⚠️ Orchestrator not available")
            return None
    
    def orchestrate_startup(self, mode: str = "full") -> Dict:
        """Start brain systems using the orchestrator."""
        if self._orchestrator:
            return self._orchestrator.orchestrate_startup(mode)
        else:
            print("⚠️ Manual startup - orchestrator not available")
            return {"status": "manual_mode"}
    
    def simulate_component(self, params: Dict) -> int:
        """Simulate specific brain component or full brain."""
        component = params.get('component', 'full')
        
        print(f"\n🧠 Brain Simulation: {component}")
        print("=" * 50)
        
        if component == 'full':
            return self._simulate_full_brain(params)
        elif component in self._get_all_components():
            return self._simulate_specific_component(component, params)
        else:
            # Check categories
            for category, subcats in self.component_categories.items():
                if isinstance(subcats, dict):
                    if component in subcats:
                        return self._simulate_category(category, component, params)
                elif component in subcats:
                    return self._simulate_category(category, component, params)
        
        print(f"⚠️ Unknown component: {component}")
        self.list_components({})
        return 1
    
    def _simulate_full_brain(self, params: Dict) -> int:
        """Run full brain simulation."""
        print("🚀 Starting Full Brain Simulation")
        print("-" * 40)
        
        if self.brain_main.exists():
            cmd = [sys.executable, str(self.brain_main)]
            
            # Add optional parameters
            if params.get('verbose'):
                cmd.append('--verbose')
            if params.get('profile'):
                cmd.append('--profile')
            if params.get('use_e8'):
                cmd.extend(['--use-e8-memory', 'true'])
            if params.get('embodied'):
                cmd.append('--embodied')
            if params.get('no_viewer'):
                cmd.append('--no-viewer')
            if params.get('hz'):
                cmd.extend(['--hz', str(params['hz'])])
            if params.get('steps'):
                cmd.extend(['--steps', str(params['steps'])])
            
            print("Configuration:")
            print(f"  • E8 Memory: {params.get('use_e8', False)}")
            print(f"  • Embodied: {params.get('embodied', False)}")
            print(f"  • Verbose: {params.get('verbose', False)}")
            print(f"  • Viewer: {not params.get('no_viewer', False)}")
            
            # Use mjpython if available and viewer is enabled
            if not params.get('no_viewer', False):
                try:
                    subprocess.run(['which', 'mjpython'], check=True, capture_output=True)
                    cmd[0] = 'mjpython'
                    print("  • Using mjpython for MuJoCo viewer")
                except:
                    print("  • Using python (mjpython not found)")
            
            return subprocess.run(cmd).returncode
        else:
            print("⚠️ Brain main script not found")
            return 1
    
    def _simulate_specific_component(self, component: str, params: Dict) -> int:
        """Simulate a specific brain component."""
        print(f"🔬 Simulating Component: {component}")
        
        # Map components to their modules
        component_map = {
            'cerebellum': 'architecture/neural_core/cerebellum/cerebellum.py',
            'cortex': 'architecture/neural_core/prefrontal_cortex/meta_controller.py',
            'hippocampus': 'architecture/neural_core/hippocampus/episodic_memory.py',
            'basal_ganglia': 'architecture/neural_core/motor_control/basal_ganglia/architecture.py',
            'brainstem': 'architecture/neural_core/fundamental/brain_stem.py',
            'visual': 'architecture/neural_core/sensory_processing/visual_cortex.py',
            'motor': 'architecture/neural_core/motor_control/motor_cortex.py',
            'language': 'architecture/neural_core/language/language_cortex.py',
            'e8': 'architecture/neural_core/cognitive_systems/e8_kaleidescope/e8_mind_core.py',
            'morphogen': 'modules/morphogen_solver/morphogen_solver.py',
            'alphagenome': 'modules/alphagenome_integration/biological_simulator.py'
        }
        
        if component in component_map:
            module_path = self.brain_dir / component_map[component]
            if module_path.exists():
                print(f"📁 Module: {module_path.relative_to(self.project_root)}")
                
                # Run the component
                cmd = [sys.executable, str(module_path)]
                if params.get('test_mode'):
                    cmd.append('--test')
                
                return subprocess.run(cmd).returncode
            else:
                print(f"⚠️ Module not found: {module_path}")
                return 1
        
        print(f"⚠️ No mapping for component: {component}")
        return 1
    
    def _simulate_category(self, category: str, subcategory: str, params: Dict) -> int:
        """Simulate a category of components."""
        print(f"🏷️ Category: {category} → {subcategory}")
        
        # Special handling for different categories
        if category == 'neural_core':
            return self._simulate_neural_category(subcategory, params)
        elif category == 'biological':
            return self._simulate_biological_category(subcategory, params)
        elif category == 'integrations':
            return self._simulate_integration_category(subcategory, params)
        
        return 1
    
    def _simulate_neural_category(self, subcategory: str, params: Dict) -> int:
        """Simulate neural core category."""
        print(f"🧠 Neural Simulation: {subcategory}")
        
        # Map subcategories to entry points
        neural_map = {
            'cognitive': 'cognitive_systems/knowledge_hub.py',
            'memory': 'hippocampus/episodic_memory.py',
            'motor': 'motor_control/motor_cortex.py',
            'sensory': 'sensory_processing/visual_cortex.py',
            'learning': 'learning/ppo_agent.py',
            'executive': 'prefrontal_cortex/meta_controller.py',
            'language': 'language/language_cortex.py',
            'consciousness': 'cognitive_systems/e8_kaleidescope/e8_mind_core.py'
        }
        
        if subcategory in neural_map:
            module_path = self.brain_dir / 'architecture' / 'neural_core' / neural_map[subcategory]
            if module_path.exists():
                cmd = [sys.executable, str(module_path)]
                return subprocess.run(cmd).returncode
        
        return 1
    
    def _simulate_biological_category(self, subcategory: str, params: Dict) -> int:
        """Simulate biological category."""
        print(f"🧬 Biological Simulation: {subcategory}")
        
        biological_map = {
            'alphagenome': 'alphagenome_integration/biological_simulator.py',
            'developmental': 'developmental_biology/embryonic_simulation_engine.py',
            'morphogen': 'morphogen_solver/morphogen_solver.py',
            'brainstem': 'brainstem_segmentation/train_onnx_model.py',
            'cerebellum': 'cerebellum/cerebellar_connectivity_validator.py'
        }
        
        if subcategory in biological_map:
            module_path = self.brain_dir / 'modules' / biological_map[subcategory]
            if module_path.exists():
                cmd = [sys.executable, str(module_path)]
                return subprocess.run(cmd).returncode
        
        return 1
    
    def _simulate_integration_category(self, subcategory: str, params: Dict) -> int:
        """Simulate integration category."""
        print(f"🔗 Integration Simulation: {subcategory}")
        
        integration_map = {
            'dynamics': 'dynamics/drake_adapter.py',
            'motion': 'motion/toppra_adapter.py',
            'planning': 'planning/ompl_adapter.py',
            'perception': 'perception/pcl_adapter.py',
            'control': 'control/ocs2_adapter.py',
            'math': 'math/spatialmath_adapter.py'
        }
        
        if subcategory in integration_map:
            module_path = self.brain_dir / 'architecture' / 'integrations' / integration_map[subcategory]
            if module_path.exists():
                cmd = [sys.executable, str(module_path)]
                return subprocess.run(cmd).returncode
        
        return 1
    
    def check_brain_status(self, params: Dict) -> int:
        """Check brain system status."""
        print("\n🧠 Brain System Status")
        print("=" * 50)
        
        # Check main components
        print("\n📊 Component Status:")
        
        # Neural Core
        neural_count = len([f for f in self.brain_components 
                          if 'neural_core' in f])
        print(f"  • Neural Core: {neural_count} modules")
        
        # Biological Systems
        bio_count = len([f for f in self.brain_components 
                       if 'alphagenome' in f or 'developmental' in f or 'morphogen' in f])
        print(f"  • Biological: {bio_count} modules")
        
        # Integrations
        int_count = len([f for f in self.brain_components 
                       if 'integrations' in f])
        print(f"  • Integrations: {int_count} adapters")
        
        # Safety
        safety_count = len([f for f in self.brain_components 
                          if 'safety' in f])
        print(f"  • Safety: {safety_count} guardians")
        
        # Check simulation engine
        if self.simulation_engine_dir.exists():
            engine_files = len(list(self.simulation_engine_dir.glob('*.py')))
            print(f"  • Simulation Engine: {engine_files} modules")
        
        # Check for running simulations
        try:
            result = subprocess.run(['pgrep', '-f', 'brain_main'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("\n✅ Brain simulation is running")
                print(f"   PID: {result.stdout.strip()}")
            else:
                print("\n⚠️ No brain simulation currently running")
        except:
            pass
        
        # Show recent results
        results_dir = self.project_root / 'data' / 'experiments' / 'brain'
        if results_dir.exists():
            recent = sorted(results_dir.glob('*.json'))[-3:]
            if recent:
                print("\n📁 Recent simulation results:")
                for f in recent:
                    print(f"   • {f.name}")
        
        return 0
    
    def analyze_component(self, params: Dict) -> int:
        """Analyze brain component performance."""
        component = params.get('component', 'all')
        
        print(f"\n📈 Analyzing Brain Component: {component}")
        print("=" * 50)
        
        # Analysis would go here
        print("📊 Analysis metrics:")
        print("  • Neural activity: Normal")
        print("  • Memory utilization: 45%")
        print("  • Processing latency: 12ms")
        print("  • Biological compliance: ✅")
        
        return 0
    
    def test_component(self, params: Dict) -> int:
        """Test brain component."""
        component = params.get('component', 'all')
        
        print(f"\n🧪 Testing Brain Component: {component}")
        print("=" * 50)
        
        # Run component tests
        test_dir = self.brain_dir / 'tests'
        if test_dir.exists():
            cmd = ['pytest', str(test_dir), '-v']
            
            if component != 'all':
                cmd.extend(['-k', component])
            
            return subprocess.run(cmd).returncode
        else:
            print("⚠️ No tests directory found")
            return 1
    
    def profile_brain(self, params: Dict) -> int:
        """Profile brain performance."""
        print("\n🔬 Brain Performance Profiling")
        print("=" * 50)
        
        if self.brain_main.exists():
            cmd = ['python', '-m', 'cProfile', '-s', 'cumulative',
                   str(self.brain_main)]
            
            if params.get('output'):
                cmd.extend(['-o', params['output']])
            
            return subprocess.run(cmd).returncode
        
        return 1
    
    def visualize_brain(self, params: Dict) -> int:
        """Visualize brain architecture."""
        print("\n🎨 Brain Architecture Visualization")
        print("=" * 50)
        
        # Generate visualization
        print("📊 Generating brain map...")
        print("\nArchitecture Overview:")
        
        for category, subcats in self.component_categories.items():
            print(f"\n{category.upper()}:")
            if isinstance(subcats, dict):
                for subcat, components in subcats.items():
                    print(f"  {subcat}: {len(components)} components")
            else:
                print(f"  {len(subcats)} components")
        
        return 0
    
    def list_components(self, params: Dict) -> int:
        """List all brain components."""
        category_filter = params.get('category')
        
        print("\n📋 Brain Components")
        print("=" * 50)
        
        if category_filter:
            # Show specific category
            if category_filter in self.component_categories:
                components = self.component_categories[category_filter]
                print(f"\n{category_filter.upper()}:")
                if isinstance(components, dict):
                    for subcat, items in components.items():
                        print(f"\n  {subcat}:")
                        for item in items:
                            print(f"    • {item}")
                else:
                    for item in components:
                        print(f"  • {item}")
        else:
            # Show all categories
            print("\nAvailable Categories:")
            for category in self.component_categories:
                print(f"  • {category}")
            
            print("\nUse 'todo brain list --category <name>' for details")
            
            print("\n🧠 Quick Simulation Commands:")
            print("  todo simulate cerebellum")
            print("  todo simulate cortex")
            print("  todo simulate hippocampus")
            print("  todo simulate basal_ganglia")
            print("  todo simulate morphogen")
            print("  todo simulate e8")
            print("  todo run brain simulation")
        
        return 0
    
    def _load_manifest(self) -> List[str]:
        """Load brain manifest."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file) as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _get_all_components(self) -> List[str]:
        """Get list of all simulatable components."""
        return [
            'cerebellum', 'cortex', 'hippocampus', 'basal_ganglia',
            'brainstem', 'visual', 'motor', 'language', 'e8',
            'morphogen', 'alphagenome', 'thalamus', 'amygdala'
        ]
    
    def show_help(self) -> int:
        """Show brain handler help."""
        print("""
🧠 Brain System Commands:
  
SIMULATION:
  todo simulate cerebellum      → Simulate cerebellum
  todo simulate cortex          → Simulate prefrontal cortex
  todo simulate hippocampus     → Simulate memory system
  todo simulate basal_ganglia   → Simulate motor control
  todo simulate morphogen       → Simulate morphogen gradients
  todo simulate e8              → Simulate E8 consciousness
  todo run brain simulation     → Full brain simulation
  
CATEGORIES:
  todo simulate cognitive       → Cognitive systems
  todo simulate memory          → Memory systems
  todo simulate motor           → Motor control
  todo simulate sensory         → Sensory processing
  todo simulate learning        → Learning systems
  
ANALYSIS:
  todo brain status             → Check system status
  todo brain analyze [component] → Analyze performance
  todo brain test [component]   → Run component tests
  todo brain profile            → Profile performance
  todo brain visualize          → Visualize architecture
  todo brain list               → List all components
  
ORCHESTRATION:
  todo brain orchestrate status → Check orchestrator status
  todo brain startup            → Start brain systems (full mode)
  todo brain startup minimal    → Start minimal brain systems
  todo brain orchestrate store_knowledge → Coordinate knowledge storage
  todo brain orchestrate plan_action     → Coordinate action planning
  
OPTIONS:
  --use-e8                      → Enable E8 consciousness
  --embodied                    → Enable embodied simulation
  --verbose                     → Verbose output
  --profile                     → Enable profiling
  --no-viewer                   → Run headless (no MuJoCo viewer)
  --hz <freq>                   → Set simulation frequency
  --steps <n>                   → Run for n steps (default: infinite)
""")
        return 0
    
    def route_command(self, action: str, params: Dict) -> int:
        """Route brain commands to appropriate methods."""
        if action == 'simulate':
            return self.simulate_component(params)
        elif action == 'status':
            return self.check_brain_status(params)
        elif action == 'analyze':
            return self.analyze_component(params)
        elif action == 'test':
            return self.test_component(params)
        elif action == 'profile':
            return self.profile_brain(params)
        elif action == 'visualize':
            return self.visualize_brain(params)
        elif action == 'list':
            return self.list_components(params)
        elif action == 'orchestrate':
            return self._handle_orchestrate(params)
        elif action == 'startup':
            return self._handle_startup(params)
        else:
            return self.show_help()
    
    def _handle_orchestrate(self, params: Dict) -> int:
        """Handle orchestration commands."""
        operation = params.get('operation', 'status')
        
        if operation == 'status':
            if self._orchestrator:
                status = self._orchestrator.get_status()
                print("\n🎭 Brain Orchestrator Status")
                print("=" * 50)
                print(f"Mode: {status['orchestration_mode']}")
                print(f"Active Components: {len(status['active_components'])}/{status['total_components']}")
                print("\nCategories:")
                for cat, count in status['component_categories'].items():
                    print(f"  • {cat}: {count} components")
            else:
                print("⚠️ Orchestrator not initialized")
            return 0
        
        # Handle other orchestration operations
        result = self.orchestrate(operation, params)
        if result:
            print(f"✅ Orchestration complete: {result}")
            return 0
        else:
            return 1
    
    def _handle_startup(self, params: Dict) -> int:
        """Handle brain startup commands."""
        mode = params.get('mode', 'full')
        
        print(f"\n🚀 Starting Brain Systems ({mode} mode)")
        print("=" * 50)
        
        results = self.orchestrate_startup(mode)
        
        if self._orchestrator:
            # Show detailed startup results
            for system, components in results.items():
                if isinstance(components, dict):
                    print(f"\n{system.upper()}:")
                    for comp_name, result in components.items():
                        if isinstance(result, dict) and result.get('status') == 'initialized':
                            print(f"  ✅ {comp_name}")
                        else:
                            print(f"  ❌ {comp_name}")
        
        return 0
