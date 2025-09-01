"""Dataset Integration Module for Quark's Learning Systems
Integrates training datasets from:
1. LLM-IK: Inverse kinematics solutions from LLMs
2. LLM Articulated Manipulation: Human demonstrations and LLM planning data
3. PartNet-Mobility: Large-scale articulated object dataset

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

class DatasetIntegration:
    """
    Manages integration of multiple robotics datasets for Quark's training.
    
    This module provides a unified interface to:
    - LLM-IK solutions database
    - Human manipulation demonstrations  
    - Articulated object manipulation prompts
    - Performance metrics and learning trajectories
    """
    
    def __init__(self, base_path: str = "/Users/camdouglas/quark"):
        """
        Initialize dataset integration system.
        
        Args:
            base_path: Base path to the Quark project
        """
        self.base_path = Path(base_path)
        self.external_path = self.base_path / "external"
        
        # Dataset paths
        self.llm_ik_path = self.external_path / "llm-ik"
        self.manipulation_path = self.external_path / "llm-articulated-manipulation"
        
        # Dataset registries
        self.ik_solutions_registry = {}
        self.manipulation_demos_registry = {}
        self.prompt_templates_registry = {}
        self.partnet_objects_registry = {}
        
        # Training data cache
        self.cached_datasets = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("📚 Dataset Integration System initialized")
        print(f"   Base path: {base_path}")
        print(f"   LLM-IK path: {self.llm_ik_path}")
        print(f"   Manipulation path: {self.manipulation_path}")
        
        # Load available datasets
        self._discover_datasets()
    
    def _discover_datasets(self):
        """Discover and catalog all available datasets."""
        self.logger.info("🔍 Discovering available datasets...")
        
        # Discover LLM-IK solutions
        if self.llm_ik_path.exists():
            self._discover_ik_solutions()
        
        # Discover manipulation demonstrations  
        if self.manipulation_path.exists():
            self._discover_manipulation_demos()
            self._discover_prompt_templates()
        
        self.logger.info(f"✅ Dataset discovery complete")
        self.logger.info(f"   IK solutions: {len(self.ik_solutions_registry)}")
        self.logger.info(f"   Manipulation demos: {len(self.manipulation_demos_registry)}")
        self.logger.info(f"   Prompt templates: {len(self.prompt_templates_registry)}")
    
    def _discover_ik_solutions(self):
        """Discover LLM-generated IK solutions."""
        solutions_path = self.llm_ik_path / "Solutions"
        if not solutions_path.exists():
            return
        
        for robot_dir in solutions_path.iterdir():
            if robot_dir.is_dir():
                robot_name = robot_dir.name
                self.ik_solutions_registry[robot_name] = {}
                
                for model_dir in robot_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        solutions = []
                        
                        for solution_file in model_dir.glob("*.py"):
                            solution_info = self._parse_ik_solution_filename(solution_file.name)
                            solution_info['file_path'] = solution_file
                            solutions.append(solution_info)
                        
                        if solutions:
                            self.ik_solutions_registry[robot_name][model_name] = solutions
        
        self.logger.info(f"📊 Found IK solutions for {len(self.ik_solutions_registry)} robots")
    
    def _discover_manipulation_demos(self):
        """Discover human manipulation demonstrations."""
        demos_path = self.manipulation_path / "src" / "rotate_records"
        if not demos_path.exists():
            return
        
        for demo_file in demos_path.glob("*.json"):
            demo_info = self._parse_manipulation_demo(demo_file)
            if demo_info:
                demo_id = demo_file.stem
                self.manipulation_demos_registry[demo_id] = {
                    'file_path': demo_file,
                    'task_type': demo_info['task_type'],
                    'object_id': demo_info['object_id'],
                    'action_sequence': demo_info['action_sequence'],
                    'joint_info': demo_info['joint_info']
                }
        
        self.logger.info(f"🤖 Found {len(self.manipulation_demos_registry)} manipulation demonstrations")
    
    def _discover_prompt_templates(self):
        """Discover LLM prompt templates for manipulation tasks."""
        prompt_path = self.manipulation_path / "src" / "prompt_config"
        if not prompt_path.exists():
            return
        
        for prompt_file in prompt_path.glob("*.json"):
            with open(prompt_file, 'r') as f:
                prompt_data = json.load(f)
            
            template_name = prompt_file.stem
            self.prompt_templates_registry[template_name] = {
                'file_path': prompt_file,
                'templates': prompt_data
            }
        
        self.logger.info(f"💬 Found {len(self.prompt_templates_registry)} prompt template sets")
    
    def _parse_ik_solution_filename(self, filename: str) -> Dict[str, Any]:
        """
        Parse IK solution filename to extract metadata.
        Format: LOWER-UPPER-SOLVING-MODE.py
        Example: 0-5-Transform-Normal.py
        """
        parts = filename.replace('.py', '').split('-')
        if len(parts) >= 4:
            return {
                'lower_joint': int(parts[0]),
                'upper_joint': int(parts[1]),
                'solving_type': parts[2],  # Position or Transform
                'solving_mode': parts[3],  # Normal, Extend, Dynamic, etc.
                'degrees_of_freedom': int(parts[1]) - int(parts[0]) + 1
            }
        return {}
    
    def _parse_manipulation_demo(self, demo_file: Path) -> Optional[Dict[str, Any]]:
        """Parse manipulation demonstration file."""
        try:
            with open(demo_file, 'r') as f:
                demo_data = json.load(f)
            
            # Extract task and object info from filename
            filename = demo_file.stem
            if '_' in filename:
                parts = filename.split('_')
                task_type = parts[0]
                object_type = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                object_id = parts[-1]
            else:
                task_type = filename
                object_type = "unknown"
                object_id = "unknown"
            
            # Extract action sequence
            actions = demo_data.get('scene', {}).get('actions', [])
            action_sequence = []
            for action in actions:
                if isinstance(action, list) and len(action) >= 7:
                    # 7DOF pose: [x, y, z, qx, qy, qz, qw]
                    action_sequence.append({
                        'position': action[:3],
                        'orientation': action[3:7],
                        'type': 'pose'
                    })
                elif isinstance(action, str):
                    action_sequence.append({
                        'command': action,
                        'type': 'command'
                    })
            
            # Extract joint information
            joint_info = demo_data.get('observations', {}).get('urdf_conclusion', {})
            
            return {
                'task_type': task_type,
                'object_type': object_type,
                'object_id': object_id,
                'action_sequence': action_sequence,
                'joint_info': joint_info,
                'full_data': demo_data
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse {demo_file}: {e}")
            return None
    
    def load_ik_training_data(self, robot_name: str = "UR5", 
                             solving_modes: List[str] = None) -> Dict[str, Any]:
        """
        Load IK solutions as training data for neural networks.
        
        Args:
            robot_name: Target robot (e.g., "UR5")
            solving_modes: List of solving modes to include
            
        Returns:
            Structured training data with solutions and metadata
        """
        if solving_modes is None:
            solving_modes = ['Normal', 'Extend', 'Dynamic', 'Cumulative']
        
        cache_key = f"ik_{robot_name}_{'_'.join(solving_modes)}"
        if cache_key in self.cached_datasets:
            return self.cached_datasets[cache_key]
        
        training_data = {
            'solutions': [],
            'metadata': [],
            'performance_metrics': [],
            'solving_progressions': {}
        }
        
        if robot_name not in self.ik_solutions_registry:
            self.logger.warning(f"No IK solutions found for robot: {robot_name}")
            return training_data
        
        robot_solutions = self.ik_solutions_registry[robot_name]
        
        for model_name, solutions in robot_solutions.items():
            for solution in solutions:
                if solution['solving_mode'] in solving_modes:
                    # Load the actual solution code
                    solution_code = self._load_solution_code(solution['file_path'])
                    
                    # Load performance metrics if available
                    metrics = self._load_solution_metrics(solution['file_path'])
                    
                    training_data['solutions'].append({
                        'model': model_name,
                        'dof': solution['degrees_of_freedom'],
                        'solving_mode': solution['solving_mode'],
                        'solving_type': solution['solving_type'],
                        'code': solution_code,
                        'joints': f"{solution['lower_joint']}-{solution['upper_joint']}"
                    })
                    
                    training_data['metadata'].append(solution)
                    
                    if metrics:
                        training_data['performance_metrics'].append(metrics)
        
        # Analyze solving progressions (how solutions build on each other)
        training_data['solving_progressions'] = self._analyze_solving_progressions(
            training_data['solutions']
        )
        
        self.cached_datasets[cache_key] = training_data
        
        self.logger.info(f"📊 Loaded IK training data for {robot_name}")
        self.logger.info(f"   Solutions: {len(training_data['solutions'])}")
        self.logger.info(f"   Models: {len(set(s['model'] for s in training_data['solutions']))}")
        
        return training_data
    
    def load_manipulation_training_data(self, task_types: List[str] = None) -> Dict[str, Any]:
        """
        Load manipulation demonstrations as training data.
        
        Args:
            task_types: List of task types to include (e.g., ['open', 'close'])
            
        Returns:
            Structured training data with demonstrations and prompts
        """
        cache_key = f"manipulation_{'_'.join(task_types) if task_types else 'all'}"
        if cache_key in self.cached_datasets:
            return self.cached_datasets[cache_key]
        
        training_data = {
            'demonstrations': [],
            'prompt_templates': {},
            'object_types': set(),
            'task_types': set(),
            'kinematic_patterns': {}
        }
        
        # Load demonstrations
        for demo_id, demo_info in self.manipulation_demos_registry.items():
            if task_types is None or demo_info['task_type'] in task_types:
                training_data['demonstrations'].append({
                    'id': demo_id,
                    'task_type': demo_info['task_type'],
                    'object_id': demo_info['object_id'],
                    'action_sequence': demo_info['action_sequence'],
                    'joint_info': demo_info['joint_info']
                })
                
                training_data['object_types'].add(demo_info['object_id'].split('_')[0])
                training_data['task_types'].add(demo_info['task_type'])
        
        # Load prompt templates
        for template_name, template_info in self.prompt_templates_registry.items():
            training_data['prompt_templates'][template_name] = template_info['templates']
        
        # Analyze kinematic patterns
        training_data['kinematic_patterns'] = self._analyze_kinematic_patterns(
            training_data['demonstrations']
        )
        
        # Convert sets to lists for JSON serialization
        training_data['object_types'] = list(training_data['object_types'])
        training_data['task_types'] = list(training_data['task_types'])
        
        self.cached_datasets[cache_key] = training_data
        
        self.logger.info(f"🤖 Loaded manipulation training data")
        self.logger.info(f"   Demonstrations: {len(training_data['demonstrations'])}")
        self.logger.info(f"   Object types: {len(training_data['object_types'])}")
        self.logger.info(f"   Task types: {len(training_data['task_types'])}")
        
        return training_data
    
    def create_unified_training_dataset(self) -> Dict[str, Any]:
        """
        Create a unified training dataset combining all available data sources.
        
        Returns:
            Comprehensive dataset for Quark's learning systems
        """
        unified_dataset = {
            'ik_solutions': self.load_ik_training_data(),
            'manipulation_demos': self.load_manipulation_training_data(),
            'learning_progressions': {},
            'cross_modal_mappings': {},
            'dataset_statistics': {}
        }
        
        # Create cross-modal mappings (IK solutions ↔ Manipulation tasks)
        unified_dataset['cross_modal_mappings'] = self._create_cross_modal_mappings(
            unified_dataset['ik_solutions'],
            unified_dataset['manipulation_demos']
        )
        
        # Analyze learning progressions across datasets
        unified_dataset['learning_progressions'] = self._analyze_cross_dataset_progressions(
            unified_dataset
        )
        
        # Generate dataset statistics
        unified_dataset['dataset_statistics'] = self._generate_dataset_statistics(
            unified_dataset
        )
        
        self.logger.info("🎯 Created unified training dataset")
        self.logger.info(f"   Total IK solutions: {len(unified_dataset['ik_solutions']['solutions'])}")
        self.logger.info(f"   Total manipulation demos: {len(unified_dataset['manipulation_demos']['demonstrations'])}")
        self.logger.info(f"   Cross-modal mappings: {len(unified_dataset['cross_modal_mappings'])}")
        
        return unified_dataset
    
    def download_partnet_mobility_dataset(self, download_path: str = None) -> bool:
        """
        Download the PartNet-Mobility dataset for articulated objects.
        
        Args:
            download_path: Where to save the dataset
            
        Returns:
            Success status
        """
        if download_path is None:
            download_path = self.base_path / "datasets" / "partnet_mobility"
        
        download_path = Path(download_path)
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset URL from the paper
        dataset_url = "https://drive.google.com/file/d/1iWoY4jmi-1mDt8Th907zNvfh0d3E9hL9/view?usp=drive_link"
        
        self.logger.info(f"📥 PartNet-Mobility dataset download initiated")
        self.logger.info(f"   URL: {dataset_url}")
        self.logger.info(f"   Target path: {download_path}")
        self.logger.info("   Manual download required - automated download from Google Drive not implemented")
        
        # For now, provide manual instructions
        instructions = f"""
        Manual Download Instructions for PartNet-Mobility Dataset:
        
        1. Visit: {dataset_url}
        2. Download the dataset file
        3. Extract to: {download_path}
        4. Run: dataset_integration.catalog_partnet_objects() to register objects
        """
        
        print(instructions)
        
        # Check if dataset already exists
        if any(download_path.iterdir()):
            self.logger.info("✅ PartNet-Mobility dataset appears to be present")
            return True
        
        return False
    
    def _load_solution_code(self, file_path: Path) -> Optional[str]:
        """Load IK solution code from file."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"Failed to load solution code from {file_path}: {e}")
            return None
    
    def _load_solution_metrics(self, solution_path: Path) -> Optional[Dict]:
        """Load performance metrics for IK solution."""
        # Look for corresponding results file
        results_path = solution_path.parent.parent.parent.parent / "Results" / solution_path.parent.parent.name / solution_path.parent.name / solution_path.stem
        
        if results_path.exists():
            try:
                # Look for CSV or JSON results
                for results_file in results_path.glob("*"):
                    if results_file.suffix == '.csv':
                        df = pd.read_csv(results_file)
                        return df.to_dict('records')
                    elif results_file.suffix == '.json':
                        with open(results_file, 'r') as f:
                            return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metrics for {solution_path}: {e}")
        
        return None
    
    def _analyze_solving_progressions(self, solutions: List[Dict]) -> Dict[str, Any]:
        """Analyze how IK solving modes build on each other."""
        progressions = {
            'complexity_progression': {},
            'mode_dependencies': {},
            'success_patterns': {}
        }
        
        # Group by DOF and analyze progression
        dof_groups = {}
        for solution in solutions:
            dof = solution['dof']
            if dof not in dof_groups:
                dof_groups[dof] = []
            dof_groups[dof].append(solution)
        
        for dof, dof_solutions in dof_groups.items():
            mode_order = ['Normal', 'Extend', 'Dynamic', 'Cumulative', 'Transfer']
            available_modes = [s['solving_mode'] for s in dof_solutions]
            
            progressions['complexity_progression'][dof] = {
                'available_modes': available_modes,
                'progression_order': [m for m in mode_order if m in available_modes]
            }
        
        return progressions
    
    def _analyze_kinematic_patterns(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Analyze kinematic patterns in manipulation demonstrations."""
        patterns = {
            'joint_type_distribution': {},
            'action_sequence_patterns': {},
            'object_manipulation_strategies': {}
        }
        
        for demo in demonstrations:
            # Analyze joint types
            joint_info = demo.get('joint_info', {})
            joint_type = joint_info.get('joint_type', 'unknown')
            
            if joint_type not in patterns['joint_type_distribution']:
                patterns['joint_type_distribution'][joint_type] = 0
            patterns['joint_type_distribution'][joint_type] += 1
            
            # Analyze action sequences
            action_count = len(demo.get('action_sequence', []))
            task_type = demo['task_type']
            
            if task_type not in patterns['action_sequence_patterns']:
                patterns['action_sequence_patterns'][task_type] = []
            patterns['action_sequence_patterns'][task_type].append(action_count)
        
        # Calculate averages
        for task_type, counts in patterns['action_sequence_patterns'].items():
            patterns['action_sequence_patterns'][task_type] = {
                'average_actions': np.mean(counts),
                'min_actions': min(counts),
                'max_actions': max(counts),
                'total_demonstrations': len(counts)
            }
        
        return patterns
    
    def _create_cross_modal_mappings(self, ik_data: Dict, manipulation_data: Dict) -> Dict[str, Any]:
        """Create mappings between IK solutions and manipulation tasks."""
        mappings = {
            'ik_to_manipulation': {},
            'manipulation_to_ik': {},
            'complexity_correlations': {}
        }
        
        # Map IK complexity to manipulation complexity
        for ik_solution in ik_data['solutions']:
            ik_dof = ik_solution['dof']
            ik_mode = ik_solution['solving_mode']
            
            # Find relevant manipulation tasks
            relevant_demos = []
            for demo in manipulation_data['demonstrations']:
                joint_info = demo.get('joint_info', {})
                if joint_info.get('joint_type') in ['revolute', 'prismatic']:
                    relevant_demos.append(demo)
            
            mappings['ik_to_manipulation'][f"{ik_dof}_{ik_mode}"] = relevant_demos[:3]  # Top 3
        
        return mappings
    
    def _analyze_cross_dataset_progressions(self, unified_dataset: Dict) -> Dict[str, Any]:
        """Analyze learning progressions across different datasets."""
        return {
            'skill_development_pathway': [
                'basic_joint_control',
                'inverse_kinematics',
                'object_interaction',
                'complex_manipulation'
            ],
            'curriculum_suggestions': {
                'phase_1': 'Simple IK problems (1-2 DOF)',
                'phase_2': 'Complex IK problems (3-6 DOF)',
                'phase_3': 'Basic manipulation (open/close)',
                'phase_4': 'Complex manipulation (multi-step tasks)'
            }
        }
    
    def _generate_dataset_statistics(self, unified_dataset: Dict) -> Dict[str, Any]:
        """Generate comprehensive statistics about the unified dataset."""
        stats = {
            'total_training_samples': 0,
            'data_distribution': {},
            'complexity_analysis': {},
            'coverage_analysis': {}
        }
        
        # Count total samples
        stats['total_training_samples'] = (
            len(unified_dataset['ik_solutions']['solutions']) +
            len(unified_dataset['manipulation_demos']['demonstrations'])
        )
        
        # Data distribution
        stats['data_distribution'] = {
            'ik_solutions': len(unified_dataset['ik_solutions']['solutions']),
            'manipulation_demos': len(unified_dataset['manipulation_demos']['demonstrations']),
            'ratio_ik_to_manipulation': len(unified_dataset['ik_solutions']['solutions']) / max(1, len(unified_dataset['manipulation_demos']['demonstrations']))
        }
        
        return stats
    
    def get_training_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for training Quark based on available datasets.
        """
        recommendations = {
            'curriculum_order': [],
            'training_priorities': {},
            'dataset_gaps': [],
            'integration_strategies': {}
        }
        
        # Analyze what we have
        ik_data = self.load_ik_training_data()
        manipulation_data = self.load_manipulation_training_data()
        
        # Curriculum recommendations
        recommendations['curriculum_order'] = [
            {
                'phase': 'Foundation',
                'focus': 'Basic motor control and proprioception',
                'datasets': ['simple_joint_movements'],
                'duration': '1000 episodes'
            },
            {
                'phase': 'Inverse Kinematics',
                'focus': 'IK problem solving with LLM guidance',
                'datasets': ['llm_ik_normal_mode'],
                'duration': '5000 episodes'
            },
            {
                'phase': 'Object Interaction',
                'focus': 'Basic manipulation tasks',
                'datasets': ['manipulation_demos_basic'],
                'duration': '10000 episodes'
            },
            {
                'phase': 'Complex Manipulation',
                'focus': 'Multi-step manipulation with planning',
                'datasets': ['manipulation_demos_complex', 'llm_planning'],
                'duration': '20000 episodes'
            }
        ]
        
        # Training priorities
        recommendations['training_priorities'] = {
            'high': ['Stabilize basic locomotion', 'Integrate proprioceptive feedback'],
            'medium': ['Implement LLM-IK solutions', 'Learn manipulation primitives'],
            'low': ['Complex multi-object tasks', 'Advanced planning']
        }
        
        # Dataset gaps
        if len(ik_data['solutions']) < 10:
            recommendations['dataset_gaps'].append('Need more IK solution examples')
        
        if len(manipulation_data['demonstrations']) < 20:
            recommendations['dataset_gaps'].append('Need more manipulation demonstrations')
        
        # Integration strategies
        recommendations['integration_strategies'] = {
            'immediate': 'Use LLM-IK for motor control hierarchy',
            'short_term': 'Implement manipulation demonstration replay',
            'long_term': 'Develop unified LLM-guided learning system'
        }
        
        return recommendations

# Global instance for easy import
dataset_integration = DatasetIntegration()

if __name__ == "__main__":
    # Demonstration
    print("🚀 Dataset Integration Demonstration")
    print("="*50)
    
    # Show available datasets
    print(f"\n📊 Available Datasets:")
    print(f"   IK Solutions: {len(dataset_integration.ik_solutions_registry)} robots")
    print(f"   Manipulation Demos: {len(dataset_integration.manipulation_demos_registry)} demonstrations")
    print(f"   Prompt Templates: {len(dataset_integration.prompt_templates_registry)} template sets")
    
    # Load sample training data
    ik_data = dataset_integration.load_ik_training_data()
    manipulation_data = dataset_integration.load_manipulation_training_data()
    
    print(f"\n🎯 Sample Training Data:")
    print(f"   IK Solutions: {len(ik_data['solutions'])}")
    print(f"   Manipulation Demos: {len(manipulation_data['demonstrations'])}")
    
    # Get training recommendations
    recommendations = dataset_integration.get_training_recommendations()
    print(f"\n💡 Training Recommendations:")
    print(f"   Curriculum phases: {len(recommendations['curriculum_order'])}")
    print(f"   High priority items: {len(recommendations['training_priorities']['high'])}")
    print(f"   Dataset gaps: {len(recommendations['dataset_gaps'])}")
    
    # Create unified dataset
    unified = dataset_integration.create_unified_training_dataset()
    print(f"\n🎯 Unified Dataset Created:")
    print(f"   Total training samples: {unified['dataset_statistics']['total_training_samples']}")
    print(f"   Cross-modal mappings: {len(unified['cross_modal_mappings'])}")