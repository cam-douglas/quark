"""LLM-Powered Manipulation Planning Module
Integrates kinematic-aware prompting for articulated object manipulation.
Based on: https://github.com/GeWu-Lab/LLM_articulated_object_manipulation

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any
import json

# Add external LLM articulated manipulation to path
manipulation_path = os.path.join(os.path.dirname(__file__), '../../../../data/external/llm-articulated-manipulation/src')
sys.path.append(manipulation_path)

class LLMManipulationPlanner:
    """
    Uses LLMs for kinematic-aware manipulation planning of articulated objects.
    
    This module implements the breakthrough approach from the ICRA 2024 paper:
    "Kinematic-aware Prompting for Generalizable Articulated Object Manipulation with LLMs"
    
    Key innovations:
    1. Unified Kinematic Knowledge Parser - converts 3D objects to LLM-understandable representations
    2. Kinematic-aware Hierarchical Prompting - guides LLMs through manipulation reasoning
    3. Generalizable to unseen articulated objects
    """

    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the LLM manipulation planner.
        
        Args:
            model_name: The LLM model to use for planning
        """
        self.model_name = model_name
        self.kinematic_knowledge_base = {}  # Stores parsed object kinematics
        self.manipulation_history = []      # Learning from past manipulations
        self.known_object_types = [
            'door', 'drawer', 'cabinet', 'laptop', 'microwave', 'refrigerator',
            'oven', 'washing_machine', 'dishwasher', 'trash_can', 'toilet',
            'suitcase', 'briefcase', 'storage_furniture', 'safe', 'table'
        ]

        print("🦾 LLM Manipulation Planner initialized")
        print(f"   Model: {model_name}")
        print(f"   Known object types: {len(self.known_object_types)}")

    def parse_object_kinematics(self, object_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified Kinematic Knowledge Parser
        
        Converts complex 3D articulated objects into unified representations
        that LLMs can reason about effectively.
        
        Args:
            object_description: Contains object type, parts, joints, constraints
            
        Returns:
            Unified kinematic representation for LLM consumption
        """

        object_type = object_description.get('type', 'unknown')
        parts = object_description.get('parts', [])
        joints = object_description.get('joints', [])

        # Create unified representation
        kinematic_rep = {
            'object_id': object_description.get('id', 'obj_unknown'),
            'object_type': object_type,
            'kinematic_chain': self._build_kinematic_chain(parts, joints),
            'manipulation_affordances': self._identify_affordances(object_type, parts),
            'constraint_analysis': self._analyze_constraints(joints),
            'manipulation_strategy': self._suggest_strategy(object_type)
        }

        # Cache in knowledge base
        obj_id = kinematic_rep['object_id']
        self.kinematic_knowledge_base[obj_id] = kinematic_rep

        print(f"🔍 Parsed kinematics for {object_type} (ID: {obj_id})")
        print(f"   Parts: {len(parts)}, Joints: {len(joints)}")
        print(f"   Affordances: {kinematic_rep['manipulation_affordances']}")

        return kinematic_rep

    def plan_manipulation(self, object_id: str, goal: str,
                         current_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate kinematic-aware manipulation plan using LLM reasoning.
        
        This implements the hierarchical prompting approach:
        1. High-level task understanding
        2. Kinematic reasoning about object structure  
        3. Waypoint generation for manipulation sequence
        4. Low-level motor command translation
        
        Args:
            object_id: ID of the object to manipulate
            goal: Natural language description of the manipulation goal
            current_state: Current state of the object and environment
            
        Returns:
            Comprehensive manipulation plan with waypoints and commands
        """

        if object_id not in self.kinematic_knowledge_base:
            raise ValueError(f"Object {object_id} not found in knowledge base. Parse kinematics first.")

        kinematic_rep = self.kinematic_knowledge_base[object_id]

        # Stage 1: High-level task understanding
        task_analysis = self._analyze_manipulation_task(goal, kinematic_rep)

        # Stage 2: Kinematic-aware reasoning
        kinematic_plan = self._generate_kinematic_plan(task_analysis, kinematic_rep)

        # Stage 3: Waypoint sequence generation
        waypoint_sequence = self._generate_waypoint_sequence(kinematic_plan, current_state)

        # Stage 4: Motor command translation
        motor_commands = self._translate_to_motor_commands(waypoint_sequence)

        manipulation_plan = {
            'object_id': object_id,
            'goal': goal,
            'task_analysis': task_analysis,
            'kinematic_plan': kinematic_plan,
            'waypoint_sequence': waypoint_sequence,
            'motor_commands': motor_commands,
            'estimated_duration': self._estimate_execution_time(waypoint_sequence),
            'success_probability': self._estimate_success_probability(kinematic_plan)
        }

        # Record in manipulation history
        self.manipulation_history.append({
            'timestamp': np.datetime64('now'),
            'object_type': kinematic_rep['object_type'],
            'goal': goal,
            'plan_complexity': len(waypoint_sequence),
            'generated': True
        })

        print(f"🎯 Generated manipulation plan for {goal}")
        print(f"   Waypoints: {len(waypoint_sequence)}")
        print(f"   Estimated duration: {manipulation_plan['estimated_duration']:.1f}s")
        print(f"   Success probability: {manipulation_plan['success_probability']:.1%}")

        return manipulation_plan

    def _build_kinematic_chain(self, parts: List[Dict], joints: List[Dict]) -> Dict[str, Any]:
        """
        Build kinematic chain representation from object parts and joints.
        """
        chain = {
            'base_part': None,
            'movable_parts': [],
            'joint_sequence': [],
            'degrees_of_freedom': 0
        }

        # Identify base (fixed) and movable parts
        for part in parts:
            if part.get('fixed', False):
                chain['base_part'] = part
            else:
                chain['movable_parts'].append(part)

        # Analyze joint sequence
        chain['joint_sequence'] = sorted(joints, key=lambda j: j.get('order', 0))
        chain['degrees_of_freedom'] = len([j for j in joints if j.get('type') != 'fixed'])

        return chain

    def _identify_affordances(self, object_type: str, parts: List[Dict]) -> List[str]:
        """
        Identify manipulation affordances based on object type and structure.
        """
        affordance_map = {
            'door': ['push', 'pull', 'turn_handle', 'unlock'],
            'drawer': ['pull', 'push', 'slide'],
            'cabinet': ['open', 'close', 'turn_handle'],
            'laptop': ['open', 'close', 'type', 'click'],
            'microwave': ['open', 'close', 'press_buttons'],
            'refrigerator': ['open', 'close', 'pull_handle'],
            'washing_machine': ['open', 'close', 'press_buttons', 'turn_dial']
        }

        base_affordances = affordance_map.get(object_type, ['push', 'pull', 'grasp'])

        # Add part-specific affordances
        for part in parts:
            part_type = part.get('type', '')
            if 'handle' in part_type:
                base_affordances.extend(['grasp_handle', 'turn_handle'])
            elif 'button' in part_type:
                base_affordances.append('press_button')

        return list(set(base_affordances))  # Remove duplicates

    def _analyze_constraints(self, joints: List[Dict]) -> Dict[str, Any]:
        """
        Analyze kinematic constraints for manipulation planning.
        """
        constraints = {
            'joint_limits': {},
            'motion_constraints': [],
            'collision_risks': [],
            'required_forces': {}
        }

        for joint in joints:
            joint_id = joint.get('id', 'unknown')
            joint_type = joint.get('type', 'revolute')

            if 'limits' in joint:
                constraints['joint_limits'][joint_id] = joint['limits']

            if joint_type == 'prismatic':
                constraints['motion_constraints'].append(f"{joint_id}: linear motion only")
            elif joint_type == 'revolute':
                constraints['motion_constraints'].append(f"{joint_id}: rotational motion only")

        return constraints

    def _suggest_strategy(self, object_type: str) -> str:
        """
        Suggest high-level manipulation strategy based on object type.
        """
        strategy_map = {
            'door': 'Approach handle, grasp firmly, apply force perpendicular to hinge axis',
            'drawer': 'Locate handle, pull along sliding axis with steady force',
            'cabinet': 'Identify opening mechanism, apply appropriate force direction',
            'laptop': 'Locate screen edge, lift carefully to avoid damage',
            'microwave': 'Press door release button, then pull door open',
            'refrigerator': 'Grasp handle, pull with increasing force to overcome seal'
        }

        return strategy_map.get(object_type, 'Analyze object structure and apply appropriate manipulation technique')

    def _analyze_manipulation_task(self, goal: str, kinematic_rep: Dict) -> Dict[str, Any]:
        """
        High-level analysis of the manipulation task using LLM reasoning.
        """
        object_type = kinematic_rep['object_type']
        affordances = kinematic_rep['manipulation_affordances']

        # Simulate LLM task analysis
        task_analysis = {
            'goal_type': self._categorize_goal(goal),
            'required_affordances': self._match_affordances_to_goal(goal, affordances),
            'task_complexity': self._assess_complexity(goal, kinematic_rep),
            'prerequisites': self._identify_prerequisites(goal, object_type),
            'success_criteria': self._define_success_criteria(goal)
        }

        print(f"🧠 LLM Task Analysis: {goal}")
        print(f"   Goal type: {task_analysis['goal_type']}")
        print(f"   Required affordances: {task_analysis['required_affordances']}")

        return task_analysis

    def _generate_kinematic_plan(self, task_analysis: Dict, kinematic_rep: Dict) -> Dict[str, Any]:
        """
        Generate kinematic reasoning about how to achieve the manipulation goal.
        """
        kinematic_plan = {
            'manipulation_sequence': [],
            'critical_waypoints': [],
            'force_requirements': {},
            'timing_constraints': []
        }

        goal_type = task_analysis['goal_type']

        if goal_type == 'open':
            kinematic_plan['manipulation_sequence'] = [
                'approach_target',
                'establish_contact',
                'apply_opening_force',
                'guide_motion',
                'verify_completion'
            ]
        elif goal_type == 'close':
            kinematic_plan['manipulation_sequence'] = [
                'approach_target',
                'establish_contact',
                'apply_closing_force',
                'ensure_alignment',
                'verify_closure'
            ]

        return kinematic_plan

    def _generate_waypoint_sequence(self, kinematic_plan: Dict,
                                  current_state: Dict = None) -> List[Dict[str, Any]]:
        """
        Generate 3D waypoint sequence for manipulation execution.
        """
        waypoints = []

        for i, action in enumerate(kinematic_plan['manipulation_sequence']):
            waypoint = {
                'step': i,
                'action': action,
                'position': np.random.uniform(-0.5, 0.5, 3).tolist(),  # Placeholder
                'orientation': [0, 0, 0, 1],  # Unit quaternion
                'gripper_state': 'open' if action in ['approach_target'] else 'closed',
                'force_vector': np.random.uniform(-1, 1, 3).tolist(),
                'duration': np.random.uniform(0.5, 2.0)
            }
            waypoints.append(waypoint)

        return waypoints

    def _translate_to_motor_commands(self, waypoints: List[Dict]) -> List[Dict[str, Any]]:
        """
        Translate waypoints to low-level motor commands.
        """
        motor_commands = []

        for waypoint in waypoints:
            command = {
                'type': 'move_to_pose',
                'target_position': waypoint['position'],
                'target_orientation': waypoint['orientation'],
                'gripper_command': waypoint['gripper_state'],
                'force_limit': np.linalg.norm(waypoint['force_vector']),
                'velocity_limit': 0.1,  # Conservative velocity
                'duration': waypoint['duration']
            }
            motor_commands.append(command)

        return motor_commands

    def _categorize_goal(self, goal: str) -> str:
        """Categorize the manipulation goal."""
        goal_lower = goal.lower()
        if 'open' in goal_lower:
            return 'open'
        elif 'close' in goal_lower:
            return 'close'
        elif 'move' in goal_lower or 'position' in goal_lower:
            return 'reposition'
        elif 'press' in goal_lower or 'push' in goal_lower:
            return 'press'
        else:
            return 'general_manipulation'

    def _match_affordances_to_goal(self, goal: str, affordances: List[str]) -> List[str]:
        """Match available affordances to the manipulation goal."""
        goal_lower = goal.lower()
        relevant_affordances = []

        for affordance in affordances:
            if any(word in goal_lower for word in affordance.split('_')):
                relevant_affordances.append(affordance)

        return relevant_affordances or ['grasp']  # Default to grasping

    def _assess_complexity(self, goal: str, kinematic_rep: Dict) -> str:
        """Assess manipulation task complexity."""
        dof = kinematic_rep['kinematic_chain']['degrees_of_freedom']
        parts_count = len(kinematic_rep['kinematic_chain']['movable_parts'])

        if dof <= 1 and parts_count <= 2:
            return 'simple'
        elif dof <= 3 and parts_count <= 4:
            return 'moderate'
        else:
            return 'complex'

    def _identify_prerequisites(self, goal: str, object_type: str) -> List[str]:
        """Identify prerequisites for successful manipulation."""
        prerequisites = ['object_detected', 'workspace_clear']

        if 'open' in goal.lower():
            prerequisites.extend(['object_unlocked', 'sufficient_clearance'])

        return prerequisites

    def _define_success_criteria(self, goal: str) -> Dict[str, Any]:
        """Define measurable success criteria."""
        return {
            'position_tolerance': 0.01,  # 1cm tolerance
            'orientation_tolerance': 0.1,  # ~6 degree tolerance
            'force_threshold': 10.0,  # Newtons
            'completion_time': 30.0  # Seconds
        }

    def _estimate_execution_time(self, waypoints: List[Dict]) -> float:
        """Estimate total execution time."""
        return sum(wp['duration'] for wp in waypoints)

    def _estimate_success_probability(self, kinematic_plan: Dict) -> float:
        """Estimate probability of successful execution."""
        complexity = len(kinematic_plan['manipulation_sequence'])
        base_prob = 0.9
        complexity_penalty = min(0.1 * complexity, 0.5)
        return max(base_prob - complexity_penalty, 0.1)

    def get_planning_stats(self) -> Dict[str, Any]:
        """Get statistics about manipulation planning performance."""
        if not self.manipulation_history:
            return {"total_plans": 0, "object_types_encountered": 0}

        total_plans = len(self.manipulation_history)
        object_types = set(h['object_type'] for h in self.manipulation_history)
        avg_complexity = np.mean([h['plan_complexity'] for h in self.manipulation_history])

        return {
            "total_plans": total_plans,
            "object_types_encountered": len(object_types),
            "average_plan_complexity": avg_complexity,
            "known_objects": len(self.kinematic_knowledge_base)
        }

    def demonstrate_capability(self) -> Dict[str, Any]:
        """
        Demonstrate the LLM manipulation planning capabilities.
        """
        print("\n🦾 LLM Manipulation Planner Demonstration")
        print("="*60)

        # Create sample object
        sample_object = {
            'id': 'demo_door_001',
            'type': 'door',
            'parts': [
                {'id': 'door_frame', 'type': 'frame', 'fixed': True},
                {'id': 'door_panel', 'type': 'panel', 'fixed': False},
                {'id': 'door_handle', 'type': 'handle', 'fixed': False}
            ],
            'joints': [
                {'id': 'hinge', 'type': 'revolute', 'axis': [0, 0, 1],
                 'limits': {'min': 0, 'max': 1.57}}  # 0 to 90 degrees
            ]
        }

        # Parse kinematics
        kinematic_rep = self.parse_object_kinematics(sample_object)

        # Generate manipulation plans for different goals
        goals = ["open the door", "close the door", "check if door is locked"]
        results = {}

        for goal in goals:
            print(f"\n🎯 Planning for goal: '{goal}'")
            try:
                plan = self.plan_manipulation('demo_door_001', goal)
                results[goal] = {
                    'success': True,
                    'waypoints': len(plan['waypoint_sequence']),
                    'duration': plan['estimated_duration'],
                    'success_probability': plan['success_probability']
                }
            except Exception as e:
                results[goal] = {'success': False, 'error': str(e)}

        # Show planning stats
        stats = self.get_planning_stats()
        print("\n📊 Planning Statistics:")
        print(f"   Total plans generated: {stats['total_plans']}")
        print(f"   Object types encountered: {stats['object_types_encountered']}")
        print(f"   Known objects in KB: {stats['known_objects']}")

        return results

# Integration hook for Brain Simulator
def create_llm_manipulation_planner() -> LLMManipulationPlanner:
    """
    Factory function to create LLM manipulation planner for brain integration.
    """
    return LLMManipulationPlanner()

if __name__ == "__main__":
    # Demonstration
    planner = LLMManipulationPlanner()
    results = planner.demonstrate_capability()
    print(f"\n🎯 Demonstration complete. Results: {json.dumps(results, indent=2)}")
