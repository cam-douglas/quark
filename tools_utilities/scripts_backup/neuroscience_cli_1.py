"""
CLI interface for Neuroscience Domain Experts

This module provides command-line tools for interacting with the neuroscience expert system,
allowing users to execute neuroscience tasks and manage expert configurations.
"""

import argparse
import json
import logging
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.neuroscience_experts import (
    NeuroscienceExpertManager,
    NeuroscienceTask,
    NeuroscienceTaskType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuroscienceCLI:
    """Command-line interface for neuroscience experts"""
    
    def __init__(self):
        self.manager = NeuroscienceExpertManager()
    
    def list_experts(self, verbose: bool = False):
        """List all available neuroscience experts"""
        experts = self.manager.get_available_experts()
        capabilities = self.manager.get_expert_capabilities()
        
        print(f"\n🧠 Available Neuroscience Experts: {len(experts)}")
        print("=" * 50)
        
        if not experts:
            print("❌ No neuroscience experts available")
            print("   Install required dependencies to enable experts")
            return
        
        for expert_name in experts:
            expert_info = capabilities[expert_name]
            print(f"\n🔬 {expert_name}")
            print(f"   Status: {'✅ Available' if expert_info['available'] else '❌ Unavailable'}")
            print(f"   Task Types: {', '.join(expert_info['task_types'])}")
            
            if verbose:
                print(f"   Dependencies: {', '.join(expert_info['dependencies'])}")
    
    def show_system_status(self):
        """Show overall system status"""
        status = self.manager.get_system_status()
        
        print(f"\n📊 Neuroscience Expert System Status")
        print("=" * 40)
        print(f"Total Experts: {status['total_experts']}")
        print(f"System Health: {status['system_health']}")
        print(f"Available Experts: {', '.join(status['available_experts'])}")
        
        if status['expert_capabilities']:
            print(f"\nExpert Capabilities:")
            for name, caps in status['expert_capabilities'].items():
                print(f"  {name}: {len(caps['task_types'])} task types")
    
    def execute_task(self, task_description: str, task_type: str, parameters: Dict[str, Any]):
        """Execute a neuroscience task"""
        try:
            # Parse task type
            task_type_enum = NeuroscienceTaskType(task_type)
        except ValueError:
            print(f"❌ Invalid task type: {task_type}")
            print(f"Available types: {[t.value for t in NeuroscienceTaskType]}")
            return
        
        # Create task
        task = NeuroscienceTask(
            task_type=task_type_enum,
            description=task_description,
            parameters=parameters,
            expected_output="Task execution result"
        )
        
        print(f"\n🚀 Executing Neuroscience Task")
        print("=" * 40)
        print(f"Description: {task_description}")
        print(f"Task Type: {task_type}")
        print(f"Parameters: {json.dumps(parameters, indent=2)}")
        
        # Execute task
        try:
            result = self.manager.execute_task(task)
            
            if result.get('success'):
                print(f"\n✅ Task completed successfully!")
                print(f"Expert Used: {result.get('routed_to_expert', 'Unknown')}")
                print(f"Confidence: {result.get('routing_confidence', 0.0):.2f}")
                
                # Display results based on task type
                if task_type == NeuroscienceTaskType.BIOMEDICAL_LITERATURE.value:
                    self._display_biomedical_results(result)
                elif task_type == NeuroscienceTaskType.SPIKING_NETWORKS.value:
                    self._display_spiking_results(result)
                elif task_type == NeuroscienceTaskType.BIOPHYSICAL_SIMULATION.value:
                    self._display_biophysical_results(result)
                elif task_type == NeuroscienceTaskType.COGNITIVE_MODELING.value:
                    self._display_cognitive_results(result)
                elif task_type == NeuroscienceTaskType.WHOLE_BRAIN_DYNAMICS.value:
                    self._display_whole_brain_results(result)
                elif task_type == NeuroscienceTaskType.PYTORCH_SNN.value:
                    self._display_pytorch_snn_results(result)
                elif task_type == NeuroscienceTaskType.SYNTHETIC_DATA.value:
                    self._display_synthetic_data_results(result)
                elif task_type == NeuroscienceTaskType.DATA_AUGMENTATION.value:
                    self._display_synthetic_data_results(result)  # Same display method
                elif task_type == NeuroscienceTaskType.SELF_IMPROVEMENT.value:
                    self._display_self_improvement_results(result)
                elif task_type == NeuroscienceTaskType.QUALITY_ASSESSMENT.value:
                    self._display_self_improvement_results(result)  # Same display method
                else:
                    print(f"\n📋 Results: {json.dumps(result.get('results', {}), indent=2)}")
            else:
                print(f"\n❌ Task failed: {result.get('error', 'Unknown error')}")
                if 'available_experts' in result:
                    print(f"Available experts: {', '.join(result['available_experts'])}")
        
        except Exception as e:
            print(f"\n💥 Task execution error: {e}")
            logger.error(f"Task execution failed: {e}", exc_info=True)
    
    def _display_biomedical_results(self, result: Dict[str, Any]):
        """Display results from biomedical literature tasks"""
        print(f"\n📚 Biomedical Literature Results:")
        print("-" * 30)
        
        if 'generated_text' in result:
            if isinstance(result['generated_text'], list):
                for i, text in enumerate(result['generated_text']):
                    print(f"\n{i+1}. {text}")
            else:
                print(f"\n{result['generated_text']}")
        
        if 'model_info' in result:
            print(f"\nModel: {result['model_info']}")
    
    def _display_spiking_results(self, result: Dict[str, Any]):
        """Display results from spiking network simulations"""
        print(f"\n⚡ Spiking Network Results:")
        print("-" * 30)
        
        results = result.get('results', {})
        if 'total_spikes' in results:
            print(f"Total Spikes: {results['total_spikes']}")
        if 'num_neurons' in result:
            print(f"Neurons: {result['num_neurons']}")
        if 'simulation_duration' in result:
            print(f"Duration: {result['simulation_duration']} ms")
    
    def _display_biophysical_results(self, result: Dict[str, Any]):
        """Display results from biophysical simulations"""
        print(f"\n🔬 Biophysical Simulation Results:")
        print("-" * 30)
        
        results = result.get('results', {})
        if 'voltage' in results:
            print(f"Voltage traces recorded: {len(results['voltage'])} points")
        if 'simulation_duration' in result:
            print(f"Duration: {result['simulation_duration']} ms")
        if 'dt' in result:
            print(f"Time step: {result['dt']} ms")
    
    def _display_cognitive_results(self, result: Dict[str, Any]):
        """Display results from cognitive modeling"""
        print(f"\n🧠 Cognitive Modeling Results:")
        print("-" * 30)
        
        results = result.get('results', {})
        if 'simulation_duration' in result:
            print(f"Duration: {result['simulation_duration']} seconds")
        if 'dt' in result:
            print(f"Time step: {result['dt']} seconds")
        
        print(f"Data probes: {len(results)}")
    
    def _display_whole_brain_results(self, result: Dict[str, Any]):
        """Display results from whole-brain dynamics"""
        print(f"\n🌐 Whole-Brain Dynamics Results:")
        print("-" * 30)
        
        results = result.get('results', {})
        if 'num_regions' in results:
            print(f"Brain regions: {results['num_regions']}")
        if 'time_points' in results:
            print(f"Time points: {len(results['time_points'])}")
        if 'simulation_duration' in result:
            print(f"Duration: {result['simulation_duration']} ms")
    
    def _display_pytorch_snn_results(self, result: Dict[str, Any]):
        """Display results from PyTorch SNN tasks"""
        print(f"\n🔥 PyTorch SNN Results:")
        print("-" * 30)
        
        results = result.get('results', {})
        if 'output_shape' in results:
            print(f"Output shape: {results['output_shape']}")
        if 'device' in result:
            print(f"Device: {result['device']}")
        if 'input_size' in result:
            print(f"Input size: {result['input_size']}")
        if 'hidden_size' in result:
            print(f"Hidden size: {result['hidden_size']}")
    
    def _display_synthetic_data_results(self, result: Dict[str, Any]):
        """Display results from synthetic data generation tasks"""
        print(f"\n🧠 Synthetic Data Generation Results:")
        print("-" * 30)
        
        results = result.get('results', {})
        if 'generated_data' in results:
            print(f"Generated data shape: {results['generated_data'].shape}")
        if 'model_info' in result:
            print(f"Model: {result['model_info']}")
    
    def _display_self_improvement_results(self, result: Dict[str, Any]):
        """Display results from self-improvement tasks"""
        print(f"\n🧠 Self-Improvement Results:")
        print("-" * 30)
        
        results = result.get('results', {})
        if 'improvement_score' in results:
            print(f"Improvement Score: {results['improvement_score']:.2f}")
        if 'feedback_summary' in results:
            print(f"Feedback Summary: {results['feedback_summary']}")
        if 'next_task_recommendation' in results:
            print(f"Next Task Recommendation: {results['next_task_recommendation']}")
    
    def interactive_mode(self):
        """Run interactive mode for neuroscience tasks"""
        print("\n🧠 Neuroscience Expert Interactive Mode")
        print("=" * 40)
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\nneuro> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif command.lower() == 'help':
                    self._show_interactive_help()
                elif command.lower() == 'status':
                    self.show_system_status()
                elif command.lower() == 'experts':
                    self.list_experts(verbose=True)
                elif command.lower().startswith('execute'):
                    self._parse_execute_command(command)
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"💥 Error: {e}")
    
    def _show_interactive_help(self):
        """Show help for interactive mode"""
        print("\n📖 Available Commands:")
        print("  help     - Show this help")
        print("  status   - Show system status")
        print("  experts  - List available experts")
        print("  execute <type> <description> [params] - Execute a task")
        print("  quit     - Exit interactive mode")
        print("\n📝 Task Types:")
        for task_type in NeuroscienceTaskType:
            print(f"  {task_type.value}")
        print("\n💡 Example:")
        print("  execute biomedical_literature 'COVID-19 is' max_length=50")
    
    def _parse_execute_command(self, command: str):
        """Parse execute command from interactive mode"""
        parts = command.split(' ', 2)
        if len(parts) < 3:
            print("❌ Usage: execute <type> <description> [params]")
            return
        
        _, task_type, description = parts
        
        # Parse parameters if provided
        parameters = {}
        if '[' in description and ']' in description:
            desc_parts = description.rsplit('[', 1)
            description = desc_parts[0].strip()
            params_str = desc_parts[1].rstrip(']')
            
            try:
                # Simple parameter parsing
                for param in params_str.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Try to convert to appropriate type
                        try:
                            if value.lower() in ['true', 'false']:
                                parameters[key] = value.lower() == 'true'
                            elif '.' in value:
                                parameters[key] = float(value)
                            else:
                                parameters[key] = int(value)
                        except ValueError:
                            parameters[key] = value
            except Exception as e:
                print(f"⚠️  Could not parse parameters: {e}")
        
        self.execute_task(description, task_type, parameters)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Neuroscience Domain Expert CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available experts
  python neuroscience_cli.py list
  
  # Show system status
  python neuroscience_cli.py status
  
  # Execute a biomedical literature task
  python neuroscience_cli.py execute biomedical_literature "COVID-19 is" --max-length 100
  
  # Execute a spiking network simulation
  python neuroscience_cli.py execute spiking_networks "Simulate 100 neurons" --duration 2000 --num-neurons 100
  
  # Interactive mode
  python neuroscience_cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available neuroscience experts')
    list_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Execute command
    execute_parser = subparsers.add_parser('execute', help='Execute a neuroscience task')
    execute_parser.add_argument('task_type', help='Type of task to execute')
    execute_parser.add_argument('description', help='Task description')
    execute_parser.add_argument('--max-length', type=int, help='Maximum length for text generation')
    execute_parser.add_argument('--duration', type=int, help='Simulation duration')
    execute_parser.add_argument('--num-neurons', type=int, help='Number of neurons')
    execute_parser.add_argument('--connection-prob', type=float, help='Connection probability')
    execute_parser.add_argument('--dt', type=float, help='Time step')
    execute_parser.add_argument('--input-size', type=int, help='Input size for SNN')
    execute_parser.add_argument('--hidden-size', type=int, help='Hidden size for SNN')
    execute_parser.add_argument('--num-classes', type=int, help='Number of classes for SNN')
    execute_parser.add_argument('--num-steps', type=int, help='Number of time steps')
    execute_parser.add_argument('--temperature', type=float, help='Temperature for text generation')
    execute_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    execute_parser.add_argument('--iterations', type=int, help='Number of improvement iterations')
    execute_parser.add_argument('--feedback-type', type=str, help='Type of feedback for improvement')
    execute_parser.add_argument('--data-type', type=str, help='Type of data to generate')
    execute_parser.add_argument('--num-samples', type=int, help='Number of samples to generate')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = NeuroscienceCLI()
    
    try:
        if args.command == 'list':
            cli.list_experts(verbose=args.verbose)
        elif args.command == 'status':
            cli.show_system_status()
        elif args.command == 'execute':
            # Build parameters dictionary
            parameters = {}
            if args.max_length:
                parameters['max_length'] = args.max_length
            if args.duration:
                parameters['duration'] = args.duration
            if args.num_neurons:
                parameters['num_neurons'] = args.num_neurons
            if args.connection_prob:
                parameters['connection_prob'] = args.connection_prob
            if args.dt:
                parameters['dt'] = args.dt
            if args.input_size:
                parameters['input_size'] = args.input_size
            if args.hidden_size:
                parameters['hidden_size'] = args.hidden_size
            if args.num_classes:
                parameters['num_classes'] = args.num_classes
            if args.num_steps:
                parameters['num_steps'] = args.num_steps
            if args.temperature:
                parameters['temperature'] = args.temperature
            if args.seed:
                parameters['seed'] = args.seed
            if args.iterations:
                parameters['iterations'] = args.iterations
            if args.feedback_type:
                parameters['feedback_type'] = args.feedback_type
            if args.data_type:
                parameters['data_type'] = args.data_type
            if args.num_samples:
                parameters['num_samples'] = args.num_samples
            
            cli.execute_task(args.description, args.task_type, parameters)
        elif args.command == 'interactive':
            cli.interactive_mode()
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Error: {e}")
        logger.error(f"CLI error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
