"""
Baby-AGI CLI Interface

This module provides a command-line interface for the Baby-AGI system,
integrating with the existing small-mind CLI structure.
"""

import click
import json
import time
from pathlib import Path
from typing import Optional

from .....................................................baby_agi.agent import BabyAGIAgent
from .....................................................baby_agi.control import AgentController
from .....................................................baby_agi.runtime import AgentRuntime


@click.group()
def baby_agi():
    """Baby-AGI: Self-Running, Interruptible Local Agent"""
    pass


@baby_agi.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--run-dir', '-d', help='Runtime directory for the agent')
@click.option('--tick-interval', '-i', type=int, default=5, help='Tick interval in seconds')
@click.option('--max-cycles', '-m', type=int, default=1000, help='Maximum execution cycles')
@click.option('--checkpoint-interval', type=int, default=10, help='Checkpoint interval in seconds')
def start(config: Optional[str], run_dir: Optional[str], tick_interval: int, max_cycles: int, checkpoint_interval: int):
    """Start the Baby-AGI agent."""
    click.echo("Starting Baby-AGI agent...")
    
    # Load configuration if provided
    agent_config = {}
    if config:
        with open(config, 'r') as f:
            agent_config = json.load(f)
    
    # Override with command line options
    if run_dir:
        agent_config['run_dir'] = run_dir
    if tick_interval:
        agent_config['tick_interval'] = tick_interval
    if max_cycles:
        agent_config['max_cycles'] = max_cycles
    if checkpoint_interval:
        agent_config['checkpoint_interval'] = checkpoint_interval
    
    try:
        agent = BabyAGIAgent(agent_config)
        click.echo(f"Agent started with config: {agent_config}")
        click.echo(f"Runtime directory: {agent.run_dir}")
        click.echo("Press Ctrl+C to stop the agent")
        
        agent.start()
        
    except KeyboardInterrupt:
        click.echo("\nAgent stopped by user")
    except Exception as e:
        click.echo(f"Error starting agent: {e}", err=True)
        raise click.Abort()


@baby_agi.command()
@click.option('--socket-path', '-s', help='Control socket path')
def status(socket_path: Optional[str]):
    """Get the status of the Baby-AGI agent."""
    controller = AgentController(socket_path)
    response = controller.status()
    
    try:
        data = json.loads(response)
        click.echo("Baby-AGI Agent Status:")
        click.echo(f"  Running: {data.get('running', 'unknown')}")
        click.echo(f"  Paused: {data.get('paused', 'unknown')}")
        
        if 'config' in data:
            config = data['config']
            click.echo(f"  Run Directory: {config.get('run_dir', 'unknown')}")
            click.echo(f"  Tick Interval: {config.get('tick_interval', 'unknown')}s")
            click.echo(f"  Max Cycles: {config.get('max_cycles', 'unknown')}")
            
            budgets = config.get('budgets', {})
            if budgets:
                click.echo("  Budgets:")
                for budget_type, value in budgets.items():
                    click.echo(f"    {budget_type}: {value}")
    except json.JSONDecodeError:
        click.echo(response)


@baby_agi.command()
@click.option('--socket-path', '-s', help='Control socket path')
def pause(socket_path: Optional[str]):
    """Pause the Baby-AGI agent."""
    controller = AgentController(socket_path)
    response = controller.pause()
    click.echo(response)


@baby_agi.command()
@click.option('--socket-path', '-s', help='Control socket path')
def resume(socket_path: Optional[str]):
    """Resume the Baby-AGI agent."""
    controller = AgentController(socket_path)
    response = controller.resume()
    click.echo(response)


@baby_agi.command()
@click.option('--socket-path', '-s', help='Control socket path')
def stop(socket_path: Optional[str]):
    """Stop the Baby-AGI agent."""
    controller = AgentController(socket_path)
    response = controller.stop()
    click.echo(response)


@baby_agi.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--checkpoint-dir', '-d', help='Checkpoint directory path')
def runtime_demo(config: Optional[str], checkpoint_dir: Optional[str]):
    """Demonstrate the Baby-AGI runtime functionality."""
    click.echo("Baby-AGI Runtime Demo")
    
    # Load configuration
    runtime_config = {}
    if config:
        with open(config, 'r') as f:
            runtime_config = json.load(f)
    else:
        # Default demo configuration
        runtime_config = {
            "budgets": {"time": 300, "tokens": 100000, "tool_calls": 100},
            "nodes": {
                "planner": {"policy": "react+mcts", "prune_with": "prm"},
                "retriever": {"hyde": True, "topk": 5},
                "actor": {"tools": ["demo_tool"]},
                "critic": {"model": "prm_think", "accept_threshold": 0.6},
                "memory_mgr": {"tiers": ["working", "episodic"]},
                "safety": {"rails": "demo", "schema_validate": True}
            },
            "checkpoint_interval": 5
        }
    
    # Create runtime
    runtime = AgentRuntime(runtime_config, checkpoint_dir)
    
    # Register demo nodes
    def demo_planner(inputs):
        return {"plan": f"Demo plan based on {inputs.get('task', 'default task')}"}
    
    def demo_retriever(inputs):
        return {"retrieved": f"Demo retrieval for {inputs.get('plan', 'no plan')}"}
    
    def demo_actor(inputs):
        return {"action": f"Demo action using {inputs.get('retrieved', 'no data')}"}
    
    def demo_critic(inputs):
        return {"score": 0.8, "feedback": "Demo feedback"}
    
    def demo_memory_mgr(inputs):
        return {"stored": f"Stored: {inputs.get('action', 'no action')}"}
    
    def demo_safety(inputs):
        return {"approved": True, "checks": ["demo_check_1", "demo_check_2"]}
    
    runtime.register_node("planner", demo_planner)
    runtime.register_node("retriever", demo_retriever)
    runtime.register_node("actor", demo_actor)
    runtime.register_node("critic", demo_critic)
    runtime.register_node("memory_mgr", demo_memory_mgr)
    runtime.register_node("safety", demo_safety)
    
    click.echo("Demo nodes registered")
    
    # Execute a cycle
    click.echo("\nExecuting demo cycle...")
    try:
        outputs = runtime.execute_cycle({"task": "demo task"})
        click.echo("Cycle completed successfully!")
        
        for node_name, output in outputs.items():
            click.echo(f"  {node_name}: {output}")
            
    except Exception as e:
        click.echo(f"Error during execution: {e}", err=True)
    
    # Show status
    click.echo("\nRuntime Status:")
    status = runtime.get_status()
    click.echo(f"  Node States: {len(status['node_states'])} nodes")
    click.echo(f"  Checkpoints: {len(runtime.list_checkpoints())} available")
    
    # List checkpoints
    checkpoints = runtime.list_checkpoints()
    if checkpoints:
        click.echo("\nAvailable Checkpoints:")
        for checkpoint in checkpoints:
            click.echo(f"  {checkpoint}")


@baby_agi.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
def create_config(config: Optional[str]):
    """Create a sample Baby-AGI configuration file."""
    if config:
        config_path = Path(config)
    else:
        config_path = Path("baby_agi_config.json")
    
    sample_config = {
        "run_dir": "~/.babyagi",
        "tick_interval": 5,
        "max_cycles": 1000,
        "checkpoint_interval": 10,
        "budgets": {
            "time": 600,
            "tokens": 200000,
            "tool_calls": 200
        },
        "nodes": {
            "planner": {
                "policy": "react+mcts",
                "prune_with": "prm"
            },
            "retriever": {
                "hyde": True,
                "topk": 8
            },
            "actor": {
                "tools": ["search", "code_exec", "browser", "http", "fs"]
            },
            "critic": {
                "model": "prm_think",
                "accept_threshold": 0.62
            },
            "memory_mgr": {
                "tiers": ["working", "episodic", "semantic"],
                "backend": "memgpt+graph"
            },
            "safety": {
                "rails": "nemo",
                "schema_validate": True
            }
        },
        "interrupts": {
            "before_tools": ["email_send", "payment", "prod_write"],
            "on_stop": "save_state+quiesce"
        },
        "sandbox": {
            "jail": "nsjail",
            "limits": {
                "cpu": "2000m",
                "mem": "2Gi",
                "io": "10MB/s"
            }
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    click.echo(f"Sample configuration created: {config_path}")
    click.echo("You can modify this file and use it with: baby-agi start -c baby_agi_config.json")


if __name__ == '__main__':
    baby_agi()
