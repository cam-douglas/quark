#!/usr/bin/env python3
"""
Exponential Growth CLI

Command-line interface for managing and monitoring the exponential growth system.
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

from ................................................exponential_orchestrator import ExponentialOrchestrator
from ................................................command_database import CommandDatabase

class GrowthCLI:
    """CLI for exponential growth management."""
    
    def __init__(self):
        self.orchestrator = None
        
    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="SmallMind Exponential Growth System - Naturally curious command discovery",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
🧠 Exponential Growth Commands:

  start     - Start exponential growth with curious agents
  stop      - Stop exponential growth process
  status    - Show current growth status and metrics
  boost     - Boost curiosity and learning parameters
  evolve    - Trigger manual evolution cycle
  agents    - Show curious agent status
  metrics   - Show detailed growth metrics
  demo      - Run interactive demonstration

Examples:
  # Start exponential growth
  python -m smallmind.commands.growth start

  # Monitor growth in real-time
  python -m smallmind.commands.growth status --watch

  # Boost curiosity when growth slows
  python -m smallmind.commands.growth boost --factor 2.0

  # See agent activity
  python -m smallmind.commands.growth agents --verbose
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Growth commands')
        
        # Start command
        start_parser = subparsers.add_parser('start', help='Start exponential growth')
        start_parser.add_argument('--project-root', help='Project root directory')
        start_parser.add_argument('--aggressive', action='store_true', 
                                help='Use aggressive growth parameters')
        
        # Stop command
        subparsers.add_parser('stop', help='Stop exponential growth')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show growth status')
        status_parser.add_argument('--json', action='store_true', help='JSON output')
        status_parser.add_argument('--watch', action='store_true', help='Watch mode')
        
        # Boost command
        boost_parser = subparsers.add_parser('boost', help='Boost growth parameters')
        boost_parser.add_argument('--factor', type=float, default=1.5, 
                                help='Boost factor (default: 1.5)')
        boost_parser.add_argument('--agent', help='Specific agent to boost')
        
        # Evolve command
        evolve_parser = subparsers.add_parser('evolve', help='Trigger evolution cycle')
        evolve_parser.add_argument('--cycles', type=int, default=1, 
                                 help='Number of evolution cycles')
        
        # Agents command
        agents_parser = subparsers.add_parser('agents', help='Show agent status')
        agents_parser.add_argument('--verbose', action='store_true', help='Detailed info')
        
        # Metrics command
        metrics_parser = subparsers.add_parser('metrics', help='Show growth metrics')
        metrics_parser.add_argument('--export', help='Export metrics to file')
        
        # Demo command
        demo_parser = subparsers.add_parser('demo', help='Interactive demonstration')
        demo_parser.add_argument('--duration', type=int, default=300, 
                               help='Demo duration in seconds')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return 0
        
        # Initialize orchestrator
        project_root = Path(args.project_root) if hasattr(args, 'project_root') and args.project_root else Path.cwd()
        self.orchestrator = ExponentialOrchestrator(project_root)
        
        # Execute command
        try:
            if args.command == 'start':
                return self._start_growth(args)
            elif args.command == 'stop':
                return self._stop_growth()
            elif args.command == 'status':
                return self._show_status(args)
            elif args.command == 'boost':
                return self._boost_growth(args)
            elif args.command == 'evolve':
                return self._trigger_evolution(args)
            elif args.command == 'agents':
                return self._show_agents(args)
            elif args.command == 'metrics':
                return self._show_metrics(args)
            elif args.command == 'demo':
                return self._run_demo(args)
            else:
                print(f"Unknown command: {args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            if self.orchestrator and self.orchestrator.is_active:
                self.orchestrator.stop_exponential_growth()
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
        finally:
            if self.orchestrator:
                self.orchestrator.db.close()
    
    def _start_growth(self, args) -> int:
        """Start exponential growth."""
        print("🚀 Starting Exponential Growth System")
        print("=" * 50)
        
        # Show initial state
        initial_status = self.orchestrator.get_exponential_status()
        print(f"📊 Initial commands: {initial_status['current_commands']}")
        print(f"🤖 Curious agents: {initial_status['curious_agents']}")
        print(f"🔍 Exploration targets: {initial_status['exploration_targets']}")
        
        # Apply aggressive parameters if requested
        if args.aggressive:
            print("⚡ Applying aggressive growth parameters...")
            self.orchestrator.exponential_factor = 1.5
            self.orchestrator.cycle_interval = 30  # 30 seconds between cycles
            self.orchestrator.learning_acceleration_factor = 2.0
        
        # Start growth
        self.orchestrator.start_exponential_growth()
        
        print("✅ Exponential growth started!")
        print("📈 Commands will now grow exponentially through:")
        print("   • Naturally curious agents exploring your codebase")
        print("   • Adaptive learning from usage patterns")
        print("   • Intelligent command evolution and mutation")
        print("   • Neuro-assisted discovery of new command sources")
        print()
        print("💡 Use 'status --watch' to monitor growth in real-time")
        print("🛑 Use 'stop' to halt the growth process")
        
        return 0
    
    def _stop_growth(self) -> int:
        """Stop exponential growth."""
        print("🛑 Stopping Exponential Growth System")
        
        if not self.orchestrator.is_active:
            print("⚠️  Growth system is not currently active")
            return 1
        
        self.orchestrator.stop_exponential_growth()
        
        # Show final summary
        final_status = self.orchestrator.get_exponential_status()
        print(f"📊 Final commands: {final_status['current_commands']}")
        print(f"🎯 Total discoveries: {final_status['total_discoveries']}")
        print(f"🔄 Growth cycles: {final_status['growth_cycles']}")
        print("✅ Growth system stopped")
        
        return 0
    
    def _show_status(self, args) -> int:
        """Show growth status."""
        if args.watch:
            return self._watch_status()
        
        status = self.orchestrator.get_exponential_status()
        
        if args.json:
            print(json.dumps(status, indent=2))
            return 0
        
        print("📊 Exponential Growth Status")
        print("=" * 40)
        
        # Growth state
        state_emoji = "🟢" if status['is_active'] else "🔴"
        print(f"{state_emoji} Status: {'ACTIVE' if status['is_active'] else 'INACTIVE'}")
        print(f"📈 Current commands: {status['current_commands']}")
        print(f"🎯 Total discoveries: {status['total_discoveries']}")
        print(f"🔄 Growth cycles: {status['growth_cycles']}")
        print(f"⚡ Exponential factor: {status['exponential_factor']:.2f}")
        print(f"📊 Exponential growth: {status['exponential_growth']:.2f}x")
        
        print(f"\n🤖 System Components:")
        print(f"   • Curious agents: {status['curious_agents']}")
        print(f"   • Exploration targets: {status['exploration_targets']}")
        print(f"   • Learning patterns: {status['learning_patterns']}")
        print(f"   • Evolution events: {status['evolution_events']}")
        
        neuro_status = "✅" if status['neuro_available'] else "❌"
        print(f"   • Neuro integration: {neuro_status}")
        
        # Recent metrics
        if status['recent_metrics']:
            print(f"\n📈 Recent Growth Metrics:")
            for i, metric in enumerate(status['recent_metrics'][-3:], 1):
                print(f"   {i}. {metric.new_discoveries} discoveries, "
                      f"{metric.growth_rate:.1%} growth rate")
        
        return 0
    
    def _watch_status(self) -> int:
        """Watch status in real-time."""
        print("👀 Watching exponential growth (Ctrl+C to stop)")
        print("=" * 50)
        
        try:
            while True:
                # Clear screen (simple version)
                print("\033[2J\033[H")
                
                status = self.orchestrator.get_exponential_status()
                
                print("🧠 SmallMind Exponential Growth - Real-time Monitor")
                print("=" * 60)
                
                # Main stats
                state = "🟢 GROWING" if status['is_active'] else "🔴 STOPPED"
                print(f"Status: {state}")
                print(f"Commands: {status['current_commands']} "
                      f"(+{status['total_discoveries']} discovered)")
                print(f"Growth Factor: {status['exponential_factor']:.2f}x")
                print(f"Cycles: {status['growth_cycles']}")
                
                # Agent activity
                print(f"\n🤖 Agent Activity:")
                print(f"   Active Agents: {status['curious_agents']}")
                print(f"   Exploration Targets: {status['exploration_targets']}")
                
                # Learning progress
                print(f"\n🧠 Learning Progress:")
                print(f"   Patterns Discovered: {status['learning_patterns']}")
                print(f"   Evolution Events: {status['evolution_events']}")
                
                # Recent activity
                if status['recent_metrics']:
                    latest = status['recent_metrics'][-1]
                    print(f"\n📊 Latest Cycle:")
                    print(f"   New Discoveries: {latest.new_discoveries}")
                    print(f"   Growth Rate: {latest.growth_rate:.1%}")
                    print(f"   Curiosity Level: {latest.curiosity_level:.1%}")
                
                print(f"\n⏰ {time.strftime('%H:%M:%S')} - Press Ctrl+C to exit")
                
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n👋 Stopping watch mode")
            return 0
    
    def _boost_growth(self, args) -> int:
        """Boost growth parameters."""
        print(f"⚡ Boosting growth by factor {args.factor}")
        
        if args.agent:
            # Boost specific agent
            self.orchestrator.growth_engine.boost_curiosity(args.agent, args.factor)
            print(f"🤖 Boosted agent: {args.agent}")
        else:
            # Boost all agents
            for agent_id in self.orchestrator.growth_engine.agents.keys():
                self.orchestrator.growth_engine.boost_curiosity(agent_id, args.factor)
            
            # Boost learning
            self.orchestrator.learning_engine.boost_learning(args.factor)
            
            print(f"🚀 Boosted all {len(self.orchestrator.growth_engine.agents)} agents")
            print("🧠 Boosted learning parameters")
        
        return 0
    
    def _trigger_evolution(self, args) -> int:
        """Trigger manual evolution cycles."""
        print(f"🧬 Triggering {args.cycles} evolution cycle(s)")
        
        total_evolutions = 0
        for i in range(args.cycles):
            print(f"   Cycle {i + 1}/{args.cycles}...", end=" ")
            
            new_commands = self.orchestrator.learning_engine.evolve_commands()
            evolutions = len(new_commands)
            total_evolutions += evolutions
            
            print(f"{evolutions} new commands evolved")
            
            if new_commands:
                for cmd in new_commands[:3]:  # Show first 3
                    print(f"      • {cmd.name}: {cmd.description}")
        
        print(f"✅ Evolution complete: {total_evolutions} total new commands")
        return 0
    
    def _show_agents(self, args) -> int:
        """Show curious agent status."""
        growth_stats = self.orchestrator.growth_engine.get_growth_stats()
        
        print("🤖 Curious Agent Status")
        print("=" * 40)
        
        if not growth_stats['agents']:
            print("No agents currently active")
            return 0
        
        for agent_id, agent_data in growth_stats['agents'].items():
            status_emoji = "🟢" if agent_data['energy'] > 50 else "🟡" if agent_data['energy'] > 20 else "🔴"
            
            print(f"\n{status_emoji} {agent_id}")
            print(f"   Energy: {agent_data['energy']:.1f}/100")
            print(f"   Discoveries: {agent_data['discoveries']}")
            print(f"   Learning Rate: {agent_data['learning_rate']:.3f}")
            print(f"   Specializations: {', '.join(agent_data['specializations'])}")
            
            if args.verbose:
                # Show more detailed info
                print(f"   Status: {'Active' if agent_data['energy'] > 10 else 'Low Energy'}")
        
        print(f"\n📊 Summary: {len(growth_stats['agents'])} agents, "
              f"{growth_stats['total_discoveries']} total discoveries")
        
        return 0
    
    def _show_metrics(self, args) -> int:
        """Show detailed growth metrics."""
        status = self.orchestrator.get_exponential_status()
        learning_stats = self.orchestrator.learning_engine.get_learning_stats()
        
        metrics = {
            "growth_overview": {
                "current_commands": status['current_commands'],
                "total_discoveries": status['total_discoveries'],
                "growth_cycles": status['growth_cycles'],
                "exponential_factor": status['exponential_factor'],
                "exponential_growth": status['exponential_growth']
            },
            "agent_metrics": {
                "total_agents": status['curious_agents'],
                "exploration_targets": status['exploration_targets']
            },
            "learning_metrics": learning_stats,
            "recent_performance": status['recent_metrics']
        }
        
        if args.export:
            # Export to file
            with open(args.export, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"📄 Metrics exported to {args.export}")
        else:
            # Display metrics
            print("📊 Detailed Growth Metrics")
            print("=" * 50)
            print(json.dumps(metrics, indent=2, default=str))
        
        return 0
    
    def _run_demo(self, args) -> int:
        """Run interactive demonstration."""
        print("🎭 Exponential Growth Demonstration")
        print("=" * 50)
        
        print("This demo will show the exponential growth system in action:")
        print("1. 🤖 Curious agents exploring your codebase")
        print("2. 🧠 Adaptive learning from simulated usage")
        print("3. 🧬 Command evolution and mutation")
        print("4. 📈 Real-time growth metrics")
        
        input("\nPress Enter to start the demo...")
        
        # Start growth if not already active
        if not self.orchestrator.is_active:
            print("\n🚀 Starting exponential growth...")
            self.orchestrator.start_exponential_growth()
            time.sleep(2)
        
        # Run demo for specified duration
        start_time = time.time()
        demo_duration = args.duration
        
        print(f"\n📊 Running demo for {demo_duration} seconds...")
        print("Watch as commands grow exponentially!\n")
        
        try:
            while time.time() - start_time < demo_duration:
                status = self.orchestrator.get_exponential_status()
                
                print(f"\r⏰ {int(time.time() - start_time):03d}s | "
                      f"Commands: {status['current_commands']} | "
                      f"Discoveries: {status['total_discoveries']} | "
                      f"Cycles: {status['growth_cycles']}", end="")
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            pass
        
        # Show final results
        print("\n\n🎉 Demo Complete!")
        final_status = self.orchestrator.get_exponential_status()
        print(f"📈 Commands: {final_status['current_commands']}")
        print(f"🎯 Discoveries: {final_status['total_discoveries']}")
        print(f"🔄 Cycles: {final_status['growth_cycles']}")
        
        # Stop growth
        self.orchestrator.stop_exponential_growth()
        
        return 0

def main():
    """Entry point for growth CLI."""
    cli = GrowthCLI()
    return cli.main()

if __name__ == "__main__":
    sys.exit(main())
