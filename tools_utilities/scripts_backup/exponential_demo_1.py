#!/usr/bin/env python3
"""
Exponential Growth Demonstration

This script demonstrates the naturally curious agents and exponential command growth
in an interactive and visually appealing way.
"""

import os, sys
import time
import random
import threading
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from command_database import CommandDatabase
    from exponential_orchestrator import ExponentialOrchestrator
    from curious_agents import CuriousAgent
    from adaptive_learning import AdaptiveLearningEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from the project root or ensure modules are available")
    sys.exit(1)

class ExponentialDemo:
    """Interactive demonstration of exponential command growth."""
    
    def __init__(self):
        self.project_root = Path("/Users/camdouglas/quark")
        self.demo_running = False
        self.orchestrator = None
        
    def run_full_demo(self):
        """Run the complete exponential growth demonstration."""
        print("🧠 SmallMind Exponential Command Growth Demonstration")
        print("=" * 70)
        print()
        print("This demo showcases:")
        print("📊 1. Current command database state")
        print("🤖 2. Naturally curious agents in action")
        print("🧬 3. Adaptive learning and command evolution")
        print("🚀 4. Real-time exponential growth")
        print("📈 5. Growth metrics and analytics")
        print()
        
        input("Press Enter to begin the demonstration...")
        
        try:
            # Phase 1: Show initial state
            self._demo_initial_state()
            
            # Phase 2: Introduce curious agents
            self._demo_curious_agents()
            
            # Phase 3: Show adaptive learning
            self._demo_adaptive_learning()
            
            # Phase 4: Start exponential growth
            self._demo_exponential_growth()
            
            # Phase 5: Real-time monitoring
            self._demo_real_time_growth()
            
            print("\n🎉 Demonstration Complete!")
            print("The exponential growth system is now actively discovering")
            print("and evolving commands to expand your resource base!")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
        finally:
            self._cleanup_demo()
    
    def _demo_initial_state(self):
        """Demonstrate the initial command database state."""
        print("\n📊 Phase 1: Initial Command Database State")
        print("-" * 50)
        
        # Initialize database
        db = CommandDatabase()
        
        # Show stats
        stats = db.get_stats()
        print(f"📋 Current commands: {stats['total_commands']}")
        print(f"📁 Categories: {stats['total_categories']}")
        
        print(f"\n📈 Commands by category:")
        for cat in stats['categories'][:5]:  # Show top 5
            print(f"   • {cat['name']}: {cat['count']} commands")
        
        print(f"\n🎯 Commands by complexity:")
        for comp in stats['complexity']:
            print(f"   • {comp['complexity'].title()}: {comp['count']} commands")
        
        # Show some example commands
        print(f"\n🔍 Example commands:")
        commands = db.search_commands("")[:5]
        for cmd in commands:
            print(f"   {cmd.number} {cmd.name}: {cmd.description}")
        
        db.close()
        
        input("\n✅ Press Enter to continue to curious agents...")
    
    def _demo_curious_agents(self):
        """Demonstrate curious agents discovering commands."""
        print("\n🤖 Phase 2: Naturally Curious Agents")
        print("-" * 50)
        
        print("Initializing curious agents with different specializations...")
        
        # Create sample agents
        agents = [
            CuriousAgent("neural_explorer", ["neuroscience", "ai", "ml"]),
            CuriousAgent("dev_tools_hunter", ["development", "tools", "scripts"]),
            CuriousAgent("cloud_seeker", ["cloud", "aws", "deployment"]),
            CuriousAgent("data_miner", ["data", "analysis", "jupyter"])
        ]
        
        print(f"✅ Created {len(agents)} curious agents:")
        for agent in agents:
            specializations = ", ".join(agent.specializations)
            print(f"   🤖 {agent.agent_id}: {specializations}")
        
        print(f"\n🔍 Agents exploring project structure...")
        
        # Simulate exploration
        for i, agent in enumerate(agents, 1):
            print(f"   Agent {i}/4: {agent.agent_id} exploring...", end=" ")
            
            # Simulate discovery process
            for j in range(3):
                time.sleep(0.5)
                print(".", end="", flush=True)
            
            # Simulate finding commands
            discoveries = random.randint(2, 8)
            print(f" found {discoveries} potential commands!")
            
            # Show what agent discovered
            discovery_types = ["Python scripts", "shell scripts", "config files", "Makefiles", "package.json"]
            discovered = random.sample(discovery_types, min(3, len(discovery_types)))
            for discovery in discovered:
                print(f"      • {discovery}")
        
        total_discoveries = sum(random.randint(2, 8) for _ in agents)
        print(f"\n🎯 Total potential commands discovered: {total_discoveries}")
        
        input("\n✅ Press Enter to see adaptive learning...")
    
    def _demo_adaptive_learning(self):
        """Demonstrate adaptive learning and evolution."""
        print("\n🧬 Phase 3: Adaptive Learning & Command Evolution")
        print("-" * 50)
        
        print("Simulating command usage patterns for learning...")
        
        # Simulate learning process
        learning_scenarios = [
            "User runs neural network training commands",
            "Developer uses deployment automation",
            "Data scientist analyzes datasets",
            "System admin monitors performance"
        ]
        
        for i, scenario in enumerate(learning_scenarios, 1):
            print(f"\n📊 Scenario {i}: {scenario}")
            
            # Simulate usage pattern
            commands_used = random.randint(3, 7)
            success_rate = random.uniform(0.7, 0.95)
            
            print(f"   Commands executed: {commands_used}")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Learning pattern: ", end="")
            
            # Simulate pattern discovery
            time.sleep(1)
            patterns = ["Sequential usage", "Context correlation", "Temporal preference"]
            discovered_pattern = random.choice(patterns)
            print(f"{discovered_pattern} ✓")
        
        print(f"\n🧠 Learning Engine Analysis:")
        print(f"   • Patterns discovered: {random.randint(8, 15)}")
        print(f"   • Command fitness improved: {random.randint(12, 25)} commands")
        print(f"   • Evolution candidates: {random.randint(3, 8)} commands")
        
        print(f"\n🧬 Attempting command evolution...")
        time.sleep(2)
        
        evolved_commands = [
            "enhanced neural training (mutation of existing command)",
            "smart deployment workflow (crossover of deploy + monitor)",
            "auto data pipeline (pattern-based generation)"
        ]
        
        for i, evolved in enumerate(evolved_commands, 1):
            print(f"   ✨ Evolution {i}: {evolved}")
        
        print(f"\n🎯 {len(evolved_commands)} new commands evolved from existing ones!")
        
        input("\n✅ Press Enter to start exponential growth...")
    
    def _demo_exponential_growth(self):
        """Demonstrate starting exponential growth."""
        print("\n🚀 Phase 4: Exponential Growth Activation")
        print("-" * 50)
        
        print("Initializing Exponential Growth Orchestrator...")
        
        # Initialize orchestrator
        self.orchestrator = ExponentialOrchestrator(self.project_root)
        
        print("✅ Orchestrator initialized with:")
        status = self.orchestrator.get_exponential_status()
        print(f"   📊 Base commands: {status['current_commands']}")
        print(f"   🤖 Curious agents: {status['curious_agents']}")
        print(f"   🔍 Exploration targets: {status['exploration_targets']}")
        print(f"   🧠 Neuro integration: {'✅' if status['neuro_available'] else '❌'}")
        
        print(f"\n⚡ Configuring exponential parameters...")
        self.orchestrator.exponential_factor = 1.3  # 30% growth per cycle
        self.orchestrator.cycle_interval = 30  # 30 seconds between cycles
        print(f"   Growth factor: {self.orchestrator.exponential_factor}x")
        print(f"   Cycle interval: {self.orchestrator.cycle_interval}s")
        
        print(f"\n🎯 Starting exponential growth process...")
        self.orchestrator.start_exponential_growth()
        
        print("✅ Exponential growth ACTIVATED!")
        print("📈 Commands will now grow exponentially through:")
        print("   • Curious agents exploring new areas")
        print("   • Learning from usage patterns")
        print("   • Evolving successful commands")
        print("   • Discovering external resources")
        
        input("\n✅ Press Enter to monitor real-time growth...")
    
    def _demo_real_time_growth(self):
        """Demonstrate real-time growth monitoring."""
        print("\n📈 Phase 5: Real-time Exponential Growth")
        print("-" * 50)
        
        print("Monitoring exponential growth in real-time...")
        print("(Watch as curious agents discover and evolve commands)")
        print("\nPress Ctrl+C to stop monitoring\n")
        
        start_time = time.time()
        initial_status = self.orchestrator.get_exponential_status()
        initial_commands = initial_status['current_commands']
        
        try:
            cycle = 0
            while True:
                status = self.orchestrator.get_exponential_status()
                elapsed = int(time.time() - start_time)
                
                # Calculate growth
                current_commands = status['current_commands']
                growth = current_commands - initial_commands
                growth_rate = (growth / initial_commands * 100) if initial_commands > 0 else 0
                
                # Display status
                print(f"\r⏰ {elapsed:03d}s | "
                      f"📊 Commands: {current_commands} (+{growth}) | "
                      f"📈 Growth: {growth_rate:.1f}% | "
                      f"🔄 Cycles: {status['growth_cycles']} | "
                      f"🎯 Discoveries: {status['total_discoveries']}", end="")
                
                # Show agent activity periodically
                if elapsed > 0 and elapsed % 15 == 0 and cycle != elapsed // 15:
                    cycle = elapsed // 15
                    print(f"\n🤖 Agent Activity:")
                    
                    # Simulate agent reports
                    agent_activities = [
                        "neural_explorer discovered 3 ML training scripts",
                        "dev_tools_hunter found new automation tools",
                        "cloud_seeker identified deployment configurations",
                        "data_miner uncovered analysis notebooks"
                    ]
                    
                    activity = random.choice(agent_activities)
                    print(f"   • {activity}")
                    
                    # Show learning events
                    if random.random() < 0.7:  # 70% chance
                        learning_events = [
                            "🧬 Evolved hybrid command from usage patterns",
                            "🧠 Discovered new command sequence pattern",
                            "⚡ Boosted agent curiosity based on discoveries",
                            "🔗 Created workflow command from frequent usage"
                        ]
                        event = random.choice(learning_events)
                        print(f"   • {event}")
                    
                    print()
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped by user")
        
        # Show final growth summary
        final_status = self.orchestrator.get_exponential_status()
        final_commands = final_status['current_commands']
        total_growth = final_commands - initial_commands
        
        print(f"\n📊 Growth Summary:")
        print(f"   Initial commands: {initial_commands}")
        print(f"   Final commands: {final_commands}")
        print(f"   Total growth: +{total_growth} commands")
        print(f"   Growth cycles: {final_status['growth_cycles']}")
        print(f"   Discovery rate: {final_status['total_discoveries']} discoveries")
        
        if initial_commands > 0:
            growth_percentage = (total_growth / initial_commands) * 100
            print(f"   Growth rate: {growth_percentage:.1f}%")
    
    def _cleanup_demo(self):
        """Clean up demo resources."""
        if self.orchestrator and self.orchestrator.is_active:
            print(f"\n🧹 Cleaning up exponential growth system...")
            self.orchestrator.stop_exponential_growth()
            print("✅ Growth system stopped")

def run_quick_demo():
    """Run a quick demonstration."""
    print("⚡ Quick Exponential Growth Demo")
    print("=" * 40)
    
    try:
        # Initialize basic components
        db = CommandDatabase()
        orchestrator = ExponentialOrchestrator()
        
        initial_commands = len(db.search_commands(""))
        print(f"📊 Starting with {initial_commands} commands")
        
        # Show agents
        status = orchestrator.get_exponential_status()
        print(f"🤖 {status['curious_agents']} curious agents ready")
        
        # Simulate a few cycles
        print(f"\n🔄 Running 3 growth cycles...")
        for i in range(3):
            print(f"   Cycle {i + 1}: ", end="")
            cycle_results = orchestrator._run_growth_cycle()
            
            discoveries = sum(cycle_results.values())
            print(f"{discoveries} discoveries")
            time.sleep(1)
        
        final_commands = len(db.search_commands(""))
        growth = final_commands - initial_commands
        print(f"\n📈 Result: {growth} new commands discovered!")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")

def main():
    """Main demo entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_demo()
    else:
        demo = ExponentialDemo()
        demo.run_full_demo()

if __name__ == "__main__":
    main()
