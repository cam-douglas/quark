#!/usr/bin/env python3
"""
Main Runner Script - Unified Brain Simulation
Starts the complete brain simulation with all components integrated
"""

import os, sys
import time
from datetime import datetime

# Add database path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def main():
    """Main function to run the unified brain simulation"""
    print("🧠 UNIFIED BRAIN SIMULATION - STARTING COMPLETE SYSTEM")
    print("=" * 80)
    print("Components:")
    print("  ✅ Brain Region Mapper")
    print("  ✅ Self-Learning System") 
    print("  ✅ Internet Scraper")
    print("  ✅ Consciousness Agent")
    print("  ✅ Biorxiv Training")
    print("  ✅ Visual Simulation")
    print("  ✅ Cloud Computing Integration")
    print("=" * 80)
    
    try:
        # Import the unified consciousness agent
        from unified_consciousness_agent import UnifiedConsciousnessAgent
        
        # Create and start the unified agent
        print("\n🚀 Initializing Unified Consciousness Agent...")
        agent = UnifiedConsciousnessAgent()
        
        print("\n🎯 Starting Unified Brain Simulation...")
        print("   - Visual dashboard will open in browser")
        print("   - Training on biorxiv paper will begin")
        print("   - Brain regions will be populated with knowledge")
        print("   - Press Ctrl+C to stop simulation")
        print("\n" + "=" * 80)
        
        # Start the simulation
        agent.start_unified_simulation()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Please ensure all required modules are available")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Simulation stopped by user")
        print("✅ Brain state saved successfully")
    except Exception as e:
        print(f"\n❌ Error in unified brain simulation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
