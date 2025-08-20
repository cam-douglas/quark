#!/usr/bin/env python3
"""
Organization Agent Startup Script
Purpose: Initialize and start organization agent with connectome integration
Inputs: Configuration files, workspace state
Outputs: Running organization service
Seeds: N/A (startup script)
Dependencies: organization_agent, auto_organization_service
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from architecture.orchestrator.auto_organization_service import start_auto_organization, get_auto_organization_status
    from architecture.orchestrator.organization_agent import OrganizationAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the quark root directory")
    sys.exit(1)

def initialize_organization_system():
    """Initialize the organization system"""
    print("🤖 Initializing Quark Organization Agent with Connectome Integration...")
    
    # Create organization agent
    agent = OrganizationAgent()
    
    # Run initial validation
    print("📊 Validating current directory structure...")
    report = agent.validate_structure()
    
    print(f"Structure Status: {'✅ VALID' if report['valid'] else '❌ NEEDS ORGANIZATION'}")
    print(f"Health Score: {report['structure_health'].upper()}")
    
    if report['issues']:
        print(f"Found {len(report['issues'])} organizational issues")
        
        # Ask user if they want to run initial organization
        response = input("Run initial organization? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("🔄 Running semantic organization with connectome integration...")
            summary = agent.organize_by_semantic_clusters(dry_run=False)
            print(f"✅ Organized {summary['moved']} files into {summary['clusters_created']} semantic clusters")
    
    # Load connectome metadata
    print("🧠 Loading connectome metadata for intelligent classification...")
    connectome_meta = agent.load_connectome_metadata()
    if connectome_meta:
        print(f"✅ Loaded connectome data for {len(connectome_meta)} brain modules")
    else:
        print("⚠️  No connectome data found - using pattern-based classification only")
    
    print("🎯 Organization agent initialized successfully!")
    return agent

def start_background_service():
    """Start the background organization service"""
    print("\n🚀 Starting background auto-organization service...")
    
    try:
        service = start_auto_organization()
        status = get_auto_organization_status()
        
        print("✅ Auto-organization service started!")
        print(f"📁 Monitoring: {Path.cwd()}")
        print(f"🔍 Filesystem Watcher: {'✅ Active' if status['filesystem_watcher'] else '❌ Disabled'}")
        print(f"⏰ Periodic Checks: {'✅ Active' if status['periodic_thread'] else '❌ Disabled'}")
        print(f"🧠 Semantic Mode: {'✅ Enabled' if status['config']['semantic_organization'] else '❌ Disabled'}")
        
        return service
        
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        return None

def main():
    """Main startup function"""
    print("=" * 60)
    print("🧠 QUARK ORGANIZATION AGENT WITH CONNECTOME INTEGRATION")
    print("=" * 60)
    
    # Initialize the organization system
    agent = initialize_organization_system()
    
    # Start background service
    service = start_background_service()
    
    if service:
        print("\n" + "=" * 60)
        print("🎉 ORGANIZATION SYSTEM FULLY ACTIVE")
        print("=" * 60)
        print("The organization agent is now monitoring your workspace and will:")
        print("• 📂 Automatically organize new files based on semantic content")
        print("• 🧠 Use connectome relationships for brain module placement")
        print("• 🔄 Run periodic organization checks")
        print("• 🧹 Clean up temporary files automatically")
        print("• 📊 Maintain directory structure health")
        print("\nFiles will be automatically organized into appropriate directories")
        print("based on their content, imports, and neural concepts.")
        print("\n💡 Use 'python -m architecture.orchestrator.organization_agent --help' for manual commands")
        print("💡 Use 'python -m architecture.orchestrator.auto_organization_service --status' for service status")
        print("\nPress Ctrl+C to stop the service")
        
        try:
            # Keep the service running
            import time
            while service.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping organization service...")
            service.stop()
            print("✅ Organization service stopped")
    else:
        print("\n❌ Failed to start background service")
        print("You can still use manual organization commands:")
        print("python -m architecture.orchestrator.organization_agent --semantic --dry-run")

if __name__ == "__main__":
    main()
