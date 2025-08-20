#!/usr/bin/env python3
"""
Agent Connector - Integrates all agents with Unified Consciousness Agent
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
import logging
import sys

# Add database path
sys.path.append(os.path.join(os.path.dirname(__file__)))

class AgentConnector:
    def __init__(self, database_path: str = "database"):
        self.database_path = database_path
        
        # Initialize all agents
        self.agents = {}
        self.agent_status = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize agent types
        self._initialize_agent_types()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = ["agent_connections", "collaborative_learning"]
        for directory in directories:
            dir_path = os.path.join(self.database_path, directory)
            os.makedirs(dir_path, exist_ok=True)
    
    def _initialize_agent_types(self):
        """Initialize agent types and their connections"""
        self.agent_types = {
            "brain_region_mapper": {
                "dependencies": [],
                "outputs": ["knowledge_mapping", "region_status"]
            },
            "self_learning_system": {
                "dependencies": ["brain_region_mapper"],
                "outputs": ["synthetic_data", "learning_metrics"]
            },
            "internet_scraper": {
                "dependencies": [],
                "outputs": ["discovered_datasets", "metadata"]
            },
            "consciousness_simulator": {
                "dependencies": ["brain_region_mapper", "self_learning_system"],
                "outputs": ["consciousness_state", "learning_iterations"]
            },
            "biorxiv_trainer": {
                "dependencies": ["brain_region_mapper", "self_learning_system"],
                "outputs": ["training_results", "synthetic_data"]
            }
        }
    
    def connect_agent(self, agent_name: str, agent_instance: Any) -> bool:
        """Connect an agent instance"""
        if agent_name not in self.agent_types:
            self.logger.error(f"Unknown agent type: {agent_name}")
            return False
        
        try:
            self.agents[agent_name] = agent_instance
            self.agent_status[agent_name] = "connected"
            self.logger.info(f"Agent {agent_name} connected successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting agent {agent_name}: {e}")
            return False
    
    def coordinate_learning_session(self) -> Dict[str, Any]:
        """Coordinate learning session between all agents"""
        self.logger.info("Starting coordinated learning session...")
        
        session_data = {
            "session_id": f"coordinated_session_{int(time.time())}",
            "started_at": datetime.now().isoformat(),
            "participating_agents": list(self.agents.keys()),
            "learning_activities": []
        }
        
        # Coordinate activities for each agent
        for agent_name in self.agents.keys():
            if agent_name == "internet_scraper":
                self._coordinate_dataset_discovery(agent_name, session_data)
            elif agent_name == "biorxiv_trainer":
                self._coordinate_paper_training(agent_name, session_data)
            elif agent_name == "consciousness_simulator":
                self._coordinate_consciousness_simulation(agent_name, session_data)
        
        # Save session data
        self._save_session_data(session_data)
        return session_data
    
    def _coordinate_dataset_discovery(self, agent_name: str, session_data: Dict[str, Any]):
        """Coordinate dataset discovery"""
        session_data["learning_activities"].append({
            "agent": agent_name,
            "activity": "dataset_discovery",
            "timestamp": datetime.now().isoformat(),
            "status": "initiated"
        })
    
    def _coordinate_paper_training(self, agent_name: str, session_data: Dict[str, Any]):
        """Coordinate paper training"""
        session_data["learning_activities"].append({
            "agent": agent_name,
            "activity": "paper_training",
            "timestamp": datetime.now().isoformat(),
            "status": "initiated"
        })
    
    def _coordinate_consciousness_simulation(self, agent_name: str, session_data: Dict[str, Any]):
        """Coordinate consciousness simulation"""
        session_data["learning_activities"].append({
            "agent": agent_name,
            "activity": "consciousness_simulation",
            "timestamp": datetime.now().isoformat(),
            "status": "initiated"
        })
    
    def _save_session_data(self, session_data: Dict[str, Any]):
        """Save session data"""
        session_file = os.path.join(self.database_path, "agent_connections", 
                                  f"coordinated_session_{session_data['session_id']}.json")
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "total_agents": len(self.agent_types),
            "connected_agents": len(self.agents),
            "system_health": "good" if len(self.agents) > 0 else "poor"
        }

def main():
    """Test the agent connector"""
    connector = AgentConnector()
    
    print("🔗 Agent Connector - Testing Agent Integration")
    print("=" * 60)
    
    # Show agent types
    print("📊 Available Agent Types:")
    for agent_name, info in connector.agent_types.items():
        print(f"  {agent_name}: {info['dependencies']}")
    
    # Get system status
    status = connector.get_system_status()
    print(f"\n🏥 System Status:")
    print(f"   Total Agents: {status['total_agents']}")
    print(f"   Connected Agents: {status['connected_agents']}")
    print(f"   System Health: {status['system_health']}")
    
    print("\n✅ Agent Connector initialized successfully!")

if __name__ == "__main__":
    main()
