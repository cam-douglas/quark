#!/usr/bin/env python3
"""
🧠 Brain Capabilities Module

This module provides comprehensive tracking and biological awareness of all brain capabilities.
It ensures the brain knows all its capabilities at all times, with real-time monitoring,
biological integration, and capability optimization.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainCapability:
    """Represents a single brain capability"""
    
    def __init__(self, name: str, description: str, category: str, 
                 biological_requirements: Dict[str, float], 
                 current_status: str = "active"):
        self.name = name
        self.description = description
        self.category = category
        self.biological_requirements = biological_requirements
        self.current_status = current_status
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.success_rate = 1.0
        self.performance_history = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "biological_requirements": self.biological_requirements,
            "current_status": self.current_status,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "success_rate": self.success_rate,
            "performance_history": self.performance_history[-10:]  # Last 10 entries
        }
    
    def update_performance(self, success: bool, execution_time: float, 
                          cognitive_load: float, memory_used: float):
        """Update capability performance metrics"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Update success rate
        if self.access_count > 1:
            self.success_rate = (self.success_rate * (self.access_count - 1) + (1.0 if success else 0.0)) / self.access_count
        else:
            self.success_rate = 1.0 if success else 0.0
        
        # Record performance
        performance_entry = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "execution_time": execution_time,
            "cognitive_load": cognitive_load,
            "memory_used": memory_used
        }
        
        self.performance_history.append(performance_entry)
        
        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]

class BrainCapabilitiesModule:
    """Comprehensive brain capabilities management system"""
    
    def __init__(self, brain_agent_path: str = "../../brain_architecture/neural_core/biological_brain_agent.py"):
        self.brain_agent_path = Path(brain_agent_path)
        self.capabilities: Dict[str, BrainCapability] = {}
        self.capability_categories = {}
        self.biological_state = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize default capabilities
        self._initialize_default_capabilities()
        
        # Capability awareness settings
        self.awareness_settings = {
            "real_time_monitoring": True,
            "biological_integration": True,
            "performance_tracking": True,
            "capability_optimization": True,
            "auto_recovery": True,
            "learning_integration": True
        }
        
        # Biological integration parameters
        self.biological_params = {
            "capability_activation_threshold": 0.3,
            "capability_deactivation_threshold": 0.1,
            "performance_optimization_rate": 0.01,
            "capability_learning_rate": 0.005,
            "biological_constraint_weight": 0.8
        }
        
        logger.info("🧠 Brain Capabilities Module initialized")
    
    def _initialize_default_capabilities(self):
        """Initialize default brain capabilities"""
        
        # Core Cognitive Capabilities
        core_capabilities = [
            {
                "name": "executive_control",
                "description": "Planning, decision-making, and goal-directed behavior",
                "category": "core_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.4,
                    "working_memory": 0.3,
                    "energy_level": 0.5,
                    "attention_focus": 0.6
                }
            },
            {
                "name": "working_memory",
                "description": "Short-term information storage and manipulation",
                "category": "core_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.2,
                    "working_memory": 0.1,
                    "energy_level": 0.3,
                    "attention_focus": 0.4
                }
            },
            {
                "name": "action_selection",
                "description": "Choosing and executing appropriate actions",
                "category": "core_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.3,
                    "working_memory": 0.2,
                    "energy_level": 0.4,
                    "attention_focus": 0.5
                }
            },
            {
                "name": "information_relay",
                "description": "Processing and routing sensory information",
                "category": "core_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.2,
                    "working_memory": 0.1,
                    "energy_level": 0.2,
                    "attention_focus": 0.3
                }
            },
            {
                "name": "episodic_memory",
                "description": "Long-term memory formation and retrieval",
                "category": "core_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.3,
                    "working_memory": 0.2,
                    "energy_level": 0.4,
                    "attention_focus": 0.4
                }
            }
        ]
        
        # Task Management Capabilities
        task_capabilities = [
            {
                "name": "task_loading",
                "description": "Loading and parsing tasks from external systems",
                "category": "task_management",
                "biological_requirements": {
                    "cognitive_load": 0.2,
                    "working_memory": 0.2,
                    "energy_level": 0.3,
                    "attention_focus": 0.4
                }
            },
            {
                "name": "task_analysis",
                "description": "Analyzing task priorities and requirements",
                "category": "task_management",
                "biological_requirements": {
                    "cognitive_load": 0.4,
                    "working_memory": 0.3,
                    "energy_level": 0.4,
                    "attention_focus": 0.6
                }
            },
            {
                "name": "task_decisions",
                "description": "Making intelligent decisions about task execution",
                "category": "task_management",
                "biological_requirements": {
                    "cognitive_load": 0.5,
                    "working_memory": 0.4,
                    "energy_level": 0.5,
                    "attention_focus": 0.7
                }
            },
            {
                "name": "task_execution",
                "description": "Executing tasks based on brain state and resources",
                "category": "task_management",
                "biological_requirements": {
                    "cognitive_load": 0.6,
                    "working_memory": 0.5,
                    "energy_level": 0.6,
                    "attention_focus": 0.8
                }
            },
            {
                "name": "resource_management",
                "description": "Managing cognitive and computational resources",
                "category": "task_management",
                "biological_requirements": {
                    "cognitive_load": 0.3,
                    "working_memory": 0.2,
                    "energy_level": 0.3,
                    "attention_focus": 0.4
                }
            }
        ]
        
        # Biological Integration Capabilities
        biological_capabilities = [
            {
                "name": "constraint_monitoring",
                "description": "Monitoring biological constraint compliance",
                "category": "biological_integration",
                "biological_requirements": {
                    "cognitive_load": 0.1,
                    "working_memory": 0.1,
                    "energy_level": 0.2,
                    "attention_focus": 0.2
                }
            },
            {
                "name": "health_monitoring",
                "description": "Monitoring brain health and performance",
                "category": "biological_integration",
                "biological_requirements": {
                    "cognitive_load": 0.2,
                    "working_memory": 0.1,
                    "energy_level": 0.2,
                    "attention_focus": 0.3
                }
            },
            {
                "name": "capability_optimization",
                "description": "Optimizing capability performance based on biological state",
                "category": "biological_integration",
                "biological_requirements": {
                    "cognitive_load": 0.4,
                    "working_memory": 0.3,
                    "energy_level": 0.4,
                    "attention_focus": 0.5
                }
            },
            {
                "name": "learning_integration",
                "description": "Integrating learning from capability usage patterns",
                "category": "biological_integration",
                "biological_requirements": {
                    "cognitive_load": 0.3,
                    "working_memory": 0.2,
                    "energy_level": 0.3,
                    "attention_focus": 0.4
                }
            }
        ]
        
        # Advanced Capabilities
        advanced_capabilities = [
            {
                "name": "pattern_recognition",
                "description": "Recognizing patterns in data and behavior",
                "category": "advanced_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.5,
                    "working_memory": 0.4,
                    "energy_level": 0.5,
                    "attention_focus": 0.6
                }
            },
            {
                "name": "predictive_modeling",
                "description": "Creating predictive models of future states",
                "category": "advanced_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.6,
                    "working_memory": 0.5,
                    "energy_level": 0.6,
                    "attention_focus": 0.7
                }
            },
            {
                "name": "adaptive_learning",
                "description": "Adapting behavior based on experience and feedback",
                "category": "advanced_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.4,
                    "working_memory": 0.3,
                    "energy_level": 0.4,
                    "attention_focus": 0.5
                }
            },
            {
                "name": "meta_cognition",
                "description": "Thinking about thinking and self-awareness",
                "category": "advanced_cognitive",
                "biological_requirements": {
                    "cognitive_load": 0.7,
                    "working_memory": 0.6,
                    "energy_level": 0.7,
                    "attention_focus": 0.8
                }
            }
        ]
        
        # Add all capabilities
        all_capabilities = core_capabilities + task_capabilities + biological_capabilities + advanced_capabilities
        
        for cap_data in all_capabilities:
            capability = BrainCapability(
                name=cap_data["name"],
                description=cap_data["description"],
                category=cap_data["category"],
                biological_requirements=cap_data["biological_requirements"]
            )
            self.capabilities[cap_data["name"]] = capability
            
            # Organize by category
            if cap_data["category"] not in self.capability_categories:
                self.capability_categories[cap_data["category"]] = []
            self.capability_categories[cap_data["category"]].append(cap_data["name"])
        
        logger.info(f"🧠 Initialized {len(self.capabilities)} brain capabilities")
    
    def get_capability(self, name: str) -> Optional[BrainCapability]:
        """Get a specific capability by name"""
        return self.capabilities.get(name)
    
    def get_capabilities_by_category(self, category: str) -> List[BrainCapability]:
        """Get all capabilities in a specific category"""
        if category not in self.capability_categories:
            return []
        
        return [self.capabilities[name] for name in self.capability_categories[category]]
    
    def get_all_capabilities(self) -> Dict[str, BrainCapability]:
        """Get all capabilities"""
        return self.capabilities.copy()
    
    def add_capability(self, name: str, description: str, category: str, 
                      biological_requirements: Dict[str, float]) -> bool:
        """Add a new capability"""
        if name in self.capabilities:
            logger.warning(f"Capability {name} already exists")
            return False
        
        capability = BrainCapability(
            name=name,
            description=description,
            category=category,
            biological_requirements=biological_requirements
        )
        
        self.capabilities[name] = capability
        
        # Add to category
        if category not in self.capability_categories:
            self.capability_categories[category] = []
        self.capability_categories[category].append(name)
        
        logger.info(f"🧠 Added new capability: {name}")
        return True
    
    def remove_capability(self, name: str) -> bool:
        """Remove a capability"""
        if name not in self.capabilities:
            logger.warning(f"Capability {name} does not exist")
            return False
        
        capability = self.capabilities[name]
        category = capability.category
        
        # Remove from capabilities
        del self.capabilities[name]
        
        # Remove from category
        if category in self.capability_categories:
            self.capability_categories[category].remove(name)
            if not self.capability_categories[category]:
                del self.capability_categories[category]
        
        logger.info(f"🧠 Removed capability: {name}")
        return True
    
    def update_biological_state(self, biological_state: Dict[str, float]):
        """Update current biological state"""
        self.biological_state = biological_state.copy()
        
        # Update capability status based on biological state
        self._update_capability_status()
    
    def _update_capability_status(self):
        """Update capability status based on current biological state"""
        for capability in self.capabilities.values():
            # Check if biological requirements are met
            requirements_met = True
            
            for requirement, required_value in capability.biological_requirements.items():
                if requirement in self.biological_state:
                    current_value = self.biological_state[requirement]
                    
                    # Check if requirement is met
                    if requirement.startswith("min_"):
                        if current_value < required_value:
                            requirements_met = False
                    elif requirement.startswith("max_"):
                        if current_value > required_value:
                            requirements_met = False
                    else:
                        # Default to minimum requirement
                        if current_value < required_value:
                            requirements_met = False
            
            # Update capability status
            if requirements_met:
                if capability.current_status != "active":
                    capability.current_status = "active"
                    logger.info(f"🧠 Capability {capability.name} activated")
            else:
                if capability.current_status != "inactive":
                    capability.current_status = "inactive"
                    logger.info(f"🧠 Capability {capability.name} deactivated due to biological constraints")
    
    def get_available_capabilities(self) -> List[BrainCapability]:
        """Get all currently available capabilities"""
        return [cap for cap in self.capabilities.values() if cap.current_status == "active"]
    
    def get_capability_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all capabilities"""
        summary = {
            "total_capabilities": len(self.capabilities),
            "active_capabilities": len(self.get_available_capabilities()),
            "inactive_capabilities": len(self.capabilities) - len(self.get_available_capabilities()),
            "categories": {},
            "performance_metrics": {},
            "biological_state": self.biological_state.copy()
        }
        
        # Category breakdown
        for category, capability_names in self.capability_categories.items():
            active_count = sum(1 for name in capability_names 
                             if self.capabilities[name].current_status == "active")
            summary["categories"][category] = {
                "total": len(capability_names),
                "active": active_count,
                "inactive": len(capability_names) - active_count
            }
        
        # Performance metrics
        if self.capabilities:
            avg_success_rate = np.mean([cap.success_rate for cap in self.capabilities.values()])
            avg_access_count = np.mean([cap.access_count for cap in self.capabilities.values()])
            
            summary["performance_metrics"] = {
                "average_success_rate": avg_success_rate,
                "average_access_count": avg_access_count,
                "most_used_capability": max(self.capabilities.values(), key=lambda x: x.access_count).name,
                "highest_performing_capability": max(self.capabilities.values(), key=lambda x: x.success_rate).name
            }
        
        return summary
    
    def execute_capability(self, name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific capability"""
        capability = self.get_capability(name)
        if not capability:
            return {"success": False, "error": f"Capability {name} not found"}
        
        if capability.current_status != "active":
            return {"success": False, "error": f"Capability {name} is not active"}
        
        start_time = time.time()
        
        try:
            # Simulate capability execution
            result = self._simulate_capability_execution(name, parameters or {})
            
            execution_time = time.time() - start_time
            
            # Update capability performance
            capability.update_performance(
                success=result["success"],
                execution_time=execution_time,
                cognitive_load=self.biological_state.get("cognitive_load", 0.5),
                memory_used=self.biological_state.get("working_memory", 0.5)
            )
            
            result["execution_time"] = execution_time
            result["capability_name"] = name
            
            logger.info(f"🧠 Executed capability {name}: {result['success']}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update capability performance (failed execution)
            capability.update_performance(
                success=False,
                execution_time=execution_time,
                cognitive_load=self.biological_state.get("cognitive_load", 0.5),
                memory_used=self.biological_state.get("working_memory", 0.5)
            )
            
            logger.error(f"Error executing capability {name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "capability_name": name
            }
    
    def _simulate_capability_execution(self, name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate capability execution (placeholder for real implementation)"""
        # This would be replaced with actual capability execution logic
        
        # Simulate different capabilities
        if "task" in name.lower():
            return {
                "success": True,
                "result": f"Task-related capability {name} executed successfully",
                "data": {"task_count": 5, "priority": "high"}
            }
        elif "memory" in name.lower():
            return {
                "success": True,
                "result": f"Memory capability {name} executed successfully",
                "data": {"memory_usage": 0.3, "items_stored": 10}
            }
        elif "executive" in name.lower():
            return {
                "success": True,
                "result": f"Executive capability {name} executed successfully",
                "data": {"decisions_made": 3, "plans_created": 2}
            }
        else:
            return {
                "success": True,
                "result": f"Capability {name} executed successfully",
                "data": {"execution_id": f"exec_{int(time.time())}"}
            }
    
    def start_capability_monitoring(self):
        """Start continuous capability monitoring"""
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Update biological state (simulated)
                    simulated_brain_state = self._simulate_brain_state()
                    self.update_biological_state(simulated_brain_state)
                    
                    # Log capability status
                    available_capabilities = self.get_available_capabilities()
                    logger.info(f"🧠 Available capabilities: {len(available_capabilities)}/{len(self.capabilities)}")
                    
                    # Sleep until next check
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in capability monitoring loop: {e}")
                    time.sleep(30)
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🚀 Brain capability monitoring started")
    
    def stop_capability_monitoring(self):
        """Stop capability monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("🛑 Brain capability monitoring stopped")
    
    def _simulate_brain_state(self) -> Dict[str, float]:
        """Simulate current brain state for monitoring"""
        # This would be replaced with actual brain state reading
        
        current_hour = datetime.now().hour
        base_load = 0.3 + (0.2 * (current_hour % 8) / 8)
        
        import random
        random.seed(int(time.time() / 300))
        
        return {
            "cognitive_load": min(1.0, base_load + random.uniform(-0.1, 0.1)),
            "working_memory": max(0.0, 1.0 - base_load + random.uniform(-0.1, 0.1)),
            "energy_level": max(0.0, 1.0 - (current_hour / 24) + random.uniform(-0.1, 0.1)),
            "attention_focus": 0.5 + random.uniform(-0.2, 0.2)
        }
    
    def generate_capability_report(self) -> str:
        """Generate comprehensive capability report"""
        summary = self.get_capability_performance_summary()
        
        report = f"""# 🧠 Brain Capabilities Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Capabilities**: {summary['total_capabilities']}
**Active Capabilities**: {summary['active_capabilities']}
**Inactive Capabilities**: {summary['inactive_capabilities']}

## 🧬 Current Biological State

"""
        
        for metric, value in summary['biological_state'].items():
            report += f"- **{metric}**: {value:.3f}\n"
        
        report += f"""
## 📊 Capability Categories

"""
        
        for category, stats in summary['categories'].items():
            report += f"""### {category.replace('_', ' ').title()}
- **Total**: {stats['total']}
- **Active**: {stats['active']}
- **Inactive**: {stats['inactive']}

"""
        
        report += f"""
## 🎯 Performance Metrics

- **Average Success Rate**: {summary['performance_metrics'].get('average_success_rate', 0):.1%}
- **Average Access Count**: {summary['performance_metrics'].get('average_access_count', 0):.1f}
- **Most Used Capability**: {summary['performance_metrics'].get('most_used_capability', 'N/A')}
- **Highest Performing**: {summary['performance_metrics'].get('highest_performing_capability', 'N/A')}

## 🔍 Detailed Capability Status

"""
        
        for category, capability_names in self.capability_categories.items():
            report += f"### {category.replace('_', ' ').title()}\n"
            
            for name in capability_names:
                capability = self.capabilities[name]
                status_emoji = "🟢" if capability.current_status == "active" else "🔴"
                
                report += f"""**{status_emoji} {capability.name}**
- **Status**: {capability.current_status}
- **Success Rate**: {capability.success_rate:.1%}
- **Access Count**: {capability.access_count}
- **Last Accessed**: {capability.last_accessed.strftime('%Y-%m-%d %H:%M:%S')}
- **Description**: {capability.description}

"""
        
        report += f"""
---
*Generated by Brain Capabilities Module*
"""
        
        return report
    
    def save_capability_report(self, output_path: str = "brain_capabilities_report.md"):
        """Save capability report to file"""
        try:
            report_content = self.generate_capability_report()
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Capability report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving capability report: {e}")
            return False

def main():
    """Main function to demonstrate brain capabilities module"""
    print("🧠 Brain Capabilities Module")
    print("=" * 50)
    
    # Create capabilities module
    capabilities_module = BrainCapabilitiesModule()
    
    # Get initial status
    print(f"📊 Initial Status:")
    print(f"   Total Capabilities: {len(capabilities_module.capabilities)}")
    print(f"   Categories: {list(capabilities_module.capability_categories.keys())}")
    
    # Start monitoring
    print("\n🚀 Starting capability monitoring...")
    capabilities_module.start_capability_monitoring()
    
    try:
        # Let it run for a few cycles
        print("📈 Monitoring for 2 minutes...")
        time.sleep(120)
        
        # Execute some capabilities
        print("\n🧠 Executing sample capabilities...")
        
        sample_capabilities = ["executive_control", "working_memory", "task_analysis"]
        for cap_name in sample_capabilities:
            result = capabilities_module.execute_capability(cap_name)
            print(f"   {cap_name}: {'✅' if result['success'] else '❌'} {result.get('result', result.get('error', 'Unknown'))}")
        
        # Generate and display report
        print("\n📊 Generating capability report...")
        report = capabilities_module.generate_capability_report()
        print(report)
        
        # Save report
        capabilities_module.save_capability_report("brain_capabilities_report.md")
        print("\n💾 Capability report saved to brain_capabilities_report.md")
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring interrupted by user")
    
    finally:
        # Stop monitoring
        print("🛑 Stopping capability monitoring...")
        capabilities_module.stop_capability_monitoring()
        
        # Final summary
        summary = capabilities_module.get_capability_performance_summary()
        print(f"\n📊 Final Summary:")
        print(f"   Active Capabilities: {summary['active_capabilities']}/{summary['total_capabilities']}")
        print(f"   Average Success Rate: {summary['performance_metrics'].get('average_success_rate', 0):.1%}")
        
        print("\n✅ Brain capabilities demonstration complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Brain capabilities demonstration failed: {e}")
        import traceback
        traceback.print_exc()
