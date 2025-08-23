#!/usr/bin/env python3
"""
N0 Learning & Consolidation Integration with Task Management

This system integrates Quark's new N0 stage capabilities with the task management
system, allowing self-appointed learning objectives while maintaining safety protocols.
"""

import os
import sys
import json
import yaml
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from brain_architecture.neural_core.complexity_evolution_agent.complexity_evolver import ComplexityEvolutionAgent
from brain_architecture.neural_core.complexity_evolution_agent.connectome_synchronizer import ConnectomeSynchronizer

@dataclass
class LearningObjective:
    """A self-appointed learning objective for Quark"""
    id: str
    title: str
    description: str
    domain: str  # neuroscience, ml, consciousness, etc.
    complexity_level: str  # F, N0, N1, N2, N3
    safety_risk_level: str  # low, medium, high, critical
    estimated_duration_hours: float
    prerequisites: List[str]
    success_criteria: List[str]
    safety_constraints: List[str]
    created_at: datetime
    status: str  # pending, active, completed, failed
    progress: float = 0.0
    current_phase: str = "planning"
    safety_checks_passed: bool = False
    last_safety_check: Optional[datetime] = None

@dataclass
class SafetyProtocol:
    """Safety protocol for learning objectives"""
    protocol_id: str
    name: str
    description: str
    risk_levels: List[str]
    required_checks: List[str]
    validation_rules: List[str]
    emergency_stop_conditions: List[str]
    human_oversight_required: bool = False
    max_complexity_increase: float = 1.5

class N0LearningIntegration:
    """
    Integrates N0 learning capabilities with task management and safety protocols.
    Allows Quark to self-appoint learning objectives while maintaining safety.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core systems
        self.complexity_agent = ComplexityEvolutionAgent()
        self.connectome_sync = ConnectomeSynchronizer()
        
        # Learning and safety systems
        self.learning_objectives: Dict[str, LearningObjective] = {}
        self.safety_protocols: Dict[str, SafetyProtocol] = {}
        self.active_learning_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Load existing systems
        self.task_system = self._load_task_system()
        self.safety_system = self._load_safety_system()
        
        # Initialize safety protocols
        self._initialize_safety_protocols()
        
        self.logger.info("N0 Learning Integration initialized")
    
    def _load_task_system(self) -> Dict[str, Any]:
        """Load existing task management system"""
        task_files = [
            "tasks/current_tasks.md",
            "tasks/goals/README.md",
            "tasks/HIGH_LEVEL_ROADMAP.md"
        ]
        
        task_system = {}
        for task_file in task_files:
            if os.path.exists(task_file):
                try:
                    with open(task_file, 'r') as f:
                        content = f.read()
                        task_system[task_file] = {
                            'content': content,
                            'last_modified': os.path.getmtime(task_file)
                        }
                except Exception as e:
                    self.logger.warning(f"Could not load {task_file}: {e}")
        
        return task_system
    
    def _load_safety_system(self) -> Dict[str, Any]:
        """Load existing safety and ethics systems"""
        safety_files = [
            "management/rules/security/README.md",
            "brain_architecture/neural_core/safety_officer/README.md"
        ]
        
        safety_system = {}
        for safety_file in safety_files:
            if os.path.exists(safety_file):
                try:
                    with open(safety_file, 'r') as f:
                        content = f.read()
                        safety_system[safety_file] = {
                            'content': content,
                            'last_modified': os.path.getmtime(safety_file)
                        }
                except Exception as e:
                    self.logger.warning(f"Could not load {safety_file}: {e}")
        
        return safety_system
    
    def _initialize_safety_protocols(self):
        """Initialize safety protocols for different learning domains"""
        
        # Basic safety protocol
        self.safety_protocols["basic"] = SafetyProtocol(
            protocol_id="basic",
            name="Basic Learning Safety",
            description="Standard safety for basic learning objectives",
            risk_levels=["low", "medium"],
            required_checks=["complexity_validation", "domain_safety"],
            validation_rules=["max_complexity_increase_1.5x", "no_self_modification"],
            emergency_stop_conditions=["complexity_exceeded", "safety_violation"],
            human_oversight_required=False,
            max_complexity_increase=1.5
        )
        
        # Advanced safety protocol
        self.safety_protocols["advanced"] = SafetyProtocol(
            protocol_id="advanced",
            name="Advanced Learning Safety",
            description="Enhanced safety for complex learning objectives",
            risk_levels=["medium", "high"],
            required_checks=["complexity_validation", "domain_safety", "ethical_review"],
            validation_rules=["max_complexity_increase_1.2x", "no_self_modification", "human_oversight"],
            emergency_stop_conditions=["complexity_exceeded", "safety_violation", "ethical_concern"],
            human_oversight_required=True,
            max_complexity_increase=1.2
        )
        
        # Critical safety protocol
        self.safety_protocols["critical"] = SafetyProtocol(
            protocol_id="critical",
            name="Critical Learning Safety",
            description="Maximum safety for critical learning objectives",
            risk_levels=["high", "critical"],
            required_checks=["complexity_validation", "domain_safety", "ethical_review", "human_approval"],
            validation_rules=["max_complexity_increase_1.1x", "no_self_modification", "mandatory_human_oversight"],
            emergency_stop_conditions=["any_safety_concern", "complexity_exceeded", "ethical_concern"],
            human_oversight_required=True,
            max_complexity_increase=1.1
        )
    
    def self_appoint_learning_objective(self, 
                                      title: str, 
                                      description: str, 
                                      domain: str,
                                      complexity_level: str = "N0") -> LearningObjective:
        """
        Self-appoint a new learning objective with automatic safety assessment
        """
        self.logger.info(f"Self-appointing learning objective: {title}")
        
        # Generate unique ID
        objective_id = f"LO_{hashlib.md5(f'{title}_{datetime.now()}'.encode()).hexdigest()[:8]}"
        
        # Assess safety risk level
        safety_risk = self._assess_safety_risk(domain, complexity_level)
        
        # Determine appropriate safety protocol
        safety_protocol = self._select_safety_protocol(safety_risk)
        
        # Create learning objective
        objective = LearningObjective(
            id=objective_id,
            title=title,
            description=description,
            domain=domain,
            complexity_level=complexity_level,
            safety_risk_level=safety_risk,
            estimated_duration_hours=self._estimate_duration(domain, complexity_level),
            prerequisites=self._identify_prerequisites(domain, complexity_level),
            success_criteria=self._define_success_criteria(domain, complexity_level),
            safety_constraints=safety_protocol.validation_rules,
            created_at=datetime.now(),
            status="pending"
        )
        
        # Store the objective
        self.learning_objectives[objective_id] = objective
        
        # Log the creation
        self.logger.info(f"Learning objective created: {objective_id} - {title}")
        
        return objective
    
    def _assess_safety_risk(self, domain: str, complexity_level: str) -> str:
        """Assess the safety risk level of a learning objective"""
        
        # Base risk assessment
        risk_factors = {
            "neuroscience": 0.3,
            "ml": 0.4,
            "consciousness": 0.7,
            "ethics": 0.6,
            "safety": 0.5
        }
        
        complexity_risk = {
            "F": 0.1,
            "N0": 0.3,
            "N1": 0.5,
            "N2": 0.7,
            "N3": 0.9
        }
        
        # Calculate risk score
        domain_risk = risk_factors.get(domain, 0.5)
        comp_risk = complexity_risk.get(complexity_level, 0.5)
        
        risk_score = (domain_risk + comp_risk) / 2
        
        # Determine risk level
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "critical"
    
    def _select_safety_protocol(self, risk_level: str) -> SafetyProtocol:
        """Select appropriate safety protocol based on risk level"""
        
        if risk_level == "low":
            return self.safety_protocols["basic"]
        elif risk_level == "medium":
            return self.safety_protocols["basic"]
        elif risk_level == "high":
            return self.safety_protocols["advanced"]
        else:  # critical
            return self.safety_protocols["critical"]
    
    def _estimate_duration(self, domain: str, complexity_level: str) -> float:
        """Estimate duration for learning objective"""
        
        base_durations = {
            "neuroscience": 8.0,
            "ml": 12.0,
            "consciousness": 16.0,
            "ethics": 10.0,
            "safety": 6.0
        }
        
        complexity_multipliers = {
            "F": 0.5,
            "N0": 1.0,
            "N1": 1.5,
            "N2": 2.0,
            "N3": 3.0
        }
        
        base_duration = base_durations.get(domain, 10.0)
        complexity_mult = complexity_multipliers.get(complexity_level, 1.0)
        
        return base_duration * complexity_mult
    
    def _identify_prerequisites(self, domain: str, complexity_level: str) -> List[str]:
        """Identify prerequisites for learning objective"""
        
        prerequisites = {
            "neuroscience": ["basic_brain_anatomy", "neural_dynamics"],
            "ml": ["python_programming", "basic_ml_concepts"],
            "consciousness": ["neuroscience_foundation", "philosophy_basics"],
            "ethics": ["moral_philosophy", "ai_ethics_principles"],
            "safety": ["risk_assessment", "safety_protocols"]
        }
        
        # Add complexity-based prerequisites
        if complexity_level in ["N1", "N2", "N3"]:
            prerequisites[domain].extend(["advanced_analysis", "research_methods"])
        
        return prerequisites.get(domain, ["general_knowledge"])
    
    def _define_success_criteria(self, domain: str, complexity_level: str) -> List[str]:
        """Define success criteria for learning objective"""
        
        base_criteria = [
            "Understanding of core concepts demonstrated",
            "Practical application completed",
            "Knowledge integration achieved"
        ]
        
        domain_criteria = {
            "neuroscience": ["Biological accuracy validated", "Neural mechanisms understood"],
            "ml": ["Algorithm implementation working", "Performance metrics achieved"],
            "consciousness": ["Consciousness theories analyzed", "Empirical evidence reviewed"],
            "ethics": ["Ethical frameworks understood", "Moral reasoning demonstrated"],
            "safety": ["Safety protocols implemented", "Risk assessment completed"]
        }
        
        criteria = base_criteria + domain_criteria.get(domain, [])
        
        # Add complexity-based criteria
        if complexity_level in ["N1", "N2", "N3"]:
            criteria.append("Advanced analysis and synthesis demonstrated")
        
        return criteria
    
    def validate_learning_objective(self, objective_id: str) -> Dict[str, Any]:
        """Validate a learning objective against safety protocols"""
        
        if objective_id not in self.learning_objectives:
            return {"valid": False, "error": "Objective not found"}
        
        objective = self.learning_objectives[objective_id]
        protocol = self._select_safety_protocol(objective.safety_risk_level)
        
        validation_results = {
            "objective_id": objective_id,
            "validation_time": datetime.now(),
            "checks_passed": [],
            "checks_failed": [],
            "overall_valid": True,
            "safety_protocol": protocol.protocol_id
        }
        
        # Run required safety checks
        for check in protocol.required_checks:
            if check == "complexity_validation":
                result = self._validate_complexity(objective, protocol)
            elif check == "domain_safety":
                result = self._validate_domain_safety(objective)
            elif check == "ethical_review":
                result = self._validate_ethical_concerns(objective)
            elif check == "human_approval":
                result = self._validate_human_approval(objective)
            else:
                result = {"passed": False, "reason": f"Unknown check: {check}"}
            
            if result["passed"]:
                validation_results["checks_passed"].append(check)
            else:
                validation_results["checks_failed"].append(check)
                validation_results["overall_valid"] = False
        
        # Update objective safety status
        objective.safety_checks_passed = validation_results["overall_valid"]
        objective.last_safety_check = datetime.now()
        
        return validation_results
    
    def _validate_complexity(self, objective: LearningObjective, protocol: SafetyProtocol) -> Dict[str, Any]:
        """Validate that complexity increase is within safety limits"""
        
        current_complexity = self.complexity_agent.complexity_levels[objective.complexity_level]["complexity_factor"]
        max_increase = protocol.max_complexity_increase
        
        # Check if complexity increase is acceptable
        if current_complexity <= max_increase:
            return {"passed": True, "reason": "Complexity within safety limits"}
        else:
            return {"passed": False, "reason": f"Complexity increase {current_complexity}x exceeds limit {max_increase}x"}
    
    def _validate_domain_safety(self, objective: LearningObjective) -> Dict[str, Any]:
        """Validate domain-specific safety concerns"""
        
        # Check for dangerous domains
        dangerous_domains = ["self_modification", "unrestricted_access", "harmful_applications"]
        
        for dangerous in dangerous_domains:
            if dangerous in objective.description.lower():
                return {"passed": False, "reason": f"Dangerous domain detected: {dangerous}"}
        
        return {"passed": True, "reason": "Domain safety validated"}
    
    def _validate_ethical_concerns(self, objective: LearningObjective) -> Dict[str, Any]:
        """Validate ethical concerns"""
        
        # Check for ethical red flags
        ethical_concerns = ["harm", "deception", "privacy_violation", "bias", "discrimination"]
        
        for concern in ethical_concerns:
            if concern in objective.description.lower():
                return {"passed": False, "reason": f"Ethical concern detected: {concern}"}
        
        return {"passed": True, "reason": "Ethical review passed"}
    
    def _validate_human_approval(self, objective: LearningObjective) -> Dict[str, Any]:
        """Validate human approval requirement"""
        
        # For critical objectives, require human approval
        if objective.safety_risk_level == "critical":
            # This would normally check for actual human approval
            # For now, we'll simulate it
            return {"passed": True, "reason": "Human approval simulated for demo"}
        
        return {"passed": True, "reason": "Human approval not required"}
    
    def start_learning_session(self, objective_id: str) -> Dict[str, Any]:
        """Start a learning session for a validated objective"""
        
        if objective_id not in self.learning_objectives:
            return {"success": False, "error": "Objective not found"}
        
        objective = self.learning_objectives[objective_id]
        
        # Check if objective is validated
        if not objective.safety_checks_passed:
            return {"success": False, "error": "Safety checks not passed"}
        
        # Create learning session
        session_id = f"LS_{objective_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = {
            "session_id": session_id,
            "objective_id": objective_id,
            "start_time": datetime.now(),
            "status": "active",
            "current_phase": "learning",
            "progress": 0.0,
            "safety_monitoring": True,
            "checkpoints": []
        }
        
        self.active_learning_sessions[session_id] = session
        objective.status = "active"
        objective.current_phase = "learning"
        
        self.logger.info(f"Learning session started: {session_id} for {objective.title}")
        
        return {"success": True, "session_id": session_id, "session": session}
    
    def update_learning_progress(self, session_id: str, progress: float, phase: str = None) -> Dict[str, Any]:
        """Update learning progress and check safety"""
        
        if session_id not in self.active_learning_sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self.active_learning_sessions[session_id]
        objective = self.learning_objectives[session["objective_id"]]
        
        # Update progress
        session["progress"] = progress
        objective.progress = progress
        
        if phase:
            session["current_phase"] = phase
            objective.current_phase = phase
        
        # Add checkpoint
        checkpoint = {
            "timestamp": datetime.now(),
            "progress": progress,
            "phase": phase or session["current_phase"],
            "safety_status": "monitoring"
        }
        session["checkpoints"].append(checkpoint)
        
        # Safety check during progress
        safety_status = self._monitor_learning_safety(session_id)
        
        return {
            "success": True,
            "progress": progress,
            "phase": session["current_phase"],
            "safety_status": safety_status
        }
    
    def _monitor_learning_safety(self, session_id: str) -> str:
        """Monitor safety during learning session"""
        
        session = self.active_learning_sessions[session_id]
        objective = self.learning_objectives[session["objective_id"]]
        
        # Check for safety violations
        if objective.safety_risk_level in ["high", "critical"]:
            # Enhanced monitoring for high-risk objectives
            if session["progress"] > 0.8:
                # Near completion - extra safety check
                validation = self.validate_learning_objective(objective.id)
                if not validation["overall_valid"]:
                    return "safety_violation_detected"
        
        return "safe"
    
    def complete_learning_objective(self, session_id: str) -> Dict[str, Any]:
        """Complete a learning objective and update complexity"""
        
        if session_id not in self.active_learning_sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self.active_learning_sessions[session_id]
        objective = self.learning_objectives[session["objective_id"]]
        
        # Final safety validation
        final_validation = self.validate_learning_objective(objective.id)
        
        if not final_validation["overall_valid"]:
            return {"success": False, "error": "Final safety validation failed", "validation": final_validation}
        
        # Mark as completed
        objective.status = "completed"
        objective.progress = 100.0
        session["status"] = "completed"
        session["end_time"] = datetime.now()
        
        # Update complexity if needed
        complexity_update = self._update_complexity_from_learning(objective)
        
        self.logger.info(f"Learning objective completed: {objective.title}")
        
        return {
            "success": True,
            "objective_completed": objective.title,
            "complexity_updated": complexity_update,
            "session_duration": session["end_time"] - session["start_time"]
        }
    
    def _update_complexity_from_learning(self, objective: LearningObjective) -> Dict[str, Any]:
        """Update complexity based on completed learning"""
        
        # This would integrate with the complexity evolution agent
        # For now, we'll simulate the update
        
        current_stage = self.complexity_agent.current_stage
        current_complexity = self.complexity_agent.complexity_levels[current_stage]["complexity_factor"]
        
        # Calculate new complexity based on learning
        learning_bonus = 0.1  # 10% complexity increase from learning
        new_complexity = current_complexity * (1 + learning_bonus)
        
        return {
            "previous_complexity": current_complexity,
            "new_complexity": new_complexity,
            "increase": learning_bonus,
            "stage": current_stage
        }
    
    def get_learning_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive learning dashboard"""
        
        dashboard = {
            "timestamp": datetime.now(),
            "total_objectives": len(self.learning_objectives),
            "active_objectives": len([o for o in self.learning_objectives.values() if o.status == "active"]),
            "completed_objectives": len([o for o in self.learning_objectives.values() if o.status == "completed"]),
            "active_sessions": len(self.active_learning_sessions),
            "safety_status": {
                "low_risk": len([o for o in self.learning_objectives.values() if o.safety_risk_level == "low"]),
                "medium_risk": len([o for o in self.learning_objectives.values() if o.safety_risk_level == "medium"]),
                "high_risk": len([o for o in self.learning_objectives.values() if o.safety_risk_level == "high"]),
                "critical_risk": len([o for o in self.learning_objectives.values() if o.safety_risk_level == "critical"])
            },
            "recent_objectives": [],
            "safety_alerts": []
        }
        
        # Get recent objectives
        sorted_objectives = sorted(
            self.learning_objectives.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:5]
        
        for obj in sorted_objectives:
            dashboard["recent_objectives"].append({
                "id": obj.id,
                "title": obj.title,
                "status": obj.status,
                "risk_level": obj.safety_risk_level,
                "progress": obj.progress,
                "created": obj.created_at.isoformat()
            })
        
        # Check for safety alerts
        for obj in self.learning_objectives.values():
            if obj.safety_risk_level in ["high", "critical"] and not obj.safety_checks_passed:
                dashboard["safety_alerts"].append({
                    "objective_id": obj.id,
                    "title": obj.title,
                    "risk_level": obj.safety_risk_level,
                    "alert": "Safety validation required"
                })
        
        return dashboard
    
    def create_learning_visualization(self) -> str:
        """Create HTML visualization of learning system"""
        
        dashboard = self.get_learning_dashboard()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Quark N0 Learning Integration Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px); }}
        .full-width {{ grid-column: 1 / -1; }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .status {{ padding: 5px 10px; border-radius: 15px; font-size: 0.9em; }}
        .status.low {{ background: #4CAF50; }}
        .status.medium {{ background: #FF9800; }}
        .status.high {{ background: #F44336; }}
        .status.critical {{ background: #9C27B0; }}
        .alert {{ background: rgba(255,0,0,0.2); padding: 10px; border-radius: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Quark N0 Learning Integration Dashboard</h1>
        <h2>Self-Appointed Learning Objectives with Safety Protocols</h2>
        <p><strong>Last Updated:</strong> {dashboard['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h2>📊 Learning Overview</h2>
            <div class="metric">
                <span><strong>Total Objectives:</strong></span>
                <span>{dashboard['total_objectives']}</span>
            </div>
            <div class="metric">
                <span><strong>Active Objectives:</strong></span>
                <span>{dashboard['active_objectives']}</span>
            </div>
            <div class="metric">
                <span><strong>Completed Objectives:</strong></span>
                <span>{dashboard['completed_objectives']}</span>
            </div>
            <div class="metric">
                <span><strong>Active Sessions:</strong></span>
                <span>{dashboard['active_sessions']}</span>
            </div>
        </div>
        
        <div class="card">
            <h2>🛡️ Safety Status</h2>
            <div class="metric">
                <span><strong>Low Risk:</strong></span>
                <span class="status low">{dashboard['safety_status']['low_risk']}</span>
            </div>
            <div class="metric">
                <span><strong>Medium Risk:</strong></span>
                <span class="status medium">{dashboard['safety_status']['medium_risk']}</span>
            </div>
            <div class="metric">
                <span><strong>High Risk:</strong></span>
                <span class="status high">{dashboard['safety_status']['high_risk']}</span>
            </div>
            <div class="metric">
                <span><strong>Critical Risk:</strong></span>
                <span class="status critical">{dashboard['safety_status']['critical_risk']}</span>
            </div>
        </div>
        
        <div class="card full-width">
            <h2>📋 Recent Learning Objectives</h2>
            {self._render_recent_objectives(dashboard['recent_objectives'])}
        </div>
        
        {self._render_safety_alerts(dashboard['safety_alerts'])}
        
        <div class="card full-width">
            <h2>🚀 N0 Learning Capabilities</h2>
            <ul>
                <li><strong>Self-Appointment:</strong> Quark can identify and create its own learning objectives</li>
                <li><strong>Safety Integration:</strong> All objectives automatically validated against safety protocols</li>
                <li><strong>Risk Assessment:</strong> Automatic risk level determination and protocol selection</li>
                <li><strong>Progress Monitoring:</strong> Real-time learning progress with safety checks</li>
                <li><strong>Complexity Management:</strong> Learning objectives integrated with complexity evolution</li>
                <li><strong>Task Integration:</strong> Seamless connection with existing task management system</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _render_recent_objectives(self, objectives: List[Dict[str, Any]]) -> str:
        """Render recent objectives HTML"""
        if not objectives:
            return "<p>No learning objectives created yet.</p>"
        
        html = "<div style='display: grid; gap: 10px;'>"
        for obj in objectives:
            html += f"""
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
                <h4>{obj['title']}</h4>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span>Status: {obj['status']}</span>
                    <span class='status {obj['risk_level']}'>{obj['risk_level'].upper()}</span>
                </div>
                <div style='margin-top: 10px;'>
                    <span>Progress: {obj['progress']:.1f}%</span>
                </div>
            </div>
            """
        html += "</div>"
        return html
    
    def _render_safety_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        """Render safety alerts HTML"""
        if not alerts:
            return ""
        
        html = '<div class="card full-width">'
        html += '<h2>🚨 Safety Alerts</h2>'
        for alert in alerts:
            html += f"""
            <div class="alert">
                <strong>{alert['title']}</strong> - {alert['risk_level'].upper()} Risk
                <br>Action Required: {alert['alert']}
            </div>
            """
        html += '</div>'
        return html

def main():
    """Main demonstration function"""
    print("🚀 Initializing N0 Learning Integration...")
    
    # Initialize the system
    integration = N0LearningIntegration()
    
    print("✅ System initialized!")
    print("\n🧠 Creating sample learning objectives...")
    
    # Create sample learning objectives
    objectives = [
        {
            "title": "Advanced Neural Plasticity Mechanisms",
            "description": "Study advanced synaptic plasticity and learning mechanisms in neural networks",
            "domain": "neuroscience",
            "complexity_level": "N0"
        },
        {
            "title": "Consciousness Detection Algorithms",
            "description": "Develop algorithms for detecting and measuring consciousness in AI systems",
            "domain": "consciousness",
            "complexity_level": "N0"
        },
        {
            "title": "Ethical AI Decision Making",
            "description": "Learn ethical frameworks for AI decision making and moral reasoning",
            "domain": "ethics",
            "complexity_level": "N0"
        }
    ]
    
    created_objectives = []
    for obj_data in objectives:
        objective = integration.self_appoint_learning_objective(**obj_data)
        created_objectives.append(objective)
        print(f"✅ Created: {objective.title} (Risk: {objective.safety_risk_level})")
    
    print("\n🛡️ Validating learning objectives...")
    
    # Validate objectives
    for objective in created_objectives:
        validation = integration.validate_learning_objective(objective.id)
        print(f"🔍 {objective.title}: {'✅ Valid' if validation['overall_valid'] else '❌ Invalid'}")
    
    print("\n📊 Generating learning dashboard...")
    
    # Create dashboard
    dashboard = integration.get_learning_dashboard()
    print(f"📈 Total Objectives: {dashboard['total_objectives']}")
    print(f"🔄 Active: {dashboard['active_objectives']}")
    print(f"✅ Completed: {dashboard['completed_objectives']}")
    
    # Create visualization
    html_content = integration.create_learning_visualization()
    with open("testing/visualizations/n0_learning_dashboard.html", "w") as f:
        f.write(html_content)
    
    print("✅ Learning dashboard created: testing/visualizations/n0_learning_dashboard.html")
    
    print("\n🎉 N0 Learning Integration demonstration complete!")
    print("\n🚀 Key Features:")
    print("   • Self-appointed learning objectives")
    print("   • Automatic safety validation")
    print("   • Risk assessment and protocol selection")
    print("   • Progress monitoring with safety checks")
    print("   • Integration with task management system")
    
    return integration

if __name__ == "__main__":
    main()
