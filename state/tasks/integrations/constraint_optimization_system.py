#!/usr/bin/env python3
"""
🧬 Constraint Optimization System

This module provides intelligent optimization of biological constraint parameters
based on performance data, brain state analysis, and learning from task execution patterns.
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

class ConstraintOptimizer:
    """Optimizes biological constraint parameters based on performance data"""
    
    def __init__(self, config_path: str = "constraint_config.json"):
        self.config_path = Path(config_path)
        self.optimization_active = False
        self.optimizer_thread = None
        
        # Default biological constraints
        self.default_constraints = {
            "max_cognitive_load": 0.8,
            "min_working_memory": 0.3,
            "min_energy_level": 0.4,
            "max_concurrent_tasks": 3,
            "task_switching_cost": 0.1,
            "stress_threshold": 0.7,
            "recovery_rate": 0.05,
            "learning_rate": 0.01
        }
        
        # Current optimized constraints
        self.current_constraints = self.default_constraints.copy()
        
        # Performance history for optimization
        self.performance_history = {
            "task_completion_rates": [],
            "constraint_violations": [],
            "resource_efficiency": [],
            "brain_health_scores": [],
            "execution_times": []
        }
        
        # Optimization parameters
        self.optimization_params = {
            "learning_rate": 0.01,
            "exploration_rate": 0.1,
            "optimization_interval": 3600,  # 1 hour
            "min_data_points": 10,
            "optimization_window": 24 * 3600,  # 24 hours
            "constraint_bounds": {
                "max_cognitive_load": (0.6, 0.9),
                "min_working_memory": (0.2, 0.5),
                "min_energy_level": (0.3, 0.6),
                "max_concurrent_tasks": (2, 5),
                "task_switching_cost": (0.05, 0.2),
                "stress_threshold": (0.5, 0.8),
                "recovery_rate": (0.02, 0.1),
                "learning_rate": (0.005, 0.02)
            }
        }
        
        # Load existing configuration if available
        self._load_constraint_config()
        
        logger.info("🧬 Constraint Optimizer initialized")
    
    def _load_constraint_config(self):
        """Load existing constraint configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Update current constraints with loaded values
                for key, value in config.get("constraints", {}).items():
                    if key in self.current_constraints:
                        self.current_constraints[key] = value
                
                # Load performance history if available
                if "performance_history" in config:
                    self.performance_history.update(config["performance_history"])
                
                logger.info("📁 Constraint configuration loaded")
            else:
                logger.info("📁 No existing configuration found, using defaults")
                
        except Exception as e:
            logger.error(f"Error loading constraint config: {e}")
    
    def _save_constraint_config(self):
        """Save current constraint configuration"""
        try:
            config = {
                "constraints": self.current_constraints,
                "performance_history": self.performance_history,
                "last_updated": datetime.now().isoformat(),
                "optimization_params": self.optimization_params
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("💾 Constraint configuration saved")
            return True
            
        except Exception as e:
            logger.error(f"Error saving constraint config: {e}")
            return False
    
    def add_performance_data(self, data: Dict[str, Any]):
        """Add new performance data for optimization"""
        timestamp = datetime.now().isoformat()
        
        # Add timestamp to data
        data["timestamp"] = timestamp
        
        # Store in appropriate history lists
        for metric, value in data.items():
            if metric in self.performance_history and metric != "timestamp":
                self.performance_history[metric].append({
                    "timestamp": timestamp,
                    "value": value
                })
                
                # Keep only recent data within optimization window
                cutoff_time = datetime.now() - timedelta(seconds=self.optimization_params["optimization_window"])
                self.performance_history[metric] = [
                    entry for entry in self.performance_history[metric]
                    if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
                ]
        
        logger.info(f"📊 Performance data added: {list(data.keys())}")
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score from historical data"""
        if not self.performance_history["task_completion_rates"]:
            return 0.5  # Default score if no data
        
        try:
            # Calculate weighted average of recent performance metrics
            recent_data = {}
            cutoff_time = datetime.now() - timedelta(hours=6)  # Last 6 hours
            
            for metric, history in self.performance_history.items():
                recent_entries = [
                    entry for entry in history
                    if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
                ]
                
                if recent_entries:
                    values = [entry["value"] for entry in recent_entries]
                    recent_data[metric] = np.mean(values)
            
            if not recent_data:
                return 0.5
            
            # Weight different metrics
            weights = {
                "task_completion_rates": 0.3,
                "resource_efficiency": 0.25,
                "brain_health_scores": 0.25,
                "constraint_violations": 0.1,
                "execution_times": 0.1
            }
            
            # Calculate weighted score
            total_score = 0
            total_weight = 0
            
            for metric, weight in weights.items():
                if metric in recent_data:
                    # Normalize values to 0-1 range
                    if metric == "constraint_violations":
                        # Lower violations = higher score
                        normalized_value = max(0, 1 - recent_data[metric])
                    elif metric == "execution_times":
                        # Lower execution times = higher score (normalize to reasonable range)
                        normalized_value = max(0, min(1, 1 - (recent_data[metric] / 3600)))  # Normalize to 1 hour
                    else:
                        # Higher values = higher score
                        normalized_value = min(1, max(0, recent_data[metric]))
                    
                    total_score += normalized_value * weight
                    total_weight += weight
            
            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    def optimize_constraints(self) -> Dict[str, Any]:
        """Optimize constraint parameters based on performance data"""
        if len(self.performance_history["task_completion_rates"]) < self.optimization_params["min_data_points"]:
            logger.info("📊 Insufficient data for optimization")
            return {"status": "insufficient_data", "constraints": self.current_constraints}
        
        try:
            current_score = self.calculate_performance_score()
            logger.info(f"📊 Current performance score: {current_score:.3f}")
            
            # Store current constraints for comparison
            previous_constraints = self.current_constraints.copy()
            
            # Optimize each constraint parameter
            optimizations = {}
            
            for constraint_name, current_value in self.current_constraints.items():
                if constraint_name in self.optimization_params["constraint_bounds"]:
                    min_val, max_val = self.optimization_params["constraint_bounds"][constraint_name]
                    
                    # Calculate optimization direction based on performance
                    if current_score < 0.6:  # Poor performance
                        # More conservative constraints
                        if constraint_name.startswith("max_"):
                            new_value = current_value * 0.95  # Reduce maximums
                        elif constraint_name.startswith("min_"):
                            new_value = current_value * 1.05  # Increase minimums
                        else:
                            new_value = current_value * 0.98  # Slightly more conservative
                    elif current_score > 0.8:  # Good performance
                        # More aggressive constraints
                        if constraint_name.startswith("max_"):
                            new_value = current_value * 1.02  # Increase maximums
                        elif constraint_name.startswith("min_"):
                            new_value = current_value * 0.98  # Decrease minimums
                        else:
                            new_value = current_value * 1.01  # Slightly more aggressive
                    else:  # Moderate performance
                        # Small random adjustment for exploration
                        adjustment = np.random.uniform(-0.02, 0.02)
                        new_value = current_value * (1 + adjustment)
                    
                    # Apply bounds
                    new_value = max(min_val, min(max_val, new_value))
                    
                    # Apply learning rate
                    new_value = current_value + (new_value - current_value) * self.optimization_params["learning_rate"]
                    
                    # Round to appropriate precision
                    if constraint_name in ["max_cognitive_load", "min_working_memory", "min_energy_level", "stress_threshold", "recovery_rate", "learning_rate"]:
                        new_value = round(new_value, 3)
                    elif constraint_name in ["max_concurrent_tasks"]:
                        new_value = int(round(new_value))
                    else:
                        new_value = round(new_value, 2)
                    
                    # Update constraint
                    self.current_constraints[constraint_name] = new_value
                    
                    # Record optimization
                    optimizations[constraint_name] = {
                        "previous": current_value,
                        "new": new_value,
                        "change": new_value - current_value
                    }
            
            # Save updated configuration
            self._save_constraint_config()
            
            logger.info("🧬 Constraints optimized successfully")
            
            return {
                "status": "optimized",
                "previous_constraints": previous_constraints,
                "new_constraints": self.current_constraints,
                "optimizations": optimizations,
                "performance_score": current_score
            }
            
        except Exception as e:
            logger.error(f"Error optimizing constraints: {e}")
            return {"status": "error", "error": str(e), "constraints": self.current_constraints}
    
    def get_constraint_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for constraint adjustments"""
        if not self.performance_history["task_completion_rates"]:
            return {"status": "no_data", "recommendations": ["Collect more performance data"]}
        
        try:
            current_score = self.calculate_performance_score()
            recommendations = []
            
            # Analyze recent performance trends
            recent_completion_rates = [
                entry for entry in self.performance_history["task_completion_rates"]
                if datetime.fromisoformat(entry["timestamp"]) > datetime.now() - timedelta(hours=6)
            ]
            
            recent_violations = [
                entry for entry in self.performance_history["constraint_violations"]
                if datetime.fromisoformat(entry["timestamp"]) > datetime.now() - timedelta(hours=6)
            ]
            
            # Generate recommendations based on performance
            if current_score < 0.5:
                recommendations.append("Performance is poor - consider more conservative constraints")
                recommendations.append("Review recent constraint violations for patterns")
                recommendations.append("Increase minimum thresholds for safety")
            elif current_score < 0.7:
                recommendations.append("Performance is below optimal - consider constraint adjustments")
                recommendations.append("Monitor constraint violations more closely")
                recommendations.append("Consider reducing maximum thresholds")
            elif current_score > 0.9:
                recommendations.append("Performance is excellent - consider more aggressive constraints")
                recommendations.append("Current constraints may be too conservative")
                recommendations.append("Consider increasing maximum thresholds")
            
            # Specific recommendations based on violations
            if recent_violations:
                avg_violations = np.mean([entry["value"] for entry in recent_violations])
                if avg_violations > 0.1:  # More than 10% violation rate
                    recommendations.append("High constraint violation rate - review constraint settings")
                    recommendations.append("Consider increasing minimum thresholds")
            
            # Recommendations based on completion rates
            if recent_completion_rates:
                avg_completion = np.mean([entry["value"] for entry in recent_completion_rates])
                if avg_completion < 0.8:
                    recommendations.append("Low task completion rate - constraints may be too restrictive")
                    recommendations.append("Consider relaxing maximum thresholds")
            
            return {
                "status": "recommendations_generated",
                "performance_score": current_score,
                "recommendations": recommendations,
                "recent_metrics": {
                    "completion_rate": np.mean([entry["value"] for entry in recent_completion_rates]) if recent_completion_rates else 0,
                    "violation_rate": np.mean([entry["value"] for entry in recent_violations]) if recent_violations else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {"status": "error", "error": str(e)}
    
    def start_optimization_loop(self):
        """Start continuous optimization loop"""
        def optimization_loop():
            while self.optimization_active:
                try:
                    # Perform optimization
                    result = self.optimize_constraints()
                    
                    if result["status"] == "optimized":
                        logger.info("🧬 Periodic constraint optimization completed")
                        
                        # Log significant changes
                        for constraint, change_info in result["optimizations"].items():
                            if abs(change_info["change"]) > 0.01:  # Significant change
                                logger.info(f"📊 {constraint}: {change_info['previous']:.3f} → {change_info['new']:.3f}")
                    
                    # Sleep until next optimization
                    time.sleep(self.optimization_params["optimization_interval"])
                    
                except Exception as e:
                    logger.error(f"Error in optimization loop: {e}")
                    time.sleep(self.optimization_params["optimization_interval"])
        
        # Start optimization thread
        self.optimization_active = True
        self.optimizer_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimizer_thread.start()
        
        logger.info("🚀 Constraint optimization loop started")
    
    def stop_optimization_loop(self):
        """Stop optimization loop"""
        self.optimization_active = False
        if self.optimizer_thread:
            self.optimizer_thread.join(timeout=5.0)
        
        logger.info("🛑 Constraint optimization loop stopped")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            "optimization_active": self.optimization_active,
            "current_constraints": self.current_constraints,
            "default_constraints": self.default_constraints,
            "performance_score": self.calculate_performance_score(),
            "data_points": {
                metric: len(history) for metric, history in self.performance_history.items()
            },
            "last_optimization": self.config_path.stat().st_mtime if self.config_path.exists() else None,
            "optimization_params": self.optimization_params
        }
    
    def reset_to_defaults(self):
        """Reset constraints to default values"""
        self.current_constraints = self.default_constraints.copy()
        self._save_constraint_config()
        logger.info("🔄 Constraints reset to default values")
    
    def set_constraint(self, constraint_name: str, value: float):
        """Set a specific constraint value"""
        if constraint_name in self.current_constraints:
            # Validate bounds
            if constraint_name in self.optimization_params["constraint_bounds"]:
                min_val, max_val = self.optimization_params["constraint_bounds"][constraint_name]
                value = max(min_val, min(max_val, value))
            
            self.current_constraints[constraint_name] = value
            self._save_constraint_config()
            logger.info(f"📊 Constraint {constraint_name} set to {value}")
            return True
        else:
            logger.error(f"Unknown constraint: {constraint_name}")
            return False

def main():
    """Main function to demonstrate constraint optimization"""
    print("🧬 Constraint Optimization System")
    print("=" * 50)
    
    # Create optimizer
    optimizer = ConstraintOptimizer()
    
    # Add some sample performance data
    print("📊 Adding sample performance data...")
    sample_data = {
        "task_completion_rates": 0.85,
        "resource_efficiency": 0.78,
        "brain_health_scores": 0.92,
        "constraint_violations": 0.05,
        "execution_times": 1800  # 30 minutes
    }
    
    for _ in range(15):  # Add 15 data points
        optimizer.add_performance_data(sample_data)
        time.sleep(0.1)
    
    # Get initial status
    print("\n📊 Initial Optimization Status:")
    status = optimizer.get_optimization_status()
    print(f"   Performance Score: {status['performance_score']:.3f}")
    print(f"   Data Points: {status['data_points']}")
    
    # Perform optimization
    print("\n🧬 Performing constraint optimization...")
    result = optimizer.optimize_constraints()
    
    if result["status"] == "optimized":
        print("✅ Optimization completed successfully!")
        print("\n📊 Constraint Changes:")
        for constraint, change_info in result["optimizations"].items():
            print(f"   {constraint}: {change_info['previous']:.3f} → {change_info['new']:.3f}")
    
    # Get recommendations
    print("\n💡 Getting constraint recommendations...")
    recommendations = optimizer.get_constraint_recommendations()
    
    if recommendations["status"] == "recommendations_generated":
        print("✅ Recommendations generated!")
        print("\n📋 Recommendations:")
        for rec in recommendations["recommendations"]:
            print(f"   • {rec}")
    
    # Start optimization loop
    print("\n🚀 Starting continuous optimization loop...")
    optimizer.start_optimization_loop()
    
    try:
        # Let it run for a few cycles
        print("⏳ Running optimization for 2 minutes...")
        time.sleep(120)
        
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user")
    
    finally:
        # Stop optimization
        print("🛑 Stopping optimization loop...")
        optimizer.stop_optimization_loop()
        
        # Final status
        final_status = optimizer.get_optimization_status()
        print(f"\n📊 Final Status:")
        print(f"   Performance Score: {final_status['performance_score']:.3f}")
        print(f"   Current Constraints: {len(final_status['current_constraints'])} active")
        
        print("\n✅ Constraint optimization demonstration complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Constraint optimization failed: {e}")
        import traceback
        traceback.print_exc()
