#!/usr/bin/env python3
"""
🧠 Dual Validation Runner
Core rule for coordinating dual validation between Claude and DeepSeek

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Coordinate dual validation between Claude (functional) and DeepSeek (biological)
**Validation Level:** Comprehensive validation with both models
**Rule ID:** validation.dual.coordination
"""

import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import validation frameworks
from claude_validation_framework import ClaudeValidator
from deepseek_validation_framework import DeepSeekValidator

class DualValidationRunner:
    """Coordinates dual validation between Claude and DeepSeek models"""
    
    def __init__(self):
        self.claude_validator = ClaudeValidator()
        self.deepseek_validator = DeepSeekValidator()
        self.validation_results = {}
    
    def run_dual_validation(self, brain_config: Dict[str, Any], simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run both Claude and DeepSeek validation"""
        print("🧠 DUAL VALIDATION PROTOCOL")
        print("=" * 60)
        print("Following multi-model validation protocol requirements")
        print("Claude: Functional implementation and testing")
        print("DeepSeek: Biological accuracy and neuroscience validation")
        print("=" * 60)
        
        # Run Claude validation
        claude_results = self.claude_validator.run_comprehensive_validation(brain_config)
        
        # Run DeepSeek validation
        deepseek_results = self.deepseek_validator.run_comprehensive_validation(simulation_data)
        
        # Combined results
        combined_results = {
            "claude": claude_results,
            "deepseek": deepseek_results,
            "overall_status": self._determine_overall_status(claude_results, deepseek_results),
            "recommendations": self._generate_recommendations(claude_results, deepseek_results)
        }
        
        # Print final results
        print("\n🎯 DUAL VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Claude (Functional): {claude_results['status']} ({claude_results['pass_rate']:.1f}%)")
        print(f"DeepSeek (Biological): {deepseek_results['status']} ({deepseek_results['pass_rate']:.1f}%)")
        print(f"Overall Status: {combined_results['overall_status']}")
        
        if combined_results['recommendations']:
            print("\n📋 RECOMMENDATIONS:")
            for rec in combined_results['recommendations']:
                print(f"  • {rec}")
        
        return combined_results
    
    def _determine_overall_status(self, claude_results: Dict[str, Any], deepseek_results: Dict[str, Any]) -> str:
        """Determine overall validation status"""
        claude_pass = claude_results['status'] == 'PASS'
        deepseek_pass = deepseek_results['status'] == 'PASS'
        
        if claude_pass and deepseek_pass:
            return "EXCELLENT"
        elif claude_pass or deepseek_pass:
            return "GOOD"
        else:
            return "POOR"
    
    def _generate_recommendations(self, claude_results: Dict[str, Any], deepseek_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Claude recommendations
        if claude_results['pass_rate'] < 80:
            recommendations.append("Improve functional test coverage and fix failing tests")
        
        # DeepSeek recommendations
        if deepseek_results['pass_rate'] < 60:
            recommendations.append("Enhance biological accuracy based on neuroscience benchmarks")
        
        # Add specific improvements from DeepSeek
        if deepseek_results.get('improvements'):
            recommendations.extend(deepseek_results['improvements'])
        
        return recommendations
    
    def save_validation_results(self, results: Dict[str, Any], output_file: str = "validation_results.json"):
        """Save validation results to file"""
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        report = "🧠 DUAL VALIDATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Claude Results
        claude = results["claude"]
        report += f"🔧 CLAUDE VALIDATION (Functional)\n"
        report += f"Status: {claude['status']}\n"
        report += f"Pass Rate: {claude['pass_rate']:.1f}%\n"
        report += f"Tests: {claude['passed_tests']}/{claude['total_tests']}\n\n"
        
        # DeepSeek Results
        deepseek = results["deepseek"]
        report += f"🧬 DEEPSEEK VALIDATION (Biological)\n"
        report += f"Status: {deepseek['status']}\n"
        report += f"Pass Rate: {deepseek['pass_rate']:.1f}%\n"
        report += f"Tests: {deepseek['passed_tests']}/{deepseek['total_tests']}\n\n"
        
        # Overall Status
        report += f"🎯 OVERALL STATUS: {results['overall_status']}\n\n"
        
        # Recommendations
        if results['recommendations']:
            report += "📋 RECOMMENDATIONS:\n"
            for rec in results['recommendations']:
                report += f"  • {rec}\n"
        
        return report

def main():
    """Main function to run dual validation"""
    # Sample brain configuration
    brain_config = {
        "modules": {
            "architecture_agent": {"type": "architecture_agent"},
            "pfc": {"type": "pfc"},
            "basal_ganglia": {"type": "basal_ganglia"},
            "thalamus": {"type": "thalamus"},
            "working_memory": {"type": "working_memory", "slots": 3},
            "dmn": {"type": "dmn"},
            "salience": {"type": "salience"},
            "attention": {"type": "attention"}
        },
        "curriculum": {
            "ticks_per_week": 50,
            "schedule": [
                {"week": 0, "wm_slots": 3, "moe_k": 1},
                {"week": 4, "wm_slots": 4, "moe_k": 2}
            ]
        }
    }
    
    # Sample simulation data for biological validation
    simulation_data = {
        "working_memory": {"slots": 4},
        "modulators": {"DA": 0.4, "ACh": 0.6, "NE": 0.5, "5HT": 0.5},
        "sleep": {"cycle_duration_minutes": 90},
        "attention": {"switching_time_ms": 300},
        "development": {"synaptic_pruning_rate": 0.05}
    }
    
    # Run dual validation
    runner = DualValidationRunner()
    results = runner.run_dual_validation(brain_config, simulation_data)
    
    # Save results
    runner.save_validation_results(results)
    
    # Generate report
    report = runner.generate_validation_report(results)
    print("\n" + report)
    
    print(f"Overall Status: {results['overall_status']}")

if __name__ == "__main__":
    main()
