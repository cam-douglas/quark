#!/usr/bin/env python3
"""
Execute Next Steps Script
Master script that runs all the next steps for the Brain-ML Synergy Architecture

This script executes:
1. Import path updates
2. Synergy pathway testing
3. Performance optimization
4. Final validation
"""

import os, sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List

class NextStepsExecutor:
    def __init__(self):
        self.execution_results = {
            'import_updates': {},
            'synergy_testing': {},
            'performance_optimization': {},
            'final_validation': {},
            'overall_status': 'PENDING'
        }
        
        self.scripts = {
            'import_updates': 'update_import_paths.py',
            'synergy_testing': 'test_synergy_pathways.py',
            'performance_optimization': 'optimize_synergy_performance.py'
        }
    
    def run_script(self, script_name: str, description: str) -> Dict[str, Any]:
        """Run a Python script and capture results"""
        print(f"\n🚀 Running: {description}")
        print(f"📁 Script: {script_name}")
        print("-" * 50)
        
        try:
            # Check if script exists
            if not os.path.exists(script_name):
                return {
                    'status': 'ERROR',
                    'message': f"Script not found: {script_name}",
                    'output': '',
                    'error': f"File {script_name} does not exist"
                }
            
            # Run the script
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            execution_time = time.time() - start_time
            
            # Process results
            if result.returncode == 0:
                status = 'SUCCESS'
                message = f"Script completed successfully in {execution_time:.2f}s"
            else:
                status = 'ERROR'
                message = f"Script failed with return code {result.returncode}"
            
            return {
                'status': status,
                'message': message,
                'output': result.stdout,
                'error': result.stderr,
                'execution_time': execution_time,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'TIMEOUT',
                'message': f"Script timed out after 5 minutes",
                'output': '',
                'error': 'Execution timeout',
                'execution_time': 300,
                'return_code': -1
            }
        except Exception as e:
            return {
                'status': 'EXCEPTION',
                'message': f"Exception occurred: {str(e)}",
                'output': '',
                'error': str(e),
                'execution_time': 0,
                'return_code': -1
            }
    
    def execute_import_updates(self) -> Dict[str, Any]:
        """Execute Phase 3: Import Path Updates"""
        print("\n" + "="*60)
        print("🔄 PHASE 3: IMPORT PATH UPDATES")
        print("="*60)
        
        result = self.run_script(
            self.scripts['import_updates'],
            "Updating Python import statements to match new architecture"
        )
        
        self.execution_results['import_updates'] = result
        return result
    
    def execute_synergy_testing(self) -> Dict[str, Any]:
        """Execute Phase 4: Synergy Pathway Testing"""
        print("\n" + "="*60)
        print("🧪 PHASE 4: SYNERGY PATHWAY TESTING")
        print("="*60)
        
        result = self.run_script(
            self.scripts['synergy_testing'],
            "Testing communication between biological and computational domains"
        )
        
        self.execution_results['synergy_testing'] = result
        return result
    
    def execute_performance_optimization(self) -> Dict[str, Any]:
        """Execute Phase 5: Performance Optimization"""
        print("\n" + "="*60)
        print("🚀 PHASE 5: PERFORMANCE OPTIMIZATION")
        print("="*60)
        
        result = self.run_script(
            self.scripts['performance_optimization'],
            "Optimizing synergy pathways for maximum performance"
        )
        
        self.execution_results['performance_optimization'] = result
        return result
    
    def run_final_validation(self) -> Dict[str, Any]:
        """Run final validation of the complete system"""
        print("\n" + "="*60)
        print("✅ FINAL VALIDATION")
        print("="*60)
        
        validation_results = {
            'architecture_integrity': self._validate_architecture_integrity(),
            'file_organization': self._validate_file_organization(),
            'synergy_pathways': self._validate_synergy_pathways(),
            'documentation_completeness': self._validate_documentation()
        }
        
        # Overall validation status
        all_passed = all(result['status'] == 'PASS' for result in validation_results.values())
        validation_results['overall_status'] = 'PASS' if all_passed else 'FAIL'
        
        self.execution_results['final_validation'] = validation_results
        return validation_results
    
    def _validate_architecture_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the new architecture"""
        print("🔍 Validating architecture integrity...")
        
        required_dirs = [
            '🧠_BRAIN_ARCHITECTURE',
            '🤖_ML_ARCHITECTURE',
            '🔄_INTEGRATION',
            '📊_DATA_KNOWLEDGE',
            '🛠️_DEVELOPMENT',
            '📋_MANAGEMENT',
            '🧪_TESTING',
            '📚_DOCUMENTATION'
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            return {
                'status': 'FAIL',
                'message': f"Missing required directories: {', '.join(missing_dirs)}",
                'details': {'missing_dirs': missing_dirs}
            }
        else:
            return {
                'status': 'PASS',
                'message': "All required architecture directories exist",
                'details': {'total_dirs': len(required_dirs)}
            }
    
    def _validate_file_organization(self) -> Dict[str, Any]:
        """Validate that files are properly organized"""
        print("📁 Validating file organization...")
        
        total_files = 0
        organized_files = 0
        
        # Count files in new architecture
        for root, dirs, files in os.walk('.'):
            # Skip hidden directories and the new architecture
            if any(skip in root for skip in ['.git', '.cursor', '__pycache__', '.pytest_cache']):
                continue
            
            # Skip if we're in the new architecture
            if any(arch_dir in root for arch_dir in ['🧠_BRAIN_ARCHITECTURE', '🤖_ML_ARCHITECTURE', '🔄_INTEGRATION', '📊_DATA_KNOWLEDGE', '🛠️_DEVELOPMENT', '📋_MANAGEMENT', '🧪_TESTING', '📚_DOCUMENTATION']):
                organized_files += len(files)
            else:
                total_files += len(files)
        
        # Calculate organization percentage
        if total_files + organized_files > 0:
            organization_percentage = (organized_files / (total_files + organized_files)) * 100
        else:
            organization_percentage = 0
        
        if organization_percentage >= 90:  # 90% threshold
            return {
                'status': 'PASS',
                'message': f"File organization is {organization_percentage:.1f}% complete",
                'details': {
                    'organized_files': organized_files,
                    'remaining_files': total_files,
                    'organization_percentage': organization_percentage
                }
            }
        else:
            return {
                'status': 'FAIL',
                'message': f"File organization is only {organization_percentage:.1f}% complete",
                'details': {
                    'organized_files': organized_files,
                    'remaining_files': total_files,
                    'organization_percentage': organization_percentage
                }
            }
    
    def _validate_synergy_pathways(self) -> Dict[str, Any]:
        """Validate that synergy pathways are working"""
        print("🔗 Validating synergy pathways...")
        
        # Check if synergy test report exists
        if os.path.exists('SYNERGY_TEST_REPORT.md'):
            return {
                'status': 'PASS',
                'message': "Synergy pathway testing completed",
                'details': {'test_report': 'SYNERGY_TEST_REPORT.md'}
            }
        else:
            return {
                'status': 'FAIL',
                'message': "Synergy pathway testing not completed",
                'details': {'missing_report': 'SYNERGY_TEST_REPORT.md'}
            }
    
    def _validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness"""
        print("📚 Validating documentation...")
        
        required_docs = [
            'README_BRAIN_ML_SYNERGY.md',
            'BRAIN_ML_ARCHITECTURE_INDEX.md',
            'MIGRATION_COMPLETION_STATUS.md',
            'ORGANIZATION_COMPLETE.md'
        ]
        
        missing_docs = []
        for doc in required_docs:
            if not os.path.exists(doc):
                missing_docs.append(doc)
        
        if missing_docs:
            return {
                'status': 'FAIL',
                'message': f"Missing required documentation: {', '.join(missing_docs)}",
                'details': {'missing_docs': missing_docs}
            }
        else:
            return {
                'status': 'PASS',
                'message': "All required documentation exists",
                'details': {'total_docs': len(required_docs)}
            }
    
    def execute_all_phases(self) -> Dict[str, Any]:
        """Execute all phases of the next steps"""
        print("🧠🤖 Brain-ML Synergy Architecture: Next Steps Execution")
        print("=" * 70)
        print("This script will execute all remaining phases to complete the implementation.")
        print("=" * 70)
        
        start_time = time.time()
        
        # Phase 3: Import Updates
        import_result = self.execute_import_updates()
        
        # Phase 4: Synergy Testing
        synergy_result = self.execute_synergy_testing()
        
        # Phase 5: Performance Optimization
        optimization_result = self.execute_performance_optimization()
        
        # Final Validation
        validation_result = self.run_final_validation()
        
        total_time = time.time() - start_time
        
        # Determine overall status
        all_phases_successful = (
            import_result['status'] == 'SUCCESS' and
            synergy_result['status'] == 'SUCCESS' and
            optimization_result['status'] == 'SUCCESS' and
            validation_result['overall_status'] == 'PASS'
        )
        
        self.execution_results['overall_status'] = 'SUCCESS' if all_phases_successful else 'PARTIAL_SUCCESS'
        
        # Generate execution summary
        summary = {
            'total_execution_time': total_time,
            'overall_status': self.execution_results['overall_status'],
            'phase_results': {
                'import_updates': import_result['status'],
                'synergy_testing': synergy_result['status'],
                'performance_optimization': optimization_result['status'],
                'final_validation': validation_result['overall_status']
            },
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on execution results"""
        recommendations = []
        
        if self.execution_results['import_updates']['status'] != 'SUCCESS':
            recommendations.append("Fix import update issues before proceeding")
        
        if self.execution_results['synergy_testing']['status'] != 'SUCCESS':
            recommendations.append("Resolve synergy pathway issues")
        
        if self.execution_results['performance_optimization']['status'] != 'SUCCESS':
            recommendations.append("Complete performance optimization")
        
        if self.execution_results['final_validation']['overall_status'] != 'PASS':
            recommendations.append("Address validation failures")
        
        if not recommendations:
            recommendations.append("All phases completed successfully - system is ready for production use!")
        
        return recommendations
    
    def generate_execution_report(self, summary: Dict[str, Any]) -> str:
        """Generate a comprehensive execution report"""
        report = "# 🚀 Next Steps Execution Report\n\n"
        
        # Executive Summary
        report += f"## 📊 Executive Summary\n\n"
        report += f"- **Overall Status**: {summary['overall_status']}\n"
        report += f"- **Total Execution Time**: {summary['total_execution_time']:.2f} seconds\n"
        report += f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Phase Results
        report += f"## 🔄 Phase Results\n\n"
        phase_results = summary['phase_results']
        report += f"- **Import Updates**: {phase_results['import_updates']}\n"
        report += f"- **Synergy Testing**: {phase_results['synergy_testing']}\n"
        report += f"- **Performance Optimization**: {phase_results['performance_optimization']}\n"
        report += f"- **Final Validation**: {phase_results['final_validation']}\n\n"
        
        # Detailed Results
        report += f"## 📋 Detailed Results\n\n"
        
        # Import Updates
        report += f"### 🔄 Import Updates\n"
        import_result = self.execution_results['import_updates']
        report += f"- **Status**: {import_result['status']}\n"
        report += f"- **Message**: {import_result['message']}\n"
        if import_result['error']:
            report += f"- **Error**: {import_result['error']}\n"
        report += "\n"
        
        # Synergy Testing
        report += f"### 🧪 Synergy Testing\n"
        synergy_result = self.execution_results['synergy_testing']
        report += f"- **Status**: {synergy_result['status']}\n"
        report += f"- **Message**: {synergy_result['message']}\n"
        if synergy_result['error']:
            report += f"- **Error**: {synergy_result['error']}\n"
        report += "\n"
        
        # Performance Optimization
        report += f"### 🚀 Performance Optimization\n"
        opt_result = self.execution_results['performance_optimization']
        report += f"- **Status**: {opt_result['status']}\n"
        report += f"- **Message**: {opt_result['message']}\n"
        if opt_result['error']:
            report += f"- **Error**: {opt_result['error']}\n"
        report += "\n"
        
        # Final Validation
        report += f"### ✅ Final Validation\n"
        val_result = self.execution_results['final_validation']
        report += f"- **Overall Status**: {val_result['overall_status']}\n"
        for test_name, test_result in val_result.items():
            if test_name != 'overall_status':
                report += f"- **{test_name.replace('_', ' ').title()}**: {test_result['status']}\n"
        report += "\n"
        
        # Recommendations
        report += f"## 💡 Recommendations\n\n"
        for i, recommendation in enumerate(summary['recommendations'], 1):
            report += f"{i}. {recommendation}\n"
        
        return report

def main():
    """Main execution function"""
    executor = NextStepsExecutor()
    
    try:
        # Execute all phases
        summary = executor.execute_all_phases()
        
        # Generate and save report
        report = executor.generate_execution_report(summary)
        
        with open('NEXT_STEPS_EXECUTION_REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print final summary
        print("\n" + "="*70)
        print("🎯 EXECUTION COMPLETE")
        print("="*70)
        print(f"📊 Overall Status: {summary['overall_status']}")
        print(f"⏱️ Total Time: {summary['total_execution_time']:.2f} seconds")
        print(f"📄 Report saved to: NEXT_STEPS_EXECUTION_REPORT.md")
        
        if summary['overall_status'] == 'SUCCESS':
            print("\n🎉 All phases completed successfully!")
            print("🚀 The Brain-ML Synergy Architecture is now fully operational!")
        else:
            print("\n⚠️ Some phases need attention. Check the report for details.")
        
        # Print recommendations
        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Execution failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
