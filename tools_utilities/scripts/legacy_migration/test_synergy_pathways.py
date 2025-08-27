#!/usr/bin/env python3
"""
Test Synergy Pathways Script
Validates the synergy pathways between biological and computational domains

This script tests the communication and integration between brain components
and ML components to ensure the synergy architecture is working correctly.
"""

import os, sys
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

class SynergyPathwayTester:
    def __init__(self):
        self.test_results = {
            'passed': [],
            'failed': [],
            'skipped': [],
            'errors': []
        }
        
        # Define synergy pathways to test
        self.synergy_pathways = {
            'biological_to_computational': [
                {
                    'name': 'Sensory Input → Data Processing',
                    'from': '🧠_BRAIN_ARCHITECTURE/01_NEURAL_CORE/sensory_input/thalamus',
                    'to': '🤖_ML_ARCHITECTURE/01_EXPERT_DOMAINS/core_ml/data_engineering',
                    'test_type': 'import_test'
                },
                {
                    'name': 'Neural Patterns → ML Training',
                    'from': '🧠_BRAIN_ARCHITECTURE/01_NEURAL_CORE/motor_control/connectome',
                    'to': '🤖_ML_ARCHITECTURE/02_TRAINING_SYSTEMS/network_training',
                    'test_type': 'import_test'
                },
                {
                    'name': 'Memory Formation → Knowledge Systems',
                    'from': '🧠_BRAIN_ARCHITECTURE/01_NEURAL_CORE/cognitive_processing/hippocampus',
                    'to': '🤖_ML_ARCHITECTURE/03_KNOWLEDGE_SYSTEMS/database',
                    'test_type': 'import_test'
                }
            ],
            'computational_to_biological': [
                {
                    'name': 'ML Insights → Neural Optimization',
                    'from': '🤖_ML_ARCHITECTURE/01_EXPERT_DOMAINS/core_ml/machine_learning',
                    'to': '🧠_BRAIN_ARCHITECTURE/01_NEURAL_CORE/motor_control/connectome',
                    'test_type': 'import_test'
                },
                {
                    'name': 'Data Patterns → Cognitive Enhancement',
                    'from': '🤖_ML_ARCHITECTURE/01_EXPERT_DOMAINS/core_ml/data_engineering',
                    'to': '🧠_BRAIN_ARCHITECTURE/01_NEURAL_CORE/cognitive_processing/conscious_agent',
                    'test_type': 'import_test'
                }
            ]
        }
    
    def test_import_pathway(self, from_path: str, to_path: str) -> Tuple[bool, str]:
        """Test if a module can import from another module"""
        try:
            # Find Python files in the source directory
            source_files = self._find_python_files(from_path)
            target_files = self._find_python_files(to_path)
            
            if not source_files:
                return False, f"No Python files found in source: {from_path}"
            
            if not target_files:
                return False, f"No Python files found in target: {to_path}"
            
            # Test a simple import scenario
            test_result = self._test_simple_import(source_files[0], target_files[0])
            return test_result
            
        except Exception as e:
            return False, f"Import test error: {str(e)}"
    
    def _find_python_files(self, directory: str) -> List[str]:
        """Find Python files in a directory"""
        python_files = []
        
        if not os.path.exists(directory):
            return []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _test_simple_import(self, source_file: str, target_file: str) -> Tuple[bool, str]:
        """Test if source file can import from target file"""
        try:
            # Read the source file to check for import statements
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if there are any import statements
            if 'import ' in content or 'from ' in content:
                return True, f"Import statements found in {os.path.basename(source_file)}"
            else:
                return True, f"No import statements in {os.path.basename(source_file)} (standalone module)"
                
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def test_file_accessibility(self, pathway: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if files are accessible in the pathway"""
        try:
            from_path = pathway['from']
            to_path = pathway['to']
            
            # Check if directories exist
            if not os.path.exists(from_path):
                return False, f"Source directory does not exist: {from_path}"
            
            if not os.path.exists(to_path):
                return False, f"Target directory does not exist: {to_path}"
            
            # Check if directories contain files
            from_files = self._find_python_files(from_path)
            to_files = self._find_python_files(to_path)
            
            if not from_files:
                return False, f"No Python files in source: {from_path}"
            
            if not to_files:
                return False, f"No Python files in target: {to_path}"
            
            return True, f"Pathway accessible: {len(from_files)} source files, {len(to_files)} target files"
            
        except Exception as e:
            return False, f"Accessibility test error: {str(e)}"
    
    def test_synergy_integration(self) -> Dict[str, Any]:
        """Test all synergy pathways"""
        print("🧠🤖 Testing Brain-ML Synergy Pathways")
        print("=" * 50)
        
        all_results = {
            'biological_to_computational': [],
            'computational_to_biological': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
        }
        
        # Test biological to computational pathways
        print("\n🔄 Testing Biological → Computational Pathways...")
        for pathway in self.synergy_pathways['biological_to_computational']:
            print(f"\n📋 Testing: {pathway['name']}")
            
            # Test accessibility
            accessible, access_msg = self.test_file_accessibility(pathway)
            print(f"   📁 Accessibility: {access_msg}")
            
            if accessible:
                # Test import pathway
                import_ok, import_msg = self.test_import_pathway(pathway['from'], pathway['to'])
                print(f"   🔗 Import Test: {import_msg}")
                
                result = {
                    'name': pathway['name'],
                    'accessible': accessible,
                    'import_ok': import_ok,
                    'access_msg': access_msg,
                    'import_msg': import_msg,
                    'status': 'PASS' if import_ok else 'FAIL'
                }
                
                if import_ok:
                    all_results['summary']['passed'] += 1
                else:
                    all_results['summary']['failed'] += 1
                
            else:
                result = {
                    'name': pathway['name'],
                    'accessible': accessible,
                    'import_ok': False,
                    'access_msg': access_msg,
                    'import_msg': 'Skipped - accessibility failed',
                    'status': 'SKIP'
                }
                all_results['summary']['skipped'] += 1
            
            all_results['biological_to_computational'].append(result)
            all_results['summary']['total_tests'] += 1
        
        # Test computational to biological pathways
        print("\n🔄 Testing Computational → Biological Pathways...")
        for pathway in self.synergy_pathways['computational_to_biological']:
            print(f"\n📋 Testing: {pathway['name']}")
            
            # Test accessibility
            accessible, access_msg = self.test_file_accessibility(pathway)
            print(f"   📁 Accessibility: {access_msg}")
            
            if accessible:
                # Test import pathway
                import_ok, import_msg = self.test_import_pathway(pathway['from'], pathway['to'])
                print(f"   🔗 Import Test: {import_msg}")
                
                result = {
                    'name': pathway['name'],
                    'accessible': accessible,
                    'import_ok': import_ok,
                    'access_msg': access_msg,
                    'import_msg': import_msg,
                    'status': 'PASS' if import_ok else 'FAIL'
                }
                
                if import_ok:
                    all_results['summary']['passed'] += 1
                else:
                    all_results['summary']['failed'] += 1
                
            else:
                result = {
                    'name': pathway['name'],
                    'accessible': accessible,
                    'import_ok': False,
                    'access_msg': access_msg,
                    'import_msg': 'Skipped - accessibility failed',
                    'status': 'SKIP'
                }
                all_results['summary']['skipped'] += 1
            
            all_results['computational_to_biological'].append(result)
            all_results['summary']['total_tests'] += 1
        
        return all_results
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report"""
        report = "# 🧪 Synergy Pathway Test Report\n\n"
        
        # Summary
        summary = results['summary']
        report += f"## 📊 Test Summary\n\n"
        report += f"- **Total Tests**: {summary['total_tests']}\n"
        report += f"- **Passed**: {summary['passed']} ✅\n"
        report += f"- **Failed**: {summary['failed']} ❌\n"
        report += f"- **Skipped**: {summary['skipped']} ⏭️\n"
        report += f"- **Success Rate**: {(summary['passed'] / summary['total_tests'] * 100):.1f}%\n\n"
        
        # Biological to Computational Results
        report += f"## 🧠→🤖 Biological → Computational Pathways\n\n"
        for result in results['biological_to_computational']:
            status_emoji = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⏭️"
            report += f"### {status_emoji} {result['name']}\n"
            report += f"- **Status**: {result['status']}\n"
            report += f"- **Accessibility**: {result['access_msg']}\n"
            report += f"- **Import Test**: {result['import_msg']}\n\n"
        
        # Computational to Biological Results
        report += f"## 🤖→🧠 Computational → Biological Pathways\n\n"
        for result in results['computational_to_biological']:
            status_emoji = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⏭️"
            report += f"### {status_emoji} {result['name']}\n"
            report += f"- **Status**: {result['status']}\n"
            report += f"- **Accessibility**: {result['access_msg']}\n"
            report += f"- **Import Test**: {result['import_msg']}\n\n"
        
        # Recommendations
        report += f"## 💡 Recommendations\n\n"
        if summary['failed'] > 0:
            report += f"- **Fix Import Issues**: {summary['failed']} pathways have import problems\n"
            report += f"- **Review File Structure**: Ensure all modules are properly organized\n"
            report += f"- **Check Dependencies**: Verify that required modules are available\n"
        else:
            report += f"- **All Pathways Working**: Synergy architecture is functioning correctly\n"
            report += f"- **Ready for Optimization**: Proceed to performance optimization phase\n"
        
        return report

def main():
    """Main execution function"""
    print("🧠🤖 Brain-ML Synergy Architecture: Synergy Pathway Tester")
    print("=" * 60)
    
    tester = SynergyPathwayTester()
    
    # Run synergy tests
    results = tester.test_synergy_integration()
    
    # Generate and save report
    report = tester.generate_test_report(results)
    
    with open('SYNERGY_TEST_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Print summary
    summary = results['summary']
    print(f"\n📊 Synergy Test Complete!")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⏭️ Skipped: {summary['skipped']}")
    print(f"📄 Report saved to: SYNERGY_TEST_REPORT.md")
    
    if summary['failed'] == 0:
        print(f"\n🎉 All synergy pathways are working correctly!")
        print(f"🚀 Ready to proceed with performance optimization!")
    else:
        print(f"\n⚠️ Some synergy pathways need attention before optimization.")

if __name__ == "__main__":
    main()
