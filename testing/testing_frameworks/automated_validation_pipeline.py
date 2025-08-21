#!/usr/bin/env python3
"""
🔧 Automated Validation Pipeline
Establishes automated validation pipelines for QUARK's experimentation protocols
"""

import os
import sys
import time
from typing import Dict, List, Any
from dataclasses import dataclass

# Add parent directories to path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
quark_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, quark_root)

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration: float
    details: str

class AutomatedValidationPipeline:
    """Automated validation pipeline for QUARK experimentation protocols"""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        print("🔧 Automated Validation Pipeline initialized")
    
    def run_brain_architecture_tests(self) -> List[ValidationResult]:
        """Run brain architecture validation tests"""
        results = []
        
        # Test executive control
        start_time = time.time()
        try:
            # Use absolute import path
            exec_path = os.path.join(quark_root, "brain_architecture", "neural_core", "prefrontal_cortex", "executive_control.py")
            if os.path.exists(exec_path):
                # Import using exec to avoid path issues
                with open(exec_path, 'r') as f:
                    exec_code = f.read()
                
                # Create a temporary namespace
                namespace = {}
                exec(exec_code, namespace)
                
                # Test the class
                ExecutiveControl = namespace['ExecutiveControl']
                executive = ExecutiveControl()
                status = executive.get_status()
                
                if status and "active_plans" in status:
                    results.append(ValidationResult(
                        test_name="Executive Control Initialization",
                        status="PASS",
                        duration=time.time() - start_time,
                        details="Executive control module initialized successfully"
                    ))
                else:
                    results.append(ValidationResult(
                        test_name="Executive Control Initialization",
                        status="FAIL",
                        duration=time.time() - start_time,
                        details="Executive control module failed to initialize properly"
                    ))
            else:
                results.append(ValidationResult(
                    test_name="Executive Control Initialization",
                    status="FAIL",
                    duration=time.time() - start_time,
                    details="Executive control module file not found"
                ))
        except Exception as e:
            results.append(ValidationResult(
                test_name="Executive Control Initialization",
                status="FAIL",
                duration=time.time() - start_time,
                details=f"Exception: {str(e)}"
            ))
        
        # Test working memory
        start_time = time.time()
        try:
            # Use absolute import path
            wm_path = os.path.join(quark_root, "brain_architecture", "neural_core", "working_memory", "working_memory.py")
            if os.path.exists(wm_path):
                # Import using exec to avoid path issues
                with open(wm_path, 'r') as f:
                    exec_code = f.read()
                
                # Create a temporary namespace
                namespace = {}
                exec(exec_code, namespace)
                
                # Test the class
                WorkingMemory = namespace['WorkingMemory']
                wm = WorkingMemory(capacity=5)
                wm.store("test item", priority=0.8)
                retrieved = wm.retrieve("test")
                
                if retrieved and "test item" in str(retrieved.content):
                    results.append(ValidationResult(
                        test_name="Working Memory Functionality",
                        status="PASS",
                        duration=time.time() - start_time,
                        details="Working memory store/retrieve working correctly"
                    ))
                else:
                    results.append(ValidationResult(
                        test_name="Working Memory Functionality",
                        status="FAIL",
                        duration=time.time() - start_time,
                        details="Working memory store/retrieve failed"
                    ))
            else:
                results.append(ValidationResult(
                    test_name="Working Memory Functionality",
                    status="FAIL",
                    duration=time.time() - start_time,
                    details="Working memory module file not found"
                ))
        except Exception as e:
            results.append(ValidationResult(
                test_name="Working Memory Functionality",
                status="FAIL",
                duration=time.time() - start_time,
                details=f"Exception: {str(e)}"
            ))
        
        return results
    
    def run_experiment_framework_tests(self) -> List[ValidationResult]:
        """Run experiment framework validation tests"""
        results = []
        
        # Test framework structure
        start_time = time.time()
        framework_files = [
            "quark_experimentation_protocols.md",
            "main_rules_integration.md"
        ]
        
        all_files_exist = all(os.path.exists(f) for f in framework_files)
        
        if all_files_exist:
            results.append(ValidationResult(
                test_name="Experiment Framework Structure",
                status="PASS",
                duration=time.time() - start_time,
                details="All framework files present and accessible"
            ))
        else:
            results.append(ValidationResult(
                test_name="Experiment Framework Structure",
                status="FAIL",
                duration=time.time() - start_time,
                details="Some framework files missing"
            ))
        
        return results
    
    def run_file_system_tests(self) -> List[ValidationResult]:
        """Run file system validation tests"""
        results = []
        
        # Test if we can access key QUARK directories
        start_time = time.time()
        key_dirs = [
            "brain_architecture",
            "tasks", 
            "testing"
        ]
        
        accessible_dirs = []
        for dir_path in key_dirs:
            full_path = os.path.join(quark_root, dir_path)
            if os.path.exists(full_path):
                accessible_dirs.append(dir_path)
        
        if len(accessible_dirs) == len(key_dirs):
            results.append(ValidationResult(
                test_name="File System Access",
                status="PASS",
                duration=time.time() - start_time,
                details=f"All key directories accessible: {', '.join(accessible_dirs)}"
            ))
        else:
            results.append(ValidationResult(
                test_name="File System Access",
                status="FAIL",
                duration=time.time() - start_time,
                details=f"Only {len(accessible_dirs)}/{len(key_dirs)} directories accessible: {', '.join(accessible_dirs)}"
            ))
        
        return results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete validation pipeline"""
        print("🔄 Running validation pipeline...")
        
        start_time = time.time()
        
        # Run all test suites
        print("\n🧠 Testing Brain Architecture...")
        brain_results = self.run_brain_architecture_tests()
        
        print("\n📋 Testing Experiment Framework...")
        framework_results = self.run_experiment_framework_tests()
        
        print("\n💾 Testing File System...")
        filesystem_results = self.run_file_system_tests()
        
        all_results = brain_results + framework_results + filesystem_results
        
        # Calculate summary
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r.status == "PASS"])
        failed_tests = len([r for r in all_results if r.status == "FAIL"])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        execution_time = time.time() - start_time
        
        self.validation_results = all_results
        
        # Show detailed results
        print("\n📊 Detailed Test Results:")
        for result in all_results:
            status_emoji = "✅" if result.status == "PASS" else "❌"
            print(f"   {status_emoji} {result.test_name}: {result.status} ({result.duration:.3f}s)")
            if result.status == "FAIL":
                print(f"      Details: {result.details}")
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "execution_time": execution_time,
            "results": all_results
        }
        
        print(f"\n✅ Pipeline completed: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
        return summary
    
    def get_summary(self) -> str:
        """Get pipeline summary"""
        if not self.validation_results:
            return "Pipeline not yet run"
        
        passed = len([r for r in self.validation_results if r.status == "PASS"])
        failed = len([r for r in self.validation_results if r.status == "FAIL"])
        total = len(self.validation_results)
        
        return f"""
🧪 QUARK Validation Pipeline Summary
=====================================
✅ Passed: {passed}/{total}
❌ Failed: {failed}
📊 Success Rate: {passed/total:.1%}
        """

def main():
    """Main function to run validation pipeline"""
    print("🧪 QUARK Automated Validation Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = AutomatedValidationPipeline()
    
    # Run full pipeline
    summary = pipeline.run_full_pipeline()
    
    # Display results
    print(pipeline.get_summary())
    
    print("\n✅ Validation pipeline completed!")
    return pipeline

if __name__ == "__main__":
    try:
        pipeline = main()
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
