#!/usr/bin/env python3
"""
Wolfram Alpha Integration Demo for Quark Brain Simulation
========================================================

Comprehensive demonstration of Wolfram Alpha API integration with your
brain simulation project. This script showcases:

1. Basic API connectivity and validation
2. Neural dynamics computations
3. Mathematical model validation
4. Brain connectivity analysis
5. Parameter optimization
6. Statistical analysis of neural data
7. Training pipeline integration
8. Real-time computational assistance

Run this to verify your Wolfram Alpha integration is working correctly.
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.wolfram_alpha_integration import (
    BrainSimulationWolfram, 
    WolframAlphaClient, 
    WolframQuery,
    WolframResultProcessor
)
from src.core.wolfram_brain_trainer import WolframBrainTrainer, TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WolframDemoRunner:
    """
    Comprehensive demo runner for Wolfram Alpha integration
    """
    
    def __init__(self, app_id: str = "TYW5HL7G68"):
        self.app_id = app_id
        self.client = WolframAlphaClient(app_id)
        self.brain_wolfram = BrainSimulationWolfram(app_id)
        self.processor = WolframResultProcessor()
        
        # Create demo results directory
        self.demo_dir = Path("./data/wolfram_demo")
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Initialized Wolfram Demo with App ID: {app_id}")

    async def test_basic_connectivity(self):
        """Test basic API connectivity"""
        print("\n" + "="*60)
        print("🔌 TESTING BASIC API CONNECTIVITY")
        print("="*60)
        
        # Simple test query
        query = WolframQuery(input_text="2+2")
        result = await self.client.query_async(query)
        
        if result.success:
            print("✅ Basic connectivity: SUCCESS")
            print(f"   Query: {result.query}")
            print(f"   Pods returned: {len(result.pods)}")
            if result.pods:
                first_result = result.pods[0]['subpods'][0]['plaintext']
                print(f"   Result: {first_result}")
        else:
            print("❌ Basic connectivity: FAILED")
            print(f"   Error: {result.error_message}")
        
        return result.success

    async def test_mathematical_computation(self):
        """Test mathematical computations relevant to brain simulation"""
        print("\n" + "="*60)
        print("🧮 TESTING MATHEMATICAL COMPUTATIONS")
        print("="*60)
        
        test_cases = [
            "integrate exp(-x^2) from -infinity to infinity",
            "eigenvalues of {{1, 0.5}, {0.5, 1}}",
            "solve differential equation y' = -y + sin(t)",
            "fourier transform of gaussian function",
            "stability analysis of x' = -x + tanh(x)"
        ]
        
        results = {}
        
        for i, test_case in enumerate(test_cases):
            print(f"\n📊 Test {i+1}: {test_case}")
            
            query = WolframQuery(
                input_text=test_case,
                include_pods=["Result", "Solution", "Plot"]
            )
            
            result = await self.client.query_async(query)
            
            if result.success:
                print("   ✅ SUCCESS")
                print(f"   Pods: {len(result.pods)}")
                
                # Extract key information
                numerical_values = self.processor.extract_numerical_results(result)
                equations = self.processor.extract_equations(result)
                
                if numerical_values:
                    print(f"   Numerical values: {numerical_values[:3]}...")
                if equations:
                    print(f"   Equations: {equations[0][:50]}..." if equations[0] else "None")
                
                results[f"test_{i+1}"] = {
                    'query': test_case,
                    'success': True,
                    'pods': len(result.pods),
                    'numerical_values': numerical_values,
                    'equations': equations
                }
            else:
                print("   ❌ FAILED")
                print(f"   Error: {result.error_message}")
                results[f"test_{i+1}"] = {
                    'query': test_case,
                    'success': False,
                    'error': result.error_message
                }
        
        return results

    async def test_neural_dynamics_analysis(self):
        """Test neural dynamics specific computations"""
        print("\n" + "="*60)
        print("🧠 TESTING NEURAL DYNAMICS ANALYSIS")
        print("="*60)
        
        # Test Hodgkin-Huxley model analysis
        print("\n🔬 Hodgkin-Huxley Model Analysis:")
        
        hh_equation = "C_m * dV/dt = I - g_Na * m^3 * h * (V - E_Na) - g_K * n^4 * (V - E_K) - g_L * (V - E_L)"
        parameters = {
            "C_m": 1.0,
            "g_Na": 120.0,
            "g_K": 36.0,
            "g_L": 0.3,
            "E_Na": 50.0,
            "E_K": -77.0,
            "E_L": -54.4
        }
        
        result = await self.brain_wolfram.compute_neural_dynamics(hh_equation, parameters)
        
        if result.success:
            print("   ✅ Hodgkin-Huxley analysis: SUCCESS")
            print(f"   Pods returned: {len(result.pods)}")
        else:
            print("   ❌ Hodgkin-Huxley analysis: FAILED")
            print(f"   Error: {result.error_message}")
        
        # Test connectivity matrix analysis
        print("\n🔗 Brain Connectivity Matrix Analysis:")
        
        # Create a realistic brain connectivity matrix
        connectivity_matrix = np.array([
            [1.0, 0.8, 0.3, 0.1],  # Cortex connections
            [0.8, 1.0, 0.6, 0.4],  # Hippocampus connections
            [0.3, 0.6, 1.0, 0.9],  # Thalamus connections
            [0.1, 0.4, 0.9, 1.0]   # Brainstem connections
        ])
        
        connectivity_result = await self.brain_wolfram.analyze_connectivity_matrix(connectivity_matrix.tolist())
        
        if connectivity_result.success:
            print("   ✅ Connectivity analysis: SUCCESS")
            print(f"   Pods returned: {len(connectivity_result.pods)}")
            
            # Extract eigenvalues
            eigenvalues = self.processor.extract_numerical_results(connectivity_result)
            if eigenvalues:
                print(f"   Eigenvalues found: {len(eigenvalues)}")
        else:
            print("   ❌ Connectivity analysis: FAILED")
        
        return {
            'hodgkin_huxley': result.success,
            'connectivity': connectivity_result.success
        }

    async def test_optimization_capabilities(self):
        """Test parameter optimization capabilities"""
        print("\n" + "="*60)
        print("🎯 TESTING OPTIMIZATION CAPABILITIES")
        print("="*60)
        
        # Test neural network parameter optimization
        print("\n⚡ Neural Network Parameter Optimization:")
        
        objective = "minimize (learning_rate - 0.001)^2 + (batch_size - 32)^2 + (hidden_size - 128)^2"
        constraints = [
            "0.0001 <= learning_rate <= 0.1",
            "8 <= batch_size <= 256",
            "32 <= hidden_size <= 512"
        ]
        
        optimization_result = await self.brain_wolfram.optimize_parameters(objective, constraints)
        
        if optimization_result.success:
            print("   ✅ Parameter optimization: SUCCESS")
            print(f"   Pods returned: {len(optimization_result.pods)}")
            
            # Extract optimized values
            optimal_values = self.processor.extract_numerical_results(optimization_result)
            if optimal_values:
                print(f"   Optimal values: {optimal_values}")
        else:
            print("   ❌ Parameter optimization: FAILED")
            print(f"   Error: {optimization_result.error_message}")
        
        return optimization_result.success

    async def test_statistical_analysis(self):
        """Test statistical analysis of neural data"""
        print("\n" + "="*60)
        print("📈 TESTING STATISTICAL ANALYSIS")
        print("="*60)
        
        # Generate synthetic neural spike data
        np.random.seed(42)
        neural_data = np.random.poisson(5, 1000)  # Poisson spike train
        
        # Analyze the data
        print("\n📊 Neural Spike Train Analysis:")
        
        data_description = f"poisson distributed neural spikes with {len(neural_data)} samples, mean={np.mean(neural_data):.2f}"
        
        stats_result = await self.brain_wolfram.statistical_analysis(data_description, "poisson distribution test")
        
        if stats_result.success:
            print("   ✅ Statistical analysis: SUCCESS")
            print(f"   Pods returned: {len(stats_result.pods)}")
        else:
            print("   ❌ Statistical analysis: FAILED")
            print(f"   Error: {stats_result.error_message}")
        
        # Test time series analysis
        print("\n⏱️ Neural Time Series Analysis:")
        
        # Generate neural oscillation data
        t = np.linspace(0, 10, 1000)
        oscillation_data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t) + 0.1 * np.random.randn(1000)
        
        time_series_description = f"neural oscillation data with {len(oscillation_data)} points, frequency components at 10Hz and 30Hz"
        
        ts_result = await self.brain_wolfram.analyze_time_series(time_series_description)
        
        if ts_result.success:
            print("   ✅ Time series analysis: SUCCESS")
            print(f"   Pods returned: {len(ts_result.pods)}")
        else:
            print("   ❌ Time series analysis: FAILED")
        
        return {
            'statistical': stats_result.success,
            'time_series': ts_result.success
        }

    async def test_training_integration(self):
        """Test training pipeline integration"""
        print("\n" + "="*60)
        print("🏋️ TESTING TRAINING INTEGRATION")
        print("="*60)
        
        # Create a simple training configuration
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=16,
            epochs=5,  # Small number for demo
            wolfram_validation_frequency=2,
            use_wolfram_guidance=True
        )
        
        # Simple architecture
        architecture = {
            'cortex': {
                'input_size': 32,
                'hidden_size': 64,
                'output_size': 16,
                'activation': 'relu'
            }
        }
        
        trainer = WolframBrainTrainer(config, self.app_id)
        
        print("\n🎓 Running mini training session...")
        
        try:
            results = await trainer.train_with_wolfram_validation(architecture)
            
            print("   ✅ Training integration: SUCCESS")
            print(f"   Final train loss: {results['train_losses'][-1]:.6f}")
            print(f"   Final val loss: {results['val_losses'][-1]:.6f}")
            print(f"   Wolfram validations: {len(results['wolfram_validations'])}")
            
            return True
            
        except Exception as e:
            print("   ❌ Training integration: FAILED")
            print(f"   Error: {e}")
            return False

    async def generate_demo_report(self, results: Dict):
        """Generate comprehensive demo report"""
        print("\n" + "="*60)
        print("📋 GENERATING DEMO REPORT")
        print("="*60)
        
        report = {
            'demo_timestamp': datetime.now().isoformat(),
            'app_id': self.app_id,
            'results': results,
            'summary': {
                'total_tests': 0,
                'successful_tests': 0,
                'success_rate': 0.0
            }
        }
        
        # Calculate summary statistics
        for category, category_results in results.items():
            if isinstance(category_results, bool):
                report['summary']['total_tests'] += 1
                if category_results:
                    report['summary']['successful_tests'] += 1
            elif isinstance(category_results, dict):
                for test_name, test_result in category_results.items():
                    if isinstance(test_result, bool):
                        report['summary']['total_tests'] += 1
                        if test_result:
                            report['summary']['successful_tests'] += 1
                    elif isinstance(test_result, dict) and 'success' in test_result:
                        report['summary']['total_tests'] += 1
                        if test_result['success']:
                            report['summary']['successful_tests'] += 1
        
        if report['summary']['total_tests'] > 0:
            report['summary']['success_rate'] = (
                report['summary']['successful_tests'] / report['summary']['total_tests']
            ) * 100
        
        # Save report
        report_path = self.demo_dir / f"wolfram_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 DEMO SUMMARY:")
        print(f"   Total tests: {report['summary']['total_tests']}")
        print(f"   Successful: {report['summary']['successful_tests']}")
        print(f"   Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"   Report saved: {report_path}")
        
        return report

    async def run_full_demo(self):
        """Run the complete Wolfram Alpha integration demo"""
        print("🌟 WOLFRAM ALPHA INTEGRATION DEMO FOR QUARK BRAIN SIMULATION")
        print("=" * 80)
        print(f"App ID: {self.app_id}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        results = {}
        
        try:
            # Test 1: Basic connectivity
            results['basic_connectivity'] = await self.test_basic_connectivity()
            
            # Test 2: Mathematical computations
            results['mathematical_computations'] = await self.test_mathematical_computation()
            
            # Test 3: Neural dynamics
            results['neural_dynamics'] = await self.test_neural_dynamics_analysis()
            
            # Test 4: Optimization
            results['optimization'] = await self.test_optimization_capabilities()
            
            # Test 5: Statistical analysis
            results['statistical_analysis'] = await self.test_statistical_analysis()
            
            # Test 6: Training integration
            results['training_integration'] = await self.test_training_integration()
            
            # Generate final report
            report = await self.generate_demo_report(results)
            
            print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("Check the generated report for detailed results.")
            
            return report
            
        except Exception as e:
            print(f"\n💥 DEMO FAILED: {e}")
            logger.error(f"Demo failed with error: {e}", exc_info=True)
            return None


async def main():
    """Main demo function"""
    # Create demo runner with your App ID
    demo = WolframDemoRunner(app_id="TYW5HL7G68")
    
    # Run full demonstration
    report = await demo.run_full_demo()
    
    if report:
        print(f"\n✨ Demo completed with {report['summary']['success_rate']:.1f}% success rate!")
    else:
        print("\n❌ Demo failed to complete.")

if __name__ == "__main__":
    # Set event loop policy for compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the demo
    asyncio.run(main())
