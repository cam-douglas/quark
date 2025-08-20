#!/usr/bin/env python3
"""
Enhanced Small-Mind Integration Demo

Demonstrates the integration of new modules:
- NEST neural network simulation
- Optuna hyperparameter optimization  
- PyVista 3D visualization
- Enhanced ML model management
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_nest_integration():
    """Test NEST neural network simulation integration"""
    print("🧠 Testing NEST Neural Network Integration...")
    
    try:
        from physics_simulation.nest_interface import NESTInterface
        
        # Initialize NEST interface
        nest_interface = NESTInterface()
        
        # Create neuron populations
        excitatory_neurons = nest_interface.create_neuron_population(
            neuron_type="iaf_cond_alpha",
            num_neurons=100,
            params={"V_th": -55.0, "I_e": 100.0}
        )
        
        inhibitory_neurons = nest_interface.create_neuron_population(
            neuron_type="iaf_cond_alpha", 
            num_neurons=25,
            params={"V_th": -55.0, "I_e": 0.0}
        )
        
        # Connect populations
        nest_interface.connect_populations(
            source_ids=excitatory_neurons,
            target_ids=excitatory_neurons,
            connection_type="all_to_all",
            weight=1.0,
            delay=1.0
        )
        
        nest_interface.connect_populations(
            source_ids=excitatory_neurons,
            target_ids=inhibitory_neurons,
            connection_type="all_to_all",
            weight=0.5,
            delay=1.0
        )
        
        # Add external input
        nest_interface.add_external_input(
            neuron_ids=excitatory_neurons[:10],
            input_type="poisson",
            params={"rate": 20.0}
        )
        
        # Add recording devices
        nest_interface.add_recording_devices(
            neuron_ids=excitatory_neurons[:20],
            device_type="spike_detector"
        )
        
        # Run simulation
        results = nest_interface.simulate(duration=1000.0)
        
        print(f"✅ NEST simulation completed: {results['status']}")
        print(f"   Network: {len(nest_interface.nodes)} neuron types")
        print(f"   Connections: {len(nest_interface.connections)} connection groups")
        
        return True
        
    except Exception as e:
        print(f"❌ NEST integration failed: {e}")
        return False

def test_optuna_integration():
    """Test Optuna hyperparameter optimization integration"""
    print("🔍 Testing Optuna Hyperparameter Optimization...")
    
    try:
        from ml_optimization.optuna_interface import OptunaOptimizer
        
        # Initialize Optuna optimizer
        optimizer = OptunaOptimizer(
            study_name="brain_development_test",
            sampler_type="tpe"
        )
        
        # Define objective function for brain development optimization
        def brain_development_objective(trial):
            params = optimizer.suggest_brain_development_params(trial)
            
            # Simulate a simple fitness function
            fitness = (
                params["growth_rate"] * 10 +
                params["synapse_density"] * 5 +
                params["plasticity_factor"] * 8 +
                (1.0 - params["dropout_rate"]) * 3
            )
            
            # Add some noise to make it realistic
            noise = np.random.normal(0, 0.1)
            return fitness + noise
        
        # Run optimization
        results = optimizer.optimize(
            objective_func=brain_development_objective,
            n_trials=20,
            timeout=60
        )
        
        if "error" not in results:
            print(f"✅ Optuna optimization completed:")
            print(f"   Best value: {results['best_value']:.4f}")
            print(f"   Trials: {results['n_trials']}")
            print(f"   Best params: {results['best_params']}")
            
            # Get optimization history
            history = optimizer.get_optimization_history()
            print(f"   Mean value: {history['mean_value']:.4f}")
            print(f"   Std value: {history['std_value']:.4f}")
            
            return True
        else:
            print(f"❌ Optimization failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Optuna integration failed: {e}")
        return False

def test_pyvista_integration():
    """Test PyVista 3D visualization integration"""
    print("🎨 Testing PyVista 3D Visualization...")
    
    try:
        from visualization.pyvista_interface import PyVistaVisualizer
        
        # Initialize PyVista visualizer
        visualizer = PyVistaVisualizer()
        
        # Create sample brain data
        brain_size = (32, 32, 32)
        brain_data = np.zeros(brain_size)
        
        # Create a simple brain-like structure
        center = np.array(brain_size) // 2
        for i in range(brain_size[0]):
            for j in range(brain_size[1]):
                for k in range(brain_size[2]):
                    pos = np.array([i, j, k])
                    distance = np.linalg.norm(pos - center)
                    if distance < 12:
                        brain_data[i, j, k] = 1.0
                    elif distance < 15:
                        brain_data[i, j, k] = 0.5
        
        # Create brain mesh
        brain_mesh_id = visualizer.create_brain_mesh(
            brain_data=brain_data,
            mesh_name="test_brain",
            smoothing=True,
            decimation=0.7
        )
        
        # Create sample neuronal network
        num_neurons = 50
        neuron_positions = np.random.rand(num_neurons, 3) * 20 - 10
        connections = np.random.randint(0, num_neurons, (num_neurons * 2, 2))
        
        network_id = visualizer.create_neuronal_network(
            neuron_positions=neuron_positions,
            connections=connections,
            network_name="test_network"
        )
        
        # Add meshes to plot
        visualizer.add_mesh_to_plot(
            mesh_id=brain_mesh_id,
            color="pink",
            opacity=0.8
        )
        
        visualizer.add_mesh_to_plot(
            mesh_id=network_id,
            color="blue",
            opacity=0.6
        )
        
        # Add text labels
        visualizer.add_text(
            text="Brain Development Model",
            position=(0, 0, 15),
            font_size=16,
            color="black"
        )
        
        # Set camera position
        visualizer.set_camera_position(
            position=(20, 20, 20),
            focal_point=(0, 0, 0)
        )
        
        # Get mesh information
        brain_info = visualizer.get_mesh_info(brain_mesh_id)
        network_info = visualizer.get_mesh_info(network_id)
        
        print(f"✅ PyVista visualization created:")
        print(f"   Brain mesh: {brain_info.get('n_points', 'N/A')} points")
        print(f"   Network: {network_info.get('n_points', 'N/A')} points")
        
        # Save visualization (non-interactive for demo)
        screenshot_path = "pyvista_brain_visualization.png"
        visualizer.show(interactive=False, screenshot_path=screenshot_path)
        
        if os.path.exists(screenshot_path):
            print(f"   Screenshot saved: {screenshot_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyVista integration failed: {e}")
        return False

def test_enhanced_model_management():
    """Test enhanced ML model management with new dependencies"""
    print("🤖 Testing Enhanced Model Management...")
    
    try:
        from models.model_manager import MoEModelManager
        
        # Initialize model manager
        model_manager = MoEModelManager()
        
        # Test model configurations
        available_models = list(model_manager.model_configs.keys())
        print(f"   Available models: {available_models}")
        
        # Test model download (will use existing models if available)
        for model_name in available_models[:1]:  # Test first model only
            try:
                local_path = model_manager.download_model(model_name, force_redownload=False)
                print(f"   Model {model_name}: {local_path}")
            except Exception as e:
                print(f"   Model {model_name}: {e}")
        
        # Test model registry
        registry = model_manager.model_registry
        print(f"   Registry entries: {len(registry)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced model management failed: {e}")
        return False

def test_large_scale_data_processing():
    """Test large-scale data processing capabilities"""
    print("📊 Testing Large-Scale Data Processing...")
    
    try:
        # Test Dask integration
        try:
            import dask.array as da
            
            # Create large array
            large_array = da.random.random((1000, 1000, 100), chunks=(100, 100, 100))
            
            # Perform operations
            result = da.mean(large_array).compute()
            print(f"   Dask array mean: {result:.6f}")
            
        except ImportError:
            print("   Dask not available")
        
        # Test Vaex integration
        try:
            import vaex
            
            # Create sample data
            data = {
                'x': np.random.randn(10000),
                'y': np.random.randn(10000),
                'z': np.random.randn(10000)
            }
            
            df = vaex.from_dict(data)
            mean_x = df.x.mean().get()
            print(f"   Vaex data mean: {mean_x:.6f}")
            
        except ImportError:
            print("   Vaex not available (Python 3.13 compatibility issue)")
        
        return True
        
    except Exception as e:
        print(f"❌ Large-scale data processing failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Enhanced Small-Mind Integration Demo")
    print("=" * 50)
    
    # Track test results
    test_results = {}
    
    # Run integration tests
    test_results['nest'] = test_nest_integration()
    test_results['optuna'] = test_optuna_integration()
    test_results['pyvista'] = test_pyvista_integration()
    test_results['model_management'] = test_enhanced_model_management()
    test_results['data_processing'] = test_large_scale_data_processing()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Integration Test Summary")
    print("=" * 50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integrations successful! Small-Mind is ready.")
    else:
        print("⚠️  Some integrations failed. Check dependencies and try again.")
    
    # Save results
    results_file = "enhanced_integration_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": test_results,
            "summary": {
                "passed": passed,
                "total": total,
                "success_rate": passed / total
            }
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
