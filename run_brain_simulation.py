#!/usr/bin/env python3
"""
Brain Simulation Launcher with Mutex Fix
Properly configures environment to avoid mutex lock issues before launching brain simulation.
"""

import os
import sys
import subprocess
import signal

def setup_environment():
    """Set up environment variables to prevent mutex lock issues."""
    # Suppress all logging and mutex debugging
    os.environ['GLOG_minloglevel'] = '3'
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_alsologtostderr'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GRPC_VERBOSITY'] = 'NONE'
    os.environ['GRPC_TRACE'] = ''
    
    # Threading controls
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    
    # JAX configuration
    os.environ['JAX_ENABLE_X64'] = 'false'
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    # MuJoCo specific fixes
    os.environ['MUJOCO_DISABLE_MUTEXES'] = '1'
    os.environ['MUJOCO_GL'] = 'disable'  # Force disable OpenGL to avoid mutex issues
    
    print("🔧 Environment configured for mutex-free operation")

def run_brain_simulation():
    """Launch the brain simulation with proper environment."""
    setup_environment()
    
    print("🚀 Launching Brain Simulation...")
    print("💡 Press Ctrl+C during simulation to interact with Quark")
    print("💡 Type 'exit', 'quit', or 'stop' to terminate")
    print("-" * 60)
    
    # Import and run brain_main after environment is set
    sys.path.insert(0, '/Users/camdouglas/quark')
    
    # Import brain_main module and call its main function
    try:
        from brain import brain_main
        brain_main.main(['--no-viewer', '--hz', '30'])
    except KeyboardInterrupt:
        print("\n👋 Brain simulation terminated by user")
    except Exception as e:
        print(f"❌ Error running brain simulation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_brain_simulation()
