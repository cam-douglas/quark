#!/usr/bin/env python3
"""
Brain Simulation Launcher with Mutex Fix
Properly configures environment to avoid mutex lock issues before launching brain simulation.
"""

# CRITICAL: Set environment variables BEFORE any imports
import os
import sys

# Pre-configure environment immediately at module load
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_alsologtostderr'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'
os.environ['LAPACK_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['JAX_ENABLE_X64'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['MUJOCO_DISABLE_MUTEXES'] = '1'
os.environ['MUJOCO_DISABLE_TLS'] = '1'
os.environ['MUJOCO_MULTITHREAD'] = '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Additional aggressive mutex prevention
os.environ['MALLOC_ARENA_MAX'] = '1'  # Limit memory arena
os.environ['GOMP_CPU_AFFINITY'] = '0'  # Force single CPU affinity
os.environ['KMP_AFFINITY'] = 'disabled'  # Disable Intel KMP affinity

def setup_environment():
    """Set up environment variables to prevent mutex lock issues."""
    # Suppress all logging and mutex debugging
    os.environ['GLOG_minloglevel'] = '3'
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_alsologtostderr'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GRPC_VERBOSITY'] = 'NONE'
    os.environ['GRPC_TRACE'] = ''

    # Enhanced threading controls - force single threading everywhere
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['BLAS_NUM_THREADS'] = '1'
    os.environ['LAPACK_NUM_THREADS'] = '1'
    
    # Tokenizers parallelism (prevents forking issues)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # PyTorch threading
    os.environ['TORCH_NUM_THREADS'] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

    # JAX configuration - force CPU and disable parallelism
    os.environ['JAX_ENABLE_X64'] = 'false'
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'

    # MuJoCo specific fixes - enhanced mutex prevention
    os.environ['MUJOCO_DISABLE_MUTEXES'] = '1'
    os.environ['MUJOCO_DISABLE_TLS'] = '1'  # Disable thread-local storage
    os.environ['MUJOCO_MULTITHREAD'] = '0'  # Force single-threaded mode
    
    # OpenGL/rendering settings
    if not os.environ.get('QUARK_ENABLE_VIEWER', '0') == '1':
        os.environ['MUJOCO_GL'] = 'disable'  # Force disable OpenGL to avoid mutex issues
    else:
        # For viewer mode, use safe OpenGL settings
        os.environ['MUJOCO_GL'] = 'egl'  # Use EGL instead of GLX
        os.environ['DISPLAY'] = ':0'  # Ensure display is set
    
    # Force interactive mode for viewer
    os.environ['QUARK_FORCE_INTERACTIVE'] = '1'
    
    # Additional mutex prevention
    os.environ['PYTHONHASHSEED'] = '0'  # Deterministic hashing
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA completely

    print("🔧 Enhanced environment configured for mutex-free operation")
    print("🔧 All threading forced to single-threaded mode")

def run_brain_simulation():
    """Launch the brain simulation with proper environment."""
    setup_environment()

    print("🚀 Launching Brain Simulation...")
    print("💡 Press Ctrl+C during simulation to interact with Quark")
    print("💡 Type 'exit', 'quit', or 'stop' to terminate")
    print("-" * 60)

    # Import and run brain_main after environment is set
    sys.path.insert(0, '/Users/camdouglas/quark')

    # Force garbage collection and clear any cached modules
    import gc
    gc.collect()
    
    # Clear any problematic cached modules
    modules_to_clear = [name for name in sys.modules.keys() 
                       if any(lib in name.lower() for lib in 
                             ['mujoco', 'torch', 'jax', 'tensorflow', 'numpy'])]
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    print(f"🔧 Cleared {len(modules_to_clear)} cached modules to prevent mutex conflicts")

    # Pre-configure threading before any imports
    try:
        import threading
        threading.current_thread().name = "MainThread"
        print("🔧 Threading configured for main thread")
    except Exception as e:
        print(f"⚠️ Threading configuration warning: {e}")

    # Import brain_main module and call its main function
    # Check if viewer is requested
    use_viewer = os.environ.get('QUARK_ENABLE_VIEWER', '0') == '1'
    args = ['--hz', '30']
    if use_viewer:
        args.extend(['--viewer'])
        print("🎮 MuJoCo viewer enabled - window should appear shortly")
        print("🔧 Using EGL rendering to prevent mutex issues")
    else:
        args.extend(['--no-viewer'])
        print("🖥️  Running in headless mode")

    try:
        # Import with additional safety
        print("🔧 Loading brain modules with mutex prevention...")
        from brain import brain_main
        brain_main.main(args)
    except KeyboardInterrupt:
        print("\n👋 Brain simulation terminated by user")
    except Exception as e:
        print(f"❌ Error running brain simulation: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_brain_simulation()
