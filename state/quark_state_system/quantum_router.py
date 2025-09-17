


"""
Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""
def route_computation_intelligently(task_name: str, problem_size: int, **kwargs):
    """
    Intelligent routing for Quark brain simulation tasks
    Automatically decides between classical and quantum computing
    """
    from quantum_decision_engine import QuantumDecisionEngine, create_brain_simulation_task

    # Initialize decision engine
    engine = QuantumDecisionEngine()

    # Create task description
    task = create_brain_simulation_task(
        task_name,
        problem_size=problem_size,
        expected_runtime_classical=kwargs.get('expected_runtime', 60),
        **kwargs
    )

    # Get intelligent decision
    decision = engine.make_computation_decision(task)

    print(f"🧠 Task: {task_name}")
    print(f"⚛️ Routing Decision: {decision['computation_type'].value}")
    print(f"📊 Quantum Advantage Score: {decision['quantum_advantage_score']:.2f}")

    # Route to appropriate compute engine
    if decision['computation_type'] == ComputationType.CLASSICAL:
        return run_classical_computation(task_name, problem_size, **kwargs)
    elif decision['computation_type'] == ComputationType.QUANTUM_SIMULATOR:
        return run_quantum_simulation(task_name, problem_size, **kwargs)
    elif decision['computation_type'] == ComputationType.QUANTUM_HARDWARE:
        return run_quantum_hardware(task_name, problem_size, **kwargs)
    else:  # HYBRID
        return run_hybrid_computation(task_name, problem_size, **kwargs)

def run_classical_computation(task_name: str, problem_size: int, **kwargs):
    """Run computation on classical hardware"""
    print("🖥️ Running on classical hardware...")
    # Your existing classical brain simulation code here
    return {"status": "success", "compute_type": "classical"}

def run_quantum_simulation(task_name: str, problem_size: int, **kwargs):
    """Run computation on quantum simulator"""
    print("🌐 Running on quantum simulator...")
    try:
        from brain_modules.alphagenome_integration.quantum_braket_integration import QuantumBrainIntegration
        quantum_brain = QuantumBrainIntegration()
        # Use local simulator for cost-effective testing
        return quantum_brain.execute_quantum_circuit(use_simulator=True)
    except ImportError:
        print("⚠️ Quantum integration not available, falling back to classical")
        return run_classical_computation(task_name, problem_size, **kwargs)

def run_quantum_hardware(task_name: str, problem_size: int, **kwargs):
    """Run computation on quantum hardware"""
    print("⚛️ Running on quantum hardware...")
    try:
        from brain_modules.alphagenome_integration.quantum_braket_integration import QuantumBrainIntegration
        quantum_brain = QuantumBrainIntegration()
        # Use actual quantum hardware
        return quantum_brain.execute_quantum_circuit(use_simulator=False)
    except ImportError:
        print("⚠️ Quantum integration not available, falling back to simulator")
        return run_quantum_simulation(task_name, problem_size, **kwargs)

def run_hybrid_computation(task_name: str, problem_size: int, **kwargs):
    """Run hybrid quantum-classical computation"""
    print("🔄 Running hybrid quantum-classical computation...")

    # Classical preprocessing
    classical_result = run_classical_computation(f"{task_name}_preprocessing", problem_size//2, **kwargs)

    # Quantum processing for optimization/search
    quantum_result = run_quantum_simulation(f"{task_name}_quantum_core", problem_size//4, **kwargs)

    # Classical postprocessing
    return {
        "status": "success",
        "compute_type": "hybrid",
        "classical_component": classical_result,
        "quantum_component": quantum_result
    }
