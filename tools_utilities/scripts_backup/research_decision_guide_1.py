#!/usr/bin/env python3
"""
Research Decision Guide for Brain Simulation Approaches

This interactive guide helps you choose the right simulation approach
based on your specific research questions and requirements.
"""

def get_research_focus():
    """Get user's primary research focus"""
    print("\n🎯 What is your PRIMARY research focus?")
    print("1. Tissue mechanics and physical development")
    print("2. Neural connectivity and network dynamics")
    print("3. Multi-scale brain development")
    print("4. I'm not sure - help me decide")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-4): "))
            if 1 <= choice <= 4:
                return choice
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")

def get_research_questions():
    """Get specific research questions"""
    print("\n❓ What specific questions are you trying to answer?")
    print("(You can select multiple)")
    print("A. How do mechanical forces affect brain growth?")
    print("B. How do neural circuits form and develop?")
    print("C. What drives learning and synaptic plasticity?")
    print("D. How does spatial organization affect function?")
    print("E. How do physical constraints limit neural development?")
    print("F. What happens in developmental disorders?")
    print("G. How do different brain regions interact?")
    print("H. Other (describe)")
    
    choices = input("\nEnter your choices (e.g., A,B,C): ").upper().split(',')
    return [choice.strip() for choice in choices]

def get_research_scale():
    """Get research scale preference"""
    print("\n🔬 What scale are you most interested in?")
    print("1. Macroscopic (brain regions, tissue layers)")
    print("2. Microscopic (neurons, synapses, circuits)")
    print("3. Multi-scale (both levels)")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if 1 <= choice <= 3:
                return choice
            else:
                print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number.")

def get_time_scale():
    """Get time scale preference"""
    print("\n⏱️  What time scale are you studying?")
    print("1. Developmental (days to months)")
    print("2. Neural (milliseconds to hours)")
    print("3. Both scales")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if 1 <= choice <= 3:
                return choice
            else:
                print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number.")

def analyze_research_needs(focus, questions, scale, time):
    """Analyze research needs and recommend approach"""
    print("\n🔍 Analyzing your research needs...")
    
    # Score each approach
    mujoco_score = 0
    nest_score = 0
    hybrid_score = 0
    
    # Focus-based scoring
    if focus == 1:  # Tissue mechanics
        mujoco_score += 3
        hybrid_score += 2
    elif focus == 2:  # Neural connectivity
        nest_score += 3
        hybrid_score += 2
    elif focus == 3:  # Multi-scale
        hybrid_score += 3
        mujoco_score += 1
        nest_score += 1
    
    # Question-based scoring
    for question in questions:
        if question in ['A', 'D']:  # Mechanical forces, spatial organization
            mujoco_score += 2
            hybrid_score += 1
        elif question in ['B', 'C']:  # Neural circuits, learning
            nest_score += 2
            hybrid_score += 1
        elif question in ['E', 'F']:  # Physical constraints, disorders
            hybrid_score += 2
            mujoco_score += 1
            nest_score += 1
        elif question == 'G':  # Brain region interactions
            hybrid_score += 2
            mujoco_score += 1
            nest_score += 1
    
    # Scale-based scoring
    if scale == 1:  # Macroscopic
        mujoco_score += 2
        hybrid_score += 1
    elif scale == 2:  # Microscopic
        nest_score += 2
        hybrid_score += 1
    elif scale == 3:  # Multi-scale
        hybrid_score += 3
    
    # Time-based scoring
    if time == 1:  # Developmental
        mujoco_score += 2
        hybrid_score += 1
    elif time == 2:  # Neural
        nest_score += 2
        hybrid_score += 1
    elif time == 3:  # Both
        hybrid_score += 3
    
    return mujoco_score, nest_score, hybrid_score

def recommend_approach(mujoco_score, nest_score, hybrid_score):
    """Recommend the best approach based on scores"""
    print("\n📊 Approach Scores:")
    print(f"🔬 MuJoCo (Physical): {mujoco_score}")
    print(f"🧠 NEST (Neural): {nest_score}")
    print(f"🚀 Hybrid (Combined): {hybrid_score}")
    
    max_score = max(mujoco_score, nest_score, hybrid_score)
    
    if hybrid_score == max_score:
        recommended = "Hybrid"
        print("\n🎯 RECOMMENDED APPROACH: Hybrid (Combined)")
        print("Why: Your research requires understanding both physical and neural aspects")
    elif mujoco_score == max_score:
        recommended = "MuJoCo"
        print("\n🎯 RECOMMENDED APPROACH: MuJoCo (Physical)")
        print("Why: Your research focuses on tissue mechanics and physical development")
    else:
        recommended = "NEST"
        print("\n🎯 RECOMMENDED APPROACH: NEST (Neural)")
        print("Why: Your research focuses on neural connectivity and network dynamics")
    
    return recommended

def provide_implementation_guide(approach):
    """Provide implementation guide for the recommended approach"""
    print(f"\n🚀 Implementation Guide for {approach} Approach")
    print("=" * 60)
    
    if approach == "MuJoCo":
        print("🔬 MuJoCo Implementation:")
        print("1. Import required modules:")
        print("   from physics_simulation.mujoco_interface import MuJoCoInterface")
        print("   from physics_simulation.dual_mode_simulator import DualModeBrainSimulator")
        print("")
        print("2. Create simulator:")
        print("   mujoco_interface = MuJoCoInterface()")
        print("   simulator = DualModeBrainSimulator(")
        print("       simulation_mode='mujoco',")
        print("       mujoco_interface=mujoco_interface")
        print("   )")
        print("")
        print("3. Setup brain model:")
        print("   brain_regions = ['cortex', 'hippocampus', 'thalamus']")
        print("   cell_types = ['neurons', 'glia', 'endothelial']")
        print("   simulator.setup_brain_development_model(brain_regions, cell_types)")
        print("")
        print("4. Run simulation:")
        print("   results = simulator.simulate_brain_growth(duration=5.0)")
        print("")
        print("💡 Best for: Tissue mechanics, spatial development, biomechanics")
        
    elif approach == "NEST":
        print("🧠 NEST Implementation:")
        print("1. Import required modules:")
        print("   from physics_simulation.dual_mode_simulator import DualModeBrainSimulator")
        print("")
        print("2. Create simulator:")
        print("   simulator = DualModeBrainSimulator(simulation_mode='nest')")
        print("")
        print("3. Setup neural model:")
        print("   brain_regions = ['cortex', 'hippocampus', 'thalamus']")
        print("   cell_types = ['excitatory', 'inhibitory']")
        print("   region_sizes = {'cortex': 1000, 'hippocampus': 800}")
        print("   simulator.setup_brain_development_model(brain_regions, cell_types, region_sizes)")
        print("")
        print("4. Run simulation:")
        print("   results = simulator.simulate_brain_growth(duration=100.0)")
        print("")
        print("💡 Best for: Neural circuits, learning, network dynamics")
        
    else:  # Hybrid
        print("🚀 Hybrid Implementation:")
        print("1. Import required modules:")
        print("   from physics_simulation.mujoco_interface import MuJoCoInterface")
        print("   from physics_simulation.dual_mode_simulator import DualModeBrainSimulator")
        print("")
        print("2. Create simulator:")
        print("   mujoco_interface = MuJoCoInterface()")
        print("   simulator = DualModeBrainSimulator(")
        print("       simulation_mode='hybrid',")
        print("       mujoco_interface=mujoco_interface")
        print("   )")
        print("")
        print("3. Setup combined model:")
        print("   brain_regions = ['cortex', 'hippocampus', 'thalamus']")
        print("   cell_types = ['excitatory', 'inhibitory']")
        print("   region_sizes = {'cortex': 500, 'hippocampus': 400}")
        print("   simulator.setup_brain_development_model(brain_regions, cell_types, region_sizes)")
        print("")
        print("4. Run simulation:")
        print("   results = simulator.simulate_brain_growth(duration=2.0)")
        print("")
        print("💡 Best for: Multi-scale effects, physical-neural interactions")

def main():
    """Run the research decision guide"""
    print("🧠 Brain Simulation Approach Decision Guide")
    print("=" * 60)
    print("This guide helps you choose the right simulation approach")
    print("based on your research questions and requirements.")
    print("=" * 60)
    
    # Get user input
    focus = get_research_focus()
    questions = get_research_questions()
    scale = get_research_scale()
    time = get_time_scale()
    
    # Analyze needs
    mujoco_score, nest_score, hybrid_score = analyze_research_needs(focus, questions, scale, time)
    
    # Recommend approach
    recommended = recommend_approach(mujoco_score, nest_score, hybrid_score)
    
    # Provide implementation guide
    provide_implementation_guide(recommended)
    
    print(f"\n🎉 You're all set to use the {recommended} approach!")
    print("\n📚 Additional Resources:")
    print("- Check simulation_approach_guide.md for detailed information")
    print("- Run simple_approach_demo.py to test the approaches")
    print("- Use dual_approach_demo.py for comprehensive demonstrations")

if __name__ == "__main__":
    main()
