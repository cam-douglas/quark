import sys
import os
import json

# Add the root directory to the Python path to allow for imports from our project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from tasks.integrations.biological_brain_task_integration import BiologicalBrainTaskIntegration
from brain_architecture.neural_core.prefrontal_cortex.executive_control import ExecutiveControl

def test_goal_integration_pipeline():
    """
    An end-to-end test to validate the entire goal integration pipeline.
    1. Runs the task analysis.
    2. Ingests the analysis into the PFC.
    3. Runs a cognitive cycle to show goal-driven motivation and self-validation.
    """
    print("--- 🧪 STARTING GOAL INTEGRATION PIPELINE TEST 🧪 ---")

    # --- Step 1: Run the Task Analysis ---
    print("\n[Step 1] Running the Biological Brain Task Integration analysis...")
    try:
        task_integrator = BiologicalBrainTaskIntegration()
        tasks = task_integrator.load_central_task_system()
        brain_analysis = task_integrator.analyze_tasks_for_brain()
        task_integrator.send_brain_analysis_to_tasks(brain_analysis)
        print(f"✅ Analysis complete. {len(tasks)} tasks processed and saved.")
    except Exception as e:
        print(f"❌ ERROR in Step 1: {e}")
        return

    # --- Step 2: Initialize the Prefrontal Cortex (Executive Control) ---
    print("\n[Step 2] Initializing the Executive Control module (PFC)...")
    try:
        pfc = ExecutiveControl()
        print("✅ PFC initialized.")
    except Exception as e:
        print(f"❌ ERROR in Step 2: {e}")
        return

    # --- Step 3: Ingest Goals from Task Analysis ---
    print("\n[Step 3] PFC is ingesting goals from the task analysis...")
    try:
        pfc.ingest_task_analysis()
        dev_plans = [p for p in pfc.plans if p.goal.startswith("DEV_TASK:")]
        validation_plans = [p for p in pfc.plans if p.goal.startswith("SELF_VALIDATE:")]
        
        assert len(dev_plans) > 0, "No development plans were created."
        assert len(validation_plans) > 0, "The self-validation plan was not created."
        
        print(f"✅ PFC successfully ingested goals.")
        print(f"   - Found {len(dev_plans)} development task plans.")
        print(f"   - Found {len(validation_plans)} self-validation plan.")
        
    except Exception as e:
        print(f"❌ ERROR in Step 3: {e}")
        return

    # --- Step 4: Run a Cognitive Cycle ---
    print("\n[Step 4] Running a single cognitive cycle in the PFC...")
    try:
        # We pass an empty dictionary as input for this test
        pfc.step({})
        
        # Verify that the motivational bias was set
        motivational_bias = pfc.dopamine_system.motivational_bias
        assert motivational_bias > 0, "Motivational bias was not set correctly for high-priority tasks."
        print(f"✅ Goal-driven motivation confirmed: Dopamine system motivational bias is set to {motivational_bias:.2f}.")

        # Verify that the validation was run
        validation_results = pfc.validator.validation_results
        assert "neural_bench" in validation_results, "Scientific validation was not triggered."
        print(f"✅ Self-validation confirmed: '{validation_results['neural_bench']['benchmark']}' benchmark was run.")
        print(f"   - Achieved a score of {validation_results['neural_bench']['score']:.2f}")

    except Exception as e:
        print(f"❌ ERROR in Step 4: {e}")
        return

    print("\n--- ✅ PIPELINE TEST COMPLETED SUCCESSFULLY ✅ ---")

if __name__ == '__main__':
    test_goal_integration_pipeline()
