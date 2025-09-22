# Ventricular System Formation: Research and Planning

This document outlines the research and planning for a new, biologically accurate implementation of the ventricular system formation.

## 1. Research Findings

**Article 1 (PMC3003255):** "Totally Tubular: The Mystery behind Function and Origin of the Brain Ventricular System" confirms that the ventricular system develops from the **lumen of the neural tube** through a process of **morphogenesis**.
**Article 2 (PMC2737731):** "A twist of insight - the role of Twist-family bHLH factors in development" highlights the importance of gene regulation and cell differentiation in morphogenesis.

## 2. Proposed Implementation

The new implementation will be based on the biological process of **morphogenesis**. Instead of "excavating" the ventricles, the simulation will start with a hollow neural tube and then apply morphogenetic rules to shape the ventricles. This will involve:
*   **Deprecating the incorrect modules:** `voxel_excavation.py`, `excavation_validator.py`, and `excavation_parameters.py`.
*   **Creating a new `ventricular_morphogenesis.py` module:** This module will be responsible for shaping the ventricular system from the initial neural tube lumen.
*   **Refactoring `ventricular_topology.py`:** This module will be modified to define the initial topology of the neural tube lumen.
*   **Refactoring `csf_flow_dynamics.py`:** This module will be updated to work with the new `ventricular_morphogenesis.py` module.

## 3. Action Plan

1.  **Deprecate the incorrect modules:** Delete `voxel_excavation.py`, `excavation_validator.py`, and `excavation_parameters.py`.
2.  **Create a new `ventricular_morphogenesis.py` module:** This module will contain the logic for shaping the ventricular system.
3.  **Refactor `ventricular_topology.py`:** Modify the `VentricularTopology` class to define the initial neural tube lumen.
4.  **Refactor `csf_flow_dynamics.py`:** Update the `CSFFlowDynamics` class to use the new `VentricularMorphogenesis` module.
5.  **Test the new implementation:** Write unit tests for the new and refactored modules.