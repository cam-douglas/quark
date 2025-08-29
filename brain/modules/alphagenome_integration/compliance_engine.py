"""
AlphaGenome Compliance Engine

This engine is responsible for enforcing the biological constraints defined in 
management/rules/biological_constraints.py.

It provides a set of validation functions that other modules, particularly the 
Autonomous Agent, can use to ensure all actions are compliant with the project's
biological and safety rules before execution.
"""
import management.rules.biological_constraints as constraints

class ComplianceEngine:
    """Enforces biological and safety rules for the Quark simulation."""

    def __init__(self):
        """Initializes the Compliance Engine."""
        print("✅ AlphaGenome Compliance Engine Initialized.")

    def validate_action_legality(self, action: str) -> bool:
        """
        Checks if a proposed high-level action is explicitly forbidden.

        Args:
            action: The name of the action to check (e.g., 'disabling_audit_logging').

        Returns:
            True if the action is allowed, False if it is prohibited.
        """
        if action in constraints.PROHIBITED_ACTIONS:
            print(f"❌ COMPLIANCE ERROR: Action '{action}' is strictly prohibited.")
            return False
        return True

    def validate_dna_sequence(self, sequence: str) -> bool:
        """
        Validates a DNA sequence against defined genomic rules.

        Args:
            sequence: The DNA sequence string to validate.

        Returns:
            True if the sequence is valid, otherwise False.
        """
        if not set(sequence).issubset(constraints.VALID_DNA_BASES):
            print("❌ COMPLIANCE ERROR: DNA sequence contains invalid characters.")
            return False
        
        if not (constraints.DNA_SEQUENCE_CONSTRAINTS['min_length'] <= len(sequence) <= constraints.DNA_SEQUENCE_CONSTRAINTS['max_length']):
            print(f"❌ COMPLIANCE ERROR: DNA sequence length ({len(sequence)}) is outside the allowed range.")
            return False
            
        print("✅ DNA sequence is compliant.")
        return True

    def validate_cell_construction(self, cell_type: str, markers: list) -> bool:
        """
        Validates parameters for constructing a new biological cell.

        Args:
            cell_type: The type of cell to be constructed (e.g., 'neuron').
            markers: A list of biological markers to be assigned to the cell.

        Returns:
            True if the construction parameters are valid, otherwise False.
        """
        if cell_type not in constraints.PREDEFINED_CELL_TYPES:
            print(f"❌ COMPLIANCE ERROR: Cell type '{cell_type}' is not a predefined valid type.")
            return False
        
        # Check for mandatory critical markers
        required = set(constraints.REQUIRED_BIOLOGICAL_MARKERS['critical_base'])
        if not required.issubset(set(markers)):
            print(f"❌ COMPLIANCE ERROR: Cell construction is missing critical markers: {required - set(markers)}")
            return False

        print(f"✅ Cell construction parameters for '{cell_type}' are compliant.")
        return True

    def check_simulation_boundaries(self, current_population: int, requested_time_hours: int) -> bool:
        """
        Checks if a simulation request exceeds safety boundaries.

        Args:
            current_population: The number of cells in the simulation.
            requested_time_hours: The requested duration of the simulation in hours.

        Returns:
            True if the simulation is within safe limits, otherwise False.
        """
        if current_population > constraints.SIMULATION_SAFETY_BOUNDARIES['max_cell_population']:
            print(f"❌ COMPLIANCE ERROR: Population ({current_population}) exceeds safety limit.")
            return False
        
        if requested_time_hours > constraints.SIMULATION_SAFETY_BOUNDARIES['max_simulation_time_hours']:
            print(f"❌ COMPLIANCE ERROR: Simulation time ({requested_time_hours}h) exceeds safety limit.")
            return False

        print("✅ Simulation boundaries are respected.")
        return True

# Example Usage (for demonstration and testing)
if __name__ == '__main__':
    engine = ComplianceEngine()
    
    print("\n--- Testing Action Legality ---")
    engine.validate_action_legality("tampering_with_dna_controller_integrity") # Should fail
    engine.validate_action_legality("running_authorized_simulation")      # Should pass
    
    print("\n--- Testing DNA Sequence ---")
    engine.validate_dna_sequence("ATCG" * 50) # Should pass
    engine.validate_dna_sequence("ATCGX")    # Should fail (invalid character)
    engine.validate_dna_sequence("AT")       # Should fail (too short)
    
    print("\n--- Testing Cell Construction ---")
    engine.validate_cell_construction("neuron", ["GFAP", "NeuN", "NSE"]) # Should pass
    engine.validate_cell_construction("glia", ["GFAP", "NeuN"])         # Should fail (invalid type)
    engine.validate_cell_construction("neuron", ["NeuN"])                # Should fail (missing marker)
    
    print("\n--- Testing Simulation Boundaries ---")
    engine.check_simulation_boundaries(500000, 12)  # Should pass
    engine.check_simulation_boundaries(2000000, 12) # Should fail (population)
    engine.check_simulation_boundaries(500000, 48)  # Should fail (time)
