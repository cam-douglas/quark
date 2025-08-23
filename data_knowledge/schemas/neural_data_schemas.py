from pydantic import BaseModel, Field
from typing import List, Dict

# --- Core Biological Primitives ---

class Neuron(BaseModel):
    """Represents a single neuron with its essential properties."""
    id: str = Field(..., description="Unique identifier for the neuron.")
    neuron_type: str = Field(..., description="Type of neuron (e.g., 'pyramidal', 'interneuron').")
    position: tuple[float, float, float] = Field(..., description="3D coordinates (x, y, z) in the brain space.")
    properties: Dict[str, float] = Field({}, description="Additional properties like firing threshold, membrane potential, etc.")

class Synapse(BaseModel):
    """Represents a synapse connecting two neurons."""
    source_neuron_id: str = Field(..., description="ID of the presynaptic neuron.")
    target_neuron_id: str = Field(..., description="ID of the postsynaptic neuron.")
    weight: float = Field(..., description="Synaptic weight, determining the connection strength.")
    plasticity_rule: str = Field("static", description="The plasticity rule governing this synapse (e.g., 'STDP', 'static').")

# --- Data Structures for Neural Activity ---

class SpikeTrain(BaseModel):
    """Represents the spike train of a single neuron."""
    neuron_id: str = Field(..., description="Identifier of the neuron that generated the spikes.")
    timestamps: List[float] = Field(..., description="List of spike times in milliseconds.")

class LocalFieldPotential(BaseModel):
    """Represents the LFP recorded from a specific brain region."""
    electrode_id: str = Field(..., description="Identifier for the recording electrode or region.")
    timestamps: List[float] = Field(..., description="List of timestamps for each LFP measurement.")
    voltages: List[float] = Field(..., description="List of corresponding voltage measurements.")

# --- Data Structures for Connectivity ---

class Connectome(BaseModel):
    """Represents the connectome of a brain or a specific region."""
    neurons: List[Neuron] = Field(..., description="List of all neurons in the connectome.")
    synapses: List[Synapse] = Field(..., description="List of all synapses connecting the neurons.")

# --- Data Structures for Molecular Data ---

class GeneExpression(BaseModel):
    """Represents the gene expression levels for a neuron or region."""
    target_id: str = Field(..., description="Identifier of the neuron or brain region.")
    expression_levels: Dict[str, float] = Field(..., description="Dictionary mapping gene names to their expression levels.")

class NeurotransmitterConcentration(BaseModel):
    """Represents neurotransmitter concentrations in a specific region."""
    region_id: str = Field(..., description="Identifier for the brain region.")
    concentrations: Dict[str, float] = Field(..., description="Dictionary mapping neurotransmitters (e.g., 'dopamine') to their concentrations.")
