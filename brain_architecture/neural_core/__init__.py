"""
🧠 Brain Modules Package - Core Neural Components

This package contains the 7 core brain modules following biological brain architecture:

Brain Modules:
- prefrontal_cortex/: Executive control, planning, reasoning
- basal_ganglia/: Action selection, reinforcement learning  
- thalamus/: Information relay, attentional modulation
- working_memory/: Short-term memory buffers
- hippocampus/: Episodic memory, pattern completion
- default_mode_network/: Internal simulation, self-reflection
- salience_networks/: Attention allocation, novelty detection
- conscious_agent/: Main conscious agent with different versions
- connectome/: Neural connectivity management
"""

from . import prefrontal_cortex
from . import basal_ganglia
from . import thalamus
from . import working_memory
from . import hippocampus
from . import default_mode_network
from . import salience_networks
from . import conscious_agent
from . import connectome

__all__ = [
    "prefrontal_cortex",
    "basal_ganglia", 
    "thalamus",
    "working_memory",
    "hippocampus",
    "default_mode_network",
    "salience_networks",
    "conscious_agent",
    "connectome"
]






