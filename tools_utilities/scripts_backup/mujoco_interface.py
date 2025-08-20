"""
MuJoCo Physics Interface for SmallMind

Provides a high-level interface to MuJoCo physics engine for:
- Brain development simulation
- Tissue mechanics
- Morphogen diffusion
- Biomechanical modeling
"""

import mujoco
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class MuJoCoInterface:
    """High-level interface to MuJoCo physics engine"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize MuJoCo interface
        
        Args:
            model_path: Path to MuJoCo XML model file
        """
        self.model = None
        self.data = None
        self.model_path = model_path
        self.simulation_time = 0.0
        self.dt = 0.01  # Default timestep
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a MuJoCo model from XML file
        
        Args:
            model_path: Path to MuJoCo XML model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            self.model_path = model_path
            logger.info(f"Loaded MuJoCo model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load MuJoCo model: {e}")
            return False
    
    def create_brain_development_model(self, 
                                     brain_regions: List[str],
                                     cell_types: List[str]) -> str:
        """
        Create a MuJoCo model for brain development simulation
        
        Args:
            brain_regions: List of brain region names
            cell_types: List of cell type names
            
        Returns:
            Path to created XML model file
        """
        xml_content = self._generate_brain_xml(brain_regions, cell_types)
        
        # Create models directory if it doesn't exist
        models_dir = Path("models/mujoco")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "brain_development.xml"
        with open(model_path, 'w') as f:
            f.write(xml_content)
        
        logger.info(f"Created brain development model at {model_path}")
        return str(model_path)
    
    def _generate_brain_xml(self, brain_regions: List[str], 
                           cell_types: List[str]) -> str:
        """Generate MuJoCo XML for brain development simulation"""
        
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="brain_development">
  <compiler angle="radian" coordinate="local"/>
  
  <worldbody>
    <!-- Brain regions -->
"""
        
        # Add brain regions
        for i, region in enumerate(brain_regions):
            x, y, z = i * 0.3, 0, 0
            xml_content += f"""    <body name="{region}" pos="{x} {y} {z}">
      <geom type="sphere" size="0.1" rgba="0.8 0.8 1.0 0.7"/>
      <joint type="free"/>
    </body>
"""
        
        # Add neural cells
        for i, cell_type in enumerate(cell_types):
            x, y, z = (i % 3) * 0.1, (i // 3) * 0.1, 0.2
            xml_content += f"""    <body name="{cell_type}_cell_{i}" pos="{x} {y} {z}">
      <geom type="sphere" size="0.02" rgba="0.2 0.8 0.2 0.8"/>
      <joint type="free"/>
    </body>
"""
        
        xml_content += """  </worldbody>
</mujoco>"""
        
        return xml_content
    
    def step_simulation(self, steps: int = 1) -> Dict:
        """
        Step the simulation forward
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            Dictionary with simulation state
        """
        if not self.model or not self.data:
            raise RuntimeError("No model loaded")
        
        results = {
            'time': [],
            'positions': [],
            'velocities': [],
            'forces': []
        }
        
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self.simulation_time += self.dt
            
            # Collect simulation data
            results['time'].append(self.simulation_time)
            results['positions'].append(self.data.qpos.copy())
            results['velocities'].append(self.data.qvel.copy())
            results['forces'].append(self.data.qfrc_applied.copy())
        
        return results
    
    def get_brain_region_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all brain regions"""
        if not self.model or not self.data:
            return {}
        
        positions = {}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and any(region in body_name for region in ['cortex', 'hippocampus', 'amygdala', 'thalamus', 'cerebellum']):
                positions[body_name] = self.data.xpos[i].copy()
        
        return positions
    
    def apply_growth_force(self, region_name: str, force: np.ndarray):
        """Apply growth force to a specific brain region"""
        if not self.model or not self.data:
            return
        
        # Find the body index for the region
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, region_name)
        if body_id >= 0:
            # Apply force to the body
            self.data.xfrc_applied[body_id, :3] = force[:3]
            self.data.xfrc_applied[body_id, 3:6] = force[3:6]
    
    def set_morphogen_concentration(self, region_name: str, concentration: float):
        """Set morphogen concentration for a brain region"""
        if not self.model or not self.data:
            return
        
        # This would typically involve modifying material properties
        # or adding custom attributes to track morphogen levels
        logger.info(f"Setting morphogen concentration for {region_name}: {concentration}")
    
    def get_simulation_stats(self) -> Dict:
        """Get current simulation statistics"""
        if not self.model or not self.data:
            return {}
        
        return {
            'simulation_time': self.simulation_time,
            'num_bodies': self.model.nbody,
            'num_joints': self.model.njnt,
            'num_actuators': self.model.nu,
            'kinetic_energy': self.data.energy[0],
            'potential_energy': self.data.energy[1],
            'total_energy': self.data.energy[0] + self.data.energy[1]
        }
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        if self.model and self.data:
            mujoco.mj_resetData(self.model, self.data)
            self.simulation_time = 0.0
            logger.info("Simulation reset to initial state")
    
    def save_simulation_state(self, filepath: str):
        """Save current simulation state to file"""
        if not self.data:
            return
        
        state_data = {
            'time': self.simulation_time,
            'positions': self.data.qpos.tolist(),
            'velocities': self.data.qvel.tolist(),
            'forces': self.data.qfrc_applied.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"Simulation state saved to {filepath}")
    
    def load_simulation_state(self, filepath: str):
        """Load simulation state from file"""
        if not self.data:
            return
        
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.simulation_time = state_data['time']
        self.data.qpos[:] = np.array(state_data['positions'])
        self.data.qvel[:] = np.array(state_data['velocities'])
        self.data.qfrc_applied[:] = np.array(state_data['forces'])
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
        
        logger.info(f"Simulation state loaded from {filepath}")
    
    def close(self):
        """Clean up MuJoCo resources"""
        if self.model:
            del self.model
        if self.data:
            del self.data
        logger.info("MuJoCo interface closed")
