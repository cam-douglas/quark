"""
PyVista Interface for 3D Visualization

Modern Python-based 3D visualization alternative to VisIt,
providing interactive brain development visualization capabilities.
"""

import os, sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import json
from pathlib import Path

# Try to import PyVista
try:
    import pyvista as pv
    import vtk
    PYVISTA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("PyVista visualization interface loaded successfully")
except ImportError:
    PYVISTA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyVista not available - creating mock interface")

class PyVistaVisualizer:
    """PyVista-based 3D visualizer for brain development models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.plotter = None
        self.meshes = {}
        self.actors = {}
        
        if PYVISTA_AVAILABLE:
            self._setup_plotter()
        else:
            self._setup_mock()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default PyVista configuration"""
        return {
            "theme": "document",
            "window_size": [1024, 768],
            "background_color": "white",
            "show_axes": True,
            "show_grid": False,
            "anti_aliasing": "fxaa",
            "multi_samples": 4
        }
    
    def _setup_plotter(self):
        """Initialize PyVista plotter"""
        if PYVISTA_AVAILABLE:
            # Set PyVista theme
            pv.set_plot_theme(self.config["theme"])
            
            # Create plotter
            self.plotter = pv.Plotter(
                window_size=self.config["window_size"],
                off_screen=False
            )
            
            # Configure plotter
            self.plotter.set_background(self.config["background_color"])
            self.plotter.show_axes(self.config["show_axes"])
            self.plotter.show_grid(self.config["show_grid"])
            
            logger.info("PyVista plotter initialized")
        else:
            logger.info("Using mock PyVista interface")
    
    def _setup_mock(self):
        """Setup mock PyVista interface for testing"""
        logger.info("Mock PyVista interface ready")
    
    def create_brain_mesh(self, 
                          brain_data: np.ndarray,
                          mesh_name: str = "brain",
                          smoothing: bool = True,
                          decimation: float = 0.5) -> str:
        """
        Create brain mesh from volumetric data
        
        Args:
            brain_data: 3D numpy array of brain data
            mesh_name: Name for the mesh
            smoothing: Apply smoothing to the mesh
            decimation: Decimation factor (0.0 to 1.0)
            
        Returns:
            Mesh ID
        """
        if PYVISTA_AVAILABLE:
            try:
                # Create grid from numpy array
                grid = pv.wrap(brain_data)
                
                # Extract surface
                surface = grid.contour([0.5])
                
                # Apply smoothing if requested
                if smoothing:
                    surface = surface.smooth(n_iter=10, relaxation_factor=0.1)
                
                # Apply decimation if requested
                if decimation < 1.0:
                    target_reduction = 1.0 - decimation
                    surface = surface.decimate(target_reduction)
                
                # Store mesh
                mesh_id = f"{mesh_name}_{len(self.meshes)}"
                self.meshes[mesh_id] = surface
                
                logger.info(f"Created brain mesh '{mesh_id}' with {surface.n_points} points")
                return mesh_id
                
            except Exception as e:
                logger.error(f"Failed to create brain mesh: {e}")
                return None
        else:
            mesh_id = f"{mesh_name}_{len(self.meshes)}"
            self.meshes[mesh_id] = "mock_mesh"
            logger.info(f"Mock: Created brain mesh '{mesh_id}'")
            return mesh_id
    
    def create_neuronal_network(self, 
                               neuron_positions: np.ndarray,
                               connections: np.ndarray,
                               network_name: str = "neurons") -> str:
        """
        Create neuronal network visualization
        
        Args:
            neuron_positions: Nx3 array of neuron positions
            connections: Mx2 array of connection indices
            network_name: Name for the network
            
        Returns:
            Network ID
        """
        if PYVISTA_AVAILABLE:
            try:
                # Create neuron spheres
                neurons = pv.PolyData(neuron_positions)
                neuron_spheres = neurons.glyph(
                    geom=pv.Sphere(radius=0.1),
                    scale=False,
                    orient=False
                )
                
                # Create connection lines
                lines = []
                for conn in connections:
                    start = neuron_positions[conn[0]]
                    end = neuron_positions[conn[1]]
                    line = pv.Line(start, end)
                    lines.append(line)
                
                # Combine all geometries
                network = pv.PolyData()
                network = network.merge(neuron_spheres)
                for line in lines:
                    network = network.merge(line)
                
                # Store network
                network_id = f"{network_name}_{len(self.meshes)}"
                self.meshes[network_id] = network
                
                logger.info(f"Created neuronal network '{network_id}' with {len(neuron_positions)} neurons")
                return network_id
                
            except Exception as e:
                logger.error(f"Failed to create neuronal network: {e}")
                return None
        else:
            network_id = f"{network_name}_{len(self.meshes)}"
            self.meshes[network_id] = "mock_network"
            logger.info(f"Mock: Created neuronal network '{network_id}'")
            return network_id
    
    def add_mesh_to_plot(self, 
                         mesh_id: str,
                         color: str = "blue",
                         opacity: float = 1.0,
                         show_edges: bool = False) -> str:
        """
        Add mesh to the plot
        
        Args:
            mesh_id: ID of the mesh to add
            color: Color of the mesh
            opacity: Opacity (0.0 to 1.0)
            show_edges: Show mesh edges
            
        Returns:
            Actor ID
        """
        if PYVISTA_AVAILABLE and self.plotter and mesh_id in self.meshes:
            try:
                mesh = self.meshes[mesh_id]
                
                # Add mesh to plotter
                actor = self.plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=opacity,
                    show_edges=show_edges,
                    lighting=True
                )
                
                # Store actor
                actor_id = f"{mesh_id}_actor_{len(self.actors)}"
                self.actors[actor_id] = actor
                
                logger.info(f"Added mesh '{mesh_id}' to plot as '{actor_id}'")
                return actor_id
                
            except Exception as e:
                logger.error(f"Failed to add mesh to plot: {e}")
                return None
        else:
            actor_id = f"{mesh_id}_actor_{len(self.actors)}"
            self.actors[actor_id] = "mock_actor"
            logger.info(f"Mock: Added mesh '{mesh_id}' to plot as '{actor_id}'")
            return actor_id
    
    def add_scalar_field(self, 
                         mesh_id: str,
                         scalar_data: np.ndarray,
                         field_name: str = "scalar_field",
                         colormap: str = "viridis") -> str:
        """
        Add scalar field to mesh
        
        Args:
            mesh_id: ID of the mesh
            scalar_data: Scalar values for each point
            field_name: Name of the scalar field
            colormap: Colormap to use
            
        Returns:
            Field ID
        """
        if PYVISTA_AVAILABLE and mesh_id in self.meshes:
            try:
                mesh = self.meshes[mesh_id]
                
                # Add scalar data to mesh
                mesh[field_name] = scalar_data
                
                # Update plotter with new data
                if self.plotter:
                    self.plotter.update_scalars(scalar_data, mesh=mesh)
                
                field_id = f"{mesh_id}_{field_name}"
                logger.info(f"Added scalar field '{field_name}' to mesh '{mesh_id}'")
                return field_id
                
            except Exception as e:
                logger.error(f"Failed to add scalar field: {e}")
                return None
        else:
            field_id = f"{mesh_id}_{field_name}"
            logger.info(f"Mock: Added scalar field '{field_name}' to mesh '{mesh_id}'")
            return field_id
    
    def set_camera_position(self, 
                           position: Tuple[float, float, float],
                           focal_point: Tuple[float, float, float],
                           view_up: Tuple[float, float, float] = (0, 0, 1)):
        """
        Set camera position and orientation
        
        Args:
            position: Camera position (x, y, z)
            focal_point: Point to look at (x, y, z)
            view_up: Up vector (x, y, z)
        """
        if PYVISTA_AVAILABLE and self.plotter:
            try:
                self.plotter.camera_position = [position, focal_point, view_up]
                logger.info(f"Camera positioned at {position}, looking at {focal_point}")
            except Exception as e:
                logger.error(f"Failed to set camera position: {e}")
        else:
            logger.info(f"Mock: Camera positioned at {position}, looking at {focal_point}")
    
    def add_text(self, 
                 text: str,
                 position: Tuple[float, float, float],
                 font_size: int = 12,
                 color: str = "black") -> str:
        """
        Add text to the visualization
        
        Args:
            text: Text to display
            position: Position (x, y, z)
            font_size: Font size
            color: Text color
            
        Returns:
            Text ID
        """
        if PYVISTA_AVAILABLE and self.plotter:
            try:
                actor = self.plotter.add_text(
                    text,
                    position=position,
                    font_size=font_size,
                    color=color
                )
                
                text_id = f"text_{len(self.actors)}"
                self.actors[text_id] = actor
                
                logger.info(f"Added text '{text}' at position {position}")
                return text_id
                
            except Exception as e:
                logger.error(f"Failed to add text: {e}")
                return None
        else:
            text_id = f"text_{len(self.actors)}"
            self.actors[text_id] = "mock_text"
            logger.info(f"Mock: Added text '{text}' at position {position}")
            return text_id
    
    def show(self, interactive: bool = True, screenshot_path: Optional[str] = None):
        """
        Display the visualization
        
        Args:
            interactive: Whether to show interactive window
            screenshot_path: Optional path to save screenshot
        """
        if PYVISTA_AVAILABLE and self.plotter:
            try:
                if screenshot_path:
                    # Take screenshot
                    self.plotter.screenshot(screenshot_path)
                    logger.info(f"Screenshot saved to {screenshot_path}")
                
                if interactive:
                    # Show interactive window
                    self.plotter.show()
                else:
                    # Close plotter
                    self.plotter.close()
                    
            except Exception as e:
                logger.error(f"Failed to show visualization: {e}")
        else:
            logger.info("Mock: Visualization displayed")
    
    def save_mesh(self, mesh_id: str, filepath: str, file_format: str = "vtk"):
        """
        Save mesh to file
        
        Args:
            mesh_id: ID of the mesh to save
            filepath: Path to save the file
            file_format: File format (vtk, stl, obj, ply)
        """
        if PYVISTA_AVAILABLE and mesh_id in self.meshes:
            try:
                mesh = self.meshes[mesh_id]
                
                # Save based on format
                if file_format == "vtk":
                    mesh.save(filepath)
                elif file_format == "stl":
                    mesh.save(filepath, binary=True)
                elif file_format == "obj":
                    mesh.save(filepath)
                elif file_format == "ply":
                    mesh.save(filepath)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
                
                logger.info(f"Mesh '{mesh_id}' saved to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save mesh: {e}")
        else:
            logger.warning(f"Cannot save mesh '{mesh_id}' - PyVista not available")
    
    def get_mesh_info(self, mesh_id: str) -> Dict[str, Any]:
        """Get information about a mesh"""
        if PYVISTA_AVAILABLE and mesh_id in self.meshes:
            mesh = self.meshes[mesh_id]
            return {
                "n_points": mesh.n_points,
                "n_cells": mesh.n_cells,
                "bounds": mesh.bounds,
                "center": mesh.center,
                "volume": mesh.volume if hasattr(mesh, 'volume') else None
            }
        else:
            return {"error": f"Mesh '{mesh_id}' not found or PyVista not available"}
    
    def clear_plot(self):
        """Clear all meshes and actors from the plot"""
        if PYVISTA_AVAILABLE and self.plotter:
            self.plotter.clear()
            self.meshes = {}
            self.actors = {}
            logger.info("Plot cleared")
        else:
            self.meshes = {}
            self.actors = {}
            logger.info("Mock: Plot cleared")
    
    def create_animation(self, 
                        mesh_ids: List[str],
                        time_steps: int = 100,
                        output_path: str = "animation.mp4") -> str:
        """
        Create animation from multiple mesh states
        
        Args:
            mesh_ids: List of mesh IDs representing different time steps
            time_steps: Number of time steps
            output_path: Path to save the animation
            
        Returns:
            Animation file path
        """
        if PYVISTA_AVAILABLE:
            try:
                # Create plotter for animation
                plotter = pv.Plotter()
                plotter.open_movie(output_path)
                
                # Animate through mesh states
                for i, mesh_id in enumerate(mesh_ids):
                    if mesh_id in self.meshes:
                        plotter.clear()
                        plotter.add_mesh(self.meshes[mesh_id])
                        plotter.write_frame()
                
                plotter.close()
                logger.info(f"Animation saved to {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Failed to create animation: {e}")
                return None
        else:
            logger.warning("Cannot create animation - PyVista not available")
            return None
