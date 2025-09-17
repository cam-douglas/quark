"""
Brain Context Provider - Provides brain architecture context for Cline

Manages brain state information, biological constraints, and morphogen system
status for context-aware autonomous coding with biological compliance.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .cline_types import BrainContext


class BrainContextProvider:
    """
    Provides comprehensive brain architecture context for Cline integration
    
    Manages brain state information, caching, and biological constraint
    validation for autonomous coding tasks.
    """
    
    def __init__(self, workspace_path: Optional[Path] = None):
        """Initialize brain context provider"""
        self.workspace_path = workspace_path or Path("/Users/camdouglas/quark")
        self.logger = logging.getLogger(__name__)
        
        # Context caching
        self._brain_context_cache: Optional[BrainContext] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl = 300  # 5 minutes
        
        self.logger.info("Brain context provider initialized")
    
    async def get_brain_context(self) -> BrainContext:
        """
        Get comprehensive brain architecture context
        
        Returns:
            BrainContext containing current brain state and constraints
        """
        current_time = time.time()
        
        # Use cached context if still valid
        if (self._brain_context_cache and self._cache_timestamp and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._brain_context_cache
        
        # Load fresh brain context
        context = BrainContext(
            current_phase=await self._get_current_phase(),
            active_modules=await self._get_active_modules(),
            neural_architecture=await self._get_neural_architecture_info(),
            biological_constraints=await self._get_biological_constraints(),
            morphogen_status=await self._get_morphogen_status(),
            foundation_layer_status=await self._get_foundation_layer_status(),
            compliance_rules=await self._get_compliance_rules()
        )
        
        # Cache the context
        self._brain_context_cache = context
        self._cache_timestamp = current_time
        
        return context
    
    def invalidate_cache(self) -> None:
        """Invalidate brain context cache"""
        self._brain_context_cache = None
        self._cache_timestamp = None
        self.logger.info("Brain context cache invalidated")
    
    # Brain context helper methods
    async def _get_current_phase(self) -> str:
        """Get current development phase"""
        try:
            foundation_tasks_path = self.workspace_path / "state/tasks/roadmap_tasks/foundation_layer_detailed_tasks.md"
            if foundation_tasks_path.exists():
                content = foundation_tasks_path.read_text()
                if "SHH SYSTEM COMPLETED" in content:
                    return "Foundation Layer - SHH System Complete"
            return "Foundation Layer - In Progress"
        except Exception as e:
            self.logger.error(f"Error getting current phase: {e}")
            return "Unknown Phase"

    async def _get_active_modules(self) -> List[str]:
        """Get list of active brain modules"""
        try:
            modules_path = self.workspace_path / "brain/modules"
            if modules_path.exists():
                return [d.name for d in modules_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            return []
        except Exception as e:
            self.logger.error(f"Error getting active modules: {e}")
            return []

    async def _get_neural_architecture_info(self) -> Dict[str, Any]:
        """Get neural architecture information"""
        return {
            "type": "embryonic_neural_tube",
            "resolution": "1µm³",
            "morphogen_systems": ["SHH", "BMP", "WNT", "FGF"],
            "spatial_grid": "3D voxel system",
            "biological_compliance": "AlphaGenome integrated"
        }

    async def _get_biological_constraints(self) -> Dict[str, Any]:
        """Get current biological constraints"""
        return {
            "alphagenome_enabled": True,
            "developmental_stage": "neural_tube_closure",
            "cell_types_active": ["neural_stem_cell", "neuroblast", "neural_crest"],
            "prohibited_patterns": ["negative_emotion", "harmful_behavior"],
            "file_size_limit": 300,  # lines
            "neuroanatomical_naming": True
        }

    async def _get_morphogen_status(self) -> Dict[str, str]:
        """Get morphogen system status"""
        return {
            "shh_system": "completed",
            "bmp_system": "in_progress", 
            "wnt_system": "planned",
            "fgf_system": "planned"
        }

    async def _get_foundation_layer_status(self) -> Dict[str, Any]:
        """Get foundation layer development status"""
        return {
            "phase": "Phase 2 - Single Morphogen Implementation",
            "completion": "15 of 19 tasks completed",
            "shh_gradient_system": "fully implemented",
            "spatial_grid": "1µm³ resolution active",
            "biological_parameters": "refactored and validated",
            "cell_fate_specification": "completed"
        }

    async def _get_compliance_rules(self) -> List[str]:
        """Get active compliance rules"""
        return [
            "All modules must be <300 lines (architecture rule)",
            "Follow neuroanatomical naming conventions", 
            "Respect developmental stage constraints",
            "Validate against AlphaGenome biological rules",
            "Maintain morphogen solver biological accuracy",
            "No negative emotions in brain modules",
            "Focused responsibilities with clean coordinator patterns"
        ]
    
    def get_context_for_task(self, task_description: str) -> Dict[str, Any]:
        """
        Get specific brain context relevant to a task
        
        Args:
            task_description: Description of the task
            
        Returns:
            Dictionary with relevant brain context
        """
        # This would be called synchronously, but handles async internally
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            brain_context = loop.run_until_complete(self.get_brain_context())
        except RuntimeError:
            # If no event loop is running, create one
            brain_context = asyncio.run(self.get_brain_context())
        
        # Extract relevant context based on task content
        task_lower = task_description.lower()
        relevant_context = {
            "current_phase": brain_context.current_phase,
            "biological_constraints": brain_context.biological_constraints,
            "compliance_rules": brain_context.compliance_rules
        }
        
        # Add morphogen-specific context if relevant
        if any(term in task_lower for term in ["morphogen", "gradient", "bmp", "wnt", "fgf", "shh"]):
            relevant_context["morphogen_status"] = brain_context.morphogen_status
            relevant_context["foundation_layer_status"] = brain_context.foundation_layer_status
        
        # Add architecture context if relevant
        if any(term in task_lower for term in ["architecture", "neural", "brain", "module"]):
            relevant_context["neural_architecture"] = brain_context.neural_architecture
            relevant_context["active_modules"] = brain_context.active_modules
        
        return relevant_context
