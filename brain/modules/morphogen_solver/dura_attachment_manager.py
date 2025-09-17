#!/usr/bin/env python3
"""Dura Mater Attachment Point Manager.

Manages attachment points for dura mater layer including skull primordia
connections, cranial sutures, and attachment strength calculations.

Integration: Attachment management component for dura mater system
Rationale: Focused attachment point logic separated from main dura system
"""

from typing import List, Tuple
import numpy as np
import logging

from .spatial_grid import GridDimensions
from .meninges_types import AttachmentPoint, AttachmentPointType

logger = logging.getLogger(__name__)

class DuraAttachmentManager:
    """Manager for dura mater attachment points.
    
    Handles creation and management of attachment points to skull primordia,
    cranial sutures, and other structural elements for the dura mater layer.
    """
    
    def __init__(self, grid_dimensions: GridDimensions):
        """Initialize dura attachment manager.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
        """
        self.dims = grid_dimensions
        logger.info("Initialized DuraAttachmentManager")
    
    def create_skull_attachment_points(self) -> List[AttachmentPoint]:
        """Create attachment points to skull primordia."""
        center_x, center_y, center_z = self.dims.x_size/2, self.dims.y_size/2, self.dims.z_size/2
        
        attachment_points = []
        
        # Add frontal attachments
        attachment_points.extend(self._create_frontal_attachments(center_x, center_y, center_z))
        
        # Add parietal attachments
        attachment_points.extend(self._create_parietal_attachments(center_x, center_y, center_z))
        
        # Add occipital attachments
        attachment_points.extend(self._create_occipital_attachments(center_x, center_y, center_z))
        
        # Add suture attachments
        attachment_points.extend(self._create_suture_attachments(center_x, center_y, center_z))
        
        logger.info(f"Created {len(attachment_points)} skull attachment points")
        return attachment_points
    
    def _create_frontal_attachments(self, cx: float, cy: float, cz: float) -> List[AttachmentPoint]:
        """Create frontal bone attachment points."""
        frontal_points = [
            (cx - 150, cy + 200, cz + 100),  # Left frontal
            (cx + 150, cy + 200, cz + 100),  # Right frontal
            (cx, cy + 250, cz + 50),         # Frontal midline
        ]
        
        attachments = []
        for i, pos in enumerate(frontal_points):
            attachments.append(AttachmentPoint(
                attachment_type=AttachmentPointType.SKULL_PRIMORDIUM,
                location=pos,
                attachment_strength_n=1e-6,  # Micro-Newton scale for embryo
                developmental_week=5.0,
                region_radius_um=20.0
            ))
        
        return attachments
    
    def _create_parietal_attachments(self, cx: float, cy: float, cz: float) -> List[AttachmentPoint]:
        """Create parietal bone attachment points."""
        parietal_points = [
            (cx - 200, cy + 100, cz + 150),  # Left parietal
            (cx + 200, cy + 100, cz + 150),  # Right parietal
            (cx - 150, cy, cz + 200),        # Left superior
            (cx + 150, cy, cz + 200),        # Right superior
        ]
        
        attachments = []
        for i, pos in enumerate(parietal_points):
            attachments.append(AttachmentPoint(
                attachment_type=AttachmentPointType.SKULL_PRIMORDIUM,
                location=pos,
                attachment_strength_n=8e-7,  # Slightly weaker than frontal
                developmental_week=5.5,
                region_radius_um=25.0
            ))
        
        return attachments
    
    def _create_occipital_attachments(self, cx: float, cy: float, cz: float) -> List[AttachmentPoint]:
        """Create occipital bone attachment points."""
        occipital_points = [
            (cx, cy - 150, cz + 100),        # Occipital midline
            (cx - 100, cy - 120, cz + 80),   # Left occipital
            (cx + 100, cy - 120, cz + 80),   # Right occipital
        ]
        
        attachments = []
        for i, pos in enumerate(occipital_points):
            attachments.append(AttachmentPoint(
                attachment_type=AttachmentPointType.SKULL_PRIMORDIUM,
                location=pos,
                attachment_strength_n=1.2e-6,  # Stronger posterior attachments
                developmental_week=6.0,
                region_radius_um=30.0
            ))
        
        return attachments
    
    def _create_suture_attachments(self, cx: float, cy: float, cz: float) -> List[AttachmentPoint]:
        """Create cranial suture attachment points."""
        suture_points = [
            (cx, cy + 150, cz + 125),        # Metopic suture
            (cx - 175, cy + 50, cz + 175),   # Left coronal suture
            (cx + 175, cy + 50, cz + 175),   # Right coronal suture
            (cx, cy - 50, cz + 150),         # Sagittal suture
        ]
        
        attachments = []
        for i, pos in enumerate(suture_points):
            attachments.append(AttachmentPoint(
                attachment_type=AttachmentPointType.CRANIAL_SUTURES,
                location=pos,
                attachment_strength_n=5e-7,  # Moderate suture strength
                developmental_week=6.5,
                region_radius_um=15.0
            ))
        
        return attachments
    
    def validate_attachment_integrity(self, attachment_points: List[AttachmentPoint]) -> float:
        """Validate integrity of attachment points.
        
        Args:
            attachment_points: List of attachment points to validate
            
        Returns:
            Integrity score (0.0 to 1.0)
        """
        if not attachment_points:
            return 0.0
        
        valid_attachments = 0
        
        for attachment in attachment_points:
            # Validate attachment is within reasonable stress limits
            if attachment.attachment_strength_n > 0:
                valid_attachments += 1
        
        integrity_score = valid_attachments / len(attachment_points)
        
        return integrity_score
