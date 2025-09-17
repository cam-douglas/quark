#!/usr/bin/env python3
"""Atlas Validation Metrics System.

Implements validation metrics for comparing morphogen solver segmentation
against Allen Brain Atlas reference including Dice coefficient, Hausdorff
distance, and regional boundary accuracy assessment.

Integration: Validation metrics component for atlas validation system
Rationale: Focused validation metrics separated from main validation coordinator
"""

from typing import Dict, List
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
import logging

from .atlas_validation_types import ValidationResult, ValidationMetric

logger = logging.getLogger(__name__)

class AtlasValidationMetrics:
    """Validation metrics calculator for atlas comparison.
    
    Computes comprehensive validation metrics including Dice coefficient,
    Hausdorff distance, Jaccard index, and surface distance measurements
    for comparing morphogen segmentation with atlas reference.
    """
    
    def __init__(self, dice_threshold: float = 0.80):
        """Initialize validation metrics calculator.
        
        Args:
            dice_threshold: Target Dice coefficient threshold
        """
        self.dice_threshold = dice_threshold
        
        logger.info("Initialized AtlasValidationMetrics")
        logger.info(f"Dice threshold: {dice_threshold}")
    
    def calculate_dice_coefficient(self, predicted: np.ndarray, 
                                  reference: np.ndarray) -> ValidationResult:
        """Calculate Dice coefficient for segmentation comparison.
        
        Args:
            predicted: Predicted segmentation labels
            reference: Reference atlas labels
            
        Returns:
            Validation result with Dice coefficient
        """
        logger.info("Calculating Dice coefficient")
        
        # Get unique labels (excluding background)
        unique_labels = np.unique(reference)
        unique_labels = unique_labels[unique_labels > 0]
        
        region_dice_scores = {}
        dice_scores = []
        
        for label in unique_labels:
            # Create binary masks for this region
            pred_mask = (predicted == label)
            ref_mask = (reference == label)
            
            # Calculate Dice coefficient
            intersection = np.sum(pred_mask & ref_mask)
            union = np.sum(pred_mask) + np.sum(ref_mask)
            
            if union > 0:
                dice_score = 2.0 * intersection / union
            else:
                dice_score = 1.0 if np.sum(pred_mask) == 0 else 0.0
            
            region_dice_scores[f"region_{label}"] = dice_score
            dice_scores.append(dice_score)
        
        # Overall Dice coefficient
        overall_dice = np.mean(dice_scores) if dice_scores else 0.0
        validation_passed = overall_dice >= self.dice_threshold
        
        result = ValidationResult(
            metric_type=ValidationMetric.DICE_COEFFICIENT,
            metric_value=overall_dice,
            target_threshold=self.dice_threshold,
            validation_passed=validation_passed,
            region_specific_scores=region_dice_scores,
            overall_score=overall_dice
        )
        
        logger.info(f"Dice coefficient: {overall_dice:.3f} (threshold: {self.dice_threshold})")
        logger.info(f"Validation: {'PASS' if validation_passed else 'FAIL'}")
        
        return result
    
    def calculate_hausdorff_distance(self, predicted: np.ndarray, 
                                    reference: np.ndarray) -> ValidationResult:
        """Calculate Hausdorff distance for boundary accuracy.
        
        Args:
            predicted: Predicted segmentation labels
            reference: Reference atlas labels
            
        Returns:
            Validation result with Hausdorff distance
        """
        logger.info("Calculating Hausdorff distance")
        
        # Get unique labels
        unique_labels = np.unique(reference)
        unique_labels = unique_labels[unique_labels > 0]
        
        region_hausdorff_scores = {}
        hausdorff_distances = []
        
        for label in unique_labels:
            # Get boundary points for this region
            pred_boundary = self._extract_boundary_points(predicted == label)
            ref_boundary = self._extract_boundary_points(reference == label)
            
            if len(pred_boundary) > 0 and len(ref_boundary) > 0:
                # Calculate directed Hausdorff distances
                h1 = directed_hausdorff(pred_boundary, ref_boundary)[0]
                h2 = directed_hausdorff(ref_boundary, pred_boundary)[0]
                
                # Hausdorff distance is the maximum of directed distances
                hausdorff_dist = max(h1, h2)
            else:
                hausdorff_dist = 0.0 if len(pred_boundary) == len(ref_boundary) == 0 else float('inf')
            
            region_hausdorff_scores[f"region_{label}"] = hausdorff_dist
            if hausdorff_dist != float('inf'):
                hausdorff_distances.append(hausdorff_dist)
        
        # Overall Hausdorff distance (mean of finite distances)
        overall_hausdorff = np.mean(hausdorff_distances) if hausdorff_distances else 0.0
        
        # Lower Hausdorff distance is better (threshold: <10 voxels)
        hausdorff_threshold = 10.0
        validation_passed = overall_hausdorff < hausdorff_threshold
        
        # Convert to score (0-1, higher is better)
        hausdorff_score = max(0.0, 1.0 - overall_hausdorff / hausdorff_threshold)
        
        result = ValidationResult(
            metric_type=ValidationMetric.HAUSDORFF_DISTANCE,
            metric_value=overall_hausdorff,
            target_threshold=hausdorff_threshold,
            validation_passed=validation_passed,
            region_specific_scores=region_hausdorff_scores,
            overall_score=hausdorff_score
        )
        
        logger.info(f"Hausdorff distance: {overall_hausdorff:.2f} voxels")
        logger.info(f"Validation: {'PASS' if validation_passed else 'FAIL'}")
        
        return result
    
    def _extract_boundary_points(self, binary_mask: np.ndarray) -> np.ndarray:
        """Extract boundary points from binary mask."""
        # Find boundary using morphological operations
        eroded = ndimage.binary_erosion(binary_mask)
        boundary = binary_mask & ~eroded
        
        # Get coordinates of boundary points
        boundary_coords = np.array(np.where(boundary)).T
        
        return boundary_coords
    
    def calculate_jaccard_index(self, predicted: np.ndarray, 
                               reference: np.ndarray) -> ValidationResult:
        """Calculate Jaccard index (IoU) for segmentation comparison."""
        logger.info("Calculating Jaccard index")
        
        unique_labels = np.unique(reference)
        unique_labels = unique_labels[unique_labels > 0]
        
        region_jaccard_scores = {}
        jaccard_scores = []
        
        for label in unique_labels:
            pred_mask = (predicted == label)
            ref_mask = (reference == label)
            
            # Calculate Jaccard index
            intersection = np.sum(pred_mask & ref_mask)
            union = np.sum(pred_mask | ref_mask)
            
            if union > 0:
                jaccard_score = intersection / union
            else:
                jaccard_score = 1.0 if np.sum(pred_mask) == 0 else 0.0
            
            region_jaccard_scores[f"region_{label}"] = jaccard_score
            jaccard_scores.append(jaccard_score)
        
        # Overall Jaccard index
        overall_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
        jaccard_threshold = 0.7  # Threshold for Jaccard index
        validation_passed = overall_jaccard >= jaccard_threshold
        
        result = ValidationResult(
            metric_type=ValidationMetric.JACCARD_INDEX,
            metric_value=overall_jaccard,
            target_threshold=jaccard_threshold,
            validation_passed=validation_passed,
            region_specific_scores=region_jaccard_scores,
            overall_score=overall_jaccard
        )
        
        logger.info(f"Jaccard index: {overall_jaccard:.3f}")
        
        return result
    
    def calculate_comprehensive_validation(self, predicted: np.ndarray,
                                         reference: np.ndarray) -> Dict[str, ValidationResult]:
        """Calculate comprehensive validation metrics.
        
        Args:
            predicted: Predicted segmentation
            reference: Reference atlas segmentation
            
        Returns:
            Dictionary with all validation results
        """
        logger.info("Calculating comprehensive validation metrics")
        
        validation_results = {}
        
        # Dice coefficient
        validation_results["dice"] = self.calculate_dice_coefficient(predicted, reference)
        
        # Hausdorff distance
        validation_results["hausdorff"] = self.calculate_hausdorff_distance(predicted, reference)
        
        # Jaccard index
        validation_results["jaccard"] = self.calculate_jaccard_index(predicted, reference)
        
        # Overall validation score
        overall_score = np.mean([
            result.overall_score for result in validation_results.values()
        ])
        
        overall_passed = all(result.validation_passed for result in validation_results.values())
        
        logger.info(f"Comprehensive validation: score={overall_score:.3f}, "
                   f"passed={'YES' if overall_passed else 'NO'}")
        
        return validation_results
