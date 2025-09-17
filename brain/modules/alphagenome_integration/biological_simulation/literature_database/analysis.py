#!/usr/bin/env python3
"""Literature Parameter Analysis - Statistical analysis and validation of morphogen parameters.

This module provides statistical analysis tools for morphogen parameters collected from
literature, including parameter validation, statistical summaries, and expert comparison.

Integration: Analysis layer for literature_database parameter validation.
Rationale: Centralized statistical analysis with biological parameter validation.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from .types import (
    Parameter, MorphogenType, ParameterType, ConfidenceLevel, 
    validate_parameter_value, get_standard_units
)
from .database import ParameterDatabase

class ParameterAnalyzer:
    """Analyzes morphogen parameters for statistical validation and expert review."""
    
    def __init__(self, database: ParameterDatabase):
        """Initialize analyzer with parameter database."""
        self.database = database
    
    def get_parameter_statistics(self, morphogen: MorphogenType, 
                               parameter_type: ParameterType) -> Dict[str, float]:
        """Get comprehensive statistical summary for parameter across all measurements."""
        parameters = self.database.get_parameters_by_morphogen(morphogen, parameter_type)
        
        if not parameters:
            return {
                "error": "No parameters found",
                "count": 0
            }
        
        values = [p.value for p in parameters]
        high_confidence_values = [
            p.value for p in parameters 
            if p.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
        ]
        expert_validated_values = [
            p.value for p in parameters 
            if p.expert_validated
        ]
        
        stats = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf'),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "high_confidence_count": len(high_confidence_values),
            "expert_validated_count": len(expert_validated_values),
        }
        
        # High confidence statistics
        if high_confidence_values:
            stats.update({
                "high_confidence_mean": float(np.mean(high_confidence_values)),
                "high_confidence_std": float(np.std(high_confidence_values)),
                "high_confidence_cv": float(np.std(high_confidence_values) / np.mean(high_confidence_values))
            })
        
        # Expert validated statistics
        if expert_validated_values:
            stats.update({
                "expert_validated_mean": float(np.mean(expert_validated_values)),
                "expert_validated_std": float(np.std(expert_validated_values)),
                "expert_validated_cv": float(np.std(expert_validated_values) / np.mean(expert_validated_values))
            })
        
        return stats
    
    def identify_parameter_outliers(self, morphogen: MorphogenType, 
                                  parameter_type: ParameterType, 
                                  threshold: float = 2.0) -> List[str]:
        """Identify parameter outliers using z-score analysis."""
        parameters = self.database.get_parameters_by_morphogen(morphogen, parameter_type)
        
        if len(parameters) < 3:
            return []  # Need minimum 3 values for meaningful outlier detection
        
        values = np.array([p.value for p in parameters])
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []  # No variation in values
        
        z_scores = np.abs((values - mean_val) / std_val)
        outlier_indices = np.where(z_scores > threshold)[0]
        
        return [parameters[i].parameter_id for i in outlier_indices]
    
    def validate_parameter_biological_constraints(self, parameter: Parameter) -> Dict[str, Any]:
        """Validate parameter against biological constraints."""
        validation_result = {
            "parameter_id": parameter.parameter_id,
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check value range constraints
        if not validate_parameter_value(parameter.parameter_type, parameter.value, parameter.unit):
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Value {parameter.value} {parameter.unit} outside biological range"
            )
        
        # Check unit consistency
        standard_unit = get_standard_units(parameter.parameter_type)
        if parameter.unit != standard_unit and standard_unit != "unknown":
            validation_result["warnings"].append(
                f"Non-standard unit '{parameter.unit}', expected '{standard_unit}'"
            )
        
        # Check for missing standard deviation
        if parameter.std_deviation is None and parameter.confidence_level == ConfidenceLevel.HIGH:
            validation_result["warnings"].append(
                "High confidence parameter missing standard deviation"
            )
        
        # Check coefficient of variation if std_deviation is available
        if parameter.std_deviation is not None and parameter.value != 0:
            cv = parameter.std_deviation / parameter.value
            if cv > 0.5:  # CV > 50% indicates high variability
                validation_result["warnings"].append(
                    f"High coefficient of variation ({cv:.2f}) suggests uncertain measurement"
                )
        
        # Species-specific recommendations
        if parameter.species != "human":
            validation_result["recommendations"].append(
                f"Parameter from {parameter.species} may need scaling for human application"
            )
        
        return validation_result
    
    def compare_parameter_sets(self, morphogen: MorphogenType, 
                             parameter_type: ParameterType) -> Dict[str, Any]:
        """Compare parameter sets across different experimental conditions."""
        parameters = self.database.get_parameters_by_morphogen(morphogen, parameter_type)
        
        if len(parameters) < 2:
            return {"error": "Insufficient data for comparison"}
        
        # Group by experimental method
        method_groups = {}
        for param in parameters:
            method = param.experimental_method or "unknown"
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(param.value)
        
        # Group by species
        species_groups = {}
        for param in parameters:
            species = param.species
            if species not in species_groups:
                species_groups[species] = []
            species_groups[species].append(param.value)
        
        # Calculate group statistics
        comparison = {
            "by_method": {},
            "by_species": {},
            "overall_consistency": 0.0
        }
        
        for method, values in method_groups.items():
            if len(values) > 1:
                comparison["by_method"][method] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf')
                }
        
        for species, values in species_groups.items():
            if len(values) > 1:
                comparison["by_species"][species] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf')
                }
        
        # Calculate overall consistency (inverse of coefficient of variation)
        all_values = [p.value for p in parameters]
        overall_cv = np.std(all_values) / np.mean(all_values) if np.mean(all_values) != 0 else float('inf')
        comparison["overall_consistency"] = 1.0 / (1.0 + overall_cv)
        
        return comparison
    
    def generate_parameter_recommendations(self, morphogen: MorphogenType) -> Dict[str, Any]:
        """Generate recommendations for parameter usage and validation."""
        recommendations = {
            "morphogen": morphogen.value,
            "recommended_values": {},
            "data_quality_assessment": {},
            "validation_priorities": [],
            "expert_review_needed": [],
            "generated_at": datetime.now().isoformat()
        }
        
        # Analyze each parameter type for this morphogen
        for param_type in ParameterType:
            stats = self.get_parameter_statistics(morphogen, param_type)
            
            if stats.get("count", 0) > 0:
                # Recommend value based on expert validation and confidence
                if stats.get("expert_validated_count", 0) > 0:
                    recommended_value = stats["expert_validated_mean"]
                    confidence = "high"
                elif stats.get("high_confidence_count", 0) > 0:
                    recommended_value = stats["high_confidence_mean"]
                    confidence = "medium"
                else:
                    recommended_value = stats["mean"]
                    confidence = "low"
                
                recommendations["recommended_values"][param_type.value] = {
                    "value": recommended_value,
                    "confidence": confidence,
                    "unit": get_standard_units(param_type),
                    "uncertainty": stats.get("std", 0.0),
                    "data_points": stats["count"]
                }
                
                # Assess data quality
                cv = stats.get("cv", float('inf'))
                if cv > 1.0:
                    quality = "poor"
                elif cv > 0.5:
                    quality = "moderate"
                else:
                    quality = "good"
                
                recommendations["data_quality_assessment"][param_type.value] = {
                    "quality": quality,
                    "coefficient_of_variation": cv,
                    "expert_validation_ratio": stats.get("expert_validated_count", 0) / stats["count"]
                }
                
                # Identify validation priorities
                if stats.get("expert_validated_count", 0) == 0:
                    recommendations["validation_priorities"].append(param_type.value)
                
                if cv > 0.5:
                    recommendations["expert_review_needed"].append(param_type.value)
        
        return recommendations
    
    def calculate_parameter_confidence_score(self, parameter: Parameter) -> float:
        """Calculate overall confidence score for a parameter (0-1 scale)."""
        score = 0.0
        
        # Base score from confidence level
        confidence_scores = {
            ConfidenceLevel.HIGH: 0.4,
            ConfidenceLevel.MEDIUM: 0.3,
            ConfidenceLevel.LOW: 0.2,
            ConfidenceLevel.PRELIMINARY: 0.1
        }
        score += confidence_scores.get(parameter.confidence_level, 0.1)
        
        # Expert validation bonus
        if parameter.expert_validated:
            score += 0.3
        
        # Measurement precision bonus
        if parameter.std_deviation is not None and parameter.value != 0:
            cv = parameter.std_deviation / parameter.value
            if cv < 0.2:  # CV < 20% indicates good precision
                score += 0.2
            elif cv < 0.5:  # CV < 50% indicates moderate precision
                score += 0.1
        
        # Species relevance
        if parameter.species == "human":
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

# Export main class
__all__ = ["ParameterAnalyzer"]