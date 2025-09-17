#!/usr/bin/env python3
"""
Test Split Creator - Phase 3 Step 1.A1

Creates stratified test split and generates manual annotation gold standard
for brainstem segmentation validation.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
import logging
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class AnnotationSlice:
    """Represents a slice selected for manual annotation."""
    slice_index: int
    axis: str  # 'axial', 'sagittal', 'coronal'
    coordinate: int
    nucleus_count: int
    subdivision_coverage: List[str]
    difficulty_score: float
    annotation_priority: str  # 'high', 'medium', 'low'


class TestSplitCreator:
    """
    Creates stratified test splits for brainstem segmentation validation.
    
    Implements intelligent slice selection for manual annotation based on:
    - Nucleus diversity (ensure all target nuclei are represented)
    - Subdivision coverage (midbrain, pons, medulla balance)
    - Difficulty stratification (easy, medium, hard cases)
    - Anatomical representativeness
    """
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / f"test_split_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        
        # Load NextBrain data
        self.volume_path = self.data_dir / "nextbrain" / "T2w.nii.gz"
        self.labels_path = self.data_dir / "nextbrain" / "manual_segmentation.nii.gz"
        
        # Load data
        self.volume_img = nib.load(self.volume_path)
        self.labels_img = nib.load(self.labels_path)
        
        self.volume = self.volume_img.get_fdata().astype(np.float32)
        self.labels = self.labels_img.get_fdata().astype(np.int32)
        
        logging.info(f"Loaded volume: {self.volume.shape}")
        logging.info(f"Loaded labels: {self.labels.shape}")
        
        # Define target nuclei for annotation
        self.target_nuclei = {
            4: "Red Nucleus",
            9: "Brain-Stem", 
            29: "Pontine Nuclei",
            85: "Inferior Colliculus",
            99: "Medulla Oblongata"
        }
        
        # Define subdivisions
        self.subdivisions = {
            "midbrain": [4, 85],      # Red Nucleus, Inferior Colliculus
            "pons": [29],             # Pontine Nuclei  
            "medulla": [99],          # Medulla Oblongata
            "general": [9]            # General Brain-Stem
        }
    
    def analyze_slice_content(self, slice_data: np.ndarray, slice_idx: int, axis: str) -> Dict[str, Any]:
        """Analyze content of a single slice."""
        
        # Count nuclei present
        unique_labels = np.unique(slice_data)
        target_nuclei_present = [label for label in unique_labels if label in self.target_nuclei]
        
        # Calculate subdivision coverage
        subdivision_coverage = []
        for subdivision, nucleus_list in self.subdivisions.items():
            if any(nucleus in unique_labels for nucleus in nucleus_list):
                subdivision_coverage.append(subdivision)
        
        # Calculate difficulty score based on:
        # - Number of different nuclei (more = harder)
        # - Size of smallest nucleus (smaller = harder) 
        # - Boundary complexity (more boundaries = harder)
        
        nucleus_count = len(target_nuclei_present)
        
        # Boundary complexity (count label transitions)
        boundary_complexity = 0
        for i in range(slice_data.shape[0] - 1):
            for j in range(slice_data.shape[1] - 1):
                if slice_data[i, j] != slice_data[i+1, j] or slice_data[i, j] != slice_data[i, j+1]:
                    boundary_complexity += 1
        
        boundary_complexity_normalized = boundary_complexity / (slice_data.shape[0] * slice_data.shape[1])
        
        # Calculate difficulty score (0-1, higher = more difficult)
        difficulty_score = (
            0.4 * min(nucleus_count / 5.0, 1.0) +  # More nuclei = harder
            0.3 * boundary_complexity_normalized +  # More boundaries = harder
            0.3 * (1.0 - np.sum(slice_data == 0) / slice_data.size)  # Less background = harder
        )
        
        return {
            'slice_index': slice_idx,
            'axis': axis,
            'nucleus_count': nucleus_count,
            'target_nuclei_present': target_nuclei_present,
            'subdivision_coverage': subdivision_coverage,
            'difficulty_score': difficulty_score,
            'boundary_complexity': boundary_complexity_normalized,
            'foreground_ratio': 1.0 - np.sum(slice_data == 0) / slice_data.size
        }
    
    def select_annotation_slices(self, n_slices: int = 30) -> List[AnnotationSlice]:
        """Select optimal slices for manual annotation."""
        
        logging.info(f"Selecting {n_slices} slices for manual annotation...")
        
        # Analyze all slices across different axes
        slice_candidates = []
        
        # Use labels shape for iteration (they're smaller)
        h, w, d = self.labels.shape
        
        # Axial slices (through superior-inferior axis)
        for z in range(0, d, 5):  # Sample every 5th slice
            slice_data = self.labels[:, :, z]
            if np.sum(slice_data > 0) > 100:  # Skip mostly empty slices
                analysis = self.analyze_slice_content(slice_data, z, 'axial')
                if analysis['nucleus_count'] > 0:  # Only keep slices with target nuclei
                    slice_candidates.append(analysis)
        
        # Sagittal slices (through left-right axis)
        for x in range(0, h, 10):  # Sample every 10th slice
            slice_data = self.labels[x, :, :]
            if np.sum(slice_data > 0) > 100:
                analysis = self.analyze_slice_content(slice_data, x, 'sagittal')
                if analysis['nucleus_count'] > 0:
                    slice_candidates.append(analysis)
        
        # Coronal slices (through anterior-posterior axis)
        for y in range(0, w, 10):  # Sample every 10th slice
            slice_data = self.labels[:, y, :]
            if np.sum(slice_data > 0) > 100:
                analysis = self.analyze_slice_content(slice_data, y, 'coronal')
                if analysis['nucleus_count'] > 0:
                    slice_candidates.append(analysis)
        
        logging.info(f"Found {len(slice_candidates)} candidate slices")
        
        # Stratified selection to ensure diversity
        selected_slices = []
        
        # Sort by difficulty for stratification
        slice_candidates.sort(key=lambda x: x['difficulty_score'])
        
        # Select slices ensuring:
        # 1. Difficulty stratification (easy, medium, hard)
        # 2. Axis diversity (axial, sagittal, coronal)
        # 3. Subdivision coverage (all subdivisions represented)
        
        # Difficulty strata
        easy_slices = slice_candidates[:len(slice_candidates)//3]
        medium_slices = slice_candidates[len(slice_candidates)//3:2*len(slice_candidates)//3]
        hard_slices = slice_candidates[2*len(slice_candidates)//3:]
        
        # Target distribution: 40% easy, 40% medium, 20% hard
        n_easy = int(n_slices * 0.4)
        n_medium = int(n_slices * 0.4)
        n_hard = n_slices - n_easy - n_medium
        
        # Select from each stratum
        selected_slices.extend(self._select_diverse_slices(easy_slices, n_easy, "easy"))
        selected_slices.extend(self._select_diverse_slices(medium_slices, n_medium, "medium"))
        selected_slices.extend(self._select_diverse_slices(hard_slices, n_hard, "hard"))
        
        # Convert to AnnotationSlice objects
        annotation_slices = []
        for slice_info in selected_slices:
            annotation_slice = AnnotationSlice(
                slice_index=slice_info['slice_index'],
                axis=slice_info['axis'],
                coordinate=slice_info['slice_index'],
                nucleus_count=slice_info['nucleus_count'],
                subdivision_coverage=slice_info['subdivision_coverage'],
                difficulty_score=slice_info['difficulty_score'],
                annotation_priority=slice_info['priority']
            )
            annotation_slices.append(annotation_slice)
        
        logging.info(f"Selected {len(annotation_slices)} slices for annotation")
        
        return annotation_slices
    
    def _select_diverse_slices(self, candidates: List[Dict], n_select: int, priority: str) -> List[Dict]:
        """Select diverse slices from candidates ensuring axis and subdivision diversity."""
        
        if n_select >= len(candidates):
            for candidate in candidates:
                candidate['priority'] = priority
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # Ensure axis diversity
        axes_needed = ['axial', 'sagittal', 'coronal']
        axes_selected = {axis: 0 for axis in axes_needed}
        
        # First pass: select one from each axis
        for axis in axes_needed:
            if len(selected) >= n_select:
                break
                
            axis_candidates = [c for c in remaining if c['axis'] == axis]
            if axis_candidates:
                # Select the one with most nucleus diversity
                best_candidate = max(axis_candidates, key=lambda x: x['nucleus_count'])
                best_candidate['priority'] = priority
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                axes_selected[axis] += 1
        
        # Second pass: fill remaining slots with highest scoring candidates
        remaining.sort(key=lambda x: (len(x['subdivision_coverage']), x['nucleus_count']), reverse=True)
        
        while len(selected) < n_select and remaining:
            candidate = remaining.pop(0)
            candidate['priority'] = priority
            selected.append(candidate)
        
        return selected
    
    def create_manual_annotation_volume(self, annotation_slices: List[AnnotationSlice]) -> None:
        """Create gold standard volume with selected slices."""
        
        logging.info("Creating manual annotation gold standard volume...")
        
        # Create a new volume with only the selected slices
        gold_standard = np.zeros_like(self.labels)
        
        slice_info = []
        
        for i, ann_slice in enumerate(annotation_slices):
            if ann_slice.axis == 'axial':
                gold_standard[:, :, ann_slice.slice_index] = self.labels[:, :, ann_slice.slice_index]
                slice_info.append({
                    'slice_id': i,
                    'axis': ann_slice.axis,
                    'coordinate': ann_slice.slice_index,
                    'nucleus_count': ann_slice.nucleus_count,
                    'subdivisions': ann_slice.subdivision_coverage,
                    'difficulty': ann_slice.difficulty_score,
                    'priority': ann_slice.annotation_priority
                })
            elif ann_slice.axis == 'sagittal':
                gold_standard[ann_slice.slice_index, :, :] = self.labels[ann_slice.slice_index, :, :]
                slice_info.append({
                    'slice_id': i,
                    'axis': ann_slice.axis,
                    'coordinate': ann_slice.slice_index,
                    'nucleus_count': ann_slice.nucleus_count,
                    'subdivisions': ann_slice.subdivision_coverage,
                    'difficulty': ann_slice.difficulty_score,
                    'priority': ann_slice.annotation_priority
                })
            elif ann_slice.axis == 'coronal':
                gold_standard[:, ann_slice.slice_index, :] = self.labels[:, ann_slice.slice_index, :]
                slice_info.append({
                    'slice_id': i,
                    'axis': ann_slice.axis,
                    'coordinate': ann_slice.slice_index,
                    'nucleus_count': ann_slice.nucleus_count,
                    'subdivisions': ann_slice.subdivision_coverage,
                    'difficulty': ann_slice.difficulty_score,
                    'priority': ann_slice.annotation_priority
                })
        
        # Save gold standard volume
        gold_standard_img = nib.Nifti1Image(gold_standard, self.labels_img.affine, self.labels_img.header)
        output_path = self.output_dir / "test_manual.nii.gz"
        nib.save(gold_standard_img, output_path)
        
        logging.info(f"Saved manual annotation gold standard: {output_path}")
        
        # Save slice information
        slice_info_path = self.output_dir / "annotation_slices_info.json"
        with open(slice_info_path, 'w') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_slices': len(annotation_slices),
                'slice_distribution': {
                    'easy': len([s for s in annotation_slices if s.annotation_priority == 'easy']),
                    'medium': len([s for s in annotation_slices if s.annotation_priority == 'medium']),
                    'hard': len([s for s in annotation_slices if s.annotation_priority == 'hard'])
                },
                'axis_distribution': {
                    'axial': len([s for s in annotation_slices if s.axis == 'axial']),
                    'sagittal': len([s for s in annotation_slices if s.axis == 'sagittal']),
                    'coronal': len([s for s in annotation_slices if s.axis == 'coronal'])
                },
                'slices': slice_info
            }, f, indent=2)
        
        logging.info(f"Saved slice information: {slice_info_path}")
        
        return output_path
    
    def generate_annotation_report(self, annotation_slices: List[AnnotationSlice]) -> None:
        """Generate comprehensive annotation report."""
        
        report = {
            'generated': datetime.now().isoformat(),
            'phase': 'Phase 3 - Validation & Testing',
            'step': '1.A1 - Stratified Test Split & Manual Annotation',
            
            'summary': {
                'total_slices_selected': len(annotation_slices),
                'target_nuclei': list(self.target_nuclei.values()),
                'subdivisions_covered': list(self.subdivisions.keys())
            },
            
            'stratification': {
                'difficulty_distribution': {
                    'easy': len([s for s in annotation_slices if s.annotation_priority == 'easy']),
                    'medium': len([s for s in annotation_slices if s.annotation_priority == 'medium']),
                    'hard': len([s for s in annotation_slices if s.annotation_priority == 'hard'])
                },
                'axis_distribution': {
                    'axial': len([s for s in annotation_slices if s.axis == 'axial']),
                    'sagittal': len([s for s in annotation_slices if s.axis == 'sagittal']),
                    'coronal': len([s for s in annotation_slices if s.axis == 'coronal'])
                }
            },
            
            'quality_metrics': {
                'average_nucleus_count': np.mean([s.nucleus_count for s in annotation_slices]),
                'average_difficulty_score': np.mean([s.difficulty_score for s in annotation_slices]),
                'subdivision_coverage_completeness': len(set([sub for s in annotation_slices for sub in s.subdivision_coverage])) / len(self.subdivisions)
            },
            
            'annotation_guidelines': {
                'target_accuracy': '±200μm from NextBrain reference',
                'inter_annotator_agreement': 'Dice > 0.90 required',
                'annotation_time_estimate': f'{len(annotation_slices)} slices × 15 min = {len(annotation_slices) * 15} minutes',
                'quality_control': 'Cross-validation with second annotator'
            }
        }
        
        # Save report
        report_path = self.output_dir / "test_split_annotation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Generated annotation report: {report_path}")


def main():
    """Execute Phase 3 Step 1.A1: Test split creation and manual annotation."""
    
    print("🔬 PHASE 3 STEP 1.A1 - TEST SPLIT & MANUAL ANNOTATION")
    print("=" * 60)
    
    # Paths
    data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation")
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/test_splits")
    
    # Check data availability
    if not (data_dir / "nextbrain" / "T2w.nii.gz").exists():
        print("❌ NextBrain data not found!")
        return False
    
    print("✅ Data found, creating test split...")
    
    try:
        # Create test split
        creator = TestSplitCreator(data_dir, output_dir)
        
        # Select annotation slices
        annotation_slices = creator.select_annotation_slices(n_slices=30)
        
        # Create manual annotation volume
        gold_standard_path = creator.create_manual_annotation_volume(annotation_slices)
        
        # Generate report
        creator.generate_annotation_report(annotation_slices)
        
        print(f"\n✅ Phase 3 Step 1.A1 Complete!")
        print(f"   📁 Gold standard: {gold_standard_path}")
        print(f"   🎯 Annotation slices: {len(annotation_slices)}")
        
        # Summary statistics
        difficulty_dist = {}
        axis_dist = {}
        
        for slice_obj in annotation_slices:
            difficulty_dist[slice_obj.annotation_priority] = difficulty_dist.get(slice_obj.annotation_priority, 0) + 1
            axis_dist[slice_obj.axis] = axis_dist.get(slice_obj.axis, 0) + 1
        
        print(f"   📊 Difficulty distribution: {difficulty_dist}")
        print(f"   📐 Axis distribution: {axis_dist}")
        print(f"   🧠 Nuclei coverage: {len(creator.target_nuclei)} target nuclei")
        print(f"   ⏱️ Estimated annotation time: {len(annotation_slices) * 15} minutes")
        
        return True
        
    except Exception as e:
        print(f"❌ Test split creation failed: {e}")
        logging.error(f"Error in test split creation: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
