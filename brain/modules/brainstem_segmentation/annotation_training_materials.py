"""
Training materials generator for brainstem annotation experts.

Creates reference guides, checklists, and mistake prevention materials
for annotator training and certification.
"""
from __future__ import annotations

from pathlib import Path
from typing import List


def create_quick_reference(output_dir: Path) -> Path:
    """Create quick reference guide for annotators."""
    content = """# Brainstem Nuclei Quick Reference

## Midbrain (Mesencephalon)
- **Periaqueductal Grey**: Around aqueduct, autonomic functions
- **Edinger-Westphal**: Small, dorsomedial to CN III, pupil control
- **Substantia Nigra**: Dark band, motor control, dopamine
- **Red Nucleus**: Round, motor coordination

## Pons (Metencephalon)
- **Locus Coeruleus**: Small, dense, arousal/attention
- **Pontine Nuclei**: Scattered, corticopontocerebellar relay
- **Facial Nucleus**: Compact, facial motor control

## Medulla (Myelencephalon)
- **Raphe Magnus**: Midline, pain modulation
- **Nucleus Ambiguus**: Elongated, larynx/pharynx/heart
- **Pre-Bötzinger**: Small, respiratory rhythm
- **Hypoglossal**: Paired, tongue motor
- **Dorsal Motor Vagus**: Longitudinal, parasympathetic

## Key Landmarks
- Cerebral aqueduct (midbrain)
- Fourth ventricle (pons/medulla)
- Pyramidal tract (medulla)
- Superior/middle cerebellar peduncles
"""
    
    ref_file = output_dir / "nuclei_quick_reference.md"
    with open(ref_file, 'w') as f:
        f.write(content)
    
    return ref_file


def create_annotation_checklist(output_dir: Path) -> Path:
    """Create annotation quality checklist."""
    content = """# Annotation Quality Checklist

## Before Starting
- [ ] Verify volume orientation (RAS coordinate system)
- [ ] Set consistent brightness/contrast
- [ ] Load anatomical reference atlas
- [ ] Review nucleus definitions

## During Annotation
- [ ] Check each nucleus against anatomical landmarks
- [ ] Verify functional subdivision assignments
- [ ] Mark uncertain regions separately
- [ ] Cross-reference with atlas every 10 slices

## Quality Control
- [ ] All 12 target nuclei annotated
- [ ] No overlapping labels
- [ ] Boundaries follow anatomical principles
- [ ] Uncertain regions documented

## Final Review
- [ ] Export segmentation as .nii.gz
- [ ] Save annotation notes
- [ ] Complete quality metrics form
- [ ] Submit for cross-validation
"""
    
    checklist_file = output_dir / "annotation_checklist.md"
    with open(checklist_file, 'w') as f:
        f.write(content)
    
    return checklist_file


def create_common_mistakes_guide(output_dir: Path) -> Path:
    """Create guide for avoiding common annotation mistakes."""
    content = """# Common Annotation Mistakes to Avoid

## General Issues
1. **Inconsistent boundaries**: Use same intensity threshold across slices
2. **Missing small nuclei**: Locus coeruleus, Edinger-Westphal are very small
3. **White matter inclusion**: Exclude fiber tracts from nuclear boundaries
4. **Orientation errors**: Verify RAS coordinates before starting

## Nucleus-Specific Mistakes

### Periaqueductal Grey
- ❌ Including aqueduct lumen (should be background)
- ❌ Extending too far from aqueduct walls
- ✅ Follow aqueduct contour closely

### Substantia Nigra
- ❌ Including cerebral peduncle fibers
- ❌ Unclear pars compacta/reticulata distinction
- ✅ Focus on cell-dense regions, exclude fiber bundles

### Locus Coeruleus
- ❌ Overestimating size (it's very small!)
- ❌ Confusing with adjacent pontine nuclei
- ✅ Look for dense, darkly-staining cluster

### Nucleus Ambiguus
- ❌ Confusing with hypoglossal nucleus
- ❌ Missing elongated column shape
- ✅ Follow ventrolateral column through multiple slices

## Quality Metrics
- Target inter-annotator Dice: ≥ 0.90
- Landmark accuracy: ≤ 200 µm from atlas
- Completion time: ≤ 4 hours per volume
"""
    
    mistakes_file = output_dir / "common_mistakes_guide.md"
    with open(mistakes_file, 'w') as f:
        f.write(content)
    
    return mistakes_file


def generate_all_training_materials(output_dir: Path) -> List[Path]:
    """Generate all training materials for annotators."""
    materials = []
    
    materials.append(create_quick_reference(output_dir))
    materials.append(create_annotation_checklist(output_dir))
    materials.append(create_common_mistakes_guide(output_dir))
    
    return materials
