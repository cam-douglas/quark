"""
Dataset Catalog Definitions for Brainstem Segmentation

Contains metadata structures and dataset identification for embryonic brain imaging data.
Part of Step 1.F2 implementation from the brainstem segmentation roadmap.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DatasetEntry:
    """Metadata structure for each embryonic brain dataset.
    
    Attributes:
        name: Unique identifier for the dataset
        source: Organization or database providing the data
        modality: Imaging modality (MRI, histology, ISH, etc.)
        developmental_stage: Embryonic stage (E11.5-E18.5)
        resolution_um: Spatial resolution in micrometers
        url: Access URL for the dataset
        access_method: How to retrieve data (api, download, manual)
        format: File format (nifti, tiff, jpeg2000, etc.)
        size_gb: Estimated size in gigabytes
        license: Data usage license
        citation: Required citation for usage
        notes: Additional relevant information
        download_status: Current download state
        registration_status: Current registration state
    """
    name: str
    source: str
    modality: str
    developmental_stage: str
    resolution_um: float
    url: str
    access_method: str
    format: str
    size_gb: Optional[float] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    notes: Optional[str] = None
    download_status: str = "pending"
    registration_status: str = "pending"


def get_public_datasets() -> List[DatasetEntry]:
    """Identify all available public embryonic brain datasets.
    
    Returns:
        List of DatasetEntry objects for known public datasets
    """
    datasets = []
    
    # Allen Brain Atlas - Developing Mouse Brain
    for stage in ["E11.5", "E13.5", "E15.5", "E18.5"]:
        datasets.append(DatasetEntry(
            name=f"Allen_DevMouse_{stage}",
            source="Allen Brain Atlas",
            modality="ISH",
            developmental_stage=stage,
            resolution_um=100,
            url="https://developingmouse.brain-map.org/api/v2/data",
            access_method="api",
            format="jpeg2000",
            license="CC BY-NC-SA 4.0",
            citation="Allen Institute for Brain Science (2013)",
            notes="Sagittal sections, multiple gene markers"
        ))
    
    # GUDMAP datasets
    datasets.append(DatasetEntry(
        name="GUDMAP_MouseBrain_E11.5",
        source="GUDMAP",
        modality="MRI",
        developmental_stage="E11.5",
        resolution_um=50,
        url="https://www.gudmap.org/id/",
        access_method="manual",
        format="nifti",
        license="CC BY 4.0",
        notes="3D MRI volumes, requires registration"
    ))
    
    # EBRAINS resources
    datasets.append(DatasetEntry(
        name="EBRAINS_DevMouseBrain",
        source="EBRAINS",
        modality="histology",
        developmental_stage="E14-E18",
        resolution_um=20,
        url="https://ebrains.eu/data/",
        access_method="manual",
        format="tiff",
        license="CC BY-SA 4.0",
        notes="High-resolution histology sections"
    ))
    
    # BrainMaps.org
    datasets.append(DatasetEntry(
        name="BrainMaps_MouseDev",
        source="BrainMaps.org",
        modality="histology",
        developmental_stage="E12-P0",
        resolution_um=1,
        url="http://brainmaps.org/",
        access_method="api",
        format="jpeg",
        license="Public Domain",
        notes="Ultra-high resolution, multiple staining methods"
    ))
    
    # Mouse Brain Architecture Project
    datasets.append(DatasetEntry(
        name="MBA_DevMouse",
        source="Mouse Brain Architecture",
        modality="histology",
        developmental_stage="E15-P0",
        resolution_um=10,
        url="http://mouse.brainarchitecture.org/",
        access_method="download",
        format="tiff",
        license="CC BY-NC 4.0",
        notes="Nissl and tracer injections"
    ))
    
    # Developmental Common Coordinate Framework (DevCCF)
    for stage, res in [("E11.5", 8), ("E13.5", 10), ("E15.5", 12)]:
        datasets.append(DatasetEntry(
            name=f"DevCCF_{stage}",
            source="Allen Institute DevCCF",
            modality="MRI+histology",
            developmental_stage=stage,
            resolution_um=res,
            url="https://community.brain-map.org/t/developmental-common-coordinate-framework/",
            access_method="download",
            format="nifti",
            license="CC BY-NC-SA 4.0",
            citation="Kronman et al., 2024",
            notes="3D reference atlas with anatomical segmentations"
        ))
    
    return datasets
