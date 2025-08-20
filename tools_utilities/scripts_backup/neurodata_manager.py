"""
Unified Neurodata Manager

Coordinates access to truly open neuroscience data sources:
- Open Neurophysiology Databases (CRCNS, NeuroMorpho, ModelDB, OSB, NeuroElectro)
- Open Brain Imaging Databases (OpenNeuro, Brainlife, NITRC, INDI)
- CommonCrawl Web Data (WARC/ARC format, S3 public access)

All sources are publicly accessible without API keys or signups.
Provides unified interface for cross-dataset analysis and integration.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np

from .....................................................open_neurophysiology import OpenNeurophysiologyInterface
from .....................................................open_brain_imaging import OpenBrainImagingInterface
from .....................................................commoncrawl_interface import CommonCrawlInterface
from .....................................................human_brain_development import create_smallmind_brain_dev_trainer


class NeurodataManager:
    """Unified manager for truly open neuroscience data sources"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize open data source interfaces
        self.open_physiology = OpenNeurophysiologyInterface()
        self.open_imaging = OpenBrainImagingInterface()
        self.commoncrawl = CommonCrawlInterface()
        
        # Initialize SmallMind brain development trainer
        self.brain_dev_trainer = create_smallmind_brain_dev_trainer()
        
        # Configuration and caching
        self.config = self._load_config(config_path)
        self.cache_dir = Path(self.config.get("cache_dir", "./neurodata_cache"))
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger.info("Neurodata Manager initialized with SmallMind brain development trainer")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "cache_dir": "./neurodata_cache",
                "max_cache_size_gb": 100,
                "preferred_sources": ["open_physiology", "open_imaging", "commoncrawl"],
                "data_quality_thresholds": {
                    "min_cell_count": 10,
                    "min_recording_duration": 60,
                    "min_spatial_resolution": 1.0
                }
            }
    
    def search_across_sources(self, query: str = "", 
                             data_types: Optional[List[str]] = None,
                             species: Optional[str] = None,
                             brain_regions: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Search for neuroscience data across all open sources
        
        Args:
            query: Search query string
            data_types: Filter by data types (e.g., ['electrophysiology', 'fMRI', 'morphology'])
            species: Filter by species (e.g., 'human', 'mouse', 'rat')
            brain_regions: Filter by brain regions (e.g., ['cortex', 'hippocampus'])
            
        Returns:
            Dictionary mapping source names to lists of results
        """
        self.logger.info(f"Searching across all sources for: {query}")
        
        results = {}
        
        # Search neurophysiology sources
        try:
            phys_results = []
            
            # CRCNS datasets
            crcns_datasets = self.open_physiology.search_crcns_datasets(query)
            phys_results.extend([{"source": "crcns", **dataset} for dataset in crcns_datasets])
            
            # NeuroMorpho neurons
            neuromorpho_neurons = self.open_physiology.get_neuromorpho_neurons(species, brain_regions[0] if brain_regions else None)
            phys_results.extend([{"source": "neuromorpho", **neuron} for neuron in neuromorpho_neurons])
            
            # ModelDB models
            modeldb_models = self.open_physiology.search_modeldb_models(query)
            phys_results.extend([{"source": "modeldb", **model} for model in modeldb_models])
            
            # Open Source Brain models
            osb_models = self.open_physiology.get_opensourcebrain_models()
            phys_results.extend([{"source": "opensourcebrain", **model} for model in osb_models])
            
            # NeuroElectro properties
            neuroelectro_props = self.open_physiology.search_neuroelectro_properties(
                species, brain_regions[0] if brain_regions else None
            )
            phys_results.extend([{"source": "neuroelectro", **prop} for prop in neuroelectro_props])
            
            # Apply filters
            if data_types:
                phys_results = [r for r in phys_results if any(dt.lower() in str(r.get("data_type", "")).lower() or 
                                                           dt.lower() in str(r.get("model_type", "")).lower() or
                                                           dt.lower() in str(r.get("modality", "")).lower() 
                                                           for dt in data_types)]
            
            if species:
                phys_results = [r for r in phys_results if species.lower() in str(r.get("species", "")).lower()]
                
            if brain_regions:
                phys_results = [r for r in phys_results if any(br.lower() in str(r.get("brain_region", "")).lower() 
                                                             for br in brain_regions)]
            
            results["open_physiology"] = phys_results
            
        except Exception as e:
            self.logger.error(f"Error searching neurophysiology sources: {e}")
            results["open_physiology"] = []
        
        # Search brain imaging sources
        try:
            imaging_results = []
            
            # OpenNeuro datasets
            openneuro_datasets = self.open_imaging.search_openneuro_datasets(query)
            imaging_results.extend([{"source": "openneuro", **dataset} for dataset in openneuro_datasets])
            
            # Brainlife datasets
            brainlife_datasets = self.open_imaging.search_brainlife_datasets(query)
            imaging_results.extend([{"source": "brainlife", **dataset} for dataset in brainlife_datasets])
            
            # NITRC resources
            nitrc_resources = self.open_imaging.search_nitrc_resources(query)
            imaging_results.extend([{"source": "nitrc", **resource} for resource in nitrc_resources])
            
            # INDI datasets
            indi_datasets = self.open_imaging.get_indi_datasets()
            imaging_results.extend([{"source": "indi", **dataset} for dataset in indi_datasets])
            
            # Apply filters
            if data_types:
                imaging_results = [r for r in imaging_results if any(dt.lower() in str(r.get("modality", "")).lower() or
                                                                   dt.lower() in str(r.get("type", "")).lower() 
                                                                   for dt in data_types)]
            
            if species:
                imaging_results = [r for r in imaging_results if species.lower() in str(r.get("species", "")).lower()]
                
            if brain_regions:
                imaging_results = [r for r in imaging_results if any(br.lower() in str(r.get("brain_region", "")).lower() 
                                                                   for br in brain_regions)]
            
            results["open_imaging"] = imaging_results
            
        except Exception as e:
            self.logger.error(f"Error searching brain imaging sources: {e}")
            results["open_imaging"] = []
        
        # Search CommonCrawl sources
        try:
            crawl_results = []
            
            # Neuroscience content
            neuroscience_content = self.commoncrawl.search_neuroscience_content(query)
            crawl_results.extend([{"source": "commoncrawl", **content} for content in neuroscience_content])
            
            # Neuroscience datasets
            neuroscience_datasets = self.commoncrawl.get_neuroscience_datasets()
            crawl_results.extend([{"source": "commoncrawl_datasets", **dataset} for dataset in neuroscience_datasets])
            
            results["commoncrawl"] = crawl_results
            
        except Exception as e:
            self.logger.error(f"Error searching CommonCrawl sources: {e}")
            results["commoncrawl"] = []
        
        # Search human brain development training data
        try:
            brain_dev_results = []
            
            # Search development knowledge
            dev_knowledge = self.brain_dev_trainer.search_development_knowledge(query)
            
            # Add stages
            for stage in dev_knowledge.get('stages', []):
                brain_dev_results.append({
                    "source": "brain_development",
                    "type": "development_stage",
                    "name": stage.name,
                    "carnegie_stage": stage.carnegie_stage,
                    "gestational_weeks": stage.gestational_weeks,
                    "description": stage.description,
                    "data_type": "developmental_timeline"
                })
            
            # Add processes
            for process in dev_knowledge.get('processes', []):
                brain_dev_results.append({
                    "source": "brain_development",
                    "type": "developmental_process",
                    "name": process.name,
                    "timeline": process.timeline,
                    "description": process.description,
                    "data_type": "developmental_process"
                })
            
            # Add cell types
            for cell_data in dev_knowledge.get('cell_types', []):
                brain_dev_results.append({
                    "source": "brain_development",
                    "type": "cell_type_info",
                    "title": cell_data['title'],
                    "cell_types": cell_data['data']['cell_types'],
                    "content": cell_data['data']['content'],
                    "data_type": "cell_biology"
                })
            
            # Add morphogens
            for morph_data in dev_knowledge.get('morphogens', []):
                brain_dev_results.append({
                    "source": "brain_development",
                    "type": "morphogen_info",
                    "title": morph_data['title'],
                    "morphogens": morph_data['data']['morphogens'],
                    "content": morph_data['data']['content'],
                    "data_type": "molecular_biology"
                })
            
            # Apply filters
            if data_types:
                brain_dev_results = [r for r in brain_dev_results if any(dt.lower() in str(r.get("data_type", "")).lower() 
                                                                       for dt in data_types)]
            
            if species and species.lower() == 'human':
                # Only include human-specific data
                brain_dev_results = [r for r in brain_dev_results if 'human' in str(r.get("description", "")).lower()]
            
            results["brain_development"] = brain_dev_results
            
        except Exception as e:
            self.logger.error(f"Error searching brain development sources: {e}")
            results["brain_development"] = []
        
        # Log search results
        total_results = sum(len(source_results) for source_results in results.values())
        self.logger.info(f"Search completed. Found {total_results} total results across {len(results)} sources")
        
        return results
    
    def get_cross_source_metadata(self, dataset_ids: Dict[str, str]) -> Dict[str, Dict]:
        """
        Get metadata from multiple open sources for comparison
        
        Args:
            dataset_ids: Dictionary mapping source names to dataset IDs
            
        Returns:
            Dictionary mapping source names to metadata
        """
        metadata = {}
        
        for source, dataset_id in dataset_ids.items():
            try:
                if source == "crcns":
                    # Get CRCNS dataset info
                    datasets = self.open_physiology.search_crcns_datasets()
                    metadata[source] = next((d for d in datasets if d["id"] == dataset_id), {})
                    
                elif source == "openneuro":
                    metadata[source] = self.open_imaging.get_openneuro_dataset_info(dataset_id)
                    
                elif source == "commoncrawl":
                    # Get CommonCrawl index info
                    indexes = self.commoncrawl.list_crawl_indexes()
                    metadata[source] = next((i for i in indexes if i["id"] == dataset_id), {})
                    
                else:
                    metadata[source] = {"source": source, "dataset_id": dataset_id}
                    
            except Exception as e:
                self.logger.warning(f"Failed to get metadata from {source}: {e}")
                metadata[source] = {"error": str(e)}
        
        return metadata
    
    def create_unified_dataset(self, source_data: Dict[str, Any], 
                              output_path: Path) -> Path:
        """
        Create a unified dataset combining data from multiple open sources
        
        Args:
            source_data: Dictionary of data from different sources
            output_path: Path to save unified dataset
            
        Returns:
            Path to unified dataset
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create unified metadata
        unified_metadata = {
            "sources": list(source_data.keys()),
            "creation_date": pd.Timestamp.now().isoformat(),
            "data_summary": {},
            "cross_references": {}
        }
        
        # Process each source's data
        for source, data in source_data.items():
            if isinstance(data, dict):
                unified_metadata["data_summary"][source] = {
                    "data_type": data.get("data_type", "unknown"),
                    "size": data.get("size", "unknown"),
                    "quality_score": self._calculate_quality_score(data)
                }
        
        # Save unified metadata
        with open(output_path / "unified_metadata.json", 'w') as f:
            json.dump(unified_metadata, f, indent=2)
        
        return output_path
    
    def get_brain_development_timeline(self, start_stage: Optional[str] = None, 
                                     end_stage: Optional[str] = None) -> List[Dict]:
        """
        Get human brain development timeline
        
        Args:
            start_stage: Starting stage name
            end_stage: Ending stage name
            
        Returns:
            List of development stages with metadata
        """
        try:
            stages = self.brain_dev_trainer.get_development_timeline(start_stage, end_stage)
            # Convert to list of dictionaries if needed
            if stages and hasattr(stages[0], '__dict__'):
                return [stage.__dict__ for stage in stages]
            else:
                return stages
        except Exception as e:
            self.logger.error(f"Error getting brain development timeline: {e}")
            return []
    
    def get_brain_development_processes(self, process_type: Optional[str] = None) -> List[Dict]:
        """
        Get human brain development processes
        
        Args:
            process_type: Type of process to filter by
            
        Returns:
            List of developmental processes
        """
        try:
            processes = self.brain_dev_trainer.get_developmental_processes(process_type)
            # Convert to list of dictionaries if needed
            if processes and hasattr(processes[0], '__dict__'):
                return [process.__dict__ for process in processes]
            else:
                return processes
        except Exception as e:
            self.logger.error(f"Error getting brain development processes: {e}")
            return []
    
    def get_cell_types_by_development_stage(self, stage_name: str) -> List[str]:
        """
        Get cell types present at a specific development stage
        
        Args:
            stage_name: Name of the development stage
            
        Returns:
            List of cell types
        """
        try:
            return self.brain_dev_trainer.get_cell_types_by_stage(stage_name)
        except Exception as e:
            self.logger.error(f"Error getting cell types for stage {stage_name}: {e}")
            return []
    
    def get_morphogens_by_development_stage(self, stage_name: str) -> List[str]:
        """
        Get morphogens active at a specific development stage
        
        Args:
            stage_name: Name of the development stage
            
        Returns:
            List of morphogens
        """
        try:
            return self.brain_dev_trainer.get_morphogens_by_stage(stage_name)
        except Exception as e:
            self.logger.error(f"Error getting morphogens for stage {stage_name}: {e}")
            return []
    
    def search_brain_development_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Search human brain development knowledge
        
        Args:
            query: Search query
            
        Returns:
            Dictionary containing search results
        """
        try:
            return self.brain_dev_trainer.search_development_knowledge(query)
        except Exception as e:
            self.logger.error(f"Error searching brain development knowledge: {e}")
            return {}
    
    def get_brain_development_training_data(self, data_type: str = "all") -> Dict[str, Any]:
        """
        Get brain development training data for model training
        
        Args:
            data_type: Type of data to retrieve
            
        Returns:
            Dictionary containing training data
        """
        try:
            return self.brain_dev_trainer.get_training_data_for_model(data_type)
        except Exception as e:
            self.logger.error(f"Error getting brain development training data: {e}")
            return {}
    
    def export_brain_development_data(self, output_path: Path, format: str = "json") -> Path:
        """
        Export brain development data to file
        
        Args:
            output_path: Path to save the exported data
            format: Export format
            
        Returns:
            Path to the exported file
        """
        try:
            return self.brain_dev_trainer.export_training_data(output_path, format)
        except Exception as e:
            self.logger.error(f"Error exporting brain development data: {e}")
            return output_path
    
    def safe_smallmind_brain_development_query(self, question: str, max_length: int = 1000) -> Dict[str, Any]:
        """
        Safely query SmallMind human brain development knowledge
        
        Args:
            question: User question about brain development
            max_length: Maximum response length
            
        Returns:
            Safe response with citations and uncertainty
        """
        try:
            response = self.brain_dev_trainer.safe_query(question, max_length)
            self.logger.info(f"SmallMind brain development query processed safely: {len(response.get('citations', []))} citations")
            return response
        except Exception as e:
            self.logger.error(f"Error in SmallMind brain development query: {e}")
            return {
                'answer': "I encountered an error while processing your question about brain development. Please try rephrasing.",
                'citations': [],
                'uncertainty': 'high',
                'safety_warnings': ['Processing error occurred'],
                'source_modules': []
            }
    
    def get_smallmind_brain_development_summary(self) -> Dict[str, Any]:
        """Get summary of SmallMind brain development training materials"""
        try:
            return self.brain_dev_trainer.get_training_summary()
        except Exception as e:
            self.logger.error(f"Error getting SmallMind training summary: {e}")
            return {'error': str(e)}
    
    def export_smallmind_brain_development_examples(self, output_path: Path) -> Path:
        """Export safe response examples for validation"""
        try:
            return self.brain_dev_trainer.export_safe_responses(output_path)
        except Exception as e:
            self.logger.error(f"Error exporting SmallMind examples: {e}")
            return output_path
    
    def safe_brain_development_query(self, question: str, max_length: int = 1000) -> Dict[str, Any]:
        """
        Safely query human brain development knowledge
        
        Args:
            question: User question about brain development
            max_length: Maximum response length
            
        Returns:
            Safe response with citations and uncertainty
        """
        try:
            response = self.brain_dev_trainer.safe_query(question, max_length)
            self.logger.info(f"Brain development query processed safely: {len(response.get('citations', []))} citations")
            return response
        except Exception as e:
            self.logger.error(f"Error in brain development query: {e}")
            return {
                'answer': "I encountered an error while processing your question about brain development. Please try rephrasing.",
                'citations': [],
                'uncertainty': 'high',
                'safety_warnings': ['Processing error occurred'],
                'source_modules': []
            }
    
    def get_brain_development_summary(self) -> Dict[str, Any]:
        """Get summary of brain development training materials"""
        try:
            return self.brain_dev_trainer.get_training_summary()
        except Exception as e:
            self.logger.error(f"Error getting training summary: {e}")
            return {'error': str(e)}
    
    def export_brain_development_examples(self, output_path: Path) -> Path:
        """Export safe response examples for validation"""
        try:
            return self.brain_dev_trainer.export_safe_responses(output_path)
        except Exception as e:
            self.logger.error(f"Error exporting examples: {e}")
            return output_path
    
    def _calculate_quality_score(self, data: Dict) -> float:
        """Calculate quality score for dataset based on configurable criteria"""
        score = 0.0
        
        # Check data completeness
        if data.get("electrophysiology"):
            score += 0.3
        if data.get("morphology"):
            score += 0.3
        if data.get("transcriptomics"):
            score += 0.2
        if data.get("metadata"):
            score += 0.2
            
        return min(score, 1.0)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics across all open data sources
        
        Returns:
            Dictionary with data source statistics
        """
        stats = {}
        
        # Open Neurophysiology statistics
        try:
            crcns_datasets = self.open_physiology.search_crcns_datasets()
            neuromorpho_neurons = self.open_physiology.get_neuromorpho_neurons()
            modeldb_models = self.open_physiology.search_modeldb_models()
            
            stats["open_physiology"] = {
                "crcns_datasets": len(crcns_datasets),
                "neuromorpho_neurons": len(neuromorpho_neurons),
                "modeldb_models": len(modeldb_models),
                "total_sources": 5  # CRCNS, NeuroMorpho, ModelDB, OSB, NeuroElectro
            }
        except Exception as e:
            stats["open_physiology"] = {"error": str(e)}
        
        # Open Brain Imaging statistics
        try:
            openneuro_datasets = self.open_imaging.search_openneuro_datasets()
            brainlife_datasets = self.open_imaging.search_brainlife_datasets()
            nitrc_resources = self.open_imaging.search_nitrc_resources()
            
            stats["open_imaging"] = {
                "openneuro_datasets": len(openneuro_datasets),
                "brainlife_datasets": len(brainlife_datasets),
                "nitrc_resources": len(nitrc_resources),
                "total_sources": 4  # OpenNeuro, Brainlife, NITRC, INDI
            }
        except Exception as e:
            stats["open_imaging"] = {"error": str(e)}
        
        # CommonCrawl statistics
        try:
            crawl_indexes = self.commoncrawl.list_crawl_indexes()
            neuroscience_datasets = self.commoncrawl.get_neuroscience_datasets()
            
            stats["commoncrawl"] = {
                "crawl_indexes": len(crawl_indexes),
                "neuroscience_datasets": len(neuroscience_datasets),
                "data_format": "WARC/ARC",
                "access": "S3 public read"
            }
        except Exception as e:
            stats["commoncrawl"] = {"error": str(e)}
        
        return stats
    
    def export_data_catalog(self, output_path: Path) -> Path:
        """
        Export a comprehensive catalog of available open data
        
        Args:
            output_path: Path to save catalog
            
        Returns:
            Path to exported catalog
        """
        catalog = {
            "export_date": pd.Timestamp.now().isoformat(),
            "data_sources": self.get_data_statistics(),
            "search_capabilities": {
                "open_physiology": ["electrophysiology", "morphology", "models", "transcriptomics"],
                "open_imaging": ["fMRI", "MEG", "EEG", "BIDS", "neuroimaging"],
                "commoncrawl": ["web_content", "literature", "publications", "WARC/ARC"]
            },
            "access_methods": {
                "open_physiology": "Public APIs, no authentication required",
                "open_imaging": "Public APIs, no authentication required", 
                "commoncrawl": "S3 public read, HTTP downloads"
            },
            "api_keys_required": False,
            "signup_required": False
        }
        
        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        return output_path

