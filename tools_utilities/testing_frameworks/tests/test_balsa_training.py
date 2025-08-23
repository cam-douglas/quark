#!/usr/bin/env python3
"""
Test Suite for BALSA Training Pipeline
Tests the neuroimaging data training pipeline for consciousness agent

Purpose: Validate BALSA training functionality
Inputs: Test data, mock neuroimaging datasets
Outputs: Test results, validation reports
Seeds: Test session IDs, mock data generation
Dependencies: pytest, unittest.mock, numpy, pandas
"""

import pytest
import unittest.mock as mock
import json
import os, sys
import tempfile
import shutil
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'database', 'training_scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'database'))

class TestBALSATrainer:
    """Test class for BALSA training pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment with temporary directories"""
        # Create temporary database directory
        self.temp_db_dir = tempfile.mkdtemp()
        self.test_database_path = os.path.join(self.temp_db_dir, "test_database")
        os.makedirs(self.test_database_path, exist_ok=True)
        
        # Mock the required dependencies
        self.mock_brain_mapper = mock.MagicMock()
        self.mock_learning_system = mock.MagicMock()
        
        # Create mock output directories
        self.mock_output_dirs = [
            "balsa_outputs",
            "neuroimaging_data", 
            "processed_connectivity",
            "brain_atlas_maps",
            "training_results"
        ]
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_db_dir)
    
    @mock.patch('balsa_training.BrainRegionMapper')
    @mock.patch('balsa_training.SelfLearningSystem')
    def test_initialization(self, mock_learning_system, mock_brain_mapper):
        """Test BALSA trainer initialization"""
        from balsa_training import BALSATrainer
        
        # Mock the dependencies
        mock_brain_mapper.return_value = self.mock_brain_mapper
        mock_learning_system.return_value = self.mock_learning_system
        
        # Initialize trainer
        trainer = BALSATrainer(self.test_database_path)
        
        # Verify initialization
        assert trainer.database_path == self.test_database_path
        assert trainer.brain_mapper == self.mock_brain_mapper
        assert trainer.learning_system == self.mock_learning_system
        
        # Verify BALSA info
        assert trainer.balsa_info["name"] == "Brain Analysis Library of Spatial Analysis (BALSA)"
        assert trainer.balsa_info["url"] == "https://balsa.wustl.edu/"
        assert len(trainer.balsa_info["datasets"]) > 0
        
        # Verify training config
        assert trainer.training_config["model_type"] == "neuroimaging_consciousness"
        assert trainer.training_config["learning_rate"] == 0.0001
        assert trainer.training_config["batch_size"] == 16
    
    def test_create_output_directories(self):
        """Test creation of output directories"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Check if directories were created
            for dir_name in self.mock_output_dirs:
                dir_path = os.path.join(self.test_database_path, dir_name)
                assert os.path.exists(dir_path), f"Directory {dir_name} was not created"
    
    def test_fetch_balsa_datasets(self):
        """Test fetching BALSA datasets"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Test dataset fetching
            datasets = trainer.fetch_balsa_datasets()
            
            # Verify datasets
            assert len(datasets) > 0
            assert "HCP-Young Adult 2025" in datasets
            assert "HCP-Young Adult Retest 2025" in datasets
            
            # Verify dataset structure
            hcp_ya = datasets["HCP-Young Adult 2025"]
            assert hcp_ya["subjects"] == 1113
            assert "structural_mri" in hcp_ya["modalities"]
            assert hcp_ya["access_level"] == "open_access"
    
    def test_process_structural_data(self):
        """Test structural data processing"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Test structural processing
            dataset_name = "HCP-Young Adult 2025"
            structural_data = trainer.process_structural_data(dataset_name)
            
            # Verify structural data
            assert structural_data["dataset"] == dataset_name
            assert "brain_regions" in structural_data
            assert "prefrontal_cortex" in structural_data["brain_regions"]
            assert "cortical_thickness" in structural_data["metrics"]
            assert "brain_masks" in structural_data["outputs"]
    
    def test_process_functional_data(self):
        """Test functional data processing"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Test functional processing
            dataset_name = "HCP-Young Adult 2025"
            functional_data = trainer.process_functional_data(dataset_name)
            
            # Verify functional data
            assert functional_data["dataset"] == dataset_name
            assert "functional_networks" in functional_data
            assert "default_mode_network" in functional_data["functional_networks"]
            assert "connectivity_metrics" in functional_data
            assert "temporal_features" in functional_data
    
    def test_analyze_connectivity_patterns(self):
        """Test connectivity pattern analysis"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Test connectivity analysis
            dataset_name = "HCP-Young Adult 2025"
            connectivity_data = trainer.analyze_connectivity_patterns(dataset_name)
            
            # Verify connectivity data
            assert connectivity_data["dataset"] == dataset_name
            assert connectivity_data["analysis_type"] == "connectivity_patterns"
            assert "structural_connectivity" in connectivity_data["connectivity_measures"]
            assert "functional_connectivity" in connectivity_data["connectivity_measures"]
            assert "consciousness_correlations" in connectivity_data
    
    def test_integrate_brain_atlas(self):
        """Test brain atlas integration"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Test atlas integration
            dataset_name = "HCP-Young Adult 2025"
            atlas_data = trainer.integrate_brain_atlas(dataset_name)
            
            # Verify atlas data
            assert atlas_data["dataset"] == dataset_name
            assert "aal_atlas" in atlas_data["atlas_systems"]
            assert "yeo_7_network_atlas" in atlas_data["atlas_systems"]
            assert "cortical_regions" in atlas_data["region_mapping"]
            assert "mni_152" in atlas_data["coordinate_systems"]
    
    def test_train_consciousness_model(self):
        """Test consciousness model training"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Create mock processed data
            processed_data = {
                "HCP_YA_structural": {"brain_regions": ["prefrontal_cortex"]},
                "HCP_YA_functional": {"functional_networks": ["default_mode_network"]}
            }
            
            # Test model training
            training_results = trainer.train_consciousness_model(processed_data)
            
            # Verify training results
            assert training_results["model_type"] == "neuroimaging_consciousness"
            assert "training_data" in training_results
            assert "model_architecture" in training_results
            assert "performance_metrics" in training_results
            
            # Verify performance metrics
            metrics = training_results["performance_metrics"]
            assert "training_accuracy" in metrics
            assert "consciousness_correlation" in metrics
            assert metrics["consciousness_correlation"] > 0.7
    
    def test_update_brain_regions(self):
        """Test brain region updates"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Create mock processed data
            processed_data = {
                "HCP_YA_structural": {
                    "brain_regions": ["prefrontal_cortex", "temporal_lobe"]
                },
                "HCP_YA_functional": {
                    "functional_networks": ["default_mode_network", "salience_network"]
                }
            }
            
            # Test brain region updates
            updated_regions = trainer.update_brain_regions(processed_data)
            
            # Verify updates
            assert updated_regions == 4  # 2 brain regions + 2 functional networks
    
    def test_extract_knowledge(self):
        """Test knowledge extraction"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Create mock processed data
            processed_data = {
                "HCP_YA_structural": {
                    "brain_regions": ["prefrontal_cortex", "temporal_lobe"]
                },
                "HCP_YA_functional": {
                    "functional_networks": ["default_mode_network", "salience_network"]
                },
                "HCP_YA_connectivity": {
                    "connectivity_measures": ["structural", "functional"]
                },
                "HCP_YA_atlas": {
                    "atlas_systems": ["aal", "yeo_7"]
                }
            }
            
            # Test knowledge extraction
            knowledge_points = trainer.extract_knowledge(processed_data)
            
            # Verify knowledge extraction
            # 2 brain regions * 2 = 4
            # 2 functional networks * 3 = 6  
            # 2 connectivity measures * 4 = 8
            # 2 atlas systems * 2 = 4
            # Total: 22
            assert knowledge_points == 22
    
    def test_save_training_results(self):
        """Test saving training results"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Create mock data
            processed_data = {"test_data": "value"}
            training_results = {"test_results": "value"}
            
            # Test saving results
            trainer._save_training_results(processed_data, training_results)
            
            # Verify files were created
            balsa_outputs_dir = os.path.join(self.test_database_path, "balsa_outputs")
            assert os.path.exists(balsa_outputs_dir)
            
            # Check for output files
            files = os.listdir(balsa_outputs_dir)
            assert any("processed_data_" in f for f in files)
            assert any("training_results_" in f for f in files)
            assert any("training_session_" in f for f in files)
    
    def test_run_training_pipeline(self):
        """Test complete training pipeline"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Test complete pipeline
            results = trainer.run_training_pipeline()
            
            # Verify pipeline results
            assert results["status"] == "success"
            assert "processed_data" in results
            assert "training_results" in results
            assert "training_session" in results
            
            # Verify training session
            session = results["training_session"]
            assert len(session["datasets_processed"]) > 0
            assert session["knowledge_extracted"] > 0
            assert session["brain_regions_updated"] > 0
    
    def test_get_training_summary(self):
        """Test training summary generation"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Test summary generation
            summary = trainer.get_training_summary()
            
            # Verify summary structure
            assert "session_id" in summary
            assert "started_at" in summary
            assert "datasets_processed" in summary
            assert "knowledge_extracted" in summary
            assert "brain_regions_updated" in summary
    
    def test_error_handling(self):
        """Test error handling in training pipeline"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Test error handling in structural processing
            with mock.patch.object(trainer, 'process_structural_data', side_effect=Exception("Test error")):
                structural_data = trainer.process_structural_data("test_dataset")
                assert structural_data == {}
            
            # Test error handling in functional processing
            with mock.patch.object(trainer, 'process_functional_data', side_effect=Exception("Test error")):
                functional_data = trainer.process_functional_data("test_dataset")
                assert functional_data == {}
    
    def test_training_config_validation(self):
        """Test training configuration validation"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Verify training configuration
            config = trainer.training_config
            
            # Check required fields
            assert "learning_rate" in config
            assert "batch_size" in config
            assert "epochs" in config
            assert "model_type" in config
            
            # Check data processing configuration
            assert "data_processing" in config
            assert config["data_processing"]["preprocessing"] == True
            assert config["data_processing"]["connectivity_analysis"] == True
    
    def test_processing_pipeline_validation(self):
        """Test processing pipeline validation"""
        from balsa_training import BALSATrainer
        
        # Mock dependencies
        with mock.patch('balsa_training.BrainRegionMapper') as mock_brain_mapper, \
             mock.patch('balsa_training.SelfLearningSystem') as mock_learning_system:
            
            mock_brain_mapper.return_value = self.mock_brain_mapper
            mock_learning_system.return_value = self.mock_learning_system
            
            trainer = BALSATrainer(self.test_database_path)
            
            # Verify processing pipeline
            pipeline = trainer.processing_pipeline
            
            # Check structural processing
            assert "structural_processing" in pipeline
            assert "brain_extraction" in pipeline["structural_processing"]
            assert "cortical_parcellation" in pipeline["structural_processing"]
            
            # Check functional processing
            assert "functional_processing" in pipeline
            assert "motion_correction" in pipeline["functional_processing"]
            assert "spatial_normalization" in pipeline["functional_processing"]
            
            # Check connectivity analysis
            assert "connectivity_analysis" in pipeline
            assert "graph_theory_metrics" in pipeline["connectivity_analysis"]
            assert "network_analysis" in pipeline["connectivity_analysis"]

class TestBALSATrainingIntegration:
    """Integration tests for BALSA training pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup_integration_test(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline"""
        # This test would run the actual training pipeline
        # For now, we'll test the main function structure
        from balsa_training import main
        
        # Mock the main function to avoid actual execution
        with mock.patch('balsa_training.BALSATrainer') as mock_trainer_class:
            mock_trainer = mock.MagicMock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock successful training
            mock_trainer.run_training_pipeline.return_value = {
                "status": "success",
                "processed_data": {"test": "data"},
                "training_results": {"test": "results"}
            }
            
            # Mock training summary
            mock_trainer.get_training_summary.return_value = {
                "session_id": "test_session",
                "datasets_processed": ["HCP-YA"],
                "knowledge_extracted": 100,
                "brain_regions_updated": 50
            }
            
            # Test main function (this would normally execute)
            # main()  # Commented out to avoid actual execution
            
            # Verify trainer was created
            mock_trainer_class.assert_called_once()
            
            # Verify training pipeline was called
            mock_trainer.run_training_pipeline.assert_called_once()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
