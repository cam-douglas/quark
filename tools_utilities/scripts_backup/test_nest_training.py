#!/usr/bin/env python3
"""
Test script for NEST Training Module
====================================

This script provides unit and integration tests for the NEST training module,
ensuring proper functionality of spiking neural network simulations.

Author: Brain Simulation Framework Team
Date: 2024
License: Apache-2.0
"""

import os, sys
import asyncio
import pytest
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from development.src.training.nest_training import NESTConfig, NESTTrainer


class TestNESTConfig:
    """Test cases for NESTConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NESTConfig()
        
        assert config.simulation_time == 1000.0
        assert config.num_neurons == 1000
        assert config.connectivity_density == 0.1
        assert config.output_dir == "nest_outputs"
        assert config.enable_brain_simulation is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = NESTConfig(
            simulation_time=500.0,
            num_neurons=500,
            connectivity_density=0.2,
            output_dir="custom_outputs",
            enable_brain_simulation=False
        )
        
        assert config.simulation_time == 500.0
        assert config.num_neurons == 500
        assert config.connectivity_density == 0.2
        assert config.output_dir == "custom_outputs"
        assert config.enable_brain_simulation is False


class TestNESTTrainer:
    """Test cases for NESTTrainer class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return NESTConfig(
            simulation_time=100.0,
            num_neurons=100,
            connectivity_density=0.1,
            output_dir=temp_dir
        )
    
    @pytest.fixture
    def trainer(self, config):
        """Create test trainer."""
        return NESTTrainer(config)
    
    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.config is not None
        assert trainer.nest_available is False  # NEST not available in test environment
        assert trainer.simulation_results == {}
    
    @patch('src.training.nest_training.console')
    def test_nest_installation_check(self, mock_console, trainer):
        """Test NEST installation check."""
        trainer._check_nest_installation()
        
        # Should call console.print for NEST not available
        mock_console.print.assert_called()
    
    def test_create_network_model_no_nest(self, trainer):
        """Test network model creation when NEST is not available."""
        result = trainer.create_network_model("test_model")
        assert result is None
    
    @patch('src.training.nest_training.console')
    @patch('src.training.nest_training.nest')
    def test_create_network_model_with_nest(self, mock_nest, mock_console, trainer):
        """Test network model creation with NEST available."""
        # Mock NEST availability
        trainer.nest_available = True
        
        # Mock NEST objects
        mock_neurons = Mock()
        mock_spike_recorder = Mock()
        mock_nest.Create.side_effect = [mock_neurons, mock_spike_recorder]
        
        result = trainer.create_network_model("test_model")
        
        assert result is not None
        assert "neurons" in result
        assert "spike_recorder" in result
        assert "network_size" in result
        assert result["network_size"] == trainer.config.num_neurons
        
        # Verify NEST calls
        mock_nest.ResetKernel.assert_called_once()
        mock_nest.SetKernelStatus.assert_called()
        mock_nest.Create.assert_called()
        mock_nest.Connect.assert_called()
    
    @patch('src.training.nest_training.console')
    @patch('src.training.nest_training.nest')
    def test_run_simulation_no_nest(self, mock_nest, mock_console, trainer):
        """Test simulation when NEST is not available."""
        result = asyncio.run(trainer.run_simulation("test_model", {}))
        assert result is None
    
    @patch('src.training.nest_training.console')
    @patch('src.training.nest_training.nest')
    @patch('src.training.nest_training.np')
    def test_run_simulation_with_nest(self, mock_np, mock_nest, mock_console, trainer):
        """Test simulation with NEST available."""
        # Mock NEST availability
        trainer.nest_available = True
        
        # Mock model data
        model_data = {
            "spike_recorder": Mock(),
            "neurons": Mock()
        }
        
        # Mock spike data
        mock_spike_data = {
            "times": [10.0, 20.0, 30.0],
            "senders": [1, 2, 1]
        }
        model_data["spike_recorder"].get.return_value = mock_spike_data
        
        # Mock numpy unique
        mock_np.unique.return_value = [1, 2]
        
        result = asyncio.run(trainer.run_simulation("test_model", model_data))
        
        assert result is not None
        assert result["model_name"] == "test_model"
        assert "simulation_time" in result
        assert "spike_data" in result
        assert "timestamp" in result
        assert "metrics" in result
        
        # Verify metrics
        metrics = result["metrics"]
        assert metrics["total_spikes"] == 3
        assert metrics["unique_neurons"] == 2
        
        # Verify NEST calls
        mock_nest.Simulate.assert_called_once_with(trainer.config.simulation_time)
    
    @patch('src.training.nest_training.console')
    @patch('src.training.nest_training.nest')
    def test_train_network(self, mock_nest, mock_console, trainer, temp_dir):
        """Test network training."""
        # Mock NEST availability
        trainer.nest_available = True
        
        # Mock NEST objects
        mock_neurons = Mock()
        mock_spike_recorder = Mock()
        mock_nest.Create.side_effect = [mock_neurons, mock_spike_recorder]
        
        # Mock spike data
        mock_spike_data = {
            "times": [10.0, 20.0],
            "senders": [1, 2]
        }
        mock_spike_recorder.get.return_value = mock_spike_data
        
        # Run training
        results = asyncio.run(trainer.train_network("test_network", epochs=2))
        
        assert results is not None
        assert len(results) == 2
        
        # Check each epoch result
        for i, result in enumerate(results):
            assert result["epoch"] == i
            assert "results" in result
            assert "timestamp" in result
        
        # Check that results were saved
        results_file = os.path.join(temp_dir, "nest_training_results.json")
        assert os.path.exists(results_file)
        
        # Verify saved data
        with open(results_file, 'r') as f:
            saved_data = json.load(f)
        assert len(saved_data) == 2
    
    def test_save_results(self, trainer, temp_dir):
        """Test saving training results."""
        # Create test results
        test_results = [
            {
                "epoch": 0,
                "results": {"test": "data"},
                "timestamp": "2024-01-01T00:00:00"
            }
        ]
        
        trainer._save_results(test_results)
        
        # Check that file was created
        results_file = os.path.join(temp_dir, "nest_training_results.json")
        assert os.path.exists(results_file)
        
        # Verify saved data
        with open(results_file, 'r') as f:
            saved_data = json.load(f)
        assert len(saved_data) == 1
        assert saved_data[0]["epoch"] == 0


class TestIntegration:
    """Integration tests for NEST training."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('src.training.nest_training.nest')
    def test_full_training_pipeline(self, mock_nest, temp_dir):
        """Test full training pipeline with mocked NEST."""
        # Mock NEST availability
        with patch('src.training.nest_training.console'):
            config = NESTConfig(
                simulation_time=50.0,
                num_neurons=50,
                output_dir=temp_dir
            )
            
            trainer = NESTTrainer(config)
            trainer.nest_available = True
            
            # Mock NEST objects
            mock_neurons = Mock()
            mock_spike_recorder = Mock()
            mock_nest.Create.side_effect = [mock_neurons, mock_spike_recorder]
            
            # Mock spike data
            mock_spike_data = {
                "times": [10.0, 15.0, 20.0],
                "senders": [1, 2, 1]
            }
            mock_spike_recorder.get.return_value = mock_spike_data
            
            # Run training
            results = asyncio.run(trainer.train_network("test_network", epochs=3))
            
            # Verify results
            assert results is not None
            assert len(results) == 3
            
            # Verify files were created
            results_file = os.path.join(temp_dir, "nest_training_results.json")
            assert os.path.exists(results_file)
            
            # Verify NEST was called correctly
            assert mock_nest.ResetKernel.called
            assert mock_nest.SetKernelStatus.called
            assert mock_nest.Create.called
            assert mock_nest.Connect.called
            assert mock_nest.Simulate.called


def run_tests():
    """Run all tests."""
    print("Running NEST Training Tests...")
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])


if __name__ == "__main__":
    run_tests()
