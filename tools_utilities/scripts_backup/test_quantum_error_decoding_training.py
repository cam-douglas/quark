#!/usr/bin/env python3
"""
Test cases for quantum error decoding training script.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

# Import the classes to test
from quantum_error_decoding_training import (
    QuantumErrorDecodingConfig,
    SurfaceCode,
    QuantumErrorDecoder,
    QuantumErrorDecodingTrainer
)


class TestQuantumErrorDecodingConfig:
    """Test cases for QuantumErrorDecodingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QuantumErrorDecodingConfig()
        
        assert config.code_distance == 5
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 50
        assert config.hidden_dim == 256
        assert config.output_dir == "quantum_error_decoding_outputs"
        assert config.enable_brain_simulation is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = QuantumErrorDecodingConfig(
            code_distance=7,
            batch_size=64,
            learning_rate=1e-3,
            num_epochs=100,
            hidden_dim=512,
            output_dir="custom_output",
            enable_brain_simulation=False
        )
        
        assert config.code_distance == 7
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 100
        assert config.hidden_dim == 512
        assert config.output_dir == "custom_output"
        assert config.enable_brain_simulation is False


class TestSurfaceCode:
    """Test cases for SurfaceCode class."""
    
    def test_initialization(self):
        """Test surface code initialization."""
        code = SurfaceCode(code_distance=5)
        
        assert code.code_distance == 5
        assert code.data_qubits == (5, 5)
        assert code.stabilizer_qubits == (4, 4)
        assert code.data_state.shape == (5, 5)
        assert len(code.error_history) == 0
    
    def test_initialize_logical_state_0(self):
        """Test initializing logical state |0⟩."""
        code = SurfaceCode(code_distance=3)
        code.initialize_logical_state("0")
        
        # Check that all qubits are in |0⟩ state
        assert np.all(code.data_state == 0)
    
    def test_initialize_logical_state_1(self):
        """Test initializing logical state |1⟩."""
        code = SurfaceCode(code_distance=3)
        code.initialize_logical_state("1")
        
        # Check that first qubit is in |1⟩ state
        assert code.data_state[0, 0] == 1
        # Check that other qubits are in |0⟩ state
        assert np.all(code.data_state[1:, :] == 0)
        assert np.all(code.data_state[0, 1:] == 0)
    
    def test_apply_errors(self):
        """Test applying random errors."""
        code = SurfaceCode(code_distance=3)
        code.initialize_logical_state("0")
        
        # Apply errors with high probability to ensure some errors occur
        with patch('numpy.random.random') as mock_random:
            mock_random.return_value = 0.0  # Always apply errors
            code.apply_errors(error_rate=0.5)
        
        # Check that errors were applied
        assert len(code.error_history) > 0
        assert all(error_type == 'X' for error_type, _, _ in code.error_history)
    
    def test_measure_stabilizers(self):
        """Test stabilizer measurement."""
        code = SurfaceCode(code_distance=3)
        code.initialize_logical_state("0")
        
        # Apply some errors
        code.data_state[0, 0] = 1
        code.data_state[1, 1] = 1
        
        syndrome = code.measure_stabilizers()
        
        # Check syndrome shape
        assert syndrome.shape == (2, 2, 2)  # 2 types, 2x2 stabilizers
        
        # Check that syndrome contains binary values
        assert np.all(np.isin(syndrome, [0, 1]))


class TestQuantumErrorDecoder:
    """Test cases for QuantumErrorDecoder class."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        config = QuantumErrorDecodingConfig(code_distance=5)
        decoder = QuantumErrorDecoder(config)
        
        assert decoder.syndrome_height == 4
        assert decoder.syndrome_width == 4
        assert decoder.syndrome_channels == 2
    
    def test_forward_pass(self):
        """Test forward pass through the decoder."""
        config = QuantumErrorDecodingConfig(code_distance=5)
        decoder = QuantumErrorDecoder(config)
        
        # Create dummy input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 2, 4, 4)  # [batch, channels, height, width]
        
        # Forward pass
        output = decoder(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 2)  # Binary classification
        
        # Check that output contains finite values
        assert torch.isfinite(output).all()
    
    def test_network_architecture(self):
        """Test network architecture components."""
        config = QuantumErrorDecodingConfig(code_distance=5)
        decoder = QuantumErrorDecoder(config)
        
        # Check encoder layers
        assert len(decoder.encoder) == 6  # 3 Conv2d + 3 ReLU
        
        # Check classifier layers
        assert len(decoder.classifier) == 7  # Pool + Flatten + 3 Linear + 2 ReLU + Dropout
        
        # Check layer types
        assert isinstance(decoder.encoder[0], torch.nn.Conv2d)
        assert isinstance(decoder.encoder[1], torch.nn.ReLU)
        assert isinstance(decoder.classifier[0], torch.nn.AdaptiveAvgPool2d)
        assert isinstance(decoder.classifier[2], torch.nn.Linear)


class TestQuantumErrorDecodingTrainer:
    """Test cases for QuantumErrorDecodingTrainer class."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        config = QuantumErrorDecodingConfig()
        trainer = QuantumErrorDecodingTrainer(config)
        
        assert trainer.config == config
        assert trainer.surface_code is not None
        assert trainer.decoder is not None
        assert trainer.optimizer is not None
        assert len(trainer.training_history) == 0
    
    def test_generate_training_data(self):
        """Test training data generation."""
        config = QuantumErrorDecodingConfig()
        trainer = QuantumErrorDecodingTrainer(config)
        
        num_samples = 10
        syndromes, labels = trainer.generate_training_data(num_samples)
        
        # Check data types
        assert isinstance(syndromes, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        
        # Check shapes
        assert syndromes.shape[0] == num_samples
        assert labels.shape[0] == num_samples
        assert syndromes.shape[1:] == (2, 4, 4)  # 2 channels, 4x4 stabilizers
        assert labels.shape[1] == 1  # Single label per sample
        
        # Check label values
        assert torch.all(torch.isin(labels, torch.tensor([0, 1])))
    
    @pytest.mark.asyncio
    async def test_train_epoch(self):
        """Test training for one epoch."""
        config = QuantumErrorDecodingConfig()
        trainer = QuantumErrorDecodingTrainer(config)
        
        # Generate small dataset
        syndromes = torch.randn(8, 2, 4, 4)  # 8 samples
        labels = torch.randint(0, 2, (8,))  # Binary labels
        
        # Train one epoch
        loss = await trainer._train_epoch(syndromes, labels)
        
        # Check that loss is a number
        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative
    
    def test_save_training_results(self):
        """Test saving training results."""
        config = QuantumErrorDecodingConfig()
        trainer = QuantumErrorDecodingTrainer(config)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            # Create dummy training results
            training_results = [
                {"epoch": 0, "loss": 0.5, "timestamp": "2024-01-01T00:00:00"},
                {"epoch": 1, "loss": 0.4, "timestamp": "2024-01-01T00:01:00"}
            ]
            
            # Save results
            trainer._save_training_results(training_results)
            
            # Check that files were created
            results_file = os.path.join(temp_dir, "training_results.json")
            model_file = os.path.join(temp_dir, "quantum_error_decoder.pt")
            
            assert os.path.exists(results_file)
            assert os.path.exists(model_file)
            
            # Check results file content
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            
            assert len(saved_results) == 2
            assert saved_results[0]["epoch"] == 0
            assert saved_results[1]["epoch"] == 1


class TestIntegration:
    """Integration tests for the complete training pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_training_pipeline(self):
        """Test the complete training pipeline."""
        config = QuantumErrorDecodingConfig(
            code_distance=3,  # Smaller for faster testing
            num_epochs=2,     # Fewer epochs for testing
            batch_size=4      # Smaller batch size
        )
        
        trainer = QuantumErrorDecodingTrainer(config)
        
        # Run training
        results = await trainer.train_decoder()
        
        # Check results
        assert len(results) == 2  # 2 epochs
        assert all("epoch" in result for result in results)
        assert all("loss" in result for result in results)
        assert all("timestamp" in result for result in results)
        
        # Check that loss values are reasonable
        for result in results:
            assert isinstance(result["loss"], float)
            assert result["loss"] >= 0
    
    def test_surface_code_with_decoder(self):
        """Test integration between surface code and decoder."""
        config = QuantumErrorDecodingConfig(code_distance=3)
        trainer = QuantumErrorDecodingTrainer(config)
        
        # Generate data using surface code
        syndromes, labels = trainer.generate_training_data(4)
        
        # Process through decoder
        trainer.decoder.eval()
        with torch.no_grad():
            predictions = trainer.decoder(syndromes)
        
        # Check predictions
        assert predictions.shape == (4, 2)  # 4 samples, 2 classes
        assert torch.isfinite(predictions).all()


def run_tests():
    """Run all tests."""
    print("Running Quantum Error Decoding Training Tests...")
    
    # Run pytest
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "test_quantum_error_decoding_training.py", 
        "-v"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
