#!/usr/bin/env python3
"""
Test Script for PaperQA Training Module
=======================================

This script provides comprehensive testing for the PaperQA training module,
including unit tests, integration tests, and validation of brain simulation
components integration.

Author: Brain Simulation Framework Team
Date: 2024
License: Apache-2.0
"""

import os, sys
import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

# Add the training module to path
sys.path.insert(0, str(Path(__file__).parent))

from paperqa_training import PaperQABrainTrainer, TrainingConfig


class TestTrainingConfig:
    """Test cases for TrainingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.paper_directory == "papers"
        assert config.index_directory == "indexes"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.enable_brain_simulation is True
        assert config.neural_dynamics_enabled is True
        assert config.cognitive_science_enabled is True
        assert config.machine_learning_enabled is True
        assert config.max_questions == 100
        assert config.batch_size == 10
        assert config.working_memory_slots == 4
        assert config.attention_heads == 8
        assert config.neural_plasticity_rate == 0.1
        assert config.output_dir == "training_outputs"
        assert config.save_embeddings is True
        assert config.save_models is True
        assert config.evaluation_split == 0.2
        assert "accuracy" in config.metrics
        assert "precision" in config.metrics
        assert "recall" in config.metrics
        assert "f1" in config.metrics
        assert "response_time" in config.metrics
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            paper_directory="custom_papers",
            model_name="gpt-4",
            enable_brain_simulation=False,
            max_questions=50,
            working_memory_slots=8,
            output_dir="custom_output"
        )
        
        assert config.paper_directory == "custom_papers"
        assert config.model_name == "gpt-4"
        assert config.enable_brain_simulation is False
        assert config.max_questions == 50
        assert config.working_memory_slots == 8
        assert config.output_dir == "custom_output"


class TestPaperQABrainTrainer:
    """Test cases for PaperQABrainTrainer class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create a test configuration."""
        return TrainingConfig(
            paper_directory=os.path.join(temp_dir, "papers"),
            index_directory=os.path.join(temp_dir, "indexes"),
            output_dir=os.path.join(temp_dir, "output"),
            enable_brain_simulation=False  # Disable for testing without brain components
        )
    
    @pytest.fixture
    def trainer(self, config):
        """Create a test trainer instance."""
        return PaperQABrainTrainer(config)
    
    def test_trainer_initialization(self, trainer, config):
        """Test trainer initialization."""
        assert trainer.config == config
        assert trainer.console is not None
        assert trainer.docs is None
        assert trainer.neural_components is None
        assert trainer.capacity_progression is None
        assert trainer.sleep_engine is None
        assert trainer.multi_scale_integration is None
        assert trainer.biological_validator is None
        assert trainer.training_history == []
        assert trainer.performance_metrics == {}
        assert trainer.neural_state == {}
    
    @patch('paperqa_training.NeuralComponents')
    @patch('paperqa_training.CapacityProgression')
    @patch('paperqa_training.SleepConsolidationEngine')
    @patch('paperqa_training.MultiScaleIntegration')
    @patch('paperqa_training.BiologicalValidator')
    def test_brain_components_initialization(self, mock_validator, mock_integration, 
                                           mock_sleep, mock_capacity, mock_neural, config):
        """Test brain components initialization."""
        config.enable_brain_simulation = True
        config.neural_dynamics_enabled = True
        
        trainer = PaperQABrainTrainer(config)
        
        mock_neural.assert_called_once_with(
            working_memory_slots=config.working_memory_slots,
            attention_heads=config.attention_heads
        )
        mock_capacity.assert_called_once()
        mock_sleep.assert_called_once()
        mock_integration.assert_called_once()
        mock_validator.assert_called_once()
    
    @patch('paperqa_training.PaperQASettings')
    @patch('paperqa_training.Docs')
    @pytest.mark.asyncio
    async def test_paperqa_initialization(self, mock_docs, mock_settings, trainer, temp_dir):
        """Test PaperQA initialization."""
        # Create paper directory
        paper_dir = os.path.join(temp_dir, "papers")
        os.makedirs(paper_dir, exist_ok=True)
        
        # Create a dummy PDF file
        dummy_pdf = os.path.join(paper_dir, "test.pdf")
        with open(dummy_pdf, 'w') as f:
            f.write("dummy pdf content")
        
        await trainer.initialize_paperqa()
        
        mock_settings.assert_called_once()
        mock_docs.assert_called_once()
        assert trainer.docs is not None
    
    @patch('paperqa_training.Docs')
    @pytest.mark.asyncio
    async def test_paperqa_initialization_no_papers(self, mock_docs, trainer):
        """Test PaperQA initialization with no papers directory."""
        await trainer.initialize_paperqa()
        
        mock_docs.assert_called_once()
        assert trainer.docs is not None
    
    @patch('paperqa_training.Progress')
    @patch.object(PaperQABrainTrainer, '_apply_neural_dynamics')
    @patch.object(PaperQABrainTrainer, '_apply_cognitive_science')
    @patch.object(PaperQABrainTrainer, '_apply_machine_learning')
    @patch.object(PaperQABrainTrainer, '_update_neural_state')
    @patch.object(PaperQABrainTrainer, '_sleep_consolidation')
    @patch.object(PaperQABrainTrainer, '_calculate_metrics')
    @patch.object(PaperQABrainTrainer, '_save_results')
    @pytest.mark.asyncio
    async def test_train_on_questions(self, mock_save, mock_calc, mock_sleep, 
                                    mock_update, mock_ml, mock_cog, mock_neural, 
                                    mock_progress, trainer):
        """Test training on questions."""
        # Mock the docs object
        mock_answer = Mock()
        mock_answer.answer = "Test answer"
        mock_answer.confidence = 0.8
        mock_answer.sources = ["source1", "source2"]
        
        trainer.docs = AsyncMock()
        trainer.docs.aquery.return_value = mock_answer
        
        # Mock progress context manager
        mock_progress.return_value.__enter__.return_value.add_task.return_value = 1
        
        questions = [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"}
        ]
        
        results = await trainer.train_on_questions(questions)
        
        assert len(results) == 2
        assert results[0]['question'] == "What is AI?"
        assert results[0]['actual_answer'] == "Test answer"
        assert results[0]['confidence'] == 0.8
        assert results[0]['sources'] == ["source1", "source2"]
        
        # Verify method calls
        assert trainer.docs.aquery.call_count == 2
        mock_calc.assert_called_once()
        mock_save.assert_called_once()
    
    @patch.object(PaperQABrainTrainer, '_display_metrics')
    def test_calculate_metrics(self, mock_display, trainer):
        """Test metrics calculation."""
        results = [
            {
                'question': 'Test 1',
                'actual_answer': 'Answer 1',
                'confidence': 0.8,
                'timestamp': '2024-01-01T00:00:00'
            },
            {
                'question': 'Test 2',
                'actual_answer': 'Answer 2',
                'confidence': 0.9,
                'timestamp': '2024-01-01T00:01:00'
            },
            {
                'question': 'Test 3',
                'actual_answer': None,
                'confidence': None,
                'timestamp': '2024-01-01T00:02:00'
            }
        ]
        
        trainer._calculate_metrics(results)
        
        assert trainer.performance_metrics['total_questions'] == 3
        assert trainer.performance_metrics['successful_queries'] == 2
        assert trainer.performance_metrics['success_rate'] == 2/3
        assert trainer.performance_metrics['avg_confidence'] == 0.85
        assert trainer.performance_metrics['confidence_std'] == 0.05
        
        mock_display.assert_called_once()
    
    def test_display_metrics(self, trainer):
        """Test metrics display."""
        metrics = {
            'total_questions': 10,
            'success_rate': 0.8,
            'avg_confidence': 0.85
        }
        
        # This should not raise an exception
        trainer._display_metrics(metrics)
    
    def test_save_results(self, trainer, temp_dir):
        """Test results saving."""
        trainer.config.output_dir = temp_dir
        trainer.performance_metrics = {'test': 'value'}
        trainer.neural_state = {'neural': 'state'}
        
        results = [
            {'question': 'Test', 'answer': 'Test answer'}
        ]
        
        trainer._save_results(results)
        
        # Check if files were created
        assert os.path.exists(os.path.join(temp_dir, "training_results.json"))
        assert os.path.exists(os.path.join(temp_dir, "performance_metrics.json"))
        assert os.path.exists(os.path.join(temp_dir, "neural_state.json"))
    
    def test_generate_training_report(self, trainer, temp_dir):
        """Test training report generation."""
        trainer.config.output_dir = temp_dir
        trainer.performance_metrics = {
            'total_questions': 10,
            'success_rate': 0.8
        }
        trainer.neural_state = {'neural': 'state'}
        
        trainer.generate_training_report()
        
        report_file = os.path.join(temp_dir, "training_report.md")
        assert os.path.exists(report_file)
        
        # Check report content
        with open(report_file, 'r') as f:
            content = f.read()
            assert "PaperQA Training Report" in content
            assert "Configuration" in content
            assert "Performance Metrics" in content
            assert "Neural State Summary" in content


class TestIntegration:
    """Integration tests for the complete training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('paperqa_training.Docs')
    @pytest.mark.asyncio
    async def test_full_training_pipeline(self, mock_docs, temp_dir):
        """Test the complete training pipeline."""
        config = TrainingConfig(
            paper_directory=os.path.join(temp_dir, "papers"),
            index_directory=os.path.join(temp_dir, "indexes"),
            output_dir=os.path.join(temp_dir, "output"),
            enable_brain_simulation=False
        )
        
        trainer = PaperQABrainTrainer(config)
        
        # Mock the docs object
        mock_answer = Mock()
        mock_answer.answer = "Test answer"
        mock_answer.confidence = 0.8
        mock_answer.sources = ["source1"]
        
        trainer.docs = AsyncMock()
        trainer.docs.aquery.return_value = mock_answer
        
        # Initialize PaperQA
        await trainer.initialize_paperqa()
        
        # Train on questions
        questions = [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"}
        ]
        
        results = await trainer.train_on_questions(questions)
        
        # Verify results
        assert len(results) == 2
        assert all('question' in r for r in results)
        assert all('actual_answer' in r for r in results)
        assert all('confidence' in r for r in results)
        
        # Verify output files
        assert os.path.exists(os.path.join(temp_dir, "output", "training_results.json"))
        assert os.path.exists(os.path.join(temp_dir, "output", "performance_metrics.json"))
        assert os.path.exists(os.path.join(temp_dir, "output", "training_report.md"))


def run_tests():
    """Run all tests."""
    print("Running PaperQA Training Tests...")
    
    # Run pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])


if __name__ == "__main__":
    run_tests()
