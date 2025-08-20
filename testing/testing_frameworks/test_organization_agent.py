#!/usr/bin/env python3
"""
Test Organization Agent with Connectome Integration
Purpose: Test the organization agent's semantic classification and connectome integration
Inputs: Test files, mock connectome data
Outputs: Test results, validation reports
Seeds: 42 (for reproducible tests)
Dependencies: pytest, organization_agent, unittest
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import os, sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from architecture.orchestrator.organization_agent import OrganizationAgent

class TestOrganizationAgent(unittest.TestCase):
    """Test cases for the Organization Agent"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.agent = OrganizationAgent(str(self.test_dir))
        
        # Create mock directory structure
        self.create_test_structure()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        
    def create_test_structure(self):
        """Create test directory structure with sample files"""
        # Create directories
        (self.test_dir / "brain_modules" / "connectome" / "exports").mkdir(parents=True)
        (self.test_dir / "logs").mkdir(exist_ok=True)
        
        # Create sample Python files with different brain content
        self.create_sample_file("test_hippocampus.py", """
import numpy as np
import torch

class HippocampalMemorySystem:
    def __init__(self):
        self.episodic_memory = []
        self.consolidation_buffer = []
        
    def encode_episodic_memory(self, experience):
        '''Encode new episodic memory into hippocampus'''
        self.episodic_memory.append(experience)
        
    def consolidate_memory(self):
        '''Consolidate memory during sleep cycles'''
        for memory in self.consolidation_buffer:
            # Process memory consolidation
            pass
""")
        
        self.create_sample_file("test_prefrontal.py", """
import torch
from typing import List, Dict

class PrefrontalCortexExecutive:
    def __init__(self):
        self.working_memory = []
        self.executive_control = True
        
    def plan_action_sequence(self, goal):
        '''Executive planning and reasoning'''
        action_sequence = []
        return action_sequence
        
    def control_attention(self, stimuli):
        '''Prefrontal cortex attention control'''
        filtered_stimuli = []
        return filtered_stimuli
""")
        
        self.create_sample_file("test_ml_model.py", """
import torch
import torch.nn as nn
import sklearn
from sklearn.ensemble import RandomForestClassifier

class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
    def forward(self, x):
        return self.layers(x)
        
def train_model(data):
    '''Train machine learning model'''
    model = NeuralNetworkModel()
    return model
""")
        
        self.create_sample_file("test_config.yaml", """
brain_modules:
  - hippocampus
  - prefrontal_cortex
  - thalamus
  
parameters:
  learning_rate: 0.001
  batch_size: 32
""")
        
        self.create_sample_file("README.md", """
# Test Project
This is a test project for the organization agent.
""")
        
        # Create mock connectome data
        self.create_mock_connectome_data()
        
    def create_sample_file(self, filename: str, content: str):
        """Create a sample file with given content"""
        file_path = self.test_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
            
    def create_mock_connectome_data(self):
        """Create mock connectome data for testing"""
        exports_dir = self.test_dir / "brain_modules" / "connectome" / "exports"
        
        # Mock build summary
        build_summary = {
            "nodes": 1000,
            "edges": 5000,
            "modules": ["PFC", "HIP", "THA", "BG"]
        }
        with open(exports_dir / "build_summary.json", 'w') as f:
            json.dump(build_summary, f)
            
        # Mock module manifests
        hip_manifest = {
            "module_id": "HIP",
            "label": "Hippocampus",
            "role": "episodic_memory",
            "population": 200,
            "keywords": ["memory", "episodic", "consolidation", "hippocampus"]
        }
        with open(exports_dir / "hip_manifest.json", 'w') as f:
            json.dump(hip_manifest, f)
            
        pfc_manifest = {
            "module_id": "PFC", 
            "label": "Prefrontal Cortex",
            "role": "executive_control",
            "population": 300,
            "keywords": ["executive", "planning", "cortex", "prefrontal"]
        }
        with open(exports_dir / "pfc_manifest.json", 'w') as f:
            json.dump(pfc_manifest, f)
    
    def test_semantic_analysis(self):
        """Test semantic analysis of Python files"""
        # Test hippocampus file
        hip_file = self.test_dir / "test_hippocampus.py"
        analysis = self.agent.analyze_code_semantics(hip_file)
        
        self.assertGreater(analysis['brain_relevance'], 0)
        self.assertIn('hippocampus', analysis['brain_keywords'])
        self.assertIn('memory', analysis['brain_keywords'])
        self.assertGreater(analysis['complexity_score'], 0)
        
        # Test prefrontal file
        pfc_file = self.test_dir / "test_prefrontal.py"
        analysis = self.agent.analyze_code_semantics(pfc_file)
        
        self.assertGreater(analysis['brain_relevance'], 0)
        self.assertIn('cortex', analysis['brain_keywords'])
        self.assertIn('prefrontal', analysis['brain_keywords'])
        
    def test_connectome_classification(self):
        """Test connectome-based file classification"""
        # Test hippocampus file classification
        hip_file = self.test_dir / "test_hippocampus.py"
        classification = self.agent.classify_by_connectome_relationships(hip_file)
        
        self.assertEqual(classification, 'brain_modules/hippocampus')
        
        # Test prefrontal file classification
        pfc_file = self.test_dir / "test_prefrontal.py"
        classification = self.agent.classify_by_connectome_relationships(pfc_file)
        
        self.assertEqual(classification, 'brain_modules/prefrontal_cortex')
        
        # Test ML file classification
        ml_file = self.test_dir / "test_ml_model.py"
        classification = self.agent.classify_by_connectome_relationships(ml_file)
        
        # Should not be brain-related, so returns None
        self.assertIsNone(classification)
        
    def test_semantic_clustering(self):
        """Test semantic clustering functionality"""
        clusters = self.agent.build_semantic_clusters()
        
        # Should have brain modules cluster
        brain_clusters = [c for c in clusters.keys() if 'brain_modules' in c]
        self.assertGreater(len(brain_clusters), 0)
        
        # Should have machine learning cluster
        ml_clusters = [c for c in clusters.keys() if 'machine_learning' in c]
        self.assertGreater(len(ml_clusters), 0)
        
    def test_file_classification_integration(self):
        """Test integrated file classification"""
        # Test Python file with brain content
        hip_file = self.test_dir / "test_hippocampus.py"
        classification = self.agent.classify_file(hip_file)
        
        self.assertEqual(classification, 'brain_modules/hippocampus')
        
        # Test config file
        config_file = self.test_dir / "test_config.yaml"
        classification = self.agent.classify_file(config_file)
        
        self.assertEqual(classification, 'configs')
        
        # Test README file
        readme_file = self.test_dir / "README.md"
        classification = self.agent.classify_file(readme_file)
        
        # README should stay in root
        self.assertIsNone(classification)
        
    def test_connectome_metadata_loading(self):
        """Test loading connectome metadata"""
        metadata = self.agent.load_connectome_metadata()
        
        self.assertIn('build_summary', metadata)
        self.assertIn('HIP', metadata)
        self.assertIn('PFC', metadata)
        
        # Check module data
        hip_data = metadata['HIP']
        self.assertEqual(hip_data['module_id'], 'HIP')
        self.assertEqual(hip_data['role'], 'episodic_memory')
        
    def test_organization_dry_run(self):
        """Test organization in dry run mode"""
        summary = self.agent.organize_by_semantic_clusters(dry_run=True)
        
        self.assertGreater(summary['scanned'], 0)
        self.assertGreaterEqual(summary['clusters_created'], 0)
        self.assertEqual(summary['errors'], 0)
        
        # Files should not actually be moved in dry run
        hip_file = self.test_dir / "test_hippocampus.py"
        self.assertTrue(hip_file.exists())
        
    def test_validate_structure(self):
        """Test directory structure validation"""
        report = self.agent.validate_structure()
        
        self.assertIn('valid', report)
        self.assertIn('issues', report)
        self.assertIn('structure_health', report)
        
        # Should initially be invalid due to files in root
        self.assertFalse(report['valid'])
        self.assertGreater(len(report['issues']), 0)
        
    def test_temp_file_handling(self):
        """Test temporary file detection and handling"""
        # Create temporary files
        temp_files = [
            self.test_dir / "temp.tmp",
            self.test_dir / "backup.bak", 
            self.test_dir / "log.log"
        ]
        
        for temp_file in temp_files:
            temp_file.write_text("temporary content")
            
        # Test classification
        for temp_file in temp_files:
            classification = self.agent.classify_file(temp_file)
            self.assertEqual(classification, "DELETE")
            
    def test_brain_keyword_detection(self):
        """Test brain keyword detection accuracy"""
        # Create file with many brain keywords
        brain_heavy_file = self.test_dir / "brain_heavy.py"
        content = """
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.neurons = []
        self.synapses = []
        self.plasticity = 0.1
        self.cortex_layers = 6
        
    def spike_train(self):
        '''Generate spike trains for neural firing'''
        spikes = []
        return spikes
        
    def dopamine_modulation(self):
        '''Dopamine neuromodulator effects'''
        pass
        
    def synaptic_plasticity(self):
        '''STDP and LTP mechanisms'''
        pass
"""
        brain_heavy_file.write_text(content)
        
        analysis = self.agent.analyze_code_semantics(brain_heavy_file)
        
        # Should detect multiple brain keywords
        expected_keywords = ['neural', 'neuron', 'synapse', 'plasticity', 'cortex', 'spike', 'dopamine']
        found_keywords = analysis['brain_keywords']
        
        for keyword in expected_keywords:
            self.assertIn(keyword, found_keywords, f"Keyword '{keyword}' not detected")
            
        # Should have high brain relevance
        self.assertGreater(analysis['brain_relevance'], 0.01)


class TestOrganizationIntegration(unittest.TestCase):
    """Integration tests for organization system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.agent = OrganizationAgent(str(self.test_dir))
        
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.test_dir)
        
    def test_end_to_end_organization(self):
        """Test complete organization workflow"""
        # Create diverse set of files
        files_to_create = {
            "neural_network.py": "import torch\nclass NeuralNet:\n    def __init__(self):\n        self.layers = []\n",
            "hippocampus_model.py": "class HippocampusModel:\n    def consolidate_memory(self):\n        pass\n",
            "test_brain.py": "def test_neural_function():\n    assert True\n",
            "config.yaml": "settings:\n  learning_rate: 0.01\n",
            "results.json": '{"accuracy": 0.95}',
            "training_script.py": "def train_model():\n    pass\n",
            "temp.tmp": "temporary file"
        }
        
        for filename, content in files_to_create.items():
            (self.test_dir / filename).write_text(content)
            
        # Run organization
        summary = self.agent.scan_and_organize(dry_run=False)
        
        # Verify organization results
        self.assertGreater(summary['moved'] + summary['deleted'], 0)
        
        # Check that files were moved to appropriate locations
        expected_locations = {
            "config.yaml": "configs",
            "results.json": "data", 
            "test_brain.py": "tests"
        }
        
        for filename, expected_dir in expected_locations.items():
            target_path = self.test_dir / expected_dir / filename
            self.assertTrue(target_path.exists(), f"{filename} not moved to {expected_dir}")
            
        # Temp file should be deleted
        self.assertFalse((self.test_dir / "temp.tmp").exists())


def run_tests():
    """Run all organization agent tests"""
    print("🧪 Running Organization Agent Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestOrganizationAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestOrganizationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print(f"{'='*50}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
