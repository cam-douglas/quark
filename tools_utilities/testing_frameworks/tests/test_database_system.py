#!/usr/bin/env python3
"""
Test Suite for Universal Domain Database System
Comprehensive tests for all components of the self-learning database system
"""

import unittest
import json
import tempfile
import shutil
import os, sys
from pathlib import Path
from datetime import datetime

# Add the database directory to Python path
database_path = Path(__file__).parent.parent / "database"
sys.path.insert(0, str(database_path))

from learning_engine.self_learning_system import SelfLearningSystem
from scrapers.internet_scraper import InternetScraper

class TestDatabaseSystem(unittest.TestCase):
    """Test cases for the Universal Domain Database System"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.database_path = Path(self.test_dir) / "test_database"
        
        # Create test database structure
        self.database_path.mkdir(parents=True, exist_ok=True)
        (self.database_path / "domains").mkdir()
        (self.database_path / "data_sources").mkdir()
        (self.database_path / "synthetic_data").mkdir()
        (self.database_path / "analytics").mkdir()
        
        # Initialize test system
        self.system = SelfLearningSystem(str(self.database_path))
        self.scraper = InternetScraper(str(self.database_path))
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.database_path, str(self.database_path))
        self.assertIsInstance(self.system.domains, dict)
        self.assertIsInstance(self.system.data_sources, dict)
        self.assertIsInstance(self.system.learning_metrics, dict)
    
    def test_domain_discovery(self):
        """Test domain discovery functionality"""
        search_terms = ["quantum physics", "machine learning"]
        discovered_domains = self.system.discover_new_domains(search_terms)
        
        self.assertIsInstance(discovered_domains, list)
        self.assertGreater(len(discovered_domains), 0)
        
        for domain in discovered_domains:
            self.assertIn('domain_id', domain)
            self.assertIn('name', domain)
            self.assertIn('confidence', domain)
            self.assertGreaterEqual(domain['confidence'], 0.0)
            self.assertLessEqual(domain['confidence'], 1.0)
    
    def test_data_source_integration(self):
        """Test data source integration"""
        source_url = "https://example.com/dataset"
        metadata = {
            "name": "Test Dataset",
            "description": "A test dataset for unit testing",
            "domain": "test",
            "data_types": ["tabular"]
        }
        
        source_id = self.system.integrate_data_source(source_url, metadata)
        
        self.assertIsInstance(source_id, str)
        self.assertIn(source_id, self.system.data_sources)
        
        source_data = self.system.data_sources[source_id]
        self.assertEqual(source_data['url'], source_url)
        self.assertEqual(source_data['name'], "Test Dataset")
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        # First, create a test domain
        test_domain = {
            "domain_id": "test_neuroscience",
            "name": "Neuroscience",
            "description": "Test neuroscience domain",
            "parent_domain": "sciences",
            "key_concepts": ["neurons", "synapses"],
            "complexity_level": "advanced"
        }
        
        # Save test domain
        domain_file = self.database_path / "domains" / "test_neuroscience.json"
        with open(domain_file, 'w') as f:
            json.dump(test_domain, f)
        
        # Reload system to include new domain
        self.system._load_existing_data()
        
        # Generate synthetic data
        synthetic_data = self.system.generate_synthetic_data("test_neuroscience", "neural_activity")
        
        self.assertIn('generated_at', synthetic_data)
        self.assertIn('domain_id', synthetic_data)
        self.assertIn('data_type', synthetic_data)
        self.assertIn('synthetic_id', synthetic_data)
        self.assertIn('data', synthetic_data)
        
        # Check that data was saved
        data_file = self.database_path / "synthetic_data" / f"{synthetic_data['synthetic_id']}.json"
        self.assertTrue(data_file.exists())
    
    def test_learning_metrics_update(self):
        """Test learning metrics update"""
        initial_sessions = self.system.learning_metrics['learning_sessions']
        
        self.system.update_learning_metrics()
        
        self.assertEqual(
            self.system.learning_metrics['learning_sessions'], 
            initial_sessions + 1
        )
        
        # Check that metrics file was created
        metrics_file = self.database_path / "analytics" / "learning_metrics.json"
        self.assertTrue(metrics_file.exists())
    
    def test_system_status(self):
        """Test system status retrieval"""
        status = self.system.get_system_status()
        
        self.assertIn('total_domains', status)
        self.assertIn('total_data_sources', status)
        self.assertIn('learning_metrics', status)
        self.assertIn('recent_activity', status)
        
        self.assertIsInstance(status['total_domains'], int)
        self.assertIsInstance(status['total_data_sources'], int)
        self.assertIsInstance(status['learning_metrics'], dict)
        self.assertIsInstance(status['recent_activity'], list)
    
    def test_internet_scraper_initialization(self):
        """Test internet scraper initialization"""
        self.assertIsNotNone(self.scraper)
        self.assertEqual(self.scraper.database_path, str(self.database_path))
        self.assertIsInstance(self.scraper.data_platforms, dict)
        self.assertIn('github', self.scraper.data_platforms)
        self.assertIn('kaggle', self.scraper.data_platforms)
    
    def test_dataset_discovery(self):
        """Test dataset discovery functionality"""
        keywords = ["neuroscience", "brain"]
        datasets = self.scraper.discover_datasets(keywords, max_results=5)
        
        self.assertIsInstance(datasets, list)
        
        # Even if no real datasets are found, the structure should be correct
        for dataset in datasets:
            self.assertIn('source_id', dataset)
            self.assertIn('name', dataset)
            self.assertIn('url', dataset)
            self.assertIn('platform', dataset)
            self.assertIn('keywords', dataset)
            self.assertIn('data_types', dataset)
            self.assertIn('domain', dataset)
    
    def test_data_type_inference(self):
        """Test data type inference"""
        test_cases = [
            ("This dataset contains images of brain scans", ["image"]),
            ("Text corpus for natural language processing", ["text"]),
            ("Audio recordings of speech patterns", ["audio"]),
            ("Video recordings of animal behavior", ["video"]),
            ("Tabular data with CSV format", ["tabular"]),
            ("Network graph showing connections", ["graph"]),
            ("Time series data over multiple years", ["time_series"])
        ]
        
        for description, expected_types in test_cases:
            inferred_types = self.scraper._infer_data_types(description)
            for expected_type in expected_types:
                self.assertIn(expected_type, inferred_types)
    
    def test_domain_inference(self):
        """Test domain inference"""
        test_cases = [
            ("Brain imaging and neural networks", "neuroscience"),
            ("Protein structures and enzyme kinetics", "biochemistry"),
            ("Quantum mechanics and particle physics", "physics"),
            ("Molecular reactions and chemical compounds", "chemistry"),
            ("Cell biology and genetic evolution", "biology"),
            ("Mathematical algorithms and statistics", "mathematics"),
            ("Programming and artificial intelligence", "computer_science"),
            ("Behavioral psychology and cognition", "psychology"),
            ("Market economics and financial trade", "economics")
        ]
        
        for description, expected_domain in test_cases:
            inferred_domain = self.scraper._infer_domain(description, [])
            self.assertEqual(inferred_domain, expected_domain)
    
    def test_metadata_extraction(self):
        """Test metadata extraction from HTML"""
        test_html = """
        <html>
        <head>
            <title>Test Dataset Title</title>
            <meta name="description" content="This is a test dataset description">
            <meta name="keywords" content="test, dataset, neuroscience, brain">
        </head>
        <body>
            <p>This is a paragraph with additional information.</p>
        </body>
        </html>
        """
        
        title = self.scraper._extract_title(test_html)
        description = self.scraper._extract_description(test_html)
        keywords = self.scraper._extract_keywords(test_html)
        
        self.assertEqual(title, "Test Dataset Title")
        self.assertEqual(description, "This is a test dataset description")
        self.assertIn("test", keywords)
        self.assertIn("dataset", keywords)
        self.assertIn("neuroscience", keywords)
        self.assertIn("brain", keywords)
    
    def test_dataset_saving(self):
        """Test dataset saving functionality"""
        test_datasets = [
            {
                "source_id": "test_1",
                "name": "Test Dataset 1",
                "url": "https://example.com/1",
                "platform": "github"
            },
            {
                "source_id": "test_2", 
                "name": "Test Dataset 2",
                "url": "https://example.com/2",
                "platform": "kaggle"
            }
        ]
        
        self.scraper.save_discovered_datasets(test_datasets, "test_datasets.json")
        
        # Check that file was created
        dataset_file = self.database_path / "data_sources" / "test_datasets.json"
        self.assertTrue(dataset_file.exists())
        
        # Check that data was saved correctly
        with open(dataset_file, 'r') as f:
            saved_datasets = json.load(f)
        
        self.assertEqual(len(saved_datasets), 2)
        self.assertEqual(saved_datasets[0]['name'], "Test Dataset 1")
        self.assertEqual(saved_datasets[1]['name'], "Test Dataset 2")

class TestDomainSpecificDataGeneration(unittest.TestCase):
    """Test domain-specific synthetic data generation"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.database_path = Path(self.test_dir) / "test_database"
        self.database_path.mkdir(parents=True, exist_ok=True)
        (self.database_path / "domains").mkdir()
        (self.database_path / "synthetic_data").mkdir()
        
        self.system = SelfLearningSystem(str(self.database_path))
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_neuroscience_data_generation(self):
        """Test neuroscience-specific data generation"""
        # Create neuroscience domain
        neuro_domain = {
            "domain_id": "neuroscience",
            "name": "Neuroscience",
            "description": "Study of the nervous system"
        }
        
        domain_file = self.database_path / "domains" / "neuroscience.json"
        with open(domain_file, 'w') as f:
            json.dump(neuro_domain, f)
        
        self.system._load_existing_data()
        
        # Test neural activity data
        neural_data = self.system.generate_synthetic_data("neuroscience", "neural_activity")
        data = neural_data['data']
        
        self.assertIn('neuron_ids', data)
        self.assertIn('spike_times', data)
        self.assertIn('firing_rates', data)
        self.assertIn('brain_region', data)
        
        # Test connectivity data
        connectivity_data = self.system.generate_synthetic_data("neuroscience", "connectivity")
        data = connectivity_data['data']
        
        self.assertIn('source_neurons', data)
        self.assertIn('target_neurons', data)
        self.assertIn('connection_strengths', data)
        self.assertIn('synapse_types', data)
    
    def test_biochemistry_data_generation(self):
        """Test biochemistry-specific data generation"""
        # Create biochemistry domain
        bio_domain = {
            "domain_id": "biochemistry",
            "name": "Biochemistry",
            "description": "Study of chemical processes in living organisms"
        }
        
        domain_file = self.database_path / "domains" / "biochemistry.json"
        with open(domain_file, 'w') as f:
            json.dump(bio_domain, f)
        
        self.system._load_existing_data()
        
        # Test metabolic pathway data
        pathway_data = self.system.generate_synthetic_data("biochemistry", "metabolic_pathway")
        data = pathway_data['data']
        
        self.assertIn('enzymes', data)
        self.assertIn('substrates', data)
        self.assertIn('products', data)
        self.assertIn('reaction_rates', data)
        self.assertIn('pathway_name', data)
        
        # Test protein structure data
        protein_data = self.system.generate_synthetic_data("biochemistry", "protein_structure")
        data = protein_data['data']
        
        self.assertIn('protein_id', data)
        self.assertIn('sequence', data)
        self.assertIn('structure_type', data)
        self.assertIn('molecular_weight', data)
        self.assertIn('isoelectric_point', data)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDatabaseSystem))
    test_suite.addTest(unittest.makeSuite(TestDomainSpecificDataGeneration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
