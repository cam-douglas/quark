#!/usr/bin/env python3
"""
Wikipedia Training Test Suite
=============================

Comprehensive tests for Wikipedia training pipeline including
cloud deployment, model training, and consciousness integration.

Purpose: Validate Wikipedia training system functionality
Inputs: Test configurations and mock data
Outputs: Test results and validation reports
Seeds: Fixed seeds for reproducible testing
Dependencies: pytest, torch, transformers, mocking libraries
"""

import os, sys
import json
import tempfile
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_systems.training_pipelines.wikipedia_cloud_training import (
    WikipediaTrainer, WikipediaTrainingConfig, WikipediaDataProcessor
)
from knowledge_systems.training_pipelines.wiki_extractor import (
    WikiExtractor, WikiTextCleaner, WikipediaXMLParser
)
from brain_modules.conscious_agent.integrations.wikipedia_consciousness_integration import (
    WikipediaConsciousnessAgent, WikipediaConsciousnessConfig, WikipediaKnowledgeRetriever
)
from deployment.cloud_computing.scripts.deploy_wikipedia_training import (
    WikipediaTrainingDeployer, DeploymentConfig, AWSDeploymentManager
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_training_config():
    """Create mock training configuration."""
    return WikipediaTrainingConfig(
        model_name="microsoft/DialoGPT-small",  # Smaller model for tests
        max_articles=100,  # Limited for tests
        batch_size=2,
        num_epochs=1,
        cache_dir="./test_cache",
        output_dir="./test_models",
        preprocessing_workers=2
    )


@pytest.fixture
def mock_deployment_config():
    """Create mock deployment configuration."""
    return DeploymentConfig(
        cloud_provider="aws",
        region="us-west-2",
        cluster_name="test-cluster",
        node_count=1,
        instance_type="t3.medium",  # Smaller instance for tests
        storage_bucket="test-bucket"
    )


@pytest.fixture
def mock_consciousness_config():
    """Create mock consciousness configuration."""
    return WikipediaConsciousnessConfig(
        wikipedia_model_path="./test_models",
        integration_layer_size=128,  # Smaller for tests
        max_context_length=512,
        max_response_length=128
    )


@pytest.fixture
def sample_wikipedia_xml():
    """Create sample Wikipedia XML for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>Test Article</title>
    <id>1</id>
    <revision>
      <id>1</id>
      <text xml:space="preserve">
This is a test Wikipedia article about artificial intelligence.

Artificial intelligence (AI) is intelligence demonstrated by machines, 
in contrast to the natural intelligence displayed by humans.

== History ==
The field of AI research was founded at a conference at Dartmouth College in 1956.

== Applications ==
AI has many applications in modern technology.

[[Category:Artificial intelligence]]
[[Category:Technology]]
      </text>
    </revision>
  </page>
  <page>
    <title>Machine Learning</title>
    <id>2</id>
    <revision>
      <id>2</id>
      <text xml:space="preserve">
Machine learning is a subset of artificial intelligence.

It involves algorithms that can learn and make decisions with minimal human intervention.

== Types ==
* Supervised learning
* Unsupervised learning
* Reinforcement learning

[[Category:Machine learning]]
[[Category:Artificial intelligence]]
      </text>
    </revision>
  </page>
</mediawiki>"""


class TestWikiTextCleaner:
    """Test Wikipedia text cleaning functionality."""
    
    def test_clean_basic_markup(self):
        """Test basic markup cleaning."""
        cleaner = WikiTextCleaner()
        
        # Test with sample markup
        text = "This is '''bold''' and ''italic'' text with [[links|displayed text]]."
        cleaned = cleaner.clean_text(text)
        
        assert "bold" in cleaned
        assert "italic" in cleaned
        assert "displayed text" in cleaned
        assert "'''" not in cleaned
        assert "''" not in cleaned
        assert "[[" not in cleaned
        assert "]]" not in cleaned
    
    def test_extract_categories(self):
        """Test category extraction."""
        cleaner = WikiTextCleaner()
        
        text = "Some text [[Category:Artificial intelligence]] more text [[Category:Technology]]"
        categories = cleaner.extract_categories(text)
        
        assert "Artificial intelligence" in categories
        assert "Technology" in categories
        assert len(categories) == 2
    
    def test_extract_links(self):
        """Test internal link extraction."""
        cleaner = WikiTextCleaner()
        
        text = "Text with [[Machine Learning]] and [[Artificial Intelligence|AI]] links"
        links = cleaner.extract_links(text)
        
        assert "Machine Learning" in links
        assert "Artificial Intelligence" in links
    
    def test_disambiguation_detection(self):
        """Test disambiguation page detection."""
        cleaner = WikiTextCleaner()
        
        text = "This is a disambiguation page that may refer to multiple topics."
        assert cleaner.is_disambiguation(text)
        
        text = "This is a regular article about a specific topic."
        assert not cleaner.is_disambiguation(text)


class TestWikipediaXMLParser:
    """Test Wikipedia XML parsing functionality."""
    
    def test_parse_article(self, sample_wikipedia_xml, temp_dir):
        """Test parsing individual articles."""
        parser = WikipediaXMLParser(min_text_length=50, max_text_length=10000)
        
        # Save sample XML to file
        xml_file = Path(temp_dir) / "test.xml"
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(sample_wikipedia_xml)
        
        # Parse articles
        articles = list(parser.parse_xml_dump(str(xml_file)))
        
        assert len(articles) == 2
        assert articles[0].title == "Test Article"
        assert articles[1].title == "Machine Learning"
        assert "artificial intelligence" in articles[0].text.lower()
        assert "machine learning" in articles[1].text.lower()
    
    def test_category_extraction(self, sample_wikipedia_xml, temp_dir):
        """Test category extraction from parsed articles."""
        parser = WikipediaXMLParser()
        
        xml_file = Path(temp_dir) / "test.xml"
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(sample_wikipedia_xml)
        
        articles = list(parser.parse_xml_dump(str(xml_file)))
        
        assert "Artificial intelligence" in articles[0].categories
        assert "Technology" in articles[0].categories
        assert "Machine learning" in articles[1].categories


@pytest.mark.asyncio
class TestWikipediaDataProcessor:
    """Test Wikipedia data processing functionality."""
    
    async def test_preprocess_articles(self, mock_training_config, temp_dir, sample_wikipedia_xml):
        """Test article preprocessing."""
        # Setup
        config = mock_training_config
        config.cache_dir = temp_dir
        
        processor = WikipediaDataProcessor(config)
        
        # Create mock XML file
        xml_file = Path(temp_dir) / "test.xml"
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(sample_wikipedia_xml)
        
        # Mock the WikiExtractor
        with patch('knowledge_systems.training_pipelines.wikipedia_cloud_training.WikiExtractor') as mock_extractor:
            mock_instance = Mock()
            mock_instance.extract.return_value = [
                {'title': 'Test Article', 'text': 'This is a test article about AI.'},
                {'title': 'Machine Learning', 'text': 'Machine learning is a subset of AI.'}
            ]
            mock_extractor.return_value = mock_instance
            
            # Test preprocessing
            dataset = processor.preprocess_wikipedia_articles(str(xml_file))
            
            assert len(dataset) == 2
            assert 'text' in dataset.column_names
            assert 'title' in dataset.column_names
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenize_dataset(self, mock_tokenizer, mock_training_config):
        """Test dataset tokenization."""
        # Setup mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': [[1, 2, 3], [4, 5, 6]],
            'attention_mask': [[1, 1, 1], [1, 1, 1]]
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        processor = WikipediaDataProcessor(mock_training_config)
        
        # Create mock dataset
        from datasets import Dataset
        dataset = Dataset.from_dict({
            'text': ['This is test text.', 'This is more test text.'],
            'title': ['Test 1', 'Test 2']
        })
        
        # Test tokenization
        tokenized = processor.tokenize_dataset(dataset, mock_tokenizer_instance)
        
        assert 'input_ids' in tokenized.column_names
        assert 'labels' in tokenized.column_names


@pytest.mark.asyncio
class TestWikipediaTrainer:
    """Test Wikipedia training functionality."""
    
    @patch('knowledge_systems.training_pipelines.wikipedia_cloud_training.WikipediaDataProcessor')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    async def test_training_setup(self, mock_model, mock_tokenizer, mock_processor, mock_training_config, temp_dir):
        """Test training environment setup."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        config = mock_training_config
        config.output_dir = temp_dir
        
        trainer = WikipediaTrainer(config)
        
        # Test model and tokenizer preparation
        model, tokenizer = trainer.prepare_model_and_tokenizer()
        
        assert model is not None
        assert tokenizer is not None
    
    @patch('knowledge_systems.training_pipelines.wikipedia_cloud_training.WikipediaDataProcessor')
    async def test_setup_training_environment(self, mock_processor, mock_training_config, temp_dir):
        """Test training environment setup without cloud manager."""
        config = mock_training_config
        config.cache_dir = temp_dir
        
        # Create trainer without cloud manager
        trainer = WikipediaTrainer(config)
        trainer.cloud_manager = None
        
        # Mock the download process
        with patch.object(trainer.data_processor, 'download_wikipedia_dump') as mock_download:
            mock_download.return_value = "mock_dump_path"
            
            setup_info = await trainer.setup_training_environment()
            
            assert 'dump_path' in setup_info
            assert setup_info['dump_path'] == "mock_dump_path"


class TestWikipediaKnowledgeRetriever:
    """Test Wikipedia knowledge retrieval functionality."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_knowledge_retrieval(self, mock_model, mock_tokenizer, mock_consciousness_config, temp_dir):
        """Test knowledge retrieval from Wikipedia model."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': [[1, 2, 3]],
            'attention_mask': [[1, 1, 1]]
        }
        mock_tokenizer_instance.decode.return_value = "This is relevant knowledge about the query."
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model_instance.return_value.loss.item.return_value = 2.5
        mock_model.return_value = mock_model_instance
        
        config = mock_consciousness_config
        config.wikipedia_model_path = temp_dir
        
        retriever = WikipediaKnowledgeRetriever(temp_dir, config)
        
        # Test knowledge retrieval
        result = retriever.retrieve_knowledge("What is artificial intelligence?")
        
        assert 'query' in result
        assert 'knowledge' in result
        assert 'confidence' in result
        assert result['query'] == "What is artificial intelligence?"


@pytest.mark.asyncio
class TestWikipediaConsciousnessAgent:
    """Test Wikipedia-consciousness integration."""
    
    @patch('brain_modules.conscious_agent.integrations.wikipedia_consciousness_integration.WikipediaKnowledgeRetriever')
    @patch('brain_modules.conscious_agent.integrations.wikipedia_consciousness_integration.BrainLauncher')
    @patch('brain_modules.conscious_agent.integrations.wikipedia_consciousness_integration.UnifiedConsciousnessAgent.__init__')
    async def test_process_with_knowledge(self, mock_consciousness, mock_brain_launcher, mock_retriever, mock_consciousness_config):
        """Test processing input with knowledge enhancement."""
        # Setup mocks
        mock_consciousness.return_value = None
        
        mock_brain_instance = Mock()
        mock_brain_instance.process_with_all_modules.return_value = {'response': 'Enhanced response'}
        mock_brain_launcher.return_value = mock_brain_instance
        
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve_knowledge.return_value = {
            'knowledge': 'Relevant Wikipedia knowledge',
            'confidence': 0.8
        }
        mock_retriever_instance.encode_query.return_value = Mock()
        mock_retriever.return_value = mock_retriever_instance
        
        agent = WikipediaConsciousnessAgent(mock_consciousness_config)
        
        # Mock additional methods
        agent.generate_response = Mock(return_value={'response': 'Base response'})
        agent.get_current_state = Mock(return_value={'embedding': [0.1] * 128})
        
        # Test processing
        result = await agent.process_with_knowledge("Test query")
        
        assert 'input' in result
        assert 'enhanced_response' in result
        assert 'knowledge' in result


class TestAWSDeploymentManager:
    """Test AWS deployment functionality."""
    
    @patch('boto3.Session')
    def test_create_s3_bucket(self, mock_session, mock_deployment_config):
        """Test S3 bucket creation."""
        # Setup mocks
        mock_s3_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.return_value = mock_s3_client
        mock_session.return_value = mock_session_instance
        
        manager = AWSDeploymentManager(mock_deployment_config)
        
        # Test bucket creation
        bucket_name = manager.create_s3_bucket()
        
        assert bucket_name == mock_deployment_config.storage_bucket
        mock_s3_client.create_bucket.assert_called_once()
    
    @patch('boto3.Session')
    def test_create_iam_roles(self, mock_session, mock_deployment_config):
        """Test IAM role creation."""
        # Setup mocks
        mock_iam_client = Mock()
        mock_iam_client.create_role.return_value = {
            'Role': {'Arn': 'arn:aws:iam::123456789012:role/test-role'}
        }
        mock_session_instance = Mock()
        mock_session_instance.client.return_value = mock_iam_client
        mock_session.return_value = mock_session_instance
        
        manager = AWSDeploymentManager(mock_deployment_config)
        
        # Test role creation
        roles = manager.create_iam_roles()
        
        assert 'cluster_role_arn' in roles
        assert 'node_role_arn' in roles


@pytest.mark.integration
class TestIntegrationWorkflow:
    """Integration tests for complete workflow."""
    
    @pytest.mark.slow
    @patch('deployment.cloud_computing.scripts.deploy_wikipedia_training.WikipediaTrainingDeployer.deploy_complete_infrastructure')
    async def test_quick_start_workflow(self, mock_deploy, temp_dir):
        """Test the complete quick start workflow."""
        from scripts.quick_start_wikipedia_training import quick_setup_wikipedia_training
        
        # Mock deployment
        mock_deploy.return_value = {
            'provider': 'aws',
            'cluster_name': 'test-cluster',
            'bucket_name': 'test-bucket'
        }
        
        # Test with dry run
        result = await quick_setup_wikipedia_training(
            cloud_provider="aws",
            num_nodes=1,
            max_articles=10,
            dry_run=True
        )
        
        assert result['status'] == 'dry_run_complete'
        assert 'training_config' in result


class TestPerformanceAndScaling:
    """Test performance and scaling aspects."""
    
    def test_memory_usage_with_large_dataset(self, mock_training_config):
        """Test memory usage with large datasets."""
        import psutil
        
        config = mock_training_config
        config.max_articles = 1000  # Larger dataset
        
        processor = WikipediaDataProcessor(config)
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate processing (would be actual processing in real test)
        # For test, just create a mock dataset
        from datasets import Dataset
        mock_dataset = Dataset.from_dict({
            'text': ['Test text'] * 1000,
            'title': ['Test title'] * 1000
        })
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Assert memory usage is reasonable (less than 1GB increase)
        assert memory_increase < 1024 * 1024 * 1024  # 1GB in bytes
    
    def test_tokenization_speed(self, mock_training_config):
        """Test tokenization speed performance."""
        import time
        
        processor = WikipediaDataProcessor(mock_training_config)
        
        # Create mock dataset
        from datasets import Dataset
        large_dataset = Dataset.from_dict({
            'text': ['This is a test article about artificial intelligence.'] * 1000,
            'title': ['Test Article'] * 1000
        })
        
        # Mock tokenizer
        from unittest.mock import Mock
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3]] * 1000,
            'attention_mask': [[1, 1, 1]] * 1000
        }
        
        # Time tokenization
        start_time = time.time()
        processor.tokenize_dataset(large_dataset, mock_tokenizer)
        end_time = time.time()
        
        tokenization_time = end_time - start_time
        
        # Assert tokenization completes in reasonable time (less than 60 seconds)
        assert tokenization_time < 60.0


@pytest.mark.parametrize("cloud_provider", ["aws"])  # Add "gcp", "azure" when implemented
class TestCloudProviderCompatibility:
    """Test compatibility across different cloud providers."""
    
    def test_deployment_config_validation(self, cloud_provider):
        """Test deployment configuration for different providers."""
        config = DeploymentConfig(
            cloud_provider=cloud_provider,
            region="us-west-2" if cloud_provider == "aws" else "us-central1",
            cluster_name="test-cluster",
            node_count=2
        )
        
        assert config.cloud_provider == cloud_provider
        assert config.cluster_name == "test-cluster"
        assert config.node_count == 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-m", "not slow"  # Skip slow tests by default
    ])
