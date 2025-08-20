#!/usr/bin/env python3
"""
Database System Initialization
Initialize the universal domain database with self-learning capabilities
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure for the database"""
    base_path = Path("database")
    
    # Main directories
    directories = [
        "domains/sciences",
        "domains/humanities", 
        "domains/technologies",
        "domains/social_sciences",
        "domains/languages",
        "domains/emerging",
        "learning_engine",
        "data_sources",
        "synthetic_data",
        "knowledge_graph",
        "scrapers",
        "analytics"
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return base_path

def initialize_learning_metrics():
    """Initialize learning metrics file"""
    metrics = {
        "system_initialized": datetime.now().isoformat(),
        "domains_discovered": 0,
        "datasets_integrated": 0,
        "synthetic_data_generated": 0,
        "learning_sessions": 0,
        "last_updated": datetime.now().isoformat(),
        "system_version": "1.0.0"
    }
    
    metrics_file = Path("database/analytics/learning_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✅ Initialized learning metrics")

def create_domain_taxonomy():
    """Create initial domain taxonomy"""
    taxonomy = {
        "sciences": {
            "description": "Natural and formal sciences",
            "subdomains": [
                "physics", "chemistry", "biology", "mathematics", 
                "astronomy", "geology", "neuroscience", "biochemistry"
            ]
        },
        "humanities": {
            "description": "Arts, literature, and philosophy",
            "subdomains": [
                "philosophy", "literature", "history", "art", 
                "music", "theology", "linguistics", "classics"
            ]
        },
        "technologies": {
            "description": "Engineering and applied sciences",
            "subdomains": [
                "computer_science", "electrical_engineering", "mechanical_engineering",
                "chemical_engineering", "civil_engineering", "biomedical_engineering"
            ]
        },
        "social_sciences": {
            "description": "Psychology, sociology, and economics",
            "subdomains": [
                "psychology", "sociology", "economics", "political_science",
                "anthropology", "geography", "education", "communication"
            ]
        },
        "languages": {
            "description": "Linguistics and communication",
            "subdomains": [
                "linguistics", "semantics", "pragmatics", "phonetics",
                "morphology", "syntax", "language_acquisition", "translation"
            ]
        },
        "emerging": {
            "description": "New and emerging fields",
            "subdomains": [
                "artificial_intelligence", "quantum_computing", "nanotechnology",
                "biotechnology", "space_exploration", "climate_science"
            ]
        }
    }
    
    taxonomy_file = Path("database/domains/taxonomy.json")
    with open(taxonomy_file, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    
    print("✅ Created domain taxonomy")

def create_knowledge_graph_schema():
    """Create knowledge graph schema"""
    schema = {
        "nodes": {
            "domain": {
                "properties": ["id", "name", "description", "parent_domain", "complexity_level"]
            },
            "concept": {
                "properties": ["id", "name", "description", "domain", "related_concepts"]
            },
            "dataset": {
                "properties": ["id", "name", "url", "domain", "data_types", "platform"]
            },
            "relationship": {
                "properties": ["source", "target", "type", "strength", "description"]
            }
        },
        "relationships": [
            "BELONGS_TO",
            "RELATED_TO", 
            "CONTAINS",
            "SIMILAR_TO",
            "DEPENDS_ON",
            "INFLUENCES"
        ]
    }
    
    schema_file = Path("database/knowledge_graph/schema.json")
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print("✅ Created knowledge graph schema")

def create_configuration():
    """Create system configuration file"""
    config = {
        "system_name": "Universal Domain Database",
        "version": "1.0.0",
        "description": "Self-learning knowledge system for all human domains",
        "settings": {
            "max_concurrent_scrapers": 5,
            "rate_limit_delay": 1.0,
            "max_datasets_per_domain": 100,
            "synthetic_data_generation": True,
            "auto_discovery": True,
            "learning_rate": 0.1
        },
        "data_sources": {
            "github": {
                "enabled": True,
                "api_rate_limit": 5000,
                "search_patterns": ["dataset", "data", "corpus"]
            },
            "kaggle": {
                "enabled": True,
                "api_required": True,
                "search_patterns": ["dataset", "competition"]
            },
            "zenodo": {
                "enabled": True,
                "api_rate_limit": 1000,
                "search_patterns": ["dataset", "research"]
            },
            "figshare": {
                "enabled": True,
                "api_rate_limit": 1000,
                "search_patterns": ["dataset", "research"]
            }
        },
        "domains": {
            "auto_categorization": True,
            "confidence_threshold": 0.7,
            "max_subdomains": 20
        }
    }
    
    config_file = Path("database/config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created system configuration")

def create_startup_script():
    """Create startup script for the system"""
    startup_script = '''#!/usr/bin/env python3
"""
Startup script for the Universal Domain Database
"""

import sys
import os
from pathlib import Path

# Add the database directory to Python path
database_path = Path(__file__).parent
sys.path.insert(0, str(database_path))

from learning_engine.self_learning_system import SelfLearningSystem
from scrapers.internet_scraper import InternetScraper

def main():
    print("🌐 Universal Domain Database - Starting Up")
    print("=" * 50)
    
    # Initialize the self-learning system
    system = SelfLearningSystem()
    
    # Initialize the internet scraper
    scraper = InternetScraper()
    
    # Show system status
    status = system.get_system_status()
    print(f"📊 System Status:")
    print(f"   Domains: {status['total_domains']}")
    print(f"   Data Sources: {status['total_data_sources']}")
    print(f"   Learning Sessions: {status['learning_metrics']['learning_sessions']}")
    
    print("\\n🚀 System ready for learning!")
    print("\\nAvailable commands:")
    print("  python database/learning_engine/self_learning_system.py")
    print("  python database/scrapers/internet_scraper.py")
    print("  python database/start_learning.py")

if __name__ == "__main__":
    main()
'''
    
    startup_file = Path("database/startup.py")
    with open(startup_file, 'w') as f:
        f.write(startup_script)
    
    # Make it executable
    os.chmod(startup_file, 0o755)
    
    print("✅ Created startup script")

def create_requirements():
    """Create requirements.txt for the system"""
    requirements = [
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "networkx>=2.8.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "nltk>=3.8.0",
        "spacy>=3.5.0",
        "transformers>=4.20.0",
        "torch>=1.13.0",
        "tensorflow>=2.10.0"
    ]
    
    requirements_file = Path("database/requirements.txt")
    with open(requirements_file, 'w') as f:
        f.write('\n'.join(requirements))
    
    print("✅ Created requirements.txt")

def main():
    """Main initialization function"""
    print("🌐 Universal Domain Database - System Initialization")
    print("=" * 60)
    
    try:
        # Create directory structure
        print("\n📁 Creating directory structure...")
        create_directory_structure()
        
        # Initialize learning metrics
        print("\n📊 Initializing learning metrics...")
        initialize_learning_metrics()
        
        # Create domain taxonomy
        print("\n🏷️ Creating domain taxonomy...")
        create_domain_taxonomy()
        
        # Create knowledge graph schema
        print("\n🕸️ Creating knowledge graph schema...")
        create_knowledge_graph_schema()
        
        # Create configuration
        print("\n⚙️ Creating system configuration...")
        create_configuration()
        
        # Create startup script
        print("\n🚀 Creating startup script...")
        create_startup_script()
        
        # Create requirements
        print("\n📦 Creating requirements...")
        create_requirements()
        
        print("\n" + "=" * 60)
        print("✅ System initialization complete!")
        print("\n🎯 Next steps:")
        print("1. Install requirements: pip install -r database/requirements.txt")
        print("2. Start the system: python database/startup.py")
        print("3. Begin learning: python database/learning_engine/self_learning_system.py")
        print("4. Discover datasets: python database/scrapers/internet_scraper.py")
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
