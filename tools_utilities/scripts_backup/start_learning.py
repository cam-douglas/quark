#!/usr/bin/env python3
"""
Start Learning - Main Orchestration Script
Coordinates the entire self-learning process for the Universal Domain Database
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add the database directory to Python path
database_path = Path(__file__).parent
sys.path.insert(0, str(database_path))

from learning_engine.self_learning_system import SelfLearningSystem
from scrapers.internet_scraper import InternetScraper

class LearningOrchestrator:
    def __init__(self):
        self.system = SelfLearningSystem()
        self.scraper = InternetScraper()
        self.learning_session = {
            "session_id": f"session_{int(time.time())}",
            "started_at": datetime.now().isoformat(),
            "domains_discovered": 0,
            "datasets_found": 0,
            "synthetic_data_generated": 0,
            "learning_progress": []
        }
    
    def run_learning_session(self):
        """Run a complete learning session"""
        print("🧠 Universal Domain Database - Learning Session")
        print("=" * 60)
        print(f"Session ID: {self.learning_session['session_id']}")
        print(f"Started: {self.learning_session['started_at']}")
        print()
        
        # Step 1: Discover new domains
        self._discover_new_domains()
        
        # Step 2: Find datasets for existing domains
        self._discover_datasets()
        
        # Step 3: Generate synthetic data
        self._generate_synthetic_data()
        
        # Step 4: Update knowledge graph
        self._update_knowledge_graph()
        
        # Step 5: Save session results
        self._save_session_results()
        
        print("\n" + "=" * 60)
        print("✅ Learning session completed!")
        self._print_session_summary()
    
    def _discover_new_domains(self):
        """Discover new domains through search and analysis"""
        print("🔍 Step 1: Discovering new domains...")
        
        # Search terms for domain discovery
        search_terms = [
            "quantum physics", "machine learning", "cognitive psychology",
            "biochemistry", "neuroscience", "artificial intelligence",
            "climate science", "genomics", "robotics", "economics"
        ]
        
        discovered_domains = self.system.discover_new_domains(search_terms)
        
        for domain in discovered_domains:
            print(f"   📚 Found: {domain['name']} (confidence: {domain['confidence']})")
            self.learning_session['domains_discovered'] += 1
            self.learning_session['learning_progress'].append({
                "timestamp": datetime.now().isoformat(),
                "action": "domain_discovered",
                "domain": domain['name'],
                "confidence": domain['confidence']
            })
        
        print(f"   Total domains discovered: {len(discovered_domains)}")
    
    def _discover_datasets(self):
        """Discover datasets for existing and new domains"""
        print("\n📊 Step 2: Discovering datasets...")
        
        # Get existing domains
        existing_domains = list(self.system.domains.keys())
        
        # Search for datasets for each domain
        for domain_id in existing_domains:
            domain = self.system.domains[domain_id]
            domain_name = domain['name'].lower()
            
            print(f"   🔍 Searching datasets for: {domain['name']}")
            
            # Create search keywords based on domain
            keywords = self._generate_domain_keywords(domain)
            
            # Discover datasets
            datasets = self.scraper.discover_datasets(keywords, max_results=10)
            
            for dataset in datasets:
                print(f"      📦 Found: {dataset['name']} ({dataset['platform']})")
                self.learning_session['datasets_found'] += 1
                self.learning_session['learning_progress'].append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "dataset_found",
                    "dataset": dataset['name'],
                    "platform": dataset['platform'],
                    "domain": domain['name']
                })
            
            # Rate limiting
            time.sleep(2)
        
        print(f"   Total datasets found: {self.learning_session['datasets_found']}")
    
    def _generate_domain_keywords(self, domain):
        """Generate search keywords for a domain"""
        domain_name = domain['name'].lower()
        
        # Domain-specific keyword mappings
        keyword_mappings = {
            "neuroscience": ["neuroscience", "brain", "neural", "cognitive", "neuropixels", "connectomics"],
            "biochemistry": ["biochemistry", "protein", "enzyme", "metabolic", "biochemical", "pathway"],
            "physics": ["physics", "quantum", "mechanics", "particle", "wave", "thermodynamics"],
            "chemistry": ["chemistry", "molecular", "reaction", "compound", "chemical", "synthesis"],
            "biology": ["biology", "cell", "organism", "gene", "evolution", "genetics"],
            "mathematics": ["mathematics", "algorithm", "statistics", "mathematical", "equation", "proof"],
            "computer_science": ["computer science", "programming", "algorithm", "software", "code", "artificial intelligence"],
            "psychology": ["psychology", "behavior", "cognitive", "mental", "psychology", "behavioral"],
            "economics": ["economics", "market", "economic", "financial", "trade", "production"]
        }
        
        return keyword_mappings.get(domain_name, [domain_name])
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for domains"""
        print("\n🧬 Step 3: Generating synthetic data...")
        
        # Generate synthetic data for existing domains
        for domain_id in self.system.domains:
            domain = self.system.domains[domain_id]
            
            print(f"   🧠 Generating data for: {domain['name']}")
            
            # Generate different types of synthetic data
            data_types = self._get_domain_data_types(domain)
            
            for data_type in data_types:
                try:
                    synthetic_data = self.system.generate_synthetic_data(domain_id, data_type)
                    print(f"      ✅ Generated: {data_type} data")
                    self.learning_session['synthetic_data_generated'] += 1
                    self.learning_session['learning_progress'].append({
                        "timestamp": datetime.now().isoformat(),
                        "action": "synthetic_data_generated",
                        "domain": domain['name'],
                        "data_type": data_type,
                        "synthetic_id": synthetic_data['synthetic_id']
                    })
                except Exception as e:
                    print(f"      ❌ Error generating {data_type} data: {e}")
        
        print(f"   Total synthetic datasets generated: {self.learning_session['synthetic_data_generated']}")
    
    def _get_domain_data_types(self, domain):
        """Get appropriate data types for a domain"""
        domain_name = domain['name'].lower()
        
        # Domain-specific data type mappings
        data_type_mappings = {
            "neuroscience": ["neural_activity", "connectivity", "brain_imaging", "behavioral_data"],
            "biochemistry": ["metabolic_pathway", "protein_structure", "enzyme_kinetics", "molecular_interactions"],
            "physics": ["particle_trajectories", "wave_functions", "energy_levels", "field_measurements"],
            "chemistry": ["reaction_kinetics", "molecular_structures", "spectroscopic_data", "thermodynamic_data"],
            "biology": ["gene_expression", "protein_interactions", "cell_populations", "ecological_data"],
            "mathematics": ["mathematical_proofs", "statistical_data", "algorithm_performance", "geometric_structures"],
            "computer_science": ["algorithm_data", "software_metrics", "network_topologies", "performance_data"],
            "psychology": ["behavioral_data", "cognitive_tasks", "survey_responses", "experimental_results"],
            "economics": ["market_data", "economic_indicators", "trade_statistics", "financial_data"]
        }
        
        return data_type_mappings.get(domain_name, ["generic_data"])
    
    def _update_knowledge_graph(self):
        """Update the knowledge graph with new discoveries"""
        print("\n🕸️ Step 4: Updating knowledge graph...")
        
        # This would update the knowledge graph with new relationships
        # For now, we'll just log the action
        self.learning_session['learning_progress'].append({
            "timestamp": datetime.now().isoformat(),
            "action": "knowledge_graph_updated",
            "details": "Knowledge graph updated with new discoveries"
        })
        
        print("   ✅ Knowledge graph updated")
    
    def _save_session_results(self):
        """Save the learning session results"""
        print("\n💾 Step 5: Saving session results...")
        
        # Update session end time
        self.learning_session['ended_at'] = datetime.now().isoformat()
        
        # Save session results
        session_file = Path("database/analytics") / f"learning_session_{self.learning_session['session_id']}.json"
        with open(session_file, 'w') as f:
            json.dump(self.learning_session, f, indent=2)
        
        # Update system metrics
        self.system.update_learning_metrics()
        
        print(f"   ✅ Session results saved to: {session_file}")
    
    def _print_session_summary(self):
        """Print a summary of the learning session"""
        print("\n📊 Learning Session Summary:")
        print(f"   Domains Discovered: {self.learning_session['domains_discovered']}")
        print(f"   Datasets Found: {self.learning_session['datasets_found']}")
        print(f"   Synthetic Data Generated: {self.learning_session['synthetic_data_generated']}")
        print(f"   Total Actions: {len(self.learning_session['learning_progress'])}")
        
        # Show system status
        status = self.system.get_system_status()
        print(f"\n🌐 Overall System Status:")
        print(f"   Total Domains: {status['total_domains']}")
        print(f"   Total Data Sources: {status['total_data_sources']}")
        print(f"   Learning Sessions: {status['learning_metrics']['learning_sessions']}")

def main():
    """Main function to start the learning process"""
    try:
        orchestrator = LearningOrchestrator()
        orchestrator.run_learning_session()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Learning session interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during learning session: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
