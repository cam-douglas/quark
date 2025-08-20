#!/usr/bin/env python3
"""
ML Resource Search Training Script
Biological AGI Development - Resource Discovery

Searches GitHub, Kaggle, Hugging Face, and other ML resources
for databases, models, and datasets relevant to biological AGI.
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLResourceSearcher:
    def __init__(self):
        self.results = []
        self.session = requests.Session()
        self.results_dir = Path("data/ml_resources")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Biological AGI keywords
        self.biological_keywords = [
            "STDP", "spike timing dependent plasticity", "neuromodulation",
            "dopamine", "serotonin", "cortical column", "neural network",
            "brain simulation", "brian simulator", "synaptic plasticity"
        ]
    
    def search_github(self):
        """Search GitHub for biological AGI repositories."""
        logger.info("Searching GitHub...")
        
        queries = [
            "brain simulation neural network",
            "STDP spike timing dependent plasticity", 
            "neuromodulation dopamine serotonin",
            "cortical column microcircuit",
            "biological neural network"
        ]
        
        for query in queries:
            try:
                url = "https://api.github.com/search/repositories"
                params = {"q": query, "sort": "stars", "order": "desc", "per_page": 20}
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for repo in data.get("items", []):
                        self.results.append({
                            "name": repo["name"],
                            "source": "github",
                            "type": "repository",
                            "url": repo["html_url"],
                            "description": repo.get("description", ""),
                            "stars": repo["stargazers_count"],
                            "relevance_score": self.calculate_relevance(repo, query)
                        })
                
                # Rate limiting
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching GitHub: {e}")
    
    def search_huggingface(self):
        """Search Hugging Face Hub for relevant models."""
        logger.info("Searching Hugging Face...")
        
        queries = ["neural network brain", "STDP implementation", "neuromodulation"]
        
        for query in queries:
            try:
                url = "https://huggingface.co/api/models"
                params = {"search": query, "limit": 20}
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    models = response.json()
                    for model in models:
                        self.results.append({
                            "name": model["modelId"],
                            "source": "huggingface",
                            "type": "model",
                            "url": f"https://huggingface.co/{model['modelId']}",
                            "description": model.get("description", ""),
                            "downloads": model.get("downloads", 0),
                            "relevance_score": self.calculate_relevance(model, query)
                        })
                
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching Hugging Face: {e}")
    
    def search_custom_resources(self):
        """Search curated biological AGI resources."""
        logger.info("Searching custom resources...")
        
        custom_resources = [
            {
                "name": "Brian Simulator",
                "source": "custom",
                "type": "simulation_framework",
                "url": "https://briansimulator.org/",
                "description": "Python package for spiking neural networks",
                "relevance_score": 0.95
            },
            {
                "name": "NEURON Simulator", 
                "source": "custom",
                "type": "simulation_framework",
                "url": "https://neuron.yale.edu/",
                "description": "Neuron simulation environment",
                "relevance_score": 0.90
            },
            {
                "name": "NEST Simulator",
                "source": "custom", 
                "type": "simulation_framework",
                "url": "https://www.nest-simulator.org/",
                "description": "Neural simulation technology",
                "relevance_score": 0.92
            },
            {
                "name": "Blue Brain Project",
                "source": "custom",
                "type": "research_project", 
                "url": "https://www.epfl.ch/research/domains/bluebrain/",
                "description": "Digital reconstruction of the brain",
                "relevance_score": 0.98
            }
        ]
        
        self.results.extend(custom_resources)
    
    def calculate_relevance(self, item, query):
        """Calculate relevance score for a resource."""
        score = 0.0
        text = ""
        
        if isinstance(item, dict):
            text += item.get("description", "")
            text += item.get("title", "")
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Exact phrase match
        if query_lower in text_lower:
            score += 0.4
        
        # Keyword matches
        for keyword in self.biological_keywords:
            if keyword.lower() in text_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def save_results(self):
        """Save search results to files."""
        # Sort by relevance
        self.results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Save to JSON
        json_file = self.results_dir / f"ml_resources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save to CSV
        csv_file = self.results_dir / f"ml_resources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {json_file} and {csv_file}")
    
    def generate_report(self):
        """Generate search report."""
        report = []
        report.append("# ML Resource Search Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Resources: {len(self.results)}")
        report.append("")
        
        # Summary by source
        sources = {}
        for resource in self.results:
            source = resource["source"]
            sources[source] = sources.get(source, 0) + 1
        
        report.append("## Summary by Source")
        for source, count in sources.items():
            report.append(f"- **{source.title()}**: {count} resources")
        report.append("")
        
        # Top resources
        report.append("## Top Resources by Relevance")
        top_resources = sorted(self.results, key=lambda x: x.get("relevance_score", 0), reverse=True)[:10]
        
        for i, resource in enumerate(top_resources, 1):
            report.append(f"{i}. **{resource['name']}** ({resource['source']})")
            report.append(f"   - URL: {resource['url']}")
            report.append(f"   - Type: {resource['type']}")
            report.append(f"   - Relevance: {resource.get('relevance_score', 0):.3f}")
            report.append(f"   - Description: {resource['description'][:100]}...")
            report.append("")
        
        # Save report
        report_file = self.results_dir / f"search_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write("\n".join(report))
        
        logger.info(f"Report saved to {report_file}")
    
    def run_search(self):
        """Run comprehensive ML resource search."""
        logger.info("Starting ML resource search...")
        
        # Search all sources
        self.search_github()
        self.search_huggingface() 
        self.search_custom_resources()
        
        # Save results and generate report
        self.save_results()
        self.generate_report()
        
        logger.info(f"Search completed! Found {len(self.results)} resources.")
        return self.results


def main():
    """Main function."""
    searcher = MLResourceSearcher()
    resources = searcher.run_search()
    
    print(f"\nFound {len(resources)} ML resources!")
    print("Check data/ml_resources/ for detailed results.")
    
    # Print top 5
    print("\nTop 5 Resources:")
    for i, resource in enumerate(resources[:5], 1):
        print(f"{i}. {resource['name']} ({resource['source']}) - Score: {resource.get('relevance_score', 0):.3f}")


if __name__ == "__main__":
    main()
