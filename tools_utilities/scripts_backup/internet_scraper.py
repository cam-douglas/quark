#!/usr/bin/env python3
"""
Internet Data Scraper
Automatically discovers and collects data from open-source datasets across the web
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from urllib.parse import urljoin, urlparse
import re

class InternetScraper:
    def __init__(self, database_path: str = "database"):
        self.database_path = database_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DomainDatabaseBot/1.0)'
        })
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Known data repositories and platforms
        self.data_platforms = {
            "github": {
                "base_url": "https://github.com",
                "api_url": "https://api.github.com",
                "search_patterns": ["dataset", "data", "corpus", "collection"]
            },
            "kaggle": {
                "base_url": "https://www.kaggle.com",
                "api_url": "https://www.kaggle.com/api/v1",
                "search_patterns": ["datasets", "competitions"]
            },
            "zenodo": {
                "base_url": "https://zenodo.org",
                "api_url": "https://zenodo.org/api",
                "search_patterns": ["dataset", "data", "research"]
            },
            "figshare": {
                "base_url": "https://figshare.com",
                "api_url": "https://api.figshare.com/v2",
                "search_patterns": ["dataset", "data", "research"]
            }
        }
    
    def discover_datasets(self, domain_keywords: List[str], max_results: int = 50) -> List[Dict[str, Any]]:
        """Discover datasets related to specific domain keywords"""
        discovered_datasets = []
        
        for keyword in domain_keywords:
            self.logger.info(f"Searching for datasets related to: {keyword}")
            
            # Search across multiple platforms
            for platform_name, platform_config in self.data_platforms.items():
                try:
                    platform_datasets = self._search_platform(platform_name, platform_config, keyword, max_results // len(self.data_platforms))
                    discovered_datasets.extend(platform_datasets)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error searching {platform_name}: {e}")
        
        return discovered_datasets
    
    def _search_platform(self, platform_name: str, platform_config: Dict[str, Any], keyword: str, max_results: int) -> List[Dict[str, Any]]:
        """Search a specific platform for datasets"""
        datasets = []
        
        if platform_name == "github":
            datasets = self._search_github(platform_config, keyword, max_results)
        elif platform_name == "kaggle":
            datasets = self._search_kaggle(platform_config, keyword, max_results)
        elif platform_name == "zenodo":
            datasets = self._search_zenodo(platform_config, keyword, max_results)
        elif platform_name == "figshare":
            datasets = self._search_figshare(platform_config, keyword, max_results)
        
        return datasets
    
    def _search_github(self, config: Dict[str, Any], keyword: str, max_results: int) -> List[Dict[str, Any]]:
        """Search GitHub for datasets"""
        datasets = []
        
        try:
            # Search repositories
            search_url = f"{config['api_url']}/search/repositories"
            params = {
                'q': f"{keyword} dataset",
                'sort': 'stars',
                'order': 'desc',
                'per_page': min(max_results, 30)
            }
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for repo in data.get('items', []):
                dataset_info = {
                    "source_id": f"github_{repo['id']}",
                    "name": repo['name'],
                    "description": repo['description'],
                    "url": repo['html_url'],
                    "platform": "github",
                    "language": repo.get('language'),
                    "stars": repo['stargazers_count'],
                    "forks": repo['forks_count'],
                    "last_updated": repo['updated_at'],
                    "discovered_at": datetime.now().isoformat(),
                    "keywords": [keyword],
                    "data_types": self._infer_data_types(repo['description'] or ""),
                    "domain": self._infer_domain(repo['description'] or "", repo['topics'] or [])
                }
                datasets.append(dataset_info)
                
        except Exception as e:
            self.logger.error(f"Error searching GitHub: {e}")
        
        return datasets
    
    def _search_kaggle(self, config: Dict[str, Any], keyword: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Kaggle for datasets"""
        datasets = []
        
        try:
            # Note: Kaggle API requires authentication
            # This is a simplified search using the web interface
            search_url = f"{config['base_url']}/datasets"
            params = {'search': keyword}
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            # Parse HTML to extract dataset information
            # This is a simplified version - in practice, you'd use BeautifulSoup
            dataset_info = {
                "source_id": f"kaggle_{int(time.time())}",
                "name": f"Kaggle Dataset - {keyword}",
                "description": f"Dataset related to {keyword} from Kaggle",
                "url": search_url,
                "platform": "kaggle",
                "discovered_at": datetime.now().isoformat(),
                "keywords": [keyword],
                "data_types": ["tabular", "structured"],
                "domain": self._infer_domain(keyword, [])
            }
            datasets.append(dataset_info)
            
        except Exception as e:
            self.logger.error(f"Error searching Kaggle: {e}")
        
        return datasets
    
    def _search_zenodo(self, config: Dict[str, Any], keyword: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Zenodo for datasets"""
        datasets = []
        
        try:
            search_url = f"{config['api_url']}/records"
            params = {
                'q': keyword,
                'type': 'dataset',
                'size': min(max_results, 20)
            }
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for record in data.get('hits', {}).get('hits', []):
                metadata = record.get('metadata', {})
                dataset_info = {
                    "source_id": f"zenodo_{record['id']}",
                    "name": metadata.get('title', 'Unknown'),
                    "description": metadata.get('description', ''),
                    "url": record['links']['html'],
                    "platform": "zenodo",
                    "doi": metadata.get('doi'),
                    "creators": metadata.get('creators', []),
                    "discovered_at": datetime.now().isoformat(),
                    "keywords": [keyword],
                    "data_types": self._infer_data_types(metadata.get('description', '')),
                    "domain": self._infer_domain(metadata.get('description', ''), metadata.get('keywords', []))
                }
                datasets.append(dataset_info)
                
        except Exception as e:
            self.logger.error(f"Error searching Zenodo: {e}")
        
        return datasets
    
    def _search_figshare(self, config: Dict[str, Any], keyword: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Figshare for datasets"""
        datasets = []
        
        try:
            search_url = f"{config['api_url']}/articles/search"
            params = {
                'search_for': keyword,
                'limit': min(max_results, 20)
            }
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for article in data:
                dataset_info = {
                    "source_id": f"figshare_{article['id']}",
                    "name": article.get('title', 'Unknown'),
                    "description": article.get('description', ''),
                    "url": article.get('url'),
                    "platform": "figshare",
                    "doi": article.get('doi'),
                    "discovered_at": datetime.now().isoformat(),
                    "keywords": [keyword],
                    "data_types": self._infer_data_types(article.get('description', '')),
                    "domain": self._infer_domain(article.get('description', ''), article.get('tags', []))
                }
                datasets.append(dataset_info)
                
        except Exception as e:
            self.logger.error(f"Error searching Figshare: {e}")
        
        return datasets
    
    def _infer_data_types(self, description: str) -> List[str]:
        """Infer data types from description"""
        data_types = []
        
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['image', 'picture', 'photo']):
            data_types.append('image')
        if any(word in description_lower for word in ['text', 'corpus', 'document']):
            data_types.append('text')
        if any(word in description_lower for word in ['audio', 'sound', 'speech']):
            data_types.append('audio')
        if any(word in description_lower for word in ['video', 'movie']):
            data_types.append('video')
        if any(word in description_lower for word in ['tabular', 'csv', 'excel', 'spreadsheet']):
            data_types.append('tabular')
        if any(word in description_lower for word in ['graph', 'network', 'relationship']):
            data_types.append('graph')
        if any(word in description_lower for word in ['time series', 'temporal']):
            data_types.append('time_series')
        
        return data_types if data_types else ['unknown']
    
    def _infer_domain(self, description: str, tags: List[str]) -> str:
        """Infer domain from description and tags"""
        text = (description + ' ' + ' '.join(tags)).lower()
        
        domain_keywords = {
            "neuroscience": ["brain", "neuron", "neural", "cognitive", "neuro"],
            "biochemistry": ["protein", "enzyme", "metabolic", "biochemical"],
            "physics": ["quantum", "mechanics", "particle", "wave"],
            "chemistry": ["molecular", "reaction", "compound", "chemical"],
            "biology": ["cell", "organism", "gene", "evolution"],
            "mathematics": ["algorithm", "statistics", "mathematical", "equation"],
            "computer_science": ["programming", "algorithm", "software", "code"],
            "psychology": ["behavior", "cognitive", "mental", "psychology"],
            "economics": ["market", "economic", "financial", "trade"],
            "medicine": ["clinical", "medical", "health", "disease"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return "general"
    
    def scrape_dataset_metadata(self, dataset_url: str) -> Optional[Dict[str, Any]]:
        """Scrape detailed metadata from a dataset URL"""
        try:
            response = self.session.get(dataset_url)
            response.raise_for_status()
            
            # Extract metadata from the page
            # This is a simplified version - in practice, you'd use BeautifulSoup
            metadata = {
                "url": dataset_url,
                "scraped_at": datetime.now().isoformat(),
                "content_length": len(response.content),
                "content_type": response.headers.get('content-type', ''),
                "last_modified": response.headers.get('last-modified', ''),
                "title": self._extract_title(response.text),
                "description": self._extract_description(response.text),
                "keywords": self._extract_keywords(response.text)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error scraping {dataset_url}: {e}")
            return None
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content"""
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE)
        return title_match.group(1).strip() if title_match else "Unknown"
    
    def _extract_description(self, html_content: str) -> str:
        """Extract description from HTML content"""
        # Look for meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', html_content, re.IGNORECASE)
        if desc_match:
            return desc_match.group(1).strip()
        
        # Look for first paragraph
        p_match = re.search(r'<p[^>]*>(.*?)</p>', html_content, re.IGNORECASE)
        return p_match.group(1).strip() if p_match else ""
    
    def _extract_keywords(self, html_content: str) -> List[str]:
        """Extract keywords from HTML content"""
        keywords = []
        
        # Look for meta keywords
        keywords_match = re.search(r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']*)["\']', html_content, re.IGNORECASE)
        if keywords_match:
            keywords.extend(keywords_match.group(1).split(','))
        
        return [kw.strip() for kw in keywords if kw.strip()]
    
    def save_discovered_datasets(self, datasets: List[Dict[str, Any]], filename: str = None):
        """Save discovered datasets to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discovered_datasets_{timestamp}.json"
        
        file_path = os.path.join(self.database_path, "data_sources", filename)
        
        with open(file_path, 'w') as f:
            json.dump(datasets, f, indent=2)
        
        self.logger.info(f"Saved {len(datasets)} discovered datasets to {file_path}")

def main():
    """Main function to demonstrate the internet scraper"""
    scraper = InternetScraper()
    
    print("🌐 Internet Data Scraper - Dataset Discovery")
    print("=" * 50)
    
    # Search for neuroscience datasets
    print("🔍 Searching for neuroscience datasets...")
    neuroscience_keywords = ["neuroscience", "brain", "neural", "cognitive", "neuropixels"]
    neuroscience_datasets = scraper.discover_datasets(neuroscience_keywords, max_results=20)
    
    print(f"Found {len(neuroscience_datasets)} neuroscience datasets:")
    for dataset in neuroscience_datasets[:5]:  # Show first 5
        print(f"  📊 {dataset['name']} ({dataset['platform']})")
        print(f"     Domain: {dataset['domain']}")
        print(f"     Data types: {', '.join(dataset['data_types'])}")
        print()
    
    # Search for biochemistry datasets
    print("🧬 Searching for biochemistry datasets...")
    biochemistry_keywords = ["biochemistry", "protein", "enzyme", "metabolic", "biochemical"]
    biochemistry_datasets = scraper.discover_datasets(biochemistry_keywords, max_results=20)
    
    print(f"Found {len(biochemistry_datasets)} biochemistry datasets:")
    for dataset in biochemistry_datasets[:5]:  # Show first 5
        print(f"  📊 {dataset['name']} ({dataset['platform']})")
        print(f"     Domain: {dataset['domain']}")
        print(f"     Data types: {', '.join(dataset['data_types'])}")
        print()
    
    # Save all discovered datasets
    all_datasets = neuroscience_datasets + biochemistry_datasets
    scraper.save_discovered_datasets(all_datasets)
    
    print(f"✅ Total datasets discovered: {len(all_datasets)}")
    print("📁 Datasets saved to database/data_sources/")

if __name__ == "__main__":
    main()
