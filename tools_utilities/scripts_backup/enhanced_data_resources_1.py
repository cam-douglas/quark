"""
Enhanced Data Resources for Baby AGI

Integrates the most recent and advanced open-source neuroscience data tools:
- DANDI Archive (latest NWB 2.0+ support)
- OpenNeuro (latest BIDS 2.0+ support)
- Allen Brain Atlas (latest API v3+)
- MICrONS/BossDB (latest CAVEclient)
- HCP (latest Workbench 1.5+)
- NeuroMorpho (latest REST API)
- CRCNS (latest datasets)
- Scientific Papers (arXiv, PubMed, bioRxiv)
- GitHub (latest neuroscience repositories)

NEW INTEGRATIONS (2025):
- FaBiAN: Synthetic fetal brain MRI phantoms (20-34.8 weeks gestation)
- 4D Embryonic Brain Atlas: Deep learning atlas (8-12 weeks gestation)
- ReWaRD: Prenatal visual signal simulation for CNN pretraining
- Multi-scale developmental modeling: CompuCell3D, COPASI
- Neural simulation frameworks: NEST, Emergent, Blue Brain Project tools
"""

import logging
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
import aiohttp
from datetime import datetime, timedelta
import re

# Import the SmallMind brain development trainer
from .....................................................human_brain_development import create_smallmind_brain_dev_trainer

logger = logging.getLogger(__name__)

class EnhancedDataResources:
    """Enhanced data resources using latest open-source neuroscience tools"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BabyAGI-Neuroscience-Data-Collector/1.0'
        })
        
        # Latest API endpoints and configurations
        self.config = {
            "dandi": {
                "base_url": "https://api.dandiarchive.org/api",
                "version": "v1",
                "latest_nwb": "2.0.0"
            },
            "openneuro": {
                "base_url": "https://openneuro.org/api",
                "version": "v1",
                "latest_bids": "2.0.0"
            },
            "allen_brain": {
                "base_url": "https://api.brain-map.org/api/v3",
                "version": "v3"
            },
            "microns": {
                "base_url": "https://bossdb.openstorage.io/api/v1",
                "version": "v1"
            },
            "hcp": {
                "base_url": "https://www.humanconnectome.org/api",
                "version": "v1"
            },
            "neuromorpho": {
                "base_url": "https://neuromorpho.org/api",
                "version": "v1"
            },
            "crcns": {
                "base_url": "https://crcns.org/api",
                "version": "v1"
            },
            "arxiv": {
                "base_url": "http://export.arxiv.org/api/query",
                "version": "v1"
            },
            "pubmed": {
                "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
                "version": "v1"
            },
            "biorxiv": {
                "base_url": "https://api.biorxiv.org",
                "version": "v1"
            },
            "github": {
                "base_url": "https://api.github.com",
                "version": "v3"
            },
            # NEW: Fetal brain simulation tools
            "fabian": {
                "base_url": "https://hub.docker.com/r/petermcgor/fabian-docker",
                "version": "2.0",
                "docker_image": "petermcgor/fabian-docker:latest",
                "dataset_url": "https://www.nature.com/articles/s41597-025-04926-9"
            },
            "embryonic_atlas": {
                "base_url": "https://arxiv.org/abs/2503.07177",
                "version": "2025",
                "gestational_range": "8-12 weeks",
                "type": "4D spatiotemporal deep learning atlas"
            },
            "fetal_atlas": {
                "base_url": "https://arxiv.org/abs/2508.04522",
                "version": "2025",
                "gestational_range": "21-37 weeks",
                "type": "conditional deep learning segmentation"
            },
            "reward": {
                "base_url": "https://arxiv.org/abs/2311.17232",
                "version": "2023",
                "type": "retinal wave simulation for CNN pretraining"
            },
            "compucell3d": {
                "base_url": "https://github.com/CompuCell3D/CompuCell3D",
                "version": "latest",
                "type": "multiscale agent-based modeling"
            },
            "copasi": {
                "base_url": "https://copasi.org/",
                "version": "latest",
                "type": "biochemical network simulation"
            },
            "nest": {
                "base_url": "https://nest-simulator.org/",
                "version": "latest",
                "type": "large-scale spiking neural networks"
            },
            "emergent": {
                "base_url": "https://github.com/emer/emergent",
                "version": "latest",
                "type": "biologically-inspired cognitive modeling"
            },
            "blue_brain": {
                "base_url": "https://bluebrain.epfl.ch/",
                "version": "latest",
                "type": "advanced anatomical and biophysical simulation"
            }
        }
        
        # Initialize brain development trainer
        try:
            self.brain_dev_trainer = create_smallmind_brain_dev_trainer()
            logger.info("Enhanced Data Resources initialized with latest API versions and brain development trainer")
        except Exception as e:
            logger.warning(f"Failed to initialize brain development trainer: {e}")
            self.brain_dev_trainer = None
            logger.info("Enhanced Data Resources initialized with latest API versions (brain development trainer disabled)")
    
    async def get_latest_neuroscience_papers(self, 
                                           query: str = "neuroscience",
                                           max_results: int = 50,
                                           days_back: int = 30) -> List[Dict]:
        """
        Get latest neuroscience papers from multiple sources using latest APIs
        
        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: How many days back to search
            
        Returns:
            List of latest neuroscience papers
        """
        papers = []
        
        # Search arXiv (latest neuroscience papers)
        try:
            arxiv_papers = await self._search_arxiv(query, max_results//3, days_back)
            papers.extend(arxiv_papers)
            logger.info(f"Found {len(arxiv_papers)} papers from arXiv")
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
        
        # Search PubMed (latest neuroscience research)
        try:
            pubmed_papers = await self._search_pubmed(query, max_results//3, days_back)
            papers.extend(pubmed_papers)
            logger.info(f"Found {len(pubmed_papers)} papers from PubMed")
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
        
        # Search bioRxiv (latest preprints)
        try:
            biorxiv_papers = await self._search_biorxiv(query, max_results//3, days_back)
            papers.extend(biorxiv_papers)
            logger.info(f"Found {len(biorxiv_papers)} papers from bioRxiv")
        except Exception as e:
            logger.warning(f"bioRxiv search failed: {e}")
        
        # Sort by date and return top results
        papers.sort(key=lambda x: x.get('date', ''), reverse=True)
        return papers[:max_results]
    
    async def _search_arxiv(self, query: str, max_results: int, days_back: int) -> List[Dict]:
        """Search arXiv for latest neuroscience papers"""
        # Use latest arXiv API
        search_query = f'all:"{query}" AND (cat:q-bio.NC OR cat:q-bio.QM OR cat:cs.AI)'
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.config["arxiv"]["base_url"], params=params) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    return self._parse_arxiv_xml(xml_content)
                else:
                    logger.warning(f"arXiv API returned status {response.status}")
                    return []
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict]:
        """Parse arXiv XML response"""
        papers = []
        
        # Extract paper information using regex (simplified parsing)
        entry_pattern = r'<entry>(.*?)</entry>'
        title_pattern = r'<title>(.*?)</title>'
        summary_pattern = r'<summary>(.*?)</summary>'
        published_pattern = r'<published>(.*?)</published>'
        id_pattern = r'<id>(.*?)</id>'
        
        entries = re.findall(entry_pattern, xml_content, re.DOTALL)
        
        for entry in entries:
            title = re.search(title_pattern, entry)
            summary = re.search(summary_pattern, entry)
            published = re.search(published_pattern, entry)
            paper_id = re.search(id_pattern, entry)
            
            if title and summary:
                papers.append({
                    'source': 'arXiv',
                    'title': title.group(1).strip(),
                    'summary': summary.group(1).strip(),
                    'date': published.group(1) if published else '',
                    'id': paper_id.group(1) if paper_id else '',
                    'url': f"https://arxiv.org/abs/{paper_id.group(1).split('/')[-1]}" if paper_id else ''
                })
        
        return papers
    
    async def _search_pubmed(self, query: str, max_results: int, days_back: int) -> List[Dict]:
        """Search PubMed for latest neuroscience research"""
        # Use latest PubMed E-utilities API
        search_params = {
            'db': 'pubmed',
            'term': f'{query}[Title/Abstract] AND neuroscience[MeSH Terms]',
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'date',
            'datetype': 'pdat'
        }
        
        # Get search results
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config['pubmed']['base_url']}/esearch.fcgi", params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                        paper_ids = data['esearchresult']['idlist'][:max_results]
                        return await self._get_pubmed_details(session, paper_ids)
        
        return []
    
    async def _get_pubmed_details(self, session: aiohttp.ClientSession, paper_ids: List[str]) -> List[Dict]:
        """Get detailed information for PubMed papers"""
        papers = []
        
        # Fetch details for each paper ID
        for paper_id in paper_ids:
            params = {
                'db': 'pubmed',
                'id': paper_id,
                'retmode': 'json'
            }
            
            async with session.get(f"{self.config['pubmed']['base_url']}/efetch.fcgi", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'pubmedarticle' in data:
                        article = data['pubmedarticle'][0]['medlinecitation']['article']
                        
                        papers.append({
                            'source': 'PubMed',
                            'title': article.get('articletitle', ''),
                            'summary': article.get('abstract', {}).get('abstracttext', ''),
                            'date': article.get('journal', {}).get('journalissue', {}).get('pubdate', {}).get('year', ''),
                            'id': paper_id,
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/"
                        })
        
        return papers
    
    async def _search_biorxiv(self, query: str, max_results: int, days_back: int) -> List[Dict]:
        """Search bioRxiv for latest neuroscience preprints"""
        # Use latest bioRxiv API
        params = {
            'query': query,
            'limit': max_results,
            'format': 'json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config['biorxiv']['base_url']}/search", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = []
                    
                    for result in data.get('collection', [])[:max_results]:
                        papers.append({
                            'source': 'bioRxiv',
                            'title': result.get('title', ''),
                            'summary': result.get('abstract', ''),
                            'date': result.get('date', ''),
                            'id': result.get('doi', ''),
                            'url': f"https://www.biorxiv.org/content/{result.get('doi', '')}"
                        })
                    
                    return papers
        
        return []
    
    async def get_latest_github_repositories(self, 
                                           query: str = "neuroscience",
                                           max_results: int = 30,
                                           days_back: int = 30) -> List[Dict]:
        """
        Get latest neuroscience GitHub repositories using latest API
        
        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: How many days back to search
            
        Returns:
            List of latest neuroscience repositories
        """
        # Use latest GitHub Search API v3
        search_query = f'{query} neuroscience brain neural'
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_str = start_date.strftime('%Y-%m-%d')
        
        params = {
            'q': f'{search_query} created:>{date_str}',
            'sort': 'stars',
            'order': 'desc',
            'per_page': min(max_results, 100)  # GitHub API limit
        }
        
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'BabyAGI-Neuroscience-Collector'
        }
        
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(f"{self.config['github']['base_url']}/search/repositories", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        repositories = []
                        
                        for repo in data.get('items', [])[:max_results]:
                            repositories.append({
                                'name': repo.get('name', ''),
                                'full_name': repo.get('full_name', ''),
                                'description': repo.get('description', ''),
                                'language': repo.get('language', ''),
                                'stars': repo.get('stargazers_count', 0),
                                'forks': repo.get('forks_count', 0),
                                'created_at': repo.get('created_at', ''),
                                'updated_at': repo.get('updated_at', ''),
                                'url': repo.get('html_url', ''),
                                'topics': repo.get('topics', [])
                            })
                        
                        return repositories
                    else:
                        logger.warning(f"GitHub API returned status {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    async def get_latest_dandi_datasets(self, 
                                       query: str = "",
                                       max_results: int = 50,
                                       days_back: int = 30) -> List[Dict]:
        """
        Get latest DANDI datasets using latest NWB 2.0+ API
        
        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: How many days back to search
            
        Returns:
            List of latest DANDI datasets
        """
        # Use latest DANDI API v1
        params = {
            'search': query,
            'page_size': max_results,
            'ordering': '-created'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['dandi']['base_url']}/dandisets/", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        datasets = []
                        
                        for dataset in data.get('results', [])[:max_results]:
                            # Check if dataset was created within specified days
                            created_date = datetime.fromisoformat(dataset.get('created', '').replace('Z', '+00:00'))
                            if (datetime.now(created_date.tzinfo) - created_date).days <= days_back:
                                datasets.append({
                                    'id': dataset.get('identifier', ''),
                                    'name': dataset.get('name', ''),
                                    'description': dataset.get('description', ''),
                                    'created': dataset.get('created', ''),
                                    'modified': dataset.get('modified', ''),
                                    'url': f"https://dandiarchive.org/dandiset/{dataset.get('identifier', '')}",
                                    'nwb_version': dataset.get('metadata', {}).get('nwb_version', ''),
                                    'subjects': dataset.get('metadata', {}).get('subjects', []),
                                    'size': dataset.get('size', 0)
                                })
                        
                        return datasets
                    else:
                        logger.warning(f"DANDI API returned status {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"DANDI search failed: {e}")
            return []
    
    async def get_latest_openneuro_datasets(self, 
                                          query: str = "",
                                          max_results: int = 50,
                                          days_back: int = 30) -> List[Dict]:
        """
        Get latest OpenNeuro datasets using latest BIDS 2.0+ API
        
        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: How many days back to search
            
        Returns:
            List of latest OpenNeuro datasets
        """
        # Use latest OpenNeuro API
        params = {
            'q': query,
            'limit': max_results,
            'sort': 'created',
            'order': 'desc'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['openneuro']['base_url']}/datasets", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        datasets = []
                        
                        for dataset in data.get('datasets', [])[:max_results]:
                            # Check if dataset was created within specified days
                            created_date = datetime.fromisoformat(dataset.get('created', '').replace('Z', '+00:00'))
                            if (datetime.now(created_date.tzinfo) - created_date).days <= days_back:
                                datasets.append({
                                    'id': dataset.get('id', ''),
                                    'name': dataset.get('name', ''),
                                    'description': dataset.get('description', ''),
                                    'created': dataset.get('created', ''),
                                    'updated': dataset.get('updated', ''),
                                    'url': f"https://openneuro.org/datasets/{dataset.get('id', '')}",
                                    'bids_version': dataset.get('bids_version', ''),
                                    'modalities': dataset.get('modalities', []),
                                    'subjects': dataset.get('subjects', []),
                                    'size': dataset.get('size', 0)
                                })
                        
                        return datasets
                    else:
                        logger.warning(f"OpenNeuro API returned status {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"OpenNeuro search failed: {e}")
            return []
    
    async def get_latest_allen_brain_data(self, 
                                         query: str = "",
                                         max_results: int = 50) -> List[Dict]:
        """
        Get latest Allen Brain Atlas data using latest API v3
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of latest Allen Brain Atlas datasets
        """
        # Use latest Allen Brain Atlas API v3
        try:
            # Get latest cell types data
            cell_types_url = f"{self.config['allen_brain']['base_url']}/data/query.json"
            params = {
                'criteria': '[{"name":"cell_type","op":"LIKE","value":"%neuron%"}]',
                'include': '["structure","donor"]',
                'limit': max_results
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(cell_types_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        datasets = []
                        
                        for item in data.get('msg', [])[:max_results]:
                            datasets.append({
                                'id': item.get('id', ''),
                                'name': item.get('name', ''),
                                'type': 'cell_type',
                                'structure': item.get('structure', {}).get('name', ''),
                                'species': item.get('donor', {}).get('species', ''),
                                'url': f"https://celltypes.brain-map.org/mouse/cell/{item.get('id', '')}",
                                'metadata': item
                            })
                        
                        return datasets
                    else:
                        logger.warning(f"Allen Brain API returned status {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"Allen Brain search failed: {e}")
            return []
    
    async def get_latest_microns_data(self, 
                                     query: str = "",
                                     max_results: int = 50) -> List[Dict]:
        """
        Get latest MICrONS data using latest BossDB API
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of latest MICrONS datasets
        """
        # Use latest BossDB API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['microns']['base_url']}/datasets") as response:
                    if response.status == 200:
                        data = await response.json()
                        datasets = []
                        
                        for dataset in data.get('datasets', [])[:max_results]:
                            if 'microns' in dataset.get('name', '').lower():
                                datasets.append({
                                    'id': dataset.get('id', ''),
                                    'name': dataset.get('name', ''),
                                    'description': dataset.get('description', ''),
                                    'url': f"https://bossdb.openstorage.io/dataset/{dataset.get('id', '')}",
                                    'metadata': dataset
                                })
                        
                        return datasets
                    else:
                        logger.warning(f"BossDB API returned status {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"MICrONS search failed: {e}")
            return []
    
    async def get_comprehensive_neuroscience_update(self, 
                                                  days_back: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive neuroscience update from all sources
        
        Args:
            days_back: How many days back to search
            
        Returns:
            Comprehensive neuroscience update
        """
        logger.info(f"Getting comprehensive neuroscience update for last {days_back} days")
        
        # Gather data from all sources concurrently
        tasks = [
            self.get_latest_neuroscience_papers("neuroscience", 30, days_back),
            self.get_latest_github_repositories("neuroscience", 20, days_back),
            self.get_latest_dandi_datasets("neuroscience", 20, days_back),
            self.get_latest_openneuro_datasets("neuroscience", 20, days_back),
            self.get_latest_allen_brain_data("neuroscience", 20),
            self.get_latest_microns_data("neuroscience", 20)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        update = {
            'timestamp': datetime.now().isoformat(),
            'days_back': days_back,
            'papers': results[0] if not isinstance(results[0], Exception) else [],
            'repositories': results[1] if not isinstance(results[1], Exception) else [],
            'dandi_datasets': results[2] if not isinstance(results[2], Exception) else [],
            'openneuro_datasets': results[3] if not isinstance(results[3], Exception) else [],
            'allen_brain_data': results[4] if not isinstance(results[4], Exception) else [],
            'microns_data': results[5] if not isinstance(results[5], Exception) else [],
            'summary': {}
        }
        
        # Generate summary
        update['summary'] = {
            'total_papers': len(update['papers']),
            'total_repositories': len(update['repositories']),
            'total_datasets': len(update['dandi_datasets']) + len(update['openneuro_datasets']),
            'total_brain_data': len(update['allen_brain_data']) + len(update['microns_data']),
            'sources': ['arXiv', 'PubMed', 'bioRxiv', 'GitHub', 'DANDI', 'OpenNeuro', 'Allen Brain', 'MICrONS']
        }
        
        logger.info(f"Comprehensive update completed: {update['summary']}")
        return update
    
    def export_update_to_file(self, update: Dict[str, Any], output_path: Path) -> Path:
        """Export neuroscience update to file"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_path / f"neuroscience_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(update, f, indent=2, default=str)
        
        # Save summary as CSV
        csv_path = output_path / f"neuroscience_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create summary dataframe
        summary_data = []
        for source in ['papers', 'repositories', 'dandi_datasets', 'openneuro_datasets', 'allen_brain_data', 'microns_data']:
            for item in update.get(source, []):
                summary_data.append({
                    'source': source,
                    'name': item.get('name', item.get('title', '')),
                    'description': item.get('description', item.get('summary', '')),
                    'url': item.get('url', ''),
                    'date': item.get('date', item.get('created', item.get('created_at', '')))
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False)
        
        return output_path
    
    async def get_comprehensive_neuroscience_update_with_brain_development(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive neuroscience update including brain development data
        
        Args:
            days_back: How many days back to search
            
        Returns:
            Comprehensive neuroscience update with brain development
        """
        # Get standard update
        standard_update = await self.get_comprehensive_neuroscience_update(days_back)
        
        # Add brain development data if available
        if self.brain_dev_trainer:
            try:
                brain_dev_data = self.brain_dev_trainer.get_training_data_for_model("all")
                standard_update['brain_development'] = {
                    'stages_count': len(brain_dev_data.get('timeline', {}).get('stages', [])),
                    'processes_count': len(brain_dev_data.get('timeline', {}).get('processes', [])),
                    'cell_types_count': len(brain_dev_data.get('cell_types', {})),
                    'morphogens_count': len(brain_dev_data.get('morphogens', {})),
                    'summary': "Human brain development training data from fertilization to birth"
                }
                standard_update['summary']['total_brain_development_sources'] = 1
                standard_update['summary']['sources'].append('Human Brain Development Training Pack')
            except Exception as e:
                logger.error(f"Failed to add brain development data: {e}")
                standard_update['brain_development'] = {'error': str(e)}
        else:
            standard_update['brain_development'] = {'error': 'Brain development trainer not available'}
        
        return standard_update
    
    def safe_brain_development_query(self, question: str, max_length: int = 1000) -> Dict[str, Any]:
        """
        Safely query human brain development knowledge
        
        Args:
            question: User question about brain development
            max_length: Maximum response length
            
        Returns:
            Safe response with citations and uncertainty
        """
        try:
            response = self.brain_dev_trainer.safe_query(question, max_length)
            logger.info(f"Brain development query processed safely: {len(response.get('citations', []))} citations")
            return response
        except Exception as e:
            logger.error(f"Error in brain development query: {e}")
            return {
                'answer': "I encountered an error while processing your question about brain development. Please try rephrasing.",
                'citations': [],
                'uncertainty': 'high',
                'safety_warnings': ['Processing error occurred'],
                'source_modules': []
            }
    
    def get_brain_development_training_summary(self) -> Dict[str, Any]:
        """Get summary of brain development training materials"""
        try:
            return self.brain_dev_trainer.get_training_summary()
        except Exception as e:
            logger.error(f"Error getting training summary: {e}")
            return {'error': str(e)}
    
    def export_brain_development_examples(self, output_path: Path) -> Path:
        """Export safe response examples for validation"""
        try:
            return self.brain_dev_trainer.export_safe_responses(output_path)
        except Exception as e:
            logger.error(f"Error exporting examples: {e}")
            return output_path

    async def get_fetal_brain_simulation_tools(self) -> Dict[str, Any]:
        """
        Get comprehensive overview of fetal brain simulation tools
        
        Returns:
            Dictionary containing all available fetal brain simulation tools
        """
        logger.info("Retrieving fetal brain simulation tools overview")
        
        fetal_tools = {
            'timestamp': datetime.now().isoformat(),
            'tools': {
                'imaging_simulation': {
                    'fabian': {
                        'name': 'FaBiAN (Fetal Brain MR Acquisition Numerical phantom)',
                        'version': '2.0 (2025)',
                        'gestational_range': '20-34.8 weeks',
                        'type': 'Synthetic T2-weighted MRI simulation',
                        'features': [
                            '594 synthetic T2 MRI series across 78 fetal brains',
                            'Motion effects and anatomical maturation simulation',
                            'Both healthy and pathological anatomies',
                            'Enhanced fidelity compared to original version',
                            'Supports fetal brain tissue segmentation'
                        ],
                        'url': self.config['fabian']['dataset_url'],
                        'docker_image': self.config['fabian']['docker_image'],
                        'applications': [
                            'Validating image processing algorithms',
                            'Augmenting scarce clinical datasets',
                            'Training deep learning models',
                            'Optimizing reconstruction techniques'
                        ]
                    }
                },
                'anatomical_atlases': {
                    'embryonic_4d': {
                        'name': '4D Human Embryonic Brain Atlas',
                        'version': '2025 (arXiv preprint)',
                        'gestational_range': '8-12 weeks',
                        'type': 'Deep learning group-wise registration',
                        'features': [
                            'Based on ultrasound imaging',
                            'Captures rapid anatomical changes',
                            'High anatomical accuracy',
                            'Spatiotemporal development mapping'
                        ],
                        'url': self.config['embryonic_atlas']['base_url'],
                        'applications': [
                            'Early brain development mapping',
                            'Morphological change tracking',
                            'Developmental timeline analysis'
                        ]
                    },
                    'fetal_segmentation': {
                        'name': 'Deep-Learning Fetal Brain Atlas',
                        'version': '2025 (arXiv preprint)',
                        'gestational_range': '21-37 weeks',
                        'type': 'Conditional deep learning segmentation',
                        'features': [
                            'Continuous age-specific atlases',
                            'Real-time segmentation capability',
                            'High structural fidelity (Dice ≈ 86%)',
                            'Conditional model architecture'
                        ],
                        'url': self.config['fetal_atlas']['base_url'],
                        'applications': [
                            'Automated fetal brain segmentation',
                            'Age-specific atlas generation',
                            'Clinical workflow integration'
                        ]
                    }
                },
                'functional_simulation': {
                    'reward': {
                        'name': 'ReWaRD (Retinal Wave-based Representation Development)',
                        'version': '2023 (arXiv preprint)',
                        'type': 'Prenatal visual signal simulation',
                        'features': [
                            'Simulates retinal waves (early prenatal visual phenomenon)',
                            'Pretrains CNN models with biological patterns',
                            'Features align with V1-level visual representation',
                            'Biological pretraining approach'
                        ],
                        'url': self.config['reward']['base_url'],
                        'applications': [
                            'Early visual cortex development modeling',
                            'Biological pretraining for neural networks',
                            'Sensory development simulation'
                        ]
                    }
                },
                'multiscale_modeling': {
                    'compucell3d': {
                        'name': 'CompuCell3D',
                        'type': 'Multiscale agent-based modeling',
                        'features': [
                            'Cellular Potts model',
                            'Reaction-diffusion systems',
                            'Morphogen gradient modeling',
                            'Open-source and extensible'
                        ],
                        'url': self.config['compucell3d']['base_url'],
                        'applications': [
                            'Neurulation simulation',
                            'Patterning and morphogenesis',
                            'Cellular behavior modeling'
                        ]
                    },
                    'copasi': {
                        'name': 'COPASI',
                        'type': 'Biochemical network simulation',
                        'features': [
                            'Cell-signaling networks',
                            'Gene-regulatory networks',
                            'SHH, WNT gradient modeling',
                            'Early brain patterning simulation'
                        ],
                        'url': self.config['copasi']['base_url'],
                        'applications': [
                            'Morphogen network dynamics',
                            'Cellular signaling pathways',
                            'Developmental biology modeling'
                        ]
                    }
                },
                'neural_simulation': {
                    'nest': {
                        'name': 'NEST',
                        'type': 'Large-scale spiking neural networks',
                        'features': [
                            'Neurons, synapses, and measurement devices',
                            'Fully open-source and scriptable',
                            'Adaptable for developmental research',
                            'High-performance simulation'
                        ],
                        'url': self.config['nest']['base_url'],
                        'applications': [
                            'Neural circuit modeling',
                            'Developmental network simulation',
                            'Large-scale brain modeling'
                        ]
                    },
                    'emergent': {
                        'name': 'Emergent (formerly PDP++)',
                        'type': 'Biologically-inspired cognitive modeling',
                        'features': [
                            'Layered architectures (Leabra)',
                            'Open-source and extensible',
                            'Cognitive development modeling',
                            'Biologically plausible learning'
                        ],
                        'url': self.config['emergent']['base_url'],
                        'applications': [
                            'Cognitive development simulation',
                            'Learning algorithm development',
                            'Neural architecture exploration'
                        ]
                    },
                    'blue_brain': {
                        'name': 'Blue Brain Project Tools',
                        'type': 'Advanced anatomical and biophysical simulation',
                        'features': [
                            'BluePyOpt, CoreNEURON, SONATA',
                            'Detailed neural modeling',
                            'High-fidelity simulation',
                            'Research-grade tools'
                        ],
                        'url': self.config['blue_brain']['base_url'],
                        'applications': [
                            'Detailed neural circuit modeling',
                            'Biophysical simulation',
                            'Advanced brain modeling'
                        ]
                    }
                }
            },
            'integration_recommendations': {
                'hybrid_pipeline': {
                    'description': 'Combine anatomical simulation with mechanistic modeling',
                    'components': [
                        'FaBiAN for structural MRI simulation',
                        '4D embryonic atlas for early development',
                        'CompuCell3D/COPASI for cellular processes',
                        'NEST/Emergent for neural activity'
                    ],
                    'workflow': 'Anatomical → Cellular → Neural → Functional'
                },
                'phased_architecture': {
                    'description': 'Phased approach for early brain processing simulation',
                    'phases': [
                        'Phase 1: Anatomical structure (FaBiAN + atlases)',
                        'Phase 2: Cell patterning (CompuCell3D + COPASI)',
                        'Phase 3: Neural network activity (NEST + Emergent)',
                        'Phase 4: Sensory primitive signals (ReWaRD approach)'
                    ]
                }
            }
        }
        
        logger.info(f"Retrieved {len(fetal_tools['tools'])} categories of fetal brain simulation tools")
        return fetal_tools

    async def create_fetal_brain_development_pipeline(self) -> Dict[str, Any]:
        """
        Create a comprehensive fetal brain development pipeline integrating all tools
        
        Returns:
            Dictionary containing the integrated pipeline configuration
        """
        logger.info("Creating integrated fetal brain development pipeline")
        
        # Get fetal simulation tools
        fetal_tools = await self.get_fetal_brain_simulation_tools()
        
        # Create integrated pipeline
        pipeline = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_name': 'SmallMind Fetal Brain Development Pipeline',
            'description': 'Integrated pipeline combining anatomical, cellular, and neural simulation tools',
            'gestational_coverage': {
                'embryonic': '8-12 weeks (4D atlas)',
                'early_fetal': '20-34.8 weeks (FaBiAN)',
                'late_fetal': '21-37 weeks (conditional atlas)',
                'total_coverage': '8-37 weeks gestation'
            },
            'pipeline_stages': {
                'stage_1': {
                    'name': 'Anatomical Structure Simulation',
                    'tools': ['FaBiAN', '4D Embryonic Atlas', 'Conditional Fetal Atlas'],
                    'output': 'Synthetic MRI data and anatomical models',
                    'gestational_range': '8-37 weeks',
                    'description': 'Generate realistic fetal brain anatomy across development'
                },
                'stage_2': {
                    'name': 'Cellular and Tissue Modeling',
                    'tools': ['CompuCell3D', 'COPASI'],
                    'output': 'Cellular behavior and signaling networks',
                    'gestational_range': '8-37 weeks',
                    'description': 'Model cellular processes, morphogen gradients, and tissue patterning'
                },
                'stage_3': {
                    'name': 'Neural Network Simulation',
                    'tools': ['NEST', 'Emergent', 'Blue Brain Project'],
                    'output': 'Neural activity and learning dynamics',
                    'gestational_range': '8-37 weeks',
                    'description': 'Simulate neural circuit development and activity patterns'
                },
                'stage_4': {
                    'name': 'Functional Development',
                    'tools': ['ReWaRD', 'Custom sensory models'],
                    'output': 'Sensory processing and functional responses',
                    'gestational_range': '8-37 weeks',
                    'description': 'Model early sensory development and functional emergence'
                }
            },
            'integration_workflow': {
                'data_flow': [
                    '1. Generate anatomical models using FaBiAN and atlases',
                    '2. Apply cellular modeling for tissue development',
                    '3. Overlay neural networks on anatomical structure',
                    '4. Integrate functional development models'
                ],
                'validation_steps': [
                    'Compare synthetic data with clinical benchmarks',
                    'Validate cellular models against developmental biology',
                    'Test neural models against known developmental patterns',
                    'Verify functional models against developmental milestones'
                ]
            },
            'implementation_guide': {
                'docker_setup': {
                    'fabian': 'docker pull petermcgor/fabian-docker:latest',
                    'compucell3d': 'Follow CompuCell3D installation guide',
                    'nest': 'Follow NEST installation guide',
                    'emergent': 'git clone https://github.com/emer/emergent'
                },
                'data_integration': {
                    'anatomical_data': 'Use FaBiAN for MRI simulation, atlases for structure',
                    'cellular_data': 'CompuCell3D for morphogenesis, COPASI for signaling',
                    'neural_data': 'NEST for spiking networks, Emergent for cognitive models',
                    'functional_data': 'ReWaRD approach for sensory development'
                },
                'pipeline_execution': {
                    'parallel_processing': 'Run anatomical and cellular stages in parallel',
                    'sequential_dependencies': 'Neural and functional stages depend on previous stages',
                    'data_synchronization': 'Ensure consistent gestational age mapping across stages',
                    'output_validation': 'Validate each stage before proceeding to next'
                }
            },
            'research_applications': {
                'clinical_research': [
                    'Fetal brain development studies',
                    'Pathological development modeling',
                    'Treatment response prediction',
                    'Developmental biomarker identification'
                ],
                'computational_research': [
                    'AI model pretraining with biological data',
                    'Developmental algorithm development',
                    'Multi-scale modeling validation',
                    'Neural architecture exploration'
                ],
                'educational_applications': [
                    'Medical training and education',
                    'Developmental biology visualization',
                    'Neuroscience research training',
                    'Clinical decision support'
                ]
            },
            'safety_considerations': {
                'data_privacy': 'All data is synthetic or publicly available',
                'ethical_guidelines': 'Follow established research ethics protocols',
                'validation_requirements': 'Compare against known developmental data',
                'uncertainty_quantification': 'Include uncertainty estimates in all outputs'
            }
        }
        
        logger.info("Fetal brain development pipeline created successfully")
        return pipeline

    async def get_comprehensive_neuroscience_update_with_fetal_tools(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive neuroscience update including fetal brain simulation tools
        
        Args:
            days_back: How many days back to search
            
        Returns:
            Comprehensive neuroscience update with fetal tools
        """
        # Get standard update
        standard_update = await self.get_comprehensive_neuroscience_update(days_back)
        
        # Add fetal brain simulation tools
        try:
            fetal_tools = await self.get_fetal_brain_simulation_tools()
            standard_update['fetal_brain_simulation'] = fetal_tools
            standard_update['summary']['total_fetal_tools'] = len(fetal_tools['tools'])
            standard_update['summary']['sources'].extend(['FaBiAN', '4D Embryonic Atlas', 'ReWaRD', 'CompuCell3D', 'COPASI', 'NEST', 'Emergent', 'Blue Brain'])
            logger.info("Added fetal brain simulation tools to comprehensive update")
        except Exception as e:
            logger.error(f"Failed to add fetal brain simulation tools: {e}")
            standard_update['fetal_brain_simulation'] = {'error': str(e)}
        
        # Add brain development data if available
        if self.brain_dev_trainer:
            try:
                brain_dev_data = self.brain_dev_trainer.get_training_data_for_model("all")
                standard_update['brain_development'] = {
                    'stages_count': len(brain_dev_data.get('timeline', {}).get('stages', [])),
                    'processes_count': len(brain_dev_data.get('timeline', {}).get('processes', [])),
                    'cell_types_count': len(brain_dev_data.get('cell_types', {})),
                    'morphogens_count': len(brain_dev_data.get('morphogens', {})),
                    'summary': "Human brain development training data from fertilization to birth"
                }
                standard_update['summary']['total_brain_development_sources'] = 1
                standard_update['summary']['sources'].append('Human Brain Development Training Pack')
            except Exception as e:
                logger.error(f"Failed to add brain development data: {e}")
                standard_update['brain_development'] = {'error': str(e)}
        else:
            standard_update['brain_development'] = {'error': 'Brain development trainer not available'}
        
        return standard_update

# Factory function
def create_enhanced_data_resources() -> EnhancedDataResources:
    """Create and return enhanced data resources instance"""
    return EnhancedDataResources()
