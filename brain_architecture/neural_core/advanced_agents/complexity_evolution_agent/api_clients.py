# brain_modules/complexity_evolution_agent/api_clients.py

"""
Purpose: External API client implementations for neuroscience, ML, and biological resources
Inputs: API endpoints, authentication keys, request parameters
Outputs: Structured data from external sources, validation results
Dependencies: aiohttp, requests, external API endpoints
"""

import os
import json
import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass
import time
import hashlib

@dataclass
class APIResponse:
    """Standardized API response format"""
    success: bool
    data: Any
    timestamp: datetime
    source: str
    validation_score: float
    error_message: Optional[str] = None

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.request_times.append(now)

class BaseAPIClient:
    """Base class for all external API clients"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                 requests_per_minute: int = 60):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.session = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "ComplexityEvolutionAgent/1.0 (Brain Simulation Project)",
                    "Accept": "application/json"
                }
            )
        return self.session
    
    async def fetch_data(self, endpoint: str, params: Dict = None, 
                        headers: Dict = None) -> APIResponse:
        """Fetch data from API endpoint with rate limiting"""
        try:
            await self.rate_limiter.wait_if_needed()
            
            session = await self._get_session()
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            if headers is None:
                headers = {}
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with session.get(url, params=params, headers=headers, timeout=30) as response:
                if response.status == 200:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        data = await response.json()
                    else:
                        data = await response.text()
                    
                    return APIResponse(
                        success=True,
                        data=data,
                        timestamp=datetime.now(),
                        source=self.__class__.__name__,
                        validation_score=1.0
                    )
                else:
                    return APIResponse(
                        success=False,
                        data=None,
                        timestamp=datetime.now(),
                        source=self.__class__.__name__,
                        validation_score=0.0,
                        error_message=f"HTTP {response.status}: {response.reason}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error fetching data from {endpoint}: {str(e)}")
            return APIResponse(
                success=False,
                data=None,
                timestamp=datetime.now(),
                source=self.__class__.__name__,
                validation_score=0.0,
                error_message=str(e)
            )
    
    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()

class AllenBrainAtlasClient(BaseAPIClient):
    """Client for Allen Brain Atlas API"""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.brain-map.org/api/v2",
            requests_per_minute=30  # Conservative rate limit
        )
    
    async def get_brain_regions(self) -> APIResponse:
        """Fetch brain region data"""
        return await self.fetch_data("data/Structure/query.json", {
            "criteria": "[graph_id$eq1]",
            "include": "structure_sets",
            "format": "json"
        })
    
    async def get_gene_expression(self, region_id: str) -> APIResponse:
        """Fetch gene expression data for a brain region"""
        return await self.fetch_data("data/Expression/query.json", {
            "criteria": f"[reference_space_id$eq1][section_data_set_id$eq{region_id}]",
            "include": "genes,section_data_sets",
            "format": "json"
        })
    
    async def get_connectivity_data(self, region_id: str) -> APIResponse:
        """Fetch connectivity data for a brain region"""
        return await self.fetch_data("data/Connectivity/query.json", {
            "criteria": f"[structure_id$eq{region_id}]",
            "include": "structure,section_data_sets",
            "format": "json"
        })

class HuggingFaceClient(BaseAPIClient):
    """Client for Hugging Face Model Hub"""
    
    def __init__(self):
        super().__init__(
            base_url="https://huggingface.co/api",
            requests_per_minute=100
        )
    
    async def get_trending_models(self, domain: str = "neuroscience", limit: int = 10) -> APIResponse:
        """Fetch trending models in neuroscience domain"""
        # Search for neuroscience-related models
        search_query = f"neuroscience OR brain OR neural OR consciousness"
        return await self.fetch_data("models", {
            "search": search_query,
            "sort": "downloads",
            "direction": "-1",
            "limit": limit
        })
    
    async def get_model_metrics(self, model_id: str) -> APIResponse:
        """Fetch performance metrics for a specific model"""
        return await self.fetch_data(f"models/{model_id}")
    
    async def get_model_benchmarks(self, model_id: str) -> APIResponse:
        """Fetch benchmark results for a model"""
        return await self.fetch_data(f"models/{model_id}/evaluation_results")

class PubMedClient(BaseAPIClient):
    """Client for NCBI PubMed API"""
    
    def __init__(self):
        super().__init__(
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            requests_per_minute=10  # NCBI rate limit
        )
    
    async def search_publications(self, query: str, max_results: int = 20) -> APIResponse:
        """Search for publications in PubMed"""
        # First, search for IDs
        search_response = await self.fetch_data("esearch.fcgi", {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        })
        
        if not search_response.success:
            return search_response
        
        # Then fetch details for each ID
        if isinstance(search_response.data, dict) and "esearchresult" in search_response.data:
            id_list = search_response.data["esearchresult"].get("idlist", [])
            if id_list:
                details_response = await self.fetch_data("esummary.fcgi", {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json"
                })
                return details_response
        
        return search_response
    
    async def get_neuroscience_publications(self, days_back: int = 30) -> APIResponse:
        """Get recent neuroscience publications"""
        query = "neuroscience[Title/Abstract] AND consciousness[Title/Abstract]"
        return await self.search_publications(query, max_results=50)
    
    async def get_publication_details(self, pubmed_id: str) -> APIResponse:
        """Get detailed information about a specific publication"""
        return await self.fetch_data("esummary.fcgi", {
            "db": "pubmed",
            "id": pubmed_id,
            "retmode": "json"
        })

class PapersWithCodeClient(BaseAPIClient):
    """Client for Papers With Code API"""
    
    def __init__(self):
        super().__init__(
            base_url="https://paperswithcode.com/api/v1",
            requests_per_minute=60
        )
    
    async def search_papers(self, query: str, items_per_page: int = 20) -> APIResponse:
        """Search for papers with code"""
        return await self.fetch_data("papers", {
            "search": query,
            "items_per_page": items_per_page
        })
    
    async def get_neuroscience_papers(self) -> APIResponse:
        """Get neuroscience papers with implementations"""
        return await self.search_papers("neuroscience consciousness brain neural")
    
    async def get_paper_implementations(self, paper_id: str) -> APIResponse:
        """Get implementations for a specific paper"""
        return await self.fetch_data(f"papers/{paper_id}/implementations")

class GitHubClient(BaseAPIClient):
    """Client for GitHub API"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            base_url="https://api.github.com",
            api_key=api_key or os.getenv("GITHUB_API_KEY"),
            requests_per_minute=30  # GitHub rate limit
        )
    
    async def search_repositories(self, query: str, sort: str = "stars", 
                                order: str = "desc", per_page: int = 20) -> APIResponse:
        """Search for GitHub repositories"""
        headers = {"Accept": "application/vnd.github.v3+json"}
        return await self.fetch_data("search/repositories", {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page
        }, headers=headers)
    
    async def get_neuroscience_repos(self) -> APIResponse:
        """Get trending neuroscience repositories"""
        return await self.search_repositories(
            "neuroscience brain simulation consciousness",
            sort="stars",
            order="desc"
        )
    
    async def get_repository_details(self, owner: str, repo: str) -> APIResponse:
        """Get detailed information about a repository"""
        headers = {"Accept": "application/vnd.github.v3+json"}
        return await self.fetch_data(f"repos/{owner}/{repo}", headers=headers)

class WikipediaClient(BaseAPIClient):
    """Client for Wikipedia API"""
    
    def __init__(self):
        super().__init__(
            base_url="https://en.wikipedia.org/api/rest_v1",
            requests_per_minute=100
        )
    
    async def search_pages(self, query: str, limit: int = 10) -> APIResponse:
        """Search for Wikipedia pages"""
        return await self.fetch_data("page/search", {
            "q": query,
            "limit": limit
        })
    
    async def get_page_summary(self, title: str) -> APIResponse:
        """Get summary of a Wikipedia page"""
        # Convert title to URL-safe format
        safe_title = title.replace(" ", "_")
        return await self.fetch_data(f"page/summary/{safe_title}")
    
    async def get_neuroscience_content(self) -> APIResponse:
        """Get neuroscience-related Wikipedia content"""
        return await self.search_pages("neuroscience consciousness brain", limit=20)

class PyPIClient(BaseAPIClient):
    """Client for PyPI API"""
    
    def __init__(self):
        super().__init__(
            base_url="https://pypi.org/pypi",
            requests_per_minute=100
        )
    
    async def search_packages(self, query: str) -> APIResponse:
        """Search for Python packages"""
        return await self.fetch_data("search", {
            "q": query
        })
    
    async def get_package_info(self, package_name: str) -> APIResponse:
        """Get information about a specific package"""
        return await self.fetch_data(f"{package_name}/json")
    
    async def get_neuroscience_packages(self) -> APIResponse:
        """Get neuroscience-related Python packages"""
        return await self.search_packages("neuroscience brain neural consciousness")

class ConsciousnessResearchClient(BaseAPIClient):
    """Client for Consciousness Research Database (simulated)"""
    
    def __init__(self):
        super().__init__(
            base_url="https://consciousness-research.org/api",
            requests_per_minute=20
        )
    
    async def get_research_studies(self, topic: str = "consciousness") -> APIResponse:
        """Get consciousness research studies"""
        # This is a simulated endpoint - in reality, you'd connect to actual API
        simulated_data = {
            "studies": [
                {
                    "id": "study_001",
                    "title": "Neural Correlates of Consciousness",
                    "authors": ["Koch, C.", "Tsuchiya, N."],
                    "year": 2023,
                    "methodology": "fMRI",
                    "sample_size": 150,
                    "findings": "Identified key neural signatures of conscious awareness"
                },
                {
                    "id": "study_002", 
                    "title": "Consciousness in Artificial Neural Networks",
                    "authors": ["Tononi, G.", "Koch, C."],
                    "year": 2024,
                    "methodology": "Computational modeling",
                    "sample_size": "N/A",
                    "findings": "Developed IIT-based consciousness measures for ANNs"
                }
            ],
            "total_count": 2,
            "query": topic
        }
        
        return APIResponse(
            success=True,
            data=simulated_data,
            timestamp=datetime.now(),
            source=self.__class__.__name__,
            validation_score=0.9
        )

# Factory function to create appropriate client
def create_api_client(resource_type: str, **kwargs) -> BaseAPIClient:
    """Create an appropriate API client based on resource type"""
    clients = {
        "allen_brain_atlas": AllenBrainAtlasClient,
        "huggingface": HuggingFaceClient,
        "pubmed": PubMedClient,
        "papers_with_code": PapersWithCodeClient,
        "github": GitHubClient,
        "wikipedia": WikipediaClient,
        "pypi": PyPIClient,
        "consciousness_research": ConsciousnessResearchClient
    }
    
    if resource_type not in clients:
        raise ValueError(f"Unknown resource type: {resource_type}")
    
    return clients[resource_type](**kwargs)

# Test function
async def test_api_clients():
    """Test all API clients"""
    print("🧪 Testing API Clients...")
    
    clients = [
        ("Allen Brain Atlas", AllenBrainAtlasClient()),
        ("Hugging Face", HuggingFaceClient()),
        ("PubMed", PubMedClient()),
        ("Papers With Code", PapersWithCodeClient()),
        ("GitHub", GitHubClient()),
        ("Wikipedia", WikipediaClient()),
        ("PyPI", PyPIClient()),
        ("Consciousness Research", ConsciousnessResearchClient())
    ]
    
    results = {}
    
    for name, client in clients:
        print(f"\n🔗 Testing {name}...")
        try:
            if name == "Allen Brain Atlas":
                response = await client.get_brain_regions()
            elif name == "Hugging Face":
                response = await client.get_trending_models("neuroscience", 5)
            elif name == "PubMed":
                response = await client.get_neuroscience_publications(7)
            elif name == "Papers With Code":
                response = await client.get_neuroscience_papers()
            elif name == "GitHub":
                response = await client.get_neuroscience_repos()
            elif name == "Wikipedia":
                response = await client.get_neuroscience_content()
            elif name == "PyPI":
                response = await client.get_neuroscience_packages()
            elif name == "Consciousness Research":
                response = await client.get_research_studies()
            
            results[name] = {
                "success": response.success,
                "validation_score": response.validation_score,
                "data_type": type(response.data).__name__,
                "timestamp": response.timestamp.isoformat()
            }
            
            if response.success:
                print(f"✅ {name}: Success (Score: {response.validation_score:.2f})")
            else:
                print(f"❌ {name}: Failed - {response.error_message}")
                
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")
            results[name] = {"success": False, "error": str(e)}
        
        await client.close()
    
    print(f"\n📊 Test Results Summary:")
    successful = sum(1 for r in results.values() if r.get("success", False))
    total = len(results)
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_api_clients())
