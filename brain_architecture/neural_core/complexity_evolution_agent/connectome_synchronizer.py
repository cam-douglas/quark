# brain_modules/complexity_evolution_agent/connectome_synchronizer.py

"""
Purpose: Synchronize project documentation and rules with external neuroscience, ML, and biological resources
Inputs: External API endpoints, project documentation, validation rules
Outputs: Updated documentation, validation reports, sync status
Dependencies: api_clients.py, external APIs, validation schemas
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass, asdict
import yaml
from pathlib import Path

# Import our API clients
try:
    from .api_clients import (
        OpenAIClient,
        AnthropicClient,
        HuggingFaceClient,
        GoogleAIClient,
        AWSClient,
        AzureClient,
        NVIDIAClient,
        CloudflareClient,
    )
except ImportError:
    from api_clients import (
        AllenBrainAtlasClient, HuggingFaceClient, PubMedClient, 
        PapersWithCodeClient, GitHubClient, WikipediaClient, 
        PyPIClient, ConsciousnessResearchClient, APIResponse
    )

@dataclass
class SyncResult:
    """Result of a synchronization operation"""
    success: bool
    resource_type: str
    timestamp: datetime
    data_count: int
    validation_score: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConnectomeSynchronizer:
    """Main synchronizer for external resources"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # Initialize API clients
        self.clients = {
            "allen_brain_atlas": AllenBrainAtlasClient(),
            "huggingface": HuggingFaceClient(),
            "pubmed": PubMedClient(),
            "papers_with_code": PapersWithCodeClient(),
            "github": GitHubClient(),
            "wikipedia": WikipediaClient(),
            "pypi": PyPIClient(),
            "consciousness_research": ConsciousnessResearchClient()
        }
        
        # Sync configuration
        self.sync_config = self._load_sync_config()
        self.sync_history = []
        
        self.logger.info("Connectome Synchronizer initialized")
    
    def _load_sync_config(self) -> Dict[str, Any]:
        """Load synchronization configuration"""
        config_path = self.project_root / "configs" / "sync_config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "sync_interval_hours": 24,
            "validation_threshold": 0.8,
            "resources": {
                "high_priority": ["allen_brain_atlas", "pubmed", "consciousness_research"],
                "medium_priority": ["huggingface", "papers_with_code", "github"],
                "low_priority": ["wikipedia", "pypi"]
            }
        }
    
    async def sync_neuroscience_resources(self) -> List[SyncResult]:
        """Synchronize neuroscience-specific resources"""
        self.logger.info("Starting neuroscience resource synchronization")
        
        results = []
        
        # Allen Brain Atlas - Brain regions and connectivity
        try:
            response = await self.clients["allen_brain_atlas"].get_brain_regions()
            if response.success:
                results.append(SyncResult(
                    success=True,
                    resource_type="allen_brain_atlas",
                    timestamp=datetime.now(),
                    data_count=len(response.data.get("msg", [])),
                    validation_score=response.validation_score
                ))
            else:
                results.append(SyncResult(
                    success=False,
                    resource_type="allen_brain_atlas",
                    timestamp=datetime.now(),
                    data_count=0,
                    validation_score=0.0,
                    error_message=response.error_message
                ))
        except Exception as e:
            self.logger.error(f"Error syncing Allen Brain Atlas: {str(e)}")
            results.append(SyncResult(
                success=False,
                resource_type="allen_brain_atlas",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(e)
            ))
        
        # PubMed - Neuroscience publications
        try:
            response = await self.clients["pubmed"].get_neuroscience_publications(7)
            if response.success:
                results.append(SyncResult(
                    success=True,
                    resource_type="pubmed",
                    timestamp=datetime.now(),
                    data_count=len(response.data.get("result", {}).get("uids", [])),
                    validation_score=response.validation_score
                ))
            else:
                results.append(SyncResult(
                    success=False,
                    resource_type="pubmed",
                    timestamp=datetime.now(),
                    data_count=0,
                    validation_score=0.0,
                    error_message=response.error_message
                ))
        except Exception as e:
            self.logger.error(f"Error syncing PubMed: {str(e)}")
            results.append(SyncResult(
                success=False,
                resource_type="pubmed",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(e)
            ))
        
        # Consciousness Research
        try:
            response = await self.clients["consciousness_research"].get_research_studies()
            if response.success:
                results.append(SyncResult(
                    success=True,
                    resource_type="consciousness_research",
                    timestamp=datetime.now(),
                    data_count=len(response.data.get("studies", [])),
                    validation_score=response.validation_score
                ))
            else:
                results.append(SyncResult(
                    success=False,
                    resource_type="consciousness_research",
                    timestamp=datetime.now(),
                    data_count=0,
                    validation_score=0.0,
                    error_message=response.error_message
                ))
        except Exception as e:
            self.logger.error(f"Error syncing Consciousness Research: {str(e)}")
            results.append(SyncResult(
                success=False,
                resource_type="consciousness_research",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(e)
            ))
        
        return results
    
    async def sync_ml_resources(self) -> List[SyncResult]:
        """Synchronize machine learning resources"""
        self.logger.info("Starting ML resource synchronization")
        
        results = []
        
        # Hugging Face - Trending neuroscience models
        try:
            response = await self.clients["huggingface"].get_trending_models("neuroscience", 10)
            if response.success:
                results.append(SyncResult(
                    success=True,
                    resource_type="huggingface",
                    timestamp=datetime.now(),
                    data_count=len(response.data),
                    validation_score=response.validation_score
                ))
            else:
                results.append(SyncResult(
                    success=False,
                    resource_type="huggingface",
                    timestamp=datetime.now(),
                    data_count=0,
                    validation_score=0.0,
                    error_message=response.error_message
                ))
        except Exception as e:
            self.logger.error(f"Error syncing Hugging Face: {str(e)}")
            results.append(SyncResult(
                success=False,
                resource_type="huggingface",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(e)
            ))
        
        # Papers With Code - Neuroscience papers
        try:
            response = await self.clients["papers_with_code"].get_neuroscience_papers()
            if response.success:
                results.append(SyncResult(
                    success=True,
                    resource_type="papers_with_code",
                    timestamp=datetime.now(),
                    data_count=len(response.data.get("results", [])),
                    validation_score=response.validation_score
                ))
            else:
                results.append(SyncResult(
                    success=False,
                    resource_type="papers_with_code",
                    timestamp=datetime.now(),
                    data_count=0,
                    validation_score=0.0,
                    error_message=response.error_message
                ))
        except Exception as e:
            self.logger.error(f"Error syncing Papers With Code: {str(e)}")
            results.append(SyncResult(
                success=False,
                resource_type="papers_with_code",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(e)
            ))
        
        # GitHub - Neuroscience repositories
        try:
            response = await self.clients["github"].get_neuroscience_repos()
            if response.success:
                results.append(SyncResult(
                    success=True,
                    resource_type="github",
                    timestamp=datetime.now(),
                    data_count=len(response.data.get("items", [])),
                    validation_score=response.validation_score
                ))
            else:
                results.append(SyncResult(
                    success=False,
                    resource_type="github",
                    timestamp=datetime.now(),
                    data_count=0,
                    validation_score=0.0,
                    error_message=response.error_message
                ))
        except Exception as e:
            self.logger.error(f"Error syncing GitHub: {str(e)}")
            results.append(SyncResult(
                success=False,
                resource_type="github",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(e)
            ))
        
        return results
    
    async def sync_knowledge_resources(self) -> List[SyncResult]:
        """Synchronize knowledge resources"""
        self.logger.info("Starting knowledge resource synchronization")
        
        results = []
        
        # Wikipedia - Neuroscience content
        try:
            response = await self.clients["wikipedia"].get_neuroscience_content()
            if response.success:
                results.append(SyncResult(
                    success=True,
                    resource_type="wikipedia",
                    timestamp=datetime.now(),
                    data_count=len(response.data.get("pages", [])),
                    validation_score=response.validation_score
                ))
            else:
                results.append(SyncResult(
                    success=False,
                    resource_type="wikipedia",
                    timestamp=datetime.now(),
                    data_count=0,
                    validation_score=0.0,
                    error_message=response.error_message
                ))
        except Exception as e:
            self.logger.error(f"Error syncing Wikipedia: {str(e)}")
            results.append(SyncResult(
                success=False,
                resource_type="wikipedia",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(e)
            ))
        
        # PyPI - Neuroscience packages
        try:
            response = await self.clients["pypi"].get_neuroscience_packages()
            if response.success:
                results.append(SyncResult(
                    success=True,
                    resource_type="pypi",
                    timestamp=datetime.now(),
                    data_count=len(response.data.get("results", [])),
                    validation_score=response.validation_score
                ))
            else:
                results.append(SyncResult(
                    success=False,
                    resource_type="pypi",
                    timestamp=datetime.now(),
                    data_count=0,
                    validation_score=0.0,
                    error_message=response.error_message
                ))
        except Exception as e:
            self.logger.error(f"Error syncing PyPI: {str(e)}")
            results.append(SyncResult(
                success=False,
                resource_type="pypi",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(e)
            ))
        
        return results
    
    async def full_sync(self) -> Dict[str, Any]:
        """Perform full synchronization of all resources"""
        self.logger.info("Starting full external resource synchronization")
        
        start_time = datetime.now()
        
        # Run all sync operations concurrently
        neuroscience_results, ml_results, knowledge_results = await asyncio.gather(
            self.sync_neuroscience_resources(),
            self.sync_ml_resources(),
            self.sync_knowledge_resources(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(neuroscience_results, Exception):
            neuroscience_results = [SyncResult(
                success=False,
                resource_type="neuroscience_sync",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(neuroscience_results)
            )]
        
        if isinstance(ml_results, Exception):
            ml_results = [SyncResult(
                success=False,
                resource_type="ml_sync",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(ml_results)
            )]
        
        if isinstance(knowledge_results, Exception):
            knowledge_results = [SyncResult(
                success=False,
                resource_type="knowledge_sync",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(knowledge_results)
            )]
        
        # Ensure all results are lists of SyncResult
        if isinstance(neuroscience_results, list):
            neuro_list = neuroscience_results
        else:
            neuro_list = [SyncResult(
                success=False,
                resource_type="neuroscience_sync",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(neuroscience_results)
            )]
        
        if isinstance(ml_results, list):
            ml_list = ml_results
        else:
            ml_list = [SyncResult(
                success=False,
                resource_type="ml_sync",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(ml_results)
            )]
        
        if isinstance(knowledge_results, list):
            knowledge_list = knowledge_results
        else:
            knowledge_list = [SyncResult(
                success=False,
                resource_type="knowledge_sync",
                timestamp=datetime.now(),
                data_count=0,
                validation_score=0.0,
                error_message=str(knowledge_results)
            )]
        
        all_results = neuro_list + ml_list + knowledge_list
        
        # Calculate metrics
        total_syncs = len(all_results)
        successful_syncs = sum(1 for r in all_results if r.success)
        avg_validation_score = sum(r.validation_score for r in all_results) / total_syncs if total_syncs > 0 else 0
        total_data_count = sum(r.data_count for r in all_results)
        
        sync_summary = {
            "timestamp": start_time.isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "total_syncs": total_syncs,
            "successful_syncs": successful_syncs,
            "success_rate": successful_syncs / total_syncs if total_syncs > 0 else 0,
            "avg_validation_score": avg_validation_score,
            "total_data_count": total_data_count,
            "results": [asdict(r) for r in all_results]
        }
        
        # Store in sync history
        self.sync_history.append(sync_summary)
        
        # Save sync report
        self._save_sync_report(sync_summary)
        
        self.logger.info(f"Full sync completed: {successful_syncs}/{total_syncs} successful")
        
        return sync_summary
    
    def _save_sync_report(self, sync_summary: Dict[str, Any]):
        """Save synchronization report to file"""
        reports_dir = self.project_root / "docs" / "sync_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"sync_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(sync_summary, f, indent=2, default=str)
        
        self.logger.info(f"Sync report saved to {report_path}")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status"""
        if not self.sync_history:
            return {
                "status": "no_syncs_performed",
                "last_sync": None,
                "success_rate": 0.0,
                "avg_validation_score": 0.0
            }
        
        latest_sync = self.sync_history[-1]
        
        return {
            "status": "active" if latest_sync["success_rate"] > 0.8 else "needs_attention",
            "last_sync": latest_sync["timestamp"],
            "success_rate": latest_sync["success_rate"],
            "avg_validation_score": latest_sync["avg_validation_score"],
            "total_syncs_performed": len(self.sync_history)
        }
    
    async def close(self):
        """Close all API client sessions"""
        for client in self.clients.values():
            await client.close()

# Test function
async def test_connectome_synchronizer():
    """Test the Connectome Synchronizer"""
    print("🧠 Testing Connectome Synchronizer...")
    
    synchronizer = ConnectomeSynchronizer()
    
    try:
        # Test individual sync operations
        print("\n🔬 Testing Neuroscience Resources...")
        neuro_results = await synchronizer.sync_neuroscience_resources()
        print(f"✅ Neuroscience sync: {len(neuro_results)} results")
        
        print("\n🤖 Testing ML Resources...")
        ml_results = await synchronizer.sync_ml_resources()
        print(f"✅ ML sync: {len(ml_results)} results")
        
        print("\n📚 Testing Knowledge Resources...")
        knowledge_results = await synchronizer.sync_knowledge_resources()
        print(f"✅ Knowledge sync: {len(knowledge_results)} results")
        
        # Test full sync
        print("\n🚀 Testing Full Sync...")
        full_sync_result = await synchronizer.full_sync()
        
        print(f"\n📊 Full Sync Results:")
        print(f"Success Rate: {full_sync_result['success_rate']:.2%}")
        print(f"Avg Validation Score: {full_sync_result['avg_validation_score']:.2f}")
        print(f"Total Data Count: {full_sync_result['total_data_count']}")
        
        # Get sync status
        status = synchronizer.get_sync_status()
        print(f"\n📈 Sync Status: {status['status']}")
        
        print("\n✅ Connectome Synchronizer test completed!")
        
    except Exception as e:
        print(f"❌ Error testing Connectome Synchronizer: {str(e)}")
    
    finally:
        await synchronizer.close()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_connectome_synchronizer())
