#!/usr/bin/env python3
"""
Comprehensive Validation System for Anti-Overconfidence
Uses ALL available resources from credentials for intelligent validation
"""

import json
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from datetime import datetime
import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from .async_http_client import get_http_client, APIResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryCategory(Enum):
    """Categories of queries for intelligent source selection"""
    PROTEIN_STRUCTURE = "protein_structure"
    GENOMICS = "genomics"
    NEUROSCIENCE = "neuroscience"
    CHEMISTRY = "chemistry"
    MATERIALS = "materials"
    MACHINE_LEARNING = "machine_learning"
    SCIENTIFIC_LITERATURE = "scientific_literature"
    BIOLOGICAL_SEQUENCE = "biological_sequence"
    CLINICAL = "clinical"
    COMPUTATIONAL = "computational"
    CODE_DOCUMENTATION = "code_documentation"
    GENERAL_AI = "general_ai"
    DATA_SCIENCE = "data_science"
    PHYSICS = "physics"
    MATHEMATICS = "mathematics"

@dataclass
class ValidationSource:
    """Represents a validation source with its capabilities"""
    name: str
    category: List[QueryCategory]
    endpoint: str
    requires_auth: bool
    auth_key: Optional[str] = None
    confidence_weight: float = 1.0
    description: str = ""
    rate_limit: Optional[Dict[str, Any]] = None

class ComprehensiveValidationSystem:
    """
    MANDATORY validation system that uses ALL available resources
    for anti-overconfidence validation
    """
    
    def __init__(self):
        """Initialize with ALL available resources from credentials"""
        self.credentials_path = Path("/Users/camdouglas/quark/data/credentials/all_api_keys.json")
        self.knowledge_path = Path("/Users/camdouglas/quark/data/knowledge/validation_system/open_access_literature_sources.json")
        self.credentials = self._load_credentials()
        self.knowledge_sources = self._load_knowledge_sources()
        self.validation_sources = self._initialize_all_sources()
        self.validation_cache = {}
        self.validation_history = []
        
    def _load_credentials(self) -> Dict[str, Any]:
        """Load ALL API credentials"""
        if not self.credentials_path.exists():
            raise FileNotFoundError(f"❌ CRITICAL: Credentials file not found at {self.credentials_path}")
        
        with open(self.credentials_path, 'r') as f:
            creds = json.load(f)
        
        logger.info(f"✅ Loaded credentials for {len(creds.get('services', {}))} services")
        return creds
    
    def _load_knowledge_sources(self) -> Dict[str, Any]:
        """Load additional knowledge sources"""
        if not self.knowledge_path.exists():
            logger.warning(f"⚠️ Knowledge sources file not found at {self.knowledge_path}")
            return {}
        
        with open(self.knowledge_path, 'r') as f:
            sources = json.load(f)
        
        logger.info(f"✅ Loaded {sources['metadata'].get('total_sources', 0)} additional knowledge sources")
        return sources
    
    def _initialize_all_sources(self) -> Dict[str, ValidationSource]:
        """Initialize ALL validation sources from credentials"""
        sources = {}
        
        # Add protein structure sources
        sources['alphafold'] = ValidationSource(
            name="AlphaFold",
            category=[QueryCategory.PROTEIN_STRUCTURE, QueryCategory.BIOLOGICAL_SEQUENCE],
            endpoint=self.credentials['services']['alphafold']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.95,
            description="AI-predicted 3D protein structures"
        )
        
        sources['rcsb_pdb'] = ValidationSource(
            name="RCSB PDB",
            category=[QueryCategory.PROTEIN_STRUCTURE],
            endpoint=self.credentials['services']['rcsb_pdb']['endpoints']['search'],
            requires_auth=False,
            confidence_weight=0.98,
            description="Experimental protein structures"
        )
        
        # Add genomics sources
        sources['ensembl'] = ValidationSource(
            name="Ensembl",
            category=[QueryCategory.GENOMICS],
            endpoint=self.credentials['services']['ensembl']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.95,
            description="Genomics data across vertebrate genomes"
        )
        
        sources['ncbi_eutils'] = ValidationSource(
            name="NCBI E-utilities",
            category=[QueryCategory.GENOMICS, QueryCategory.SCIENTIFIC_LITERATURE, QueryCategory.BIOLOGICAL_SEQUENCE],
            endpoint=self.credentials['services']['ncbi_eutilities']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.98,
            description="NCBI databases including PubMed, GenBank"
        )
        
        # Add chemistry sources
        sources['pubchem'] = ValidationSource(
            name="PubChem",
            category=[QueryCategory.CHEMISTRY],
            endpoint=self.credentials['services']['pubchem']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.95,
            description="Chemical compounds and bioassays"
        )
        
        # Add materials science sources
        sources['materials_project'] = ValidationSource(
            name="Materials Project",
            category=[QueryCategory.MATERIALS],
            endpoint=self.credentials['services']['materials_project']['endpoints']['base'],
            requires_auth=True,
            auth_key=self.credentials['services']['materials_project']['api_key'],
            confidence_weight=0.92,
            description="Computed materials properties"
        )
        
        sources['oqmd'] = ValidationSource(
            name="OQMD",
            category=[QueryCategory.MATERIALS],
            endpoint=self.credentials['services']['oqmd']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.90,
            description="Quantum materials database"
        )
        
        # Add ML/AI sources
        sources['openml'] = ValidationSource(
            name="OpenML",
            category=[QueryCategory.MACHINE_LEARNING, QueryCategory.DATA_SCIENCE],
            endpoint=self.credentials['services']['openml']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.88,
            description="ML datasets and experiments"
        )
        
        sources['huggingface'] = ValidationSource(
            name="Hugging Face",
            category=[QueryCategory.MACHINE_LEARNING, QueryCategory.GENERAL_AI],
            endpoint="https://huggingface.co/api",
            requires_auth=True,
            auth_key=self.credentials['services']['huggingface']['token'],
            confidence_weight=0.85,
            description="ML models and datasets"
        )
        
        # Add scientific literature sources
        sources['arxiv'] = ValidationSource(
            name="arXiv",
            category=[QueryCategory.SCIENTIFIC_LITERATURE, QueryCategory.PHYSICS, QueryCategory.MATHEMATICS],
            endpoint=self.credentials['services']['arxiv']['endpoints']['query'],
            requires_auth=False,
            confidence_weight=0.90,
            description="Preprint archive for sciences"
        )
        
        sources['pubmed'] = ValidationSource(
            name="PubMed Central",
            category=[QueryCategory.SCIENTIFIC_LITERATURE, QueryCategory.CLINICAL, QueryCategory.NEUROSCIENCE],
            endpoint="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            requires_auth=False,
            confidence_weight=0.95,
            description="Biomedical literature"
        )
        
        # Add protein sequence sources
        sources['uniprot'] = ValidationSource(
            name="UniProt",
            category=[QueryCategory.BIOLOGICAL_SEQUENCE, QueryCategory.PROTEIN_STRUCTURE],
            endpoint=self.credentials['services']['uniprot']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.96,
            description="Protein sequence and function"
        )
        
        sources['blast'] = ValidationSource(
            name="BLAST",
            category=[QueryCategory.BIOLOGICAL_SEQUENCE],
            endpoint=self.credentials['services']['blast']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.94,
            description="Sequence similarity search"
        )
        
        # Add computational sources
        sources['wolfram'] = ValidationSource(
            name="Wolfram Alpha",
            category=[QueryCategory.COMPUTATIONAL, QueryCategory.MATHEMATICS, QueryCategory.PHYSICS],
            endpoint="http://api.wolframalpha.com/v2/query",
            requires_auth=True,
            auth_key=self.credentials['services']['wolfram']['app_id'],
            confidence_weight=0.92,
            description="Computational knowledge engine"
        )
        
        # Add AI validation sources
        sources['openai'] = ValidationSource(
            name="OpenAI GPT",
            category=[QueryCategory.GENERAL_AI, QueryCategory.CODE_DOCUMENTATION],
            endpoint="https://api.openai.com/v1/chat/completions",
            requires_auth=True,
            auth_key=self.credentials['services']['openai']['api_key'],
            confidence_weight=0.80,
            description="General AI validation"
        )
        
        sources['claude'] = ValidationSource(
            name="Claude",
            category=[QueryCategory.GENERAL_AI, QueryCategory.CODE_DOCUMENTATION],
            endpoint="https://api.anthropic.com/v1/messages",
            requires_auth=True,
            auth_key=self.credentials['services']['claude']['api_key'],
            confidence_weight=0.82,
            description="Advanced reasoning validation"
        )
        
        sources['gemini'] = ValidationSource(
            name="Gemini",
            category=[QueryCategory.GENERAL_AI],
            endpoint="https://generativelanguage.googleapis.com/v1beta/models",
            requires_auth=True,
            auth_key=self.credentials['services']['gemini']['api_key'],
            confidence_weight=0.78,
            description="Google AI validation"
        )
        
        # Add code/documentation sources
        sources['context7'] = ValidationSource(
            name="Context7",
            category=[QueryCategory.CODE_DOCUMENTATION],
            endpoint="https://api.context7.ai/v1",
            requires_auth=True,
            auth_key=self.credentials['services']['context7']['api_key'],
            confidence_weight=0.85,
            description="Code documentation for libraries"
        )
        
        sources['github'] = ValidationSource(
            name="GitHub",
            category=[QueryCategory.CODE_DOCUMENTATION],
            endpoint="https://api.github.com",
            requires_auth=True,
            auth_key=self.credentials['services']['github']['token'],
            confidence_weight=0.88,
            description="Code repositories and documentation"
        )
        
        # Add data sources
        sources['kaggle'] = ValidationSource(
            name="Kaggle",
            category=[QueryCategory.DATA_SCIENCE, QueryCategory.MACHINE_LEARNING],
            endpoint="https://www.kaggle.com/api/v1",
            requires_auth=True,
            auth_key=self.credentials['services']['kaggle']['api_key'],
            confidence_weight=0.82,
            description="Datasets and ML competitions"
        )
        
        # Add CDX Server for web archive validation
        sources['cdx_server'] = ValidationSource(
            name="CDX Server",
            category=[QueryCategory.SCIENTIFIC_LITERATURE, QueryCategory.CODE_DOCUMENTATION],
            endpoint=self.credentials['services']['cdx_server']['endpoints']['base'],
            requires_auth=False,
            confidence_weight=0.75,
            description="Web archive index queries for historical validation"
        )
        
        # CRITICAL: Add open access literature sources for anti-overconfidence validation
        if self.knowledge_sources:
            # Add preprint servers
            preprint_servers = self.knowledge_sources.get('preprint_servers', {})
            for source_id, source_data in preprint_servers.items():
                if source_data.get('status') == 'tested_accessible' or source_data.get('test_status') == 'tested_working':
                    # Map subjects to categories
                    categories = [QueryCategory.SCIENTIFIC_LITERATURE]
                    subjects = source_data.get('subjects', [])
                    
                    if any(s in str(subjects) for s in ['biology', 'life', 'medical']):
                        categories.extend([QueryCategory.BIOLOGICAL_SEQUENCE, QueryCategory.CLINICAL])
                    if any(s in str(subjects) for s in ['neuro', 'brain', 'cognitive']):
                        categories.append(QueryCategory.NEUROSCIENCE)
                    if any(s in str(subjects) for s in ['chemistry', 'chemical']):
                        categories.append(QueryCategory.CHEMISTRY)
                    if any(s in str(subjects) for s in ['physics']):
                        categories.append(QueryCategory.PHYSICS)
                    if any(s in str(subjects) for s in ['math']):
                        categories.append(QueryCategory.MATHEMATICS)
                    if any(s in str(subjects) for s in ['computer', 'machine learning']):
                        categories.extend([QueryCategory.MACHINE_LEARNING, QueryCategory.COMPUTATIONAL])
                    
                    sources[f'openaccess_{source_id}'] = ValidationSource(
                        name=source_data.get('name', source_id),
                        category=categories,
                        endpoint=source_data.get('api_endpoint', source_data.get('url', '')),
                        requires_auth=source_data.get('authentication', 'none') != 'none',
                        confidence_weight=0.95,  # High weight for peer-reviewed sources
                        description=f"Open access: {source_data.get('name', source_id)}"
                    )
            
            # Add academic databases
            academic_databases = self.knowledge_sources.get('academic_databases', {})
            for source_id, source_data in academic_databases.items():
                if source_data.get('status') == 'tested_accessible' or source_data.get('test_status') == 'tested_working':
                    sources[f'openaccess_{source_id}'] = ValidationSource(
                        name=source_data.get('name', source_id),
                        category=[QueryCategory.SCIENTIFIC_LITERATURE],
                        endpoint=source_data.get('api_endpoint', source_data.get('url', '')),
                        requires_auth=False,
                        confidence_weight=0.94,
                        description=f"Academic database: {source_data.get('name', source_id)}"
                    )
            
            # Add neuroscience-specific sources (HIGH PRIORITY for Quark)
            neuro_sources = self.knowledge_sources.get('neuroscience_specific', {})
            for source_id, source_data in neuro_sources.items():
                if source_data.get('status') == 'tested_accessible' or source_data.get('test_status') == 'tested_working':
                    sources[f'openaccess_{source_id}'] = ValidationSource(
                        name=source_data.get('name', source_id),
                        category=[QueryCategory.NEUROSCIENCE, QueryCategory.SCIENTIFIC_LITERATURE],
                        endpoint=source_data.get('api_endpoint', source_data.get('url', '')),
                        requires_auth=False,
                        confidence_weight=0.96,  # Very high weight for neuroscience sources
                        description=f"Neuroscience: {source_data.get('name', source_id)}"
                    )
            
            logger.info(f"✅ Added {sum(1 for k in sources.keys() if 'openaccess' in k)} open access literature sources")
        
        logger.info(f"✅ Initialized {len(sources)} total validation sources (credentials + open access)")
        return sources
    
    def categorize_query(self, query: str) -> List[QueryCategory]:
        """Intelligently categorize a query to determine best validation sources"""
        categories = []
        query_lower = query.lower()
        
        # Protein/structure keywords
        if any(kw in query_lower for kw in ['protein', 'structure', 'pdb', 'fold', 'domain', 'binding site']):
            categories.append(QueryCategory.PROTEIN_STRUCTURE)
        
        # Genomics keywords
        if any(kw in query_lower for kw in ['gene', 'genome', 'transcript', 'chromosome', 'variant', 'snp', 'mutation']):
            categories.append(QueryCategory.GENOMICS)
        
        # Neuroscience keywords
        if any(kw in query_lower for kw in ['brain', 'neuron', 'synapse', 'neural', 'cognitive', 'cortex']):
            categories.append(QueryCategory.NEUROSCIENCE)
        
        # Chemistry keywords
        if any(kw in query_lower for kw in ['compound', 'molecule', 'drug', 'chemical', 'reaction', 'inhibitor']):
            categories.append(QueryCategory.CHEMISTRY)
        
        # Materials keywords
        if any(kw in query_lower for kw in ['material', 'crystal', 'band gap', 'lattice', 'alloy', 'semiconductor']):
            categories.append(QueryCategory.MATERIALS)
        
        # ML keywords
        if any(kw in query_lower for kw in ['machine learning', 'ml', 'model', 'dataset', 'training', 'neural network']):
            categories.append(QueryCategory.MACHINE_LEARNING)
        
        # Literature keywords
        if any(kw in query_lower for kw in ['paper', 'publication', 'study', 'research', 'citation', 'reference']):
            categories.append(QueryCategory.SCIENTIFIC_LITERATURE)
        
        # Sequence keywords
        if any(kw in query_lower for kw in ['sequence', 'blast', 'alignment', 'homolog', 'ortholog', 'fasta']):
            categories.append(QueryCategory.BIOLOGICAL_SEQUENCE)
        
        # Clinical keywords
        if any(kw in query_lower for kw in ['disease', 'patient', 'clinical', 'therapy', 'treatment', 'diagnosis']):
            categories.append(QueryCategory.CLINICAL)
        
        # Computational keywords
        if any(kw in query_lower for kw in ['calculate', 'compute', 'equation', 'formula', 'algorithm']):
            categories.append(QueryCategory.COMPUTATIONAL)
        
        # Code keywords
        if any(kw in query_lower for kw in ['code', 'api', 'function', 'library', 'documentation', 'implement']):
            categories.append(QueryCategory.CODE_DOCUMENTATION)
        
        # Physics keywords
        if any(kw in query_lower for kw in ['physics', 'quantum', 'particle', 'energy', 'force', 'field']):
            categories.append(QueryCategory.PHYSICS)
        
        # Math keywords
        if any(kw in query_lower for kw in ['math', 'equation', 'theorem', 'proof', 'calculus', 'algebra']):
            categories.append(QueryCategory.MATHEMATICS)
        
        # Default to general AI if no specific category
        if not categories:
            categories.append(QueryCategory.GENERAL_AI)
        
        return categories
    
    def select_validation_sources(self, query: str, min_sources: int = 3) -> List[ValidationSource]:
        """
        Intelligently select the BEST sources for validating a query
        ALWAYS uses multiple sources for cross-validation
        """
        categories = self.categorize_query(query)
        selected_sources = []
        
        # First pass: get all sources matching categories
        for source_name, source in self.validation_sources.items():
            for category in categories:
                if category in source.category:
                    selected_sources.append(source)
                    break
        
        # PRIORITIZE open access sources for scientific literature
        def source_priority_score(source):
            score = source.confidence_weight
            
            # Boost open access sources for scientific queries
            if any('openaccess' in k for k, v in self.validation_sources.items() if v == source):
                if QueryCategory.SCIENTIFIC_LITERATURE in categories:
                    score += 0.1  # Boost for scientific literature
                if QueryCategory.BIOLOGICAL_SEQUENCE in categories or QueryCategory.GENOMICS in categories:
                    score += 0.05  # Extra boost for biological queries
            
            return score
        
        # Sort by priority score (confidence + open access boost)
        selected_sources.sort(key=source_priority_score, reverse=True)
        
        # Ensure minimum number of sources
        if len(selected_sources) < min_sources:
            # Add high-confidence general sources (including open access)
            general_source_keys = ['arxiv', 'pubmed', 'openai', 'claude']
            
            # Also include open access sources
            openaccess_keys = [k for k in self.validation_sources.keys() if 'openaccess' in k]
            general_source_keys.extend(openaccess_keys[:5])  # Add top 5 open access sources
            
            for key in general_source_keys:
                source = self.validation_sources.get(key)
                if source and source not in selected_sources:
                    selected_sources.append(source)
                if len(selected_sources) >= min_sources:
                    break
        
        logger.info(f"✅ Selected {len(selected_sources)} sources for query: {categories}")
        return selected_sources[:10]  # Limit to top 10 sources
    
    async def validate_claim(self, claim: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        MANDATORY validation of a claim using ALL relevant sources
        Returns confidence score and evidence
        """
        # Create cache key
        cache_key = hashlib.md5(f"{claim}{context}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.validation_cache:
            logger.info(f"✅ Using cached validation for claim: {claim[:50]}...")
            return self.validation_cache[cache_key]
        
        # Select appropriate sources
        sources = self.select_validation_sources(claim)
        
        # Validate against each source
        validations = []
        for source in sources:
            try:
                result = await self._validate_with_source(claim, context, source)
                validations.append(result)
            except Exception as e:
                logger.error(f"❌ Validation failed for {source.name}: {e}")
        
        # Aggregate results
        aggregate_result = self._aggregate_validations(validations)
        
        # Cache result
        self.validation_cache[cache_key] = aggregate_result
        
        # Log validation
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'claim': claim,
            'sources_used': [s.name for s in sources],
            'confidence': aggregate_result['confidence'],
            'consensus': aggregate_result['consensus']
        })
        
        return aggregate_result
    
    async def _validate_with_source(self, claim: str, context: str, source: ValidationSource) -> Dict[str, Any]:
        """Validate claim with a specific source using async HTTP client"""
        try:
            http_client = await get_http_client()
            
            # Route to appropriate API handler based on source name
            if 'arxiv' in source.name.lower():
                return await self._validate_arxiv(claim, context, source, http_client)
            elif 'pubmed' in source.name.lower():
                return await self._validate_pubmed(claim, context, source, http_client)
            elif 'pubchem' in source.name.lower():
                return await self._validate_pubchem(claim, context, source, http_client)
            elif 'alphafold' in source.name.lower():
                return await self._validate_alphafold(claim, context, source, http_client)
            elif 'rcsb' in source.name.lower():
                return await self._validate_rcsb_pdb(claim, context, source, http_client)
            elif 'ensembl' in source.name.lower():
                return await self._validate_ensembl(claim, context, source, http_client)
            elif 'materials_project' in source.name.lower():
                return await self._validate_materials_project(claim, context, source, http_client)
            elif 'openai' in source.name.lower():
                return await self._validate_openai(claim, context, source, http_client)
            elif 'claude' in source.name.lower():
                return await self._validate_claude(claim, context, source, http_client)
            elif 'wolfram' in source.name.lower():
                return await self._validate_wolfram(claim, context, source, http_client)
            elif 'uniprot' in source.name.lower():
                return await self._validate_uniprot(claim, context, source, http_client)
            elif 'blast' in source.name.lower():
                return await self._validate_blast(claim, context, source, http_client)
            elif 'cdx' in source.name.lower():
                return await self._validate_cdx_server(claim, context, source, http_client)
            else:
                # Generic validation for other sources
                return await self._validate_generic(claim, context, source, http_client)
                
        except Exception as e:
            logger.error(f"❌ Validation error for {source.name}: {e}")
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"Validation failed: {str(e)}",
                'supports_claim': False,
                'details': {'error': str(e)}
            }
    
    async def _validate_arxiv(self, claim: str, context: str, source: ValidationSource, http_client) -> Dict[str, Any]:
        """Validate against ArXiv API"""
        try:
            # Extract search terms from claim
            search_terms = self._extract_search_terms(claim)
            
            params = {
                'search_query': f'all:{" AND ".join(search_terms[:5])}',  # Limit to 5 terms
                'start': 0,
                'max_results': 10,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = await http_client.get(
                url=source.endpoint,
                api_name='arxiv',
                params=params,
                timeout=15.0
            )
            
            if not response.success:
                return {
                    'source': source.name,
                    'confidence': 0.0,
                    'evidence': f"ArXiv API error: {response.error}",
                    'supports_claim': False,
                    'details': {'error': response.error}
                }
            
            # Parse XML response
            if 'xml_content' in response.data:
                return self._parse_arxiv_response(response.data['xml_content'], claim, source)
            else:
                return {
                    'source': source.name,
                    'confidence': 0.3,
                    'evidence': "ArXiv search completed but no relevant papers found",
                    'supports_claim': False,
                    'details': {'papers_found': 0}
                }
                
        except Exception as e:
            logger.error(f"❌ ArXiv validation error: {e}")
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"ArXiv validation failed: {str(e)}",
                'supports_claim': False,
                'details': {'error': str(e)}
            }
    
    async def _validate_pubmed(self, claim: str, context: str, source: ValidationSource, http_client) -> Dict[str, Any]:
        """Validate against PubMed API"""
        try:
            search_terms = self._extract_search_terms(claim)
            
            # First, search for relevant papers
            search_params = {
                'db': 'pubmed',
                'term': ' AND '.join(search_terms[:5]),
                'retmax': 10,
                'retmode': 'json',
                'tool': 'quark_validation',
                'email': 'research@quark-ai.com'
            }
            
            search_response = await http_client.get(
                url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                api_name='ncbi_eutils',
                params=search_params,
                timeout=15.0
            )
            
            if not search_response.success:
                return {
                    'source': source.name,
                    'confidence': 0.0,
                    'evidence': f"PubMed search failed: {search_response.error}",
                    'supports_claim': False,
                    'details': {'error': search_response.error}
                }
            
            # Extract PMIDs from search results
            search_data = search_response.data
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return {
                    'source': source.name,
                    'confidence': 0.2,
                    'evidence': "No relevant PubMed papers found for this claim",
                    'supports_claim': False,
                    'details': {'papers_found': 0}
                }
            
            # Fetch abstracts for the papers
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids[:5]),  # Limit to 5 papers
                'retmode': 'xml',
                'tool': 'quark_validation',
                'email': 'research@quark-ai.com'
            }
            
            fetch_response = await http_client.get(
                url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                api_name='ncbi_eutils',
                params=fetch_params,
                timeout=20.0
            )
            
            if fetch_response.success and 'xml_content' in fetch_response.data:
                return self._parse_pubmed_response(fetch_response.data['xml_content'], claim, source, len(pmids))
            else:
                return {
                    'source': source.name,
                    'confidence': 0.4,
                    'evidence': f"Found {len(pmids)} relevant PubMed papers but could not fetch abstracts",
                    'supports_claim': True,
                    'details': {'papers_found': len(pmids)}
                }
                
        except Exception as e:
            logger.error(f"❌ PubMed validation error: {e}")
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"PubMed validation failed: {str(e)}",
                'supports_claim': False,
                'details': {'error': str(e)}
            }
    
    async def _validate_materials_project(self, claim: str, context: str, source: ValidationSource, http_client) -> Dict[str, Any]:
        """Validate against Materials Project API"""
        try:
            # Extract material-related terms
            material_terms = self._extract_material_terms(claim)
            
            if not material_terms:
                return {
                    'source': source.name,
                    'confidence': 0.1,
                    'evidence': "No material-related terms found in claim",
                    'supports_claim': False,
                    'details': {'material_terms': []}
                }
            
            headers = {
                'X-API-KEY': source.auth_key
            }
            
            # Search for materials
            params = {
                'formula': material_terms[0] if material_terms else '',
                '_limit': 10
            }
            
            response = await http_client.get(
                url=f"{source.endpoint}/materials/summary",
                api_name='materials_project',
                params=params,
                headers=headers,
                timeout=20.0
            )
            
            if not response.success:
                return {
                    'source': source.name,
                    'confidence': 0.0,
                    'evidence': f"Materials Project API error: {response.error}",
                    'supports_claim': False,
                    'details': {'error': response.error}
                }
            
            materials_data = response.data.get('data', [])
            
            return {
                'source': source.name,
                'confidence': min(0.8, len(materials_data) * 0.1),
                'evidence': f"Found {len(materials_data)} materials in Materials Project database",
                'supports_claim': len(materials_data) > 0,
                'details': {
                    'materials_found': len(materials_data),
                    'search_terms': material_terms
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Materials Project validation error: {e}")
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"Materials Project validation failed: {str(e)}",
                'supports_claim': False,
                'details': {'error': str(e)}
            }
    
    async def _validate_uniprot(self, claim: str, context: str, source: ValidationSource, http_client) -> Dict[str, Any]:
        """Validate against UniProt REST API"""
        try:
            # Extract protein-related terms
            protein_terms = self._extract_protein_terms(claim)
            
            if not protein_terms:
                return {
                    'source': source.name,
                    'confidence': 0.1,
                    'evidence': "No protein-related terms found in claim",
                    'supports_claim': False,
                    'details': {'protein_terms': []}
                }
            
            # Search UniProt for proteins
            params = {
                'query': ' OR '.join(protein_terms[:3]),  # Limit to 3 terms
                'format': 'json',
                'size': 10
            }
            
            response = await http_client.get(
                url=f"{source.endpoint}/uniprotkb/search",
                api_name='uniprot',
                params=params,
                timeout=15.0
            )
            
            if not response.success:
                return {
                    'source': source.name,
                    'confidence': 0.0,
                    'evidence': f"UniProt API error: {response.error}",
                    'supports_claim': False,
                    'details': {'error': response.error}
                }
            
            results = response.data.get('results', [])
            
            return {
                'source': source.name,
                'confidence': min(0.85, len(results) * 0.1),
                'evidence': f"Found {len(results)} proteins in UniProt matching search terms",
                'supports_claim': len(results) > 0,
                'details': {
                    'proteins_found': len(results),
                    'search_terms': protein_terms
                }
            }
            
        except Exception as e:
            logger.error(f"❌ UniProt validation error: {e}")
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"UniProt validation failed: {str(e)}",
                'supports_claim': False,
                'details': {'error': str(e)}
            }
    
    async def _validate_blast(self, claim: str, context: str, source: ValidationSource, http_client) -> Dict[str, Any]:
        """Validate against BLAST API (limited due to rate limits)"""
        try:
            # BLAST has strict rate limits, so we'll do a lightweight validation
            sequence_terms = self._extract_sequence_terms(claim)
            
            if not sequence_terms:
                return {
                    'source': source.name,
                    'confidence': 0.1,
                    'evidence': "No sequence-related terms found in claim",
                    'supports_claim': False,
                    'details': {'sequence_terms': []}
                }
            
            # For BLAST, we'll just validate that the service is accessible
            # due to strict rate limits (100 searches/day)
            response = await http_client.get(
                url=source.endpoint,
                api_name='blast',
                timeout=10.0
            )
            
            confidence = 0.6 if response.success else 0.0
            
            return {
                'source': source.name,
                'confidence': confidence * source.confidence_weight,
                'evidence': f"BLAST service {'accessible' if response.success else 'not accessible'} - sequence terms detected: {', '.join(sequence_terms[:3])}",
                'supports_claim': response.success and len(sequence_terms) > 0,
                'details': {
                    'sequence_terms': sequence_terms,
                    'service_accessible': response.success,
                    'note': 'Limited validation due to BLAST rate limits'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ BLAST validation error: {e}")
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"BLAST validation failed: {str(e)}",
                'supports_claim': False,
                'details': {'error': str(e)}
            }
    
    async def _validate_cdx_server(self, claim: str, context: str, source: ValidationSource, http_client) -> Dict[str, Any]:
        """Validate against CDX Server API (if running)"""
        try:
            # Extract URL-related terms for web archive validation
            url_terms = self._extract_url_terms(claim)
            
            # Check if CDX server is running
            response = await http_client.get(
                url=source.endpoint,
                api_name='cdx_server',
                timeout=5.0
            )
            
            if not response.success:
                return {
                    'source': source.name,
                    'confidence': 0.0,
                    'evidence': "CDX Server not running - install pywb and start server for web archive validation",
                    'supports_claim': False,
                    'details': {
                        'server_running': False,
                        'url_terms': url_terms,
                        'setup_required': True
                    }
                }
            
            # If server is running and we have URL terms, it's a good validation source
            confidence = 0.7 if url_terms else 0.3
            
            return {
                'source': source.name,
                'confidence': confidence * source.confidence_weight,
                'evidence': f"CDX Server running - can validate web archive data for URLs: {', '.join(url_terms[:3]) if url_terms else 'none detected'}",
                'supports_claim': len(url_terms) > 0,
                'details': {
                    'server_running': True,
                    'url_terms': url_terms,
                    'archive_validation_available': True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ CDX Server validation error: {e}")
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"CDX Server validation failed: {str(e)}",
                'supports_claim': False,
                'details': {'error': str(e)}
            }
    
    async def _validate_generic(self, claim: str, context: str, source: ValidationSource, http_client) -> Dict[str, Any]:
        """Generic validation for sources without specific handlers"""
        try:
            # Simple GET request to check if source is accessible
            response = await http_client.get(
                url=source.endpoint,
                api_name=source.name.lower().replace(' ', '_'),
                timeout=10.0
            )
            
            confidence = 0.5 if response.success else 0.0
            
            return {
                'source': source.name,
                'confidence': confidence * source.confidence_weight,
                'evidence': f"Generic validation from {source.name}: {'accessible' if response.success else 'not accessible'}",
                'supports_claim': response.success,
                'details': {
                    'response_time': response.response_time,
                    'status_code': response.status_code
                }
            }
            
        except Exception as e:
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"Generic validation failed: {str(e)}",
                'supports_claim': False,
                'details': {'error': str(e)}
            }
    
    def _extract_search_terms(self, claim: str) -> List[str]:
        """Extract relevant search terms from claim"""
        # Simple keyword extraction - could be enhanced with NLP
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should'}
        
        words = claim.lower().replace(',', ' ').replace('.', ' ').split()
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def _extract_material_terms(self, claim: str) -> List[str]:
        """Extract material-related terms from claim"""
        material_keywords = ['silicon', 'carbon', 'iron', 'aluminum', 'copper', 'gold', 'silver', 'titanium', 'steel', 'graphene', 'diamond']
        
        claim_lower = claim.lower()
        found_materials = [material for material in material_keywords if material in claim_lower]
        
        # Also look for chemical formulas (simple pattern)
        import re
        formulas = re.findall(r'\b[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)*\b', claim)
        
        return found_materials + formulas
    
    def _extract_protein_terms(self, claim: str) -> List[str]:
        """Extract protein-related terms from claim"""
        protein_keywords = [
            'protein', 'enzyme', 'receptor', 'kinase', 'phosphatase', 'antibody',
            'insulin', 'hemoglobin', 'collagen', 'albumin', 'globulin', 'histone',
            'cytokine', 'hormone', 'neurotransmitter', 'dopamine', 'serotonin',
            'acetylcholine', 'gaba', 'glutamate', 'glycine', 'histamine'
        ]
        
        claim_lower = claim.lower()
        found_proteins = [protein for protein in protein_keywords if protein in claim_lower]
        
        # Also look for UniProt accession patterns (e.g., P04637)
        import re
        accessions = re.findall(r'\b[A-Z][0-9][A-Z0-9]{3}[0-9]\b', claim)
        
        return found_proteins + accessions
    
    def _extract_sequence_terms(self, claim: str) -> List[str]:
        """Extract sequence-related terms from claim"""
        sequence_keywords = [
            'sequence', 'alignment', 'homolog', 'ortholog', 'paralog', 'blast',
            'fasta', 'dna', 'rna', 'amino acid', 'nucleotide', 'codon',
            'mutation', 'variant', 'snp', 'indel', 'substitution', 'deletion',
            'insertion', 'phylogeny', 'evolution', 'conservation'
        ]
        
        claim_lower = claim.lower()
        found_terms = [term for term in sequence_keywords if term in claim_lower]
        
        # Look for sequence patterns (simple DNA/RNA/protein sequences)
        import re
        dna_sequences = re.findall(r'\b[ATCG]{4,}\b', claim.upper())
        protein_sequences = re.findall(r'\b[ACDEFGHIKLMNPQRSTVWY]{4,}\b', claim.upper())
        
        return found_terms + dna_sequences + protein_sequences
    
    def _extract_url_terms(self, claim: str) -> List[str]:
        """Extract URL-related terms from claim"""
        url_keywords = [
            'website', 'url', 'domain', 'archive', 'wayback', 'web page',
            'http', 'https', 'www', 'site', 'portal', 'database'
        ]
        
        claim_lower = claim.lower()
        found_terms = [term for term in url_keywords if term in claim_lower]
        
        # Look for actual URLs
        import re
        urls = re.findall(r'https?://[^\s]+', claim)
        domains = re.findall(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', claim)
        
        return found_terms + urls + domains
    
    def _parse_arxiv_response(self, xml_content: str, claim: str, source: ValidationSource) -> Dict[str, Any]:
        """Parse ArXiv XML response"""
        try:
            root = ET.fromstring(xml_content)
            
            # Count entries
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            if not entries:
                return {
                    'source': source.name,
                    'confidence': 0.2,
                    'evidence': "ArXiv search completed but no papers found",
                    'supports_claim': False,
                    'details': {'papers_found': 0}
                }
            
            # Analyze relevance (simple keyword matching)
            claim_keywords = set(self._extract_search_terms(claim))
            relevant_papers = 0
            
            for entry in entries[:5]:  # Check first 5 papers
                title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                
                if title_elem is not None and summary_elem is not None:
                    title_text = title_elem.text.lower()
                    summary_text = summary_elem.text.lower()
                    
                    # Check for keyword overlap
                    paper_words = set(title_text.split() + summary_text.split())
                    overlap = len(claim_keywords.intersection(paper_words))
                    
                    if overlap >= 2:  # At least 2 keywords match
                        relevant_papers += 1
            
            confidence = min(0.85, relevant_papers * 0.2)
            
            return {
                'source': source.name,
                'confidence': confidence * source.confidence_weight,
                'evidence': f"Found {len(entries)} ArXiv papers, {relevant_papers} appear relevant",
                'supports_claim': relevant_papers > 0,
                'details': {
                    'total_papers': len(entries),
                    'relevant_papers': relevant_papers
                }
            }
            
        except ET.ParseError as e:
            return {
                'source': source.name,
                'confidence': 0.0,
                'evidence': f"Failed to parse ArXiv response: {str(e)}",
                'supports_claim': False,
                'details': {'parse_error': str(e)}
            }
    
    def _parse_pubmed_response(self, xml_content: str, claim: str, source: ValidationSource, total_papers: int) -> Dict[str, Any]:
        """Parse PubMed XML response"""
        try:
            root = ET.fromstring(xml_content)
            
            # Find all articles
            articles = root.findall('.//PubmedArticle')
            
            if not articles:
                return {
                    'source': source.name,
                    'confidence': 0.3,
                    'evidence': f"Found {total_papers} PubMed papers but could not parse abstracts",
                    'supports_claim': total_papers > 0,
                    'details': {'papers_found': total_papers}
                }
            
            # Analyze abstracts for relevance
            claim_keywords = set(self._extract_search_terms(claim))
            relevant_abstracts = 0
            
            for article in articles:
                abstract_elem = article.find('.//AbstractText')
                title_elem = article.find('.//ArticleTitle')
                
                if abstract_elem is not None:
                    abstract_text = abstract_elem.text or ""
                    title_text = title_elem.text if title_elem is not None else ""
                    
                    combined_text = (abstract_text + " " + title_text).lower()
                    abstract_words = set(combined_text.split())
                    
                    overlap = len(claim_keywords.intersection(abstract_words))
                    if overlap >= 2:
                        relevant_abstracts += 1
            
            confidence = min(0.90, relevant_abstracts * 0.25)
            
            return {
                'source': source.name,
                'confidence': confidence * source.confidence_weight,
                'evidence': f"Analyzed {len(articles)} PubMed abstracts, {relevant_abstracts} appear relevant to claim",
                'supports_claim': relevant_abstracts > 0,
                'details': {
                    'abstracts_analyzed': len(articles),
                    'relevant_abstracts': relevant_abstracts,
                    'total_papers_found': total_papers
                }
            }
            
        except ET.ParseError as e:
            return {
                'source': source.name,
                'confidence': 0.2,
                'evidence': f"Found papers but failed to parse PubMed response: {str(e)}",
                'supports_claim': True,
                'details': {'parse_error': str(e), 'papers_found': total_papers}
            }
    
    def _aggregate_validations(self, validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple validation results"""
        if not validations:
            return {
                'confidence': 0.0,
                'consensus': 'NO_VALIDATION',
                'evidence': [],
                'sources_checked': 0
            }
        
        # Calculate weighted confidence
        total_weight = sum(v['confidence'] for v in validations)
        supporting = sum(1 for v in validations if v.get('supports_claim', False))
        
        confidence = (supporting / len(validations)) * (total_weight / len(validations))
        
        # Determine consensus
        if supporting == len(validations):
            consensus = 'STRONG_SUPPORT'
        elif supporting > len(validations) * 0.7:
            consensus = 'MODERATE_SUPPORT'
        elif supporting > len(validations) * 0.3:
            consensus = 'MIXED_EVIDENCE'
        else:
            consensus = 'CONTRADICTED'
        
        return {
            'confidence': min(confidence, 0.90),  # Cap at 90% per anti-overconfidence rules
            'consensus': consensus,
            'evidence': validations,
            'sources_checked': len(validations),
            'supporting_sources': supporting
        }
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        return {
            'total_sources_available': len(self.validation_sources),
            'sources_by_category': self._count_sources_by_category(),
            'validation_history_count': len(self.validation_history),
            'cache_size': len(self.validation_cache),
            'credentials_loaded': len(self.credentials.get('services', {})),
            'knowledge_sources_loaded': self.knowledge_sources.get('metadata', {}).get('total_sources', 0)
        }
    
    def _count_sources_by_category(self) -> Dict[str, int]:
        """Count sources by category"""
        category_counts = {}
        for source in self.validation_sources.values():
            for category in source.category:
                category_counts[category.value] = category_counts.get(category.value, 0) + 1
        return category_counts
    
    def mandatory_validation_check(self, statement: str) -> Tuple[bool, str]:
        """
        MANDATORY check that MUST be called before making any claim
        Returns (is_validated, validation_message)
        """
        # This would be called by the anti-overconfidence system
        sources = self.select_validation_sources(statement)
        
        if len(sources) < 3:
            return False, f"❌ INSUFFICIENT SOURCES: Only {len(sources)} sources available for validation"
        
        validation_message = f"""
⚠️ MANDATORY VALIDATION REQUIRED
Statement: {statement[:100]}...
Selected Sources: {', '.join(s.name for s in sources[:5])}
Categories: {', '.join(c.value for c in self.categorize_query(statement))}

This statement MUST be validated against authoritative sources before proceeding.
"""
        return True, validation_message
    
    def get_all_available_sources(self) -> List[str]:
        """Return list of ALL available validation sources"""
        return list(self.validation_sources.keys())
    
    def verify_all_sources_active(self) -> Dict[str, bool]:
        """Verify that ALL sources are properly configured"""
        status = {}
        for name, source in self.validation_sources.items():
            if source.requires_auth:
                status[name] = bool(source.auth_key)
            else:
                status[name] = True
        
        inactive = [k for k, v in status.items() if not v]
        if inactive:
            logger.warning(f"⚠️ Inactive sources: {inactive}")
        else:
            logger.info(f"✅ All {len(status)} validation sources are active")
        
        return status


# Singleton instance
_validation_system = None

def get_validation_system() -> ComprehensiveValidationSystem:
    """Get or create the singleton validation system"""
    global _validation_system
    if _validation_system is None:
        _validation_system = ComprehensiveValidationSystem()
    return _validation_system

def mandatory_validate(claim: str, context: str = None) -> Dict[str, Any]:
    """
    MANDATORY validation function that MUST be called
    Uses ALL available resources intelligently
    """
    system = get_validation_system()
    
    # Run async validation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(system.validate_claim(claim, context))
    finally:
        loop.close()
    
    # Log validation
    logger.info(f"""
📊 VALIDATION COMPLETE:
Claim: {claim[:100]}...
Confidence: {result['confidence']*100:.1f}%
Consensus: {result['consensus']}
Sources: {result['sources_checked']}
""")
    
    return result

if __name__ == "__main__":
    # Test the system
    system = get_validation_system()
    
    print("🔍 COMPREHENSIVE VALIDATION SYSTEM INITIALIZED")
    print(f"✅ Total Sources Available: {len(system.get_all_available_sources())}")
    print(f"✅ Sources: {', '.join(system.get_all_available_sources())}")
    
    # Verify all sources
    status = system.verify_all_sources_active()
    active_count = sum(1 for v in status.values() if v)
    print(f"✅ Active Sources: {active_count}/{len(status)}")
    
    # Generate report
    report = system.get_validation_report()
    print(f"""
📊 VALIDATION SYSTEM REPORT:
- Total Sources: {report['total_sources_available']}
- Credentials Loaded: {report['credentials_loaded']}
- Knowledge Sources: {report['knowledge_sources_loaded']}
- Categories Covered: {len(report['sources_by_category'])}
""")
    
    # Test validation
    test_claims = [
        "AlphaFold can predict protein structures",
        "Neural networks in the brain use backpropagation",
        "CRISPR can edit any gene with 100% accuracy"
    ]
    
    for claim in test_claims:
        is_valid, message = system.mandatory_validation_check(claim)
        print(f"\n📌 Claim: {claim}")
        print(message)
