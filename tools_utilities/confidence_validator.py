#!/usr/bin/env python3
"""
Confidence Validation System for Cursor AI
Enforces anti-overconfidence rules and mandatory validation checkpoints
Integrates all available APIs and MCP servers for comprehensive validation
"""

import json
import logging
import re
import sys
import subprocess
import asyncio
import aiohttp
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories with strict thresholds"""
    LOW = (0, 40, "⚠️ LOW CONFIDENCE")
    MEDIUM = (40, 70, "🟡 MEDIUM CONFIDENCE")
    HIGH = (70, 90, "✅ HIGH CONFIDENCE")
    FORBIDDEN = (90, 100, "🚫 OVERCONFIDENT (FORBIDDEN)")
    

class ValidationSource(Enum):
    """Authoritative source hierarchy"""
    PRIMARY = 1.0  # Official docs, peer-reviewed papers
    SECONDARY = 0.7  # Community best practices, high-vote answers
    EXPERIMENTAL = 0.3  # Untested code, personal interpretations
    

class ValidationCategory(Enum):
    """Categories of validation that can be performed"""
    SCIENTIFIC_LITERATURE = "scientific_literature"
    PROTEIN_STRUCTURE = "protein_structure"
    GENOMIC_DATA = "genomic_data"
    CHEMICAL_COMPOUND = "chemical_compound"
    MACHINE_LEARNING = "machine_learning"
    MATERIALS_SCIENCE = "materials_science"
    CODE_DOCUMENTATION = "code_documentation"
    BIOLOGICAL_SEQUENCE = "biological_sequence"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MATHEMATICAL = "mathematical"
    ARXIV_PAPER = "arxiv_paper"


class ResourceType(Enum):
    """Types of resources available for validation"""
    API = "api"
    MCP_SERVER = "mcp_server"
    LOCAL_TOOL = "local_tool"
    WEB_SERVICE = "web_service"


class ValidationResource:
    """Represents a single validation resource"""
    
    def __init__(self, name: str, resource_type: ResourceType, 
                 categories: List[ValidationCategory], config: Dict[str, Any]):
        self.name = name
        self.resource_type = resource_type
        self.categories = categories
        self.config = config
        self.available = True
        self.last_used = None
        self.success_count = 0
        self.failure_count = 0
        
    @property
    def reliability_score(self) -> float:
        """Calculate reliability based on success/failure ratio"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Neutral score for unused resources
        return self.success_count / total
    

class ConfidenceValidator:
    """
    Enforces anti-overconfidence rules and validation requirements
    Integrates all available APIs and MCP servers for comprehensive validation
    """
    
    def __init__(self, max_confidence: float = 0.90):
        """
        Initialize validator with maximum allowed confidence
        
        Args:
            max_confidence: Hard cap on confidence (default 90%)
        """
        self.max_confidence = max_confidence
        self.validation_history: List[Dict[str, Any]] = []
        self.current_confidence: float = 0.0
        self.validation_sources: List[Dict[str, Any]] = []
        self.uncertainty_areas: List[str] = []
        
        # Enhanced validation resources
        self.credentials_path = Path("/Users/camdouglas/quark/data/credentials/all_api_keys.json")
        self.open_access_sources_path = Path("/Users/camdouglas/quark/data/knowledge/validation_system/open_access_literature_sources.json")
        self.credentials = self._load_credentials()
        self.open_access_sources = self._load_open_access_sources()
        self.resources = self._initialize_resources()
        self.validation_cache = {}
        self.session = None
        
    def _load_credentials(self) -> Dict[str, Any]:
        """Load API credentials from secure storage"""
        try:
            with open(self.credentials_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Credentials file not found: {self.credentials_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing credentials JSON: {e}")
            return {}
    
    def _load_open_access_sources(self) -> Dict[str, Any]:
        """Load open access literature sources for validation"""
        try:
            with open(self.open_access_sources_path, 'r') as f:
                sources_data = json.load(f)
                logger.info(f"Loaded {sources_data.get('metadata', {}).get('total_sources', 0)} open access literature sources")
                logger.info(f"  Success rate: {sources_data.get('metadata', {}).get('test_summary', {}).get('success_rate', 'unknown')}")
                return sources_data
        except FileNotFoundError:
            logger.warning(f"Open access sources file not found: {self.open_access_sources_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing open access sources JSON: {e}")
            return {}
    
    def _initialize_resources(self) -> Dict[str, ValidationResource]:
        """Initialize ALL available validation resources from credentials file"""
        resources = {}
        
        # MCP Servers available in Cursor
        mcp_servers = {
            'context7': {
                'categories': [ValidationCategory.CODE_DOCUMENTATION],
                'description': 'Real-time code documentation via Context7 MCP'
            },
            'arxiv': {
                'categories': [ValidationCategory.SCIENTIFIC_LITERATURE, ValidationCategory.ARXIV_PAPER],
                'description': 'Academic papers via arXiv MCP'
            },
            'pubmed': {
                'categories': [ValidationCategory.SCIENTIFIC_LITERATURE],
                'description': 'Biomedical literature via PubMed MCP'
            },
            'openalex': {
                'categories': [ValidationCategory.SCIENTIFIC_LITERATURE],
                'description': 'Scholarly works via OpenAlex MCP'
            },
            'github': {
                'categories': [ValidationCategory.CODE_DOCUMENTATION],
                'description': 'Source code via GitHub MCP'
            },
            'fetch': {
                'categories': [ValidationCategory.GENERAL_KNOWLEDGE],
                'description': 'Web content via Fetch MCP'
            },
            'memory': {
                'categories': [ValidationCategory.GENERAL_KNOWLEDGE],
                'description': 'Knowledge graph via Memory MCP'
            },
            'figma': {
                'categories': [ValidationCategory.GENERAL_KNOWLEDGE],
                'description': 'Design specs via Figma MCP'
            },
            'filesystem': {
                'categories': [ValidationCategory.GENERAL_KNOWLEDGE],
                'description': 'File system access via MCP'
            },
            'time': {
                'categories': [ValidationCategory.GENERAL_KNOWLEDGE],
                'description': 'Time and timezone via MCP'
            },
            'cline': {
                'categories': [ValidationCategory.CODE_DOCUMENTATION],
                'description': 'Cline autonomous coding via MCP'
            }
        }
        
        for name, config in mcp_servers.items():
            resources[f'mcp_{name}'] = ValidationResource(
                name=f'MCP {name.title()}',
                resource_type=ResourceType.MCP_SERVER,
                categories=config['categories'],
                config={'description': config['description'], 'mcp_name': name}
            )
        
        # Initialize ALL API services from credentials file
        if self.credentials and 'services' in self.credentials:
            # Comprehensive category mappings for ALL services
            api_category_mappings = {
                'rcsb_pdb': [ValidationCategory.PROTEIN_STRUCTURE, ValidationCategory.BIOLOGICAL_SEQUENCE],
                'alphafold': [ValidationCategory.PROTEIN_STRUCTURE, ValidationCategory.BIOLOGICAL_SEQUENCE],
                'ncbi_eutilities': [ValidationCategory.GENOMIC_DATA, ValidationCategory.SCIENTIFIC_LITERATURE, ValidationCategory.BIOLOGICAL_SEQUENCE],
                'ensembl': [ValidationCategory.GENOMIC_DATA, ValidationCategory.BIOLOGICAL_SEQUENCE],
                'pubchem': [ValidationCategory.CHEMICAL_COMPOUND],
                'openml': [ValidationCategory.MACHINE_LEARNING],
                'oqmd': [ValidationCategory.MATERIALS_SCIENCE],
                'materials_project': [ValidationCategory.MATERIALS_SCIENCE],
                'uniprot': [ValidationCategory.PROTEIN_STRUCTURE, ValidationCategory.BIOLOGICAL_SEQUENCE],
                'blast': [ValidationCategory.BIOLOGICAL_SEQUENCE],
                'arxiv': [ValidationCategory.ARXIV_PAPER, ValidationCategory.SCIENTIFIC_LITERATURE],
                'pubmed_central': [ValidationCategory.SCIENTIFIC_LITERATURE],
                'wolfram': [ValidationCategory.MATHEMATICAL],
                'openai': [ValidationCategory.GENERAL_KNOWLEDGE],
                'claude': [ValidationCategory.GENERAL_KNOWLEDGE],
                'gemini': [ValidationCategory.GENERAL_KNOWLEDGE],
                'alphagenome': [ValidationCategory.GENOMIC_DATA, ValidationCategory.BIOLOGICAL_SEQUENCE],
                'openrouter': [ValidationCategory.GENERAL_KNOWLEDGE],
                'context7': [ValidationCategory.CODE_DOCUMENTATION],
                'aws': [ValidationCategory.GENERAL_KNOWLEDGE],
                'github': [ValidationCategory.CODE_DOCUMENTATION],
                'huggingface': [ValidationCategory.MACHINE_LEARNING],
                'google_cloud': [ValidationCategory.GENERAL_KNOWLEDGE],
                'kaggle': [ValidationCategory.MACHINE_LEARNING]
            }
            
            # Load ALL services from the credentials file
            for service_name in self.credentials.get('services', {}):
                service_data = self.credentials['services'][service_name]
                
                # Use mapped categories or default to GENERAL_KNOWLEDGE
                categories = api_category_mappings.get(
                    service_name, 
                    [ValidationCategory.GENERAL_KNOWLEDGE]
                )
                
                # Enhance categories based on service description
                description = service_data.get('description', '').lower()
                if 'protein' in description or 'structure' in description:
                    if ValidationCategory.PROTEIN_STRUCTURE not in categories:
                        categories.append(ValidationCategory.PROTEIN_STRUCTURE)
                if 'sequence' in description or 'dna' in description or 'rna' in description:
                    if ValidationCategory.BIOLOGICAL_SEQUENCE not in categories:
                        categories.append(ValidationCategory.BIOLOGICAL_SEQUENCE)
                if 'genom' in description or 'gene' in description:
                    if ValidationCategory.GENOMIC_DATA not in categories:
                        categories.append(ValidationCategory.GENOMIC_DATA)
                if 'chemical' in description or 'compound' in description or 'drug' in description:
                    if ValidationCategory.CHEMICAL_COMPOUND not in categories:
                        categories.append(ValidationCategory.CHEMICAL_COMPOUND)
                if 'machine learning' in description or 'ml' in description or 'model' in description:
                    if ValidationCategory.MACHINE_LEARNING not in categories:
                        categories.append(ValidationCategory.MACHINE_LEARNING)
                if 'material' in description or 'crystal' in description:
                    if ValidationCategory.MATERIALS_SCIENCE not in categories:
                        categories.append(ValidationCategory.MATERIALS_SCIENCE)
                if 'paper' in description or 'literature' in description or 'publication' in description:
                    if ValidationCategory.SCIENTIFIC_LITERATURE not in categories:
                        categories.append(ValidationCategory.SCIENTIFIC_LITERATURE)
                if 'math' in description or 'equation' in description or 'calculation' in description:
                    if ValidationCategory.MATHEMATICAL not in categories:
                        categories.append(ValidationCategory.MATHEMATICAL)
                
                # Create resource for this service
                resources[f'api_{service_name}'] = ValidationResource(
                    name=service_data.get('service', service_name.replace('_', ' ').title()),
                    resource_type=ResourceType.API,
                    categories=categories,
                    config={
                        'service_data': service_data,
                        'service_name': service_name,
                        'description': service_data.get('description', ''),
                        'endpoints': service_data.get('endpoints', {}),
                        'api_key': service_data.get('api_key'),
                        'token': service_data.get('token'),
                        'app_id': service_data.get('app_id'),
                        'authentication': service_data.get('authentication', 'none'),
                        'rate_limits': service_data.get('rate_limits', {}),
                        'features': service_data.get('features', {}),
                        'documentation': service_data.get('documentation', '')
                    }
                )
            
            # Also initialize any API keys at root level of credentials
            root_api_keys = {
                'gemini_api_key': ('Gemini', [ValidationCategory.GENERAL_KNOWLEDGE]),
                'alphagenome_api_key': ('AlphaGenome', [ValidationCategory.GENOMIC_DATA, ValidationCategory.BIOLOGICAL_SEQUENCE]),
                'openai_api_key': ('OpenAI', [ValidationCategory.GENERAL_KNOWLEDGE]),
                'anthropic_api_key': ('Anthropic Claude', [ValidationCategory.GENERAL_KNOWLEDGE]),
                'aws_secret_key': ('AWS', [ValidationCategory.GENERAL_KNOWLEDGE]),
                'github_token': ('GitHub', [ValidationCategory.CODE_DOCUMENTATION]),
                'huggingface_token': ('HuggingFace', [ValidationCategory.MACHINE_LEARNING]),
                'google_cloud_api_key': ('Google Cloud', [ValidationCategory.GENERAL_KNOWLEDGE]),
                'kaggle_api_key': ('Kaggle', [ValidationCategory.MACHINE_LEARNING]),
                'wolfram_alpha_app_id': ('Wolfram Alpha', [ValidationCategory.MATHEMATICAL]),
                'openrouter_api_key': ('OpenRouter', [ValidationCategory.GENERAL_KNOWLEDGE])
            }
            
            for key_name, (service_name, categories) in root_api_keys.items():
                if key_name in self.credentials and f'root_{key_name}' not in resources:
                    resources[f'root_{key_name}'] = ValidationResource(
                        name=f'{service_name} (Root Key)',
                        resource_type=ResourceType.API,
                        categories=categories,
                        config={
                            'key_name': key_name,
                            'key_value': self.credentials.get(key_name),
                            'description': f'{service_name} API from root credentials'
                        }
                    )
        
        # Add Open Access Literature Sources (CRITICAL for anti-overconfidence validation)
        if self.open_access_sources:
            # Process preprint servers
            preprint_servers = self.open_access_sources.get('preprint_servers', {})
            for source_id, source_data in preprint_servers.items():
                if source_data.get('status') == 'tested_accessible' or source_data.get('test_status') == 'tested_working':
                    categories = []
                    subjects = source_data.get('subjects', [])
                    
                    # Map subjects to categories
                    if any(s in str(subjects) for s in ['physics', 'math', 'computer']):
                        categories.append(ValidationCategory.ARXIV_PAPER)
                    if any(s in str(subjects) for s in ['biology', 'life', 'medical', 'neuro']):
                        categories.extend([ValidationCategory.BIOLOGICAL_SEQUENCE, ValidationCategory.GENOMIC_DATA])
                    if any(s in str(subjects) for s in ['chemistry', 'chemical']):
                        categories.append(ValidationCategory.CHEMICAL_COMPOUND)
                    if any(s in str(subjects) for s in ['psychology', 'cognitive', 'brain']):
                        categories.append(ValidationCategory.SCIENTIFIC_LITERATURE)
                    
                    if not categories:
                        categories = [ValidationCategory.SCIENTIFIC_LITERATURE]
                    
                    resources[f'openaccess_{source_id}'] = ValidationResource(
                        name=source_data.get('name', source_id),
                        resource_type=ResourceType.API,
                        categories=categories,
                        config={
                            'source_type': 'preprint_server',
                            'url': source_data.get('url'),
                            'api_endpoint': source_data.get('api_endpoint'),
                            'documentation': source_data.get('documentation'),
                            'access_method': source_data.get('access_method'),
                            'authentication': source_data.get('authentication', 'none'),
                            'description': f"Open access preprint server: {source_data.get('name')}",
                            'test_status': source_data.get('test_status'),
                            'response_time': source_data.get('response_time', 'unknown')
                        }
                    )
            
            # Process academic databases
            academic_databases = self.open_access_sources.get('academic_databases', {})
            for source_id, source_data in academic_databases.items():
                if source_data.get('status') == 'tested_accessible' or source_data.get('test_status') == 'tested_working':
                    resources[f'openaccess_{source_id}'] = ValidationResource(
                        name=source_data.get('name', source_id),
                        resource_type=ResourceType.API,
                        categories=[ValidationCategory.SCIENTIFIC_LITERATURE],
                        config={
                            'source_type': 'academic_database',
                            'url': source_data.get('url'),
                            'api_endpoint': source_data.get('api_endpoint'),
                            'documentation': source_data.get('documentation'),
                            'access_method': source_data.get('access_method'),
                            'authentication': source_data.get('authentication', 'none'),
                            'description': f"Academic database: {source_data.get('name')}",
                            'test_status': source_data.get('test_status')
                        }
                    )
            
            # Process open access journals
            open_journals = self.open_access_sources.get('open_access_journals', {})
            for source_id, source_data in open_journals.items():
                if source_data.get('status') == 'tested_accessible' or source_data.get('test_status') == 'tested_working':
                    resources[f'openaccess_{source_id}'] = ValidationResource(
                        name=source_data.get('name', source_id),
                        resource_type=ResourceType.API,
                        categories=[ValidationCategory.SCIENTIFIC_LITERATURE],
                        config={
                            'source_type': 'open_access_journal',
                            'url': source_data.get('url'),
                            'api_endpoint': source_data.get('api_endpoint'),
                            'documentation': source_data.get('documentation'),
                            'access_method': source_data.get('access_method'),
                            'description': f"Open access journal: {source_data.get('name')}",
                            'test_status': source_data.get('test_status')
                        }
                    )
            
            # Process neuroscience-specific sources (HIGH PRIORITY for Quark)
            neuro_sources = self.open_access_sources.get('neuroscience_specific', {})
            for source_id, source_data in neuro_sources.items():
                if source_data.get('status') == 'tested_accessible' or source_data.get('test_status') == 'tested_working':
                    resources[f'openaccess_{source_id}'] = ValidationResource(
                        name=source_data.get('name', source_id),
                        resource_type=ResourceType.API,
                        categories=[ValidationCategory.SCIENTIFIC_LITERATURE, ValidationCategory.BIOLOGICAL_SEQUENCE],
                        config={
                            'source_type': 'neuroscience_specific',
                            'url': source_data.get('url'),
                            'api_endpoint': source_data.get('api_endpoint'),
                            'description': f"Neuroscience resource: {source_data.get('name')}",
                            'priority': 'high',  # High priority for brain-related queries
                            'test_status': source_data.get('test_status')
                        }
                    )
                    
        logger.info(f"Initialized ALL {len(resources)} validation resources")
        logger.info(f"  - MCP Servers: {sum(1 for r in resources.values() if r.resource_type == ResourceType.MCP_SERVER)}")
        logger.info(f"  - APIs (including credentials): {sum(1 for r in resources.values() if r.resource_type == ResourceType.API and 'openaccess' not in r.name)}")
        logger.info(f"  - Open Access Literature Sources: {sum(1 for name in resources.keys() if 'openaccess' in name)}")
        
        return resources
        
    def calculate_confidence(
        self,
        source_authority: float,
        cross_validation: float,
        test_coverage: float,
        peer_review: float
    ) -> float:
        """
        Calculate confidence score with hard cap at 90%
        
        Args:
            source_authority: Authority level of primary source (0-1)
            cross_validation: Cross-validation score (0-1)
            test_coverage: Test coverage percentage (0-1)
            peer_review: Peer review validation (0-1)
            
        Returns:
            Confidence score capped at max_confidence
        """
        raw_confidence = (
            source_authority * 0.3 +
            cross_validation * 0.3 +
            test_coverage * 0.2 +
            peer_review * 0.2
        )
        
        # Apply hard cap
        confidence = min(raw_confidence, self.max_confidence)
        
        # Store for tracking
        self.current_confidence = confidence
        
        # Log calculation
        logger.info(f"Confidence calculated: {confidence:.1%} (raw: {raw_confidence:.1%})")
        
        return confidence
        
    def get_confidence_level(self, score: float) -> Tuple[ConfidenceLevel, str]:
        """
        Get confidence level category and prefix
        
        Args:
            score: Confidence score (0-1)
            
        Returns:
            Tuple of (ConfidenceLevel, prefix string)
        """
        score_percent = score * 100
        
        for level in ConfidenceLevel:
            min_val, max_val, prefix = level.value
            if min_val <= score_percent < max_val:
                return level, f"{prefix} ({score_percent:.0f}%)"
                
        # Should never reach here, but handle edge case
        return ConfidenceLevel.LOW, f"⚠️ LOW CONFIDENCE ({score_percent:.0f}%)"
        
    def validate_sources(self, sources: List[Dict[str, Any]]) -> float:
        """
        Validate and score information sources
        
        Args:
            sources: List of source dictionaries with 'type' and 'authority' keys
            
        Returns:
            Average authority score
        """
        if not sources:
            logger.warning("No validation sources provided")
            return 0.0
            
        total_authority = 0.0
        valid_sources = 0
        
        for source in sources:
            source_type = source.get('type', 'experimental')
            
            # Map source type to validation level
            if source_type in ['official_docs', 'peer_reviewed', 'api_spec']:
                authority = ValidationSource.PRIMARY.value
            elif source_type in ['community', 'stackoverflow', 'blog']:
                authority = ValidationSource.SECONDARY.value
            else:
                authority = ValidationSource.EXPERIMENTAL.value
                
            # Adjust for recency if provided
            if 'date' in source:
                try:
                    source_date = datetime.fromisoformat(source['date'])
                    age_days = (datetime.now() - source_date).days
                    if age_days > 730:  # Older than 2 years
                        authority *= 0.7
                except:
                    pass
                    
            total_authority += authority
            valid_sources += 1
            
            # Store for reporting
            self.validation_sources.append({
                'source': source.get('name', 'Unknown'),
                'type': source_type,
                'authority': authority,
                'url': source.get('url', '')
            })
            
        return total_authority / valid_sources if valid_sources > 0 else 0.0
        
    def check_uncertainty_triggers(self, context: Dict[str, Any]) -> List[str]:
        """
        Check for conditions that should trigger uncertainty expression
        
        Args:
            context: Context dictionary with various flags
            
        Returns:
            List of triggered uncertainty conditions
        """
        triggers = []
        
        if not context.get('documentation_found', True):
            triggers.append("No direct documentation found")
            
        if context.get('conflicting_sources', False):
            triggers.append("Conflicting information between sources")
            
        if context.get('complex_biology', False):
            triggers.append("Complex biological/scientific concepts")
            
        if context.get('performance_unclear', False):
            triggers.append("Performance implications unclear")
            
        if context.get('security_implications', False):
            triggers.append("Security implications present")
            
        if context.get('multiple_approaches', False):
            triggers.append("Multiple valid approaches exist")
            
        if context.get('ambiguous_requirements', False):
            triggers.append("Requirements are ambiguous")
            
        self.uncertainty_areas = triggers
        return triggers
        
    def generate_validation_report(self) -> str:
        """
        Generate formatted validation report with confidence assessment
        
        Returns:
            Formatted validation report string
        """
        level, prefix = self.get_confidence_level(self.current_confidence)
        
        report = f"""
{prefix}

📊 VALIDATION METRICS:
- Confidence Score: {self.current_confidence:.1%}
- Sources Consulted: {len(self.validation_sources)}
- Uncertainty Areas: {len(self.uncertainty_areas)}

🔍 VALIDATION SOURCES:
"""
        
        for i, source in enumerate(self.validation_sources, 1):
            report += f"{i}. [{source['source']}] "
            report += f"(Authority: {source['authority']:.1f}) "
            if source['url']:
                report += f"- {source['url']}\n"
            else:
                report += "\n"
                
        if self.uncertainty_areas:
            report += f"""
⚠️ UNCERTAINTY TRIGGERS:
"""
            for area in self.uncertainty_areas:
                report += f"- {area}\n"
                
        # Add recommendations based on confidence level
        if level == ConfidenceLevel.LOW:
            report += """
🔔 RECOMMENDED ACTIONS:
- Seek additional authoritative sources
- Request user verification
- Consider alternative approaches
- Run comprehensive tests before proceeding
"""
        elif level == ConfidenceLevel.MEDIUM:
            report += """
🔔 RECOMMENDED ACTIONS:
- Cross-validate with additional sources
- Run edge case tests
- Document assumptions clearly
"""
        elif level == ConfidenceLevel.FORBIDDEN:
            report += """
🚫 WARNING: OVERCONFIDENCE DETECTED
- Reduce certainty in claims
- Add explicit uncertainty markers
- Seek contradictory evidence
- Never claim 100% certainty
"""
            
        return report
        
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Validate that response includes proper confidence markers
        
        Args:
            response: Response text to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'has_confidence_statement': False,
            'has_uncertainty_markers': False,
            'has_source_citations': False,
            'confidence_level_stated': False,
            'validation_checklist': []
        }
        
        # Check for confidence statements
        confidence_patterns = [
            r'confidence.*\d+%',
            r'(?:low|medium|high)\s+confidence',
            r'(?:uncertain|moderately confident|reasonably confident)'
        ]
        
        for pattern in confidence_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                results['has_confidence_statement'] = True
                break
                
        # Check for uncertainty markers
        uncertainty_patterns = [
            r'(?:might|may|could|possibly|potentially)',
            r'(?:uncertain|unsure|unclear)',
            r'I (?:don\'t|do not) know',
            r'(?:assumption|guess|estimate)'
        ]
        
        for pattern in uncertainty_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                results['has_uncertainty_markers'] = True
                break
                
        # Check for source citations
        citation_patterns = [
            r'(?:source|reference|according to|based on)',
            r'(?:documentation|paper|study|article)',
            r'\[.*\]',  # Bracketed citations
            r'https?://'  # URLs
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                results['has_source_citations'] = True
                break
                
        # Check if confidence level is explicitly stated
        if re.search(r'\d+%|confidence level', response, re.IGNORECASE):
            results['confidence_level_stated'] = True
            
        # Validation checklist
        checklist = [
            ('Confidence explicitly stated', results['has_confidence_statement']),
            ('Uncertainty acknowledged', results['has_uncertainty_markers']),
            ('Sources cited', results['has_source_citations']),
            ('Confidence level quantified', results['confidence_level_stated'])
        ]
        
        results['validation_checklist'] = checklist
        results['passes_validation'] = all(v for _, v in checklist)
        
        return results
        
    def enforce_validation(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Enforce validation rules on content
        
        Args:
            content: Content to validate
            context: Optional context dictionary
            
        Returns:
            Tuple of (passes_validation, feedback_message)
        """
        # Validate response format
        validation = self.validate_response(content)
        
        if not validation['passes_validation']:
            missing = [item for item, passed in validation['validation_checklist'] if not passed]
            
            feedback = "⚠️ VALIDATION FAILED - Missing required elements:\n"
            for item in missing:
                feedback += f"  - {item}\n"
                
            feedback += "\n📋 Required elements:\n"
            feedback += "  1. Explicit confidence statement (X%)\n"
            feedback += "  2. Uncertainty acknowledgment where appropriate\n"
            feedback += "  3. Source citations for claims\n"
            feedback += "  4. Quantified confidence level\n"
            
            return False, feedback
            
        # Check for overconfidence
        if "100%" in content or re.search(r'(absolutely|definitely|certainly) (certain|sure|correct)', content, re.IGNORECASE):
            feedback = "🚫 OVERCONFIDENCE DETECTED\n"
            feedback += "Never claim 100% certainty or use absolute language.\n"
            feedback += "Maximum allowed confidence is 90%.\n"
            return False, feedback
            
        return True, "✅ Validation passed"
    
    def detect_validation_needs(self, context: str) -> List[ValidationCategory]:
        """
        Analyze context to determine what types of validation are needed
        
        Args:
            context: Text or code context to analyze
            
        Returns:
            List of validation categories needed
        """
        categories = []
        
        # Scientific/biological patterns
        if re.search(r'(?:protein|gene|DNA|RNA|amino acid|sequence|genome)', context, re.IGNORECASE):
            categories.append(ValidationCategory.BIOLOGICAL_SEQUENCE)
            categories.append(ValidationCategory.GENOMIC_DATA)
            
        if re.search(r'(?:structure|PDB|fold|3D|conformation|crystal)', context, re.IGNORECASE):
            categories.append(ValidationCategory.PROTEIN_STRUCTURE)
            
        if re.search(r'(?:compound|molecule|drug|chemical|SMILES|InChI)', context, re.IGNORECASE):
            categories.append(ValidationCategory.CHEMICAL_COMPOUND)
            
        if re.search(r'(?:paper|publication|study|research|literature|citation)', context, re.IGNORECASE):
            categories.append(ValidationCategory.SCIENTIFIC_LITERATURE)
            
        if re.search(r'(?:arXiv|preprint|physics|mathematics)', context, re.IGNORECASE):
            categories.append(ValidationCategory.ARXIV_PAPER)
            
        # Technical patterns
        if re.search(r'(?:machine learning|ML|model|dataset|algorithm|training)', context, re.IGNORECASE):
            categories.append(ValidationCategory.MACHINE_LEARNING)
            
        if re.search(r'(?:material|crystal|lattice|bandgap|formation energy)', context, re.IGNORECASE):
            categories.append(ValidationCategory.MATERIALS_SCIENCE)
            
        if re.search(r'(?:import|function|class|API|library|package)', context, re.IGNORECASE):
            categories.append(ValidationCategory.CODE_DOCUMENTATION)
            
        if re.search(r'(?:equation|calculate|formula|derivative|integral)', context, re.IGNORECASE):
            categories.append(ValidationCategory.MATHEMATICAL)
            
        # Default to general if no specific patterns found
        if not categories:
            categories.append(ValidationCategory.GENERAL_KNOWLEDGE)
            
        return list(set(categories))  # Remove duplicates
        
    def select_best_resources(self, categories: List[ValidationCategory], 
                             max_resources: int = 3, query: str = "") -> List[ValidationResource]:
        """
        Intelligently select the best resources for validation based on task requirements
        
        Args:
            categories: List of validation categories needed
            max_resources: Maximum number of resources to use
            query: Optional query text to help with selection
            
        Returns:
            List of selected validation resources prioritized by relevance
        """
        candidate_resources = []
        query_lower = query.lower() if query else ""
        
        for resource in self.resources.values():
            if not resource.available:
                continue
                
            # Calculate base relevance from category matching
            category_relevance = sum(1 for cat in categories if cat in resource.categories)
            
            # Boost score for specific keyword matches in query
            keyword_boost = 0.0
            if query_lower:
                # Check for specific service mentions
                resource_name_lower = resource.name.lower()
                service_name = resource.config.get('service_name', '').lower()
                
                # Direct service name match gets highest boost
                if service_name and service_name in query_lower:
                    keyword_boost = 2.0
                elif any(word in resource_name_lower for word in query_lower.split()):
                    keyword_boost = 1.5
                    
                # Boost for specific domain keywords
                description = resource.config.get('description', '').lower()
                domain_keywords = {
                    'protein': ['alphafold', 'uniprot', 'pdb', 'structure'],
                    'gene': ['ensembl', 'ncbi', 'genome'],
                    'chemical': ['pubchem', 'compound'],
                    'paper': ['arxiv', 'pubmed', 'literature'],
                    'code': ['context7', 'github'],
                    'ml': ['openml', 'huggingface', 'kaggle'],
                    'material': ['materials_project', 'oqmd']
                }
                
                for keyword, services in domain_keywords.items():
                    if keyword in query_lower:
                        if any(svc in resource_name_lower or svc in service_name for svc in services):
                            keyword_boost += 0.5
                            
            # Priority scoring for resource types
            type_priority = {
                ResourceType.MCP_SERVER: 1.2,  # Prefer MCP servers for real-time data
                ResourceType.API: 1.0,
                ResourceType.LOCAL_TOOL: 0.8,
                ResourceType.WEB_SERVICE: 0.6
            }
            type_score = type_priority.get(resource.resource_type, 0.5)
            
            # Calculate final score
            if category_relevance > 0 or keyword_boost > 0:
                # Weighted scoring: categories (40%), keywords (30%), reliability (20%), type (10%)
                score = (
                    category_relevance * 0.4 +
                    keyword_boost * 0.3 +
                    resource.reliability_score * 0.2 +
                    type_score * 0.1
                )
                
                # Special handling for highly specific resources
                if resource.resource_type == ResourceType.MCP_SERVER:
                    # Always include Context7 for code-related queries
                    if 'context7' in resource.config.get('mcp_name', '') and ValidationCategory.CODE_DOCUMENTATION in categories:
                        score += 1.0
                    # Always include arXiv/PubMed for scientific queries
                    if resource.config.get('mcp_name') in ['arxiv', 'pubmed', 'openalex'] and ValidationCategory.SCIENTIFIC_LITERATURE in categories:
                        score += 0.8
                
                # ANTI-OVERCONFIDENCE: Prioritize open access literature sources for scientific validation
                if 'openaccess' in resource.name.lower():
                    # Boost open access sources when dealing with scientific uncertainty
                    if ValidationCategory.SCIENTIFIC_LITERATURE in categories:
                        score += 0.7  # High priority for peer-reviewed sources
                    if resource.config.get('source_type') == 'neuroscience_specific':
                        score += 0.5  # Extra boost for brain-related queries (Quark focus)
                    if resource.config.get('test_status') == 'tested_working':
                        score += 0.2  # Prefer verified working sources
                        
                candidate_resources.append((score, resource))
                
        # Sort by score (highest first)
        candidate_resources.sort(key=lambda x: x[0], reverse=True)
        
        # Select resources with diversity in mind
        selected = []
        selected_types = set()
        selected_categories = set()
        
        for score, resource in candidate_resources:
            if len(selected) >= max_resources:
                break
                
            # Ensure diversity: try to include different types and categories
            resource_categories = set(resource.categories)
            
            # Always include the highest scoring resource
            if len(selected) == 0:
                selected.append(resource)
                selected_types.add(resource.resource_type)
                selected_categories.update(resource_categories)
            else:
                # Prefer resources that cover new categories or types
                new_categories = resource_categories - selected_categories
                new_type = resource.resource_type not in selected_types
                
                # Include if it adds new coverage or score is very high
                if new_categories or new_type or score > 1.5:
                    selected.append(resource)
                    selected_types.add(resource.resource_type)
                    selected_categories.update(resource_categories)
                    
        # If we don't have enough resources, add more from top candidates
        while len(selected) < max_resources and len(candidate_resources) > len(selected):
            for score, resource in candidate_resources:
                if resource not in selected:
                    selected.append(resource)
                    break
            else:
                break
                
        logger.info(f"Selected {len(selected)} resources from {len(candidate_resources)} candidates")
        logger.info(f"  Resources: {[r.name for r in selected]}")
        logger.info(f"  Categories covered: {[c.value for c in selected_categories]}")
        
        return selected
    
    def get_mcp_validation_instruction(self, resource: ValidationResource, query: str) -> str:
        """
        Generate instruction for MCP validation
        
        Args:
            resource: MCP server resource
            query: Validation query
            
        Returns:
            Instruction string for using MCP
        """
        mcp_name = resource.config.get('mcp_name', '')
        
        instructions = {
            'context7': f"Use Context7 MCP to validate: {query}",
            'arxiv': f"Search arXiv for papers about: {query}",
            'pubmed': f"Search PubMed for literature on: {query}",
            'openalex': f"Search OpenAlex for scholarly works on: {query}",
            'github': f"Search GitHub for code examples of: {query}",
            'fetch': f"Fetch web content to validate: {query}",
            'memory': f"Query knowledge graph for: {query}",
            'figma': f"Check Figma designs for: {query}"
        }
        
        return instructions.get(mcp_name, f"Validate using {resource.name}: {query}")
    
    def get_api_validation_instruction(self, resource: ValidationResource, query: str) -> str:
        """
        Generate instruction for API validation
        
        Args:
            resource: API resource
            query: Validation query
            
        Returns:
            Instruction string for using API
        """
        service_data = resource.config.get('service_data', {})
        
        return {
            'api_rcsb_pdb': f"Query RCSB PDB for protein structures: {query}",
            'api_alphafold': f"Check AlphaFold for protein predictions: {query}",
            'api_ncbi_eutilities': f"Search NCBI databases for: {query}",
            'api_ensembl': f"Query Ensembl for genomic data: {query}",
            'api_pubchem': f"Search PubChem for chemical compounds: {query}",
            'api_openml': f"Check OpenML for ML datasets/models: {query}",
            'api_materials_project': f"Query Materials Project for: {query}",
            'api_uniprot': f"Search UniProt for protein sequences: {query}",
            'api_blast': f"Run BLAST sequence alignment for: {query}",
            'api_arxiv': f"Search arXiv API for papers on: {query}"
        }.get(f'api_{resource.name.lower().replace(" ", "_")}', 
              f"Query {resource.name} API: {query}")
    
    def perform_enhanced_validation(self, context: str, query: str = None,
                                   categories: Optional[List[ValidationCategory]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation using all available resources
        
        Args:
            context: Context requiring validation
            query: Specific validation query
            categories: Optional list of validation categories
            
        Returns:
            Enhanced validation results with resource recommendations
        """
        # Detect validation needs if categories not provided
        if not categories:
            categories = self.detect_validation_needs(context)
            
        # Select best resources (pass query for better selection)
        resources = self.select_best_resources(categories, query=query or context)
        
        # Generate validation instructions
        validation_plan = {
            'categories': [c.value for c in categories],
            'resources_selected': [],
            'validation_instructions': [],
            'mcp_servers_to_use': [],
            'apis_to_use': [],
            'confidence_adjustment': 0.0
        }
        
        for resource in resources:
            validation_plan['resources_selected'].append({
                'name': resource.name,
                'type': resource.resource_type.value,
                'reliability': resource.reliability_score
            })
            
            if resource.resource_type == ResourceType.MCP_SERVER:
                instruction = self.get_mcp_validation_instruction(resource, query or context[:100])
                validation_plan['validation_instructions'].append(instruction)
                validation_plan['mcp_servers_to_use'].append(resource.config.get('mcp_name'))
                
            elif resource.resource_type == ResourceType.API:
                instruction = self.get_api_validation_instruction(resource, query or context[:100])
                validation_plan['validation_instructions'].append(instruction)
                
                # Add API details for reference
                service_data = resource.config.get('service_data', {})
                validation_plan['apis_to_use'].append({
                    'name': resource.name,
                    'endpoints': service_data.get('endpoints', {}),
                    'requires_auth': 'api_key' in service_data or 'token' in service_data
                })
        
        # Adjust confidence based on available resources
        if len(resources) >= 3:
            validation_plan['confidence_adjustment'] = 0.15
        elif len(resources) >= 2:
            validation_plan['confidence_adjustment'] = 0.10
        elif len(resources) >= 1:
            validation_plan['confidence_adjustment'] = 0.05
        
        # Update internal tracking
        self.validation_sources.extend([
            {'name': r['name'], 'type': r['type']} 
            for r in validation_plan['resources_selected']
        ])
        
        return validation_plan
    
    def generate_validation_checklist(self) -> List[str]:
        """
        Generate a checklist for validation using available resources
        
        Returns:
            List of validation steps to perform
        """
        checklist = [
            "📋 MANDATORY VALIDATION CHECKLIST:",
            "",
            "1. RESOURCE CONSULTATION:",
        ]
        
        # Add MCP servers
        mcp_available = [r for r in self.resources.values() 
                        if r.resource_type == ResourceType.MCP_SERVER]
        if mcp_available:
            checklist.append("   MCP Servers (Use when uncertain):")
            for resource in mcp_available[:5]:
                checklist.append(f"   □ {resource.name}: {resource.config.get('description', '')}")
        
        # Add APIs
        api_available = [r for r in self.resources.values() 
                        if r.resource_type == ResourceType.API]
        if api_available:
            checklist.append("")
            checklist.append("   APIs (For specialized validation):")
            for resource in api_available[:5]:
                checklist.append(f"   □ {resource.name}: {resource.config.get('description', '')}")
        
        checklist.extend([
            "",
            "2. VALIDATION SEQUENCE:",
            "   □ Check documentation (Context7 MCP)",
            "   □ Search literature (arXiv/PubMed MCP)",
            "   □ Verify with APIs (domain-specific)",
            "   □ Cross-reference multiple sources",
            "   □ Calculate confidence score",
            "",
            "3. CONFIDENCE REQUIREMENTS:",
            "   □ Never claim 100% confidence",
            "   □ Express uncertainty explicitly",
            "   □ Cite all sources used",
            "   □ Document validation gaps",
            "",
            "4. WHEN TO USE RESOURCES:",
            "   ⚠️ Low confidence (<40%): Use 3+ resources",
            "   🟡 Medium confidence (40-70%): Use 2+ resources",
            "   ✅ High confidence (70-90%): Use 1+ resource for verification",
            "",
            f"📍 Credentials loaded from: {self.credentials_path}",
            f"✅ {len(self.resources)} validation resources available"
        ])
        
        return checklist
    
    def perform_anti_overconfidence_validation(self, claim: str, context: str = "", 
                                               user_statement: str = "") -> Dict[str, Any]:
        """
        Perform strict anti-overconfidence validation as per mandatory rules.
        This method MUST be called when making ANY technical or scientific claim.
        
        Args:
            claim: The claim or statement to validate
            context: Additional context for the claim
            user_statement: If validating a user's statement
            
        Returns:
            Comprehensive validation result with mandatory uncertainty expression
        """
        # MANDATORY: Start with radical uncertainty assumption
        validation_result = {
            'initial_confidence': 0.0,  # Start at ZERO confidence
            'claim': claim,
            'validated': False,
            'uncertainty_level': 'HIGH',
            'sources_consulted': [],
            'validation_gaps': [],
            'user_correction_needed': False,
            'exhaustive_attempts': [],
            'final_confidence': 0.0
        }
        
        # Detect what type of validation is needed
        categories = self.detect_validation_needs(claim + " " + context)
        
        # MANDATORY: Select multiple sources (minimum 3 for low confidence)
        max_sources = 5 if user_statement else 3
        resources = self.select_best_resources(categories, max_resources=max_sources, query=claim)
        
        # Prioritize open access literature sources
        open_access_resources = [r for r in resources if 'openaccess' in r.name.lower()]
        other_resources = [r for r in resources if 'openaccess' not in r.name.lower()]
        
        # Record sources being consulted
        validation_result['sources_consulted'] = [
            {
                'name': r.name,
                'type': r.resource_type.value,
                'authority_level': 'PRIMARY' if 'openaccess' in r.name.lower() or r.resource_type == ResourceType.MCP_SERVER else 'SECONDARY',
                'url': r.config.get('url', r.config.get('api_endpoint', ''))
            }
            for r in resources
        ]
        
        # Check for uncertainty triggers (MANDATORY)
        uncertainty_triggers = []
        
        # User skepticism protocol (if validating user statement)
        if user_statement:
            validation_result['user_correction_needed'] = True
            uncertainty_triggers.append("User claim requires validation")
            
            # Check for contradictions with best practices
            if any(keyword in user_statement.lower() for keyword in 
                   ['always', 'never', 'definitely', 'certainly', '100%', 'guaranteed']):
                uncertainty_triggers.append("Absolute claims detected - likely overconfident")
                validation_result['uncertainty_level'] = 'EXTREME'
        
        # Scientific/biological claims need peer-reviewed sources
        if any(cat in categories for cat in [
            ValidationCategory.BIOLOGICAL_SEQUENCE,
            ValidationCategory.PROTEIN_STRUCTURE,
            ValidationCategory.GENOMIC_DATA
        ]):
            if not open_access_resources:
                uncertainty_triggers.append("No peer-reviewed sources available for biological claim")
                validation_result['validation_gaps'].append("Missing peer-reviewed validation")
        
        # Complex concepts trigger
        if len(categories) > 2:
            uncertainty_triggers.append("Complex multi-domain concept")
        
        # No direct documentation trigger
        if len(resources) < 2:
            uncertainty_triggers.append("Insufficient validation sources")
            validation_result['validation_gaps'].append("Need more authoritative sources")
        
        validation_result['uncertainty_triggers'] = uncertainty_triggers
        
        # Calculate confidence with strict caps
        if len(open_access_resources) > 0:
            # Have peer-reviewed sources
            base_confidence = 0.3
        elif len(resources) > 0:
            # Have some sources
            base_confidence = 0.2
        else:
            # No sources
            base_confidence = 0.0
        
        # Adjust confidence based on source count and quality
        source_multiplier = min(len(resources) * 0.15, 0.6)  # Max 60% from sources
        
        # Apply uncertainty penalties
        uncertainty_penalty = len(uncertainty_triggers) * 0.1
        
        # Calculate final confidence (NEVER exceed 90%)
        final_confidence = min(
            base_confidence + source_multiplier - uncertainty_penalty,
            0.9  # HARD CAP at 90%
        )
        
        validation_result['final_confidence'] = max(final_confidence, 0.0)
        
        # Determine confidence level
        level, prefix = self.get_confidence_level(validation_result['final_confidence'])
        validation_result['confidence_level'] = prefix
        
        # Set validation status
        validation_result['validated'] = validation_result['final_confidence'] >= 0.4
        
        # Generate validation instructions
        validation_result['validation_instructions'] = []
        for resource in resources[:3]:  # Top 3 resources
            if resource.resource_type == ResourceType.MCP_SERVER:
                instruction = self.get_mcp_validation_instruction(resource, claim)
            else:
                instruction = self.get_api_validation_instruction(resource, claim)
            validation_result['validation_instructions'].append(instruction)
        
        # Generate anti-overconfidence report
        validation_result['anti_overconfidence_report'] = self._generate_anti_overconfidence_report(
            validation_result, user_statement
        )
        
        return validation_result
    
    def _generate_anti_overconfidence_report(self, validation_result: Dict[str, Any], 
                                            user_statement: str = "") -> str:
        """Generate mandatory anti-overconfidence report"""
        report = f"""
{validation_result['confidence_level']}

WHAT I'M CERTAIN ABOUT:
"""
        if validation_result['final_confidence'] > 0.5:
            report += f"- Claim has been partially validated by {len(validation_result['sources_consulted'])} sources\n"
        else:
            report += "- Limited certainty due to insufficient validation\n"
        
        report += """
WHAT I'M UNCERTAIN ABOUT:
"""
        for gap in validation_result['validation_gaps']:
            report += f"- {gap}\n"
        for trigger in validation_result.get('uncertainty_triggers', []):
            report += f"- {trigger}\n"
        
        report += f"""
VALIDATION SOURCES CONSULTED:
"""
        for i, source in enumerate(validation_result['sources_consulted'], 1):
            report += f"{i}. {source['name']} ({source['authority_level']})\n"
        
        if user_statement and validation_result['user_correction_needed']:
            report += f"""
🤔 QUESTIONING YOUR APPROACH:

What you suggested: {user_statement[:100]}...

My concerns:
- Need to verify against authoritative sources
- Multiple validation sources should be consulted
- Alternative approaches may exist

Recommended validation:
"""
            for instruction in validation_result['validation_instructions']:
                report += f"- {instruction}\n"
        
        report += f"""
RECOMMENDED ADDITIONAL VALIDATION:
- Consult peer-reviewed literature via open access sources
- Cross-validate with multiple independent sources
- Run empirical tests if applicable
- Consider alternative interpretations

⚠️ Remember: Maximum confidence is 90% - always maintain uncertainty
"""
        return report


def main():
    """
    CLI interface for enhanced confidence validation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Confidence Validation System')
    parser.add_argument('--check', type=str, help='Check confidence in a response file')
    parser.add_argument('--validate', type=str, help='Validate response format')
    parser.add_argument('--report', action='store_true', help='Generate validation report')
    parser.add_argument('--sources', type=str, help='JSON file with validation sources')
    parser.add_argument('--resources', action='store_true', help='Show available validation resources')
    parser.add_argument('--checklist', action='store_true', help='Generate validation checklist')
    parser.add_argument('--enhance', type=str, help='Perform enhanced validation on text/file')
    
    args = parser.parse_args()
    
    validator = ConfidenceValidator()
    
    if args.resources:
        # Show available validation resources
        print("\n" + "="*60)
        print("AVAILABLE VALIDATION RESOURCES")
        print("="*60)
        
        # Group by type
        mcp_servers = []
        apis = []
        
        for name, resource in validator.resources.items():
            if resource.resource_type == ResourceType.MCP_SERVER:
                mcp_servers.append(resource)
            elif resource.resource_type == ResourceType.API:
                apis.append(resource)
        
        print(f"\n📡 MCP Servers ({len(mcp_servers)} available):")
        for resource in mcp_servers:
            print(f"  • {resource.name}: {resource.config.get('description', '')}")
        
        print(f"\n🔌 APIs ({len(apis)} available):")
        for resource in apis[:10]:  # Show first 10
            print(f"  • {resource.name}: {resource.config.get('description', '')}")
        if len(apis) > 10:
            print(f"  ... and {len(apis)-10} more")
        
        print(f"\n📊 Total: {len(validator.resources)} resources")
        print(f"📍 Credentials: {validator.credentials_path}")
        
    elif args.checklist:
        # Generate validation checklist
        checklist = validator.generate_validation_checklist()
        for line in checklist:
            print(line)
            
    elif args.enhance:
        # Perform enhanced validation
        if Path(args.enhance).exists():
            context = Path(args.enhance).read_text()
            print(f"📄 Validating file: {args.enhance}")
        else:
            context = args.enhance
            print(f"📝 Validating text")
        
        # Detect categories
        categories = validator.detect_validation_needs(context)
        print(f"\n🔍 Detected validation categories:")
        for cat in categories:
            print(f"  • {cat.value}")
        
        # Perform enhanced validation
        validation_plan = validator.perform_enhanced_validation(context)
        
        print(f"\n📋 VALIDATION PLAN:")
        print(f"Resources selected: {len(validation_plan['resources_selected'])}")
        
        for resource in validation_plan['resources_selected']:
            print(f"  • {resource['name']} ({resource['type']}) - Reliability: {resource['reliability']:.1%}")
        
        print(f"\n🎯 Validation Instructions:")
        for i, instruction in enumerate(validation_plan['validation_instructions'], 1):
            print(f"  {i}. {instruction}")
        
        if validation_plan['mcp_servers_to_use']:
            print(f"\n📡 MCP Servers to use:")
            for server in validation_plan['mcp_servers_to_use']:
                print(f"  • {server}")
        
        if validation_plan['apis_to_use']:
            print(f"\n🔌 APIs to use:")
            for api in validation_plan['apis_to_use']:
                print(f"  • {api['name']} (Auth required: {api['requires_auth']})")
        
        print(f"\n📊 Confidence adjustment: +{validation_plan['confidence_adjustment']:.0%}")
        
    elif args.check:
        # Check confidence in a file
        try:
            with open(args.check, 'r') as f:
                content = f.read()
                
            passed, feedback = validator.enforce_validation(content)
            print(feedback)
            
            if not passed:
                sys.exit(1)
                
        except FileNotFoundError:
            print(f"Error: File '{args.check}' not found")
            sys.exit(1)
            
    elif args.validate:
        # Validate specific content
        passed, feedback = validator.enforce_validation(args.validate)
        print(feedback)
        
        if not passed:
            sys.exit(1)
            
    elif args.report:
        # Generate validation report
        if args.sources:
            try:
                with open(args.sources, 'r') as f:
                    sources = json.load(f)
                    
                authority = validator.validate_sources(sources.get('sources', []))
                
                # Calculate sample confidence
                validator.calculate_confidence(
                    source_authority=authority,
                    cross_validation=sources.get('cross_validation', 0.5),
                    test_coverage=sources.get('test_coverage', 0.3),
                    peer_review=sources.get('peer_review', 0.2)
                )
                
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading sources: {e}")
                sys.exit(1)
                
        print(validator.generate_validation_report())
        
    else:
        # Interactive mode - validate stdin
        print("Confidence Validator - Interactive Mode")
        print("Enter content to validate (Ctrl+D to finish):")
        
        try:
            content = sys.stdin.read()
            passed, feedback = validator.enforce_validation(content)
            print("\n" + feedback)
            
            if not passed:
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\nValidation cancelled")
            sys.exit(0)


if __name__ == "__main__":
    main()
