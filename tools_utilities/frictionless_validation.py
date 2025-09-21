#!/usr/bin/env python3
"""
Frictionless Knowledge Validation System
========================================
Simplified interface for agents to access all validation sources
without dealing with async complexity or API details.

This module provides a simple, synchronous interface to the comprehensive
validation system, making knowledge acquisition as frictionless as possible.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrictionlessValidator:
    """
    Simple, synchronous interface to all validation sources.
    Designed for maximum ease of use by AI agents.
    """
    
    def __init__(self):
        """Initialize with all available sources"""
        self.credentials_path = Path("/Users/camdouglas/quark/data/credentials/all_api_keys.json")
        self.available_sources = self._load_available_sources()
        self.validation_cache = {}
        
    def _load_available_sources(self) -> Dict[str, Dict]:
        """Load information about all available validation sources"""
        if not self.credentials_path.exists():
            logger.warning(f"⚠️ Credentials file not found: {self.credentials_path}")
            return {}
        
        with open(self.credentials_path, 'r') as f:
            creds = json.load(f)
        
        services = creds.get('services', {})
        
        # Map services to their validation capabilities
        source_map = {
            # Biological & Medical
            'uniprot': {
                'name': 'UniProt',
                'description': '250M+ protein sequences and functions',
                'categories': ['proteins', 'sequences', 'biology'],
                'confidence': 0.96,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'protein information, sequences, functional annotations'
            },
            'rcsb_pdb': {
                'name': 'RCSB PDB',
                'description': '200K+ experimental protein structures',
                'categories': ['proteins', 'structures', 'biology'],
                'confidence': 0.98,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'protein 3D structures, experimental data'
            },
            'alphafold': {
                'name': 'AlphaFold',
                'description': '200M+ AI-predicted protein structures',
                'categories': ['proteins', 'structures', 'ai'],
                'confidence': 0.95,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'predicted protein structures, coverage of proteomes'
            },
            'blast': {
                'name': 'NCBI BLAST',
                'description': 'Sequence similarity search',
                'categories': ['sequences', 'biology', 'evolution'],
                'confidence': 0.94,
                'auth_required': False,
                'rate_limit': 'strict (100/day)',
                'best_for': 'sequence homology, evolutionary relationships'
            },
            'ensembl': {
                'name': 'Ensembl',
                'description': 'Genomic data across species',
                'categories': ['genomics', 'biology', 'evolution'],
                'confidence': 0.95,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'gene annotations, comparative genomics'
            },
            'ncbi_eutilities': {
                'name': 'NCBI E-utilities',
                'description': 'PubMed, GenBank, and 30+ databases',
                'categories': ['literature', 'sequences', 'biology'],
                'confidence': 0.98,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'biomedical literature, sequence databases'
            },
            
            # Chemistry & Materials
            'pubchem': {
                'name': 'PubChem',
                'description': '100M+ chemical compounds',
                'categories': ['chemistry', 'drugs', 'molecules'],
                'confidence': 0.95,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'chemical properties, drug information'
            },
            'materials_project': {
                'name': 'Materials Project',
                'description': '150K+ computed materials',
                'categories': ['materials', 'physics', 'chemistry'],
                'confidence': 0.92,
                'auth_required': True,
                'rate_limit': 'moderate',
                'best_for': 'materials properties, electronic structure'
            },
            'oqmd': {
                'name': 'OQMD',
                'description': '700K+ quantum materials',
                'categories': ['materials', 'physics', 'quantum'],
                'confidence': 0.90,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'thermodynamic stability, phase diagrams'
            },
            
            # Data & ML
            'openml': {
                'name': 'OpenML',
                'description': '20K+ ML datasets and experiments',
                'categories': ['machine_learning', 'data', 'benchmarks'],
                'confidence': 0.88,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'ML datasets, algorithm benchmarks'
            },
            'kaggle': {
                'name': 'Kaggle',
                'description': 'Competition datasets and notebooks',
                'categories': ['machine_learning', 'data', 'competitions'],
                'confidence': 0.82,
                'auth_required': True,
                'rate_limit': 'reasonable',
                'best_for': 'real-world datasets, ML competitions'
            },
            
            # Literature & Archives
            'arxiv': {
                'name': 'arXiv',
                'description': '2M+ research preprints',
                'categories': ['literature', 'physics', 'math', 'cs', 'biology'],
                'confidence': 0.90,
                'auth_required': False,
                'rate_limit': 'reasonable',
                'best_for': 'latest research, preprints, cutting-edge science'
            },
            'cdx_server': {
                'name': 'CDX Server',
                'description': 'Web archive index queries',
                'categories': ['archives', 'web', 'historical'],
                'confidence': 0.75,
                'auth_required': False,
                'rate_limit': 'self-hosted',
                'best_for': 'historical web content, archived resources'
            }
        }
        
        # Only include sources that are actually configured
        available = {}
        for service_id, service_data in services.items():
            if service_id in source_map:
                source_info = source_map[service_id].copy()
                source_info['configured'] = True
                source_info['endpoints'] = service_data.get('endpoints', {})
                available[service_id] = source_info
        
        logger.info(f"✅ {len(available)} validation sources available: {', '.join(available.keys())}")
        return available
    
    def quick_validate(self, claim: str, max_sources: int = 5) -> Dict[str, Any]:
        """
        Quick validation using the best available sources.
        Returns simplified results for easy consumption.
        
        Args:
            claim: Statement to validate
            max_sources: Maximum number of sources to check
            
        Returns:
            Dict with confidence, supporting_sources, and summary
        """
        # Create cache key
        cache_key = f"{claim}_{max_sources}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Select best sources for this claim
        selected_sources = self._select_best_sources(claim, max_sources)
        
        # Simple validation results
        results = {
            'claim': claim,
            'confidence': 0.0,
            'supporting_sources': [],
            'available_sources': len(selected_sources),
            'categories_covered': [],
            'summary': '',
            'details': {},
            'timestamp': time.time()
        }
        
        if not selected_sources:
            results['summary'] = "❌ No suitable validation sources available for this claim"
            return results
        
        # Collect categories and build summary
        all_categories = set()
        source_names = []
        
        for source_id, source_info in selected_sources.items():
            source_names.append(source_info['name'])
            all_categories.update(source_info['categories'])
            results['supporting_sources'].append({
                'name': source_info['name'],
                'description': source_info['description'],
                'confidence': source_info['confidence'],
                'best_for': source_info['best_for']
            })
        
        results['categories_covered'] = list(all_categories)
        
        # Calculate aggregate confidence based on source quality and coverage
        avg_confidence = sum(s['confidence'] for s in selected_sources.values()) / len(selected_sources)
        coverage_bonus = min(0.1, len(all_categories) * 0.02)  # Bonus for category diversity
        
        results['confidence'] = min(0.90, avg_confidence + coverage_bonus)  # Cap at 90%
        
        # Build summary
        results['summary'] = f"""
✅ VALIDATION AVAILABLE
Sources: {', '.join(source_names)}
Categories: {', '.join(sorted(all_categories))}
Confidence: {results['confidence']*100:.1f}%

Best sources for validation:
{chr(10).join(f"• {s['name']}: {s['best_for']}" for s in results['supporting_sources'][:3])}
"""
        
        # Cache result
        self.validation_cache[cache_key] = results
        return results
    
    def _select_best_sources(self, claim: str, max_sources: int) -> Dict[str, Dict]:
        """Select the best sources for validating a specific claim"""
        claim_lower = claim.lower()
        
        # Score sources based on relevance to claim
        scored_sources = []
        
        for source_id, source_info in self.available_sources.items():
            score = 0.0
            
            # Base confidence score
            score += source_info['confidence']
            
            # Category relevance scoring
            for category in source_info['categories']:
                if self._claim_matches_category(claim_lower, category):
                    score += 0.2
            
            # Specific keyword matching
            if any(keyword in claim_lower for keyword in self._get_source_keywords(source_id)):
                score += 0.3
            
            # Penalty for auth required (less frictionless)
            if source_info['auth_required']:
                score -= 0.1
            
            # Penalty for strict rate limits
            if 'strict' in source_info['rate_limit']:
                score -= 0.2
            
            scored_sources.append((score, source_id, source_info))
        
        # Sort by score and take top sources
        scored_sources.sort(key=lambda x: x[0], reverse=True)
        
        selected = {}
        for score, source_id, source_info in scored_sources[:max_sources]:
            if score > 0.5:  # Only include reasonably relevant sources
                selected[source_id] = source_info
        
        return selected
    
    def _claim_matches_category(self, claim_lower: str, category: str) -> bool:
        """Check if claim matches a category"""
        category_keywords = {
            'proteins': ['protein', 'enzyme', 'receptor', 'antibody', 'peptide'],
            'sequences': ['sequence', 'dna', 'rna', 'gene', 'genome', 'mutation'],
            'biology': ['cell', 'organism', 'species', 'evolution', 'biological'],
            'structures': ['structure', 'fold', 'domain', 'conformation', '3d'],
            'chemistry': ['chemical', 'compound', 'molecule', 'drug', 'reaction'],
            'materials': ['material', 'crystal', 'metal', 'semiconductor', 'alloy'],
            'physics': ['quantum', 'energy', 'force', 'particle', 'wave'],
            'machine_learning': ['ml', 'ai', 'model', 'training', 'neural', 'algorithm'],
            'literature': ['paper', 'study', 'research', 'publication', 'citation'],
            'data': ['dataset', 'data', 'experiment', 'measurement', 'analysis']
        }
        
        keywords = category_keywords.get(category, [])
        return any(keyword in claim_lower for keyword in keywords)
    
    def _get_source_keywords(self, source_id: str) -> List[str]:
        """Get specific keywords that indicate a source is relevant"""
        source_keywords = {
            'uniprot': ['uniprot', 'protein', 'enzyme', 'receptor', 'accession'],
            'rcsb_pdb': ['pdb', 'structure', 'crystal', 'x-ray', 'nmr'],
            'alphafold': ['alphafold', 'predicted', 'structure', 'deepmind'],
            'blast': ['blast', 'homolog', 'similarity', 'alignment', 'sequence'],
            'ensembl': ['ensembl', 'gene', 'genome', 'transcript', 'variant'],
            'ncbi_eutilities': ['ncbi', 'pubmed', 'genbank', 'literature'],
            'pubchem': ['pubchem', 'compound', 'chemical', 'drug', 'molecule'],
            'materials_project': ['materials project', 'dft', 'electronic', 'band gap'],
            'oqmd': ['oqmd', 'thermodynamic', 'stability', 'phase'],
            'openml': ['openml', 'dataset', 'benchmark', 'machine learning'],
            'kaggle': ['kaggle', 'competition', 'dataset', 'notebook'],
            'arxiv': ['arxiv', 'preprint', 'paper', 'research'],
            'cdx_server': ['archive', 'wayback', 'web', 'historical', 'url']
        }
        
        return source_keywords.get(source_id, [])
    
    def get_source_recommendations(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get recommendations for the best sources for a specific topic.
        
        Args:
            topic: Research topic or domain
            
        Returns:
            List of recommended sources with explanations
        """
        recommendations = []
        
        # Score all sources for this topic
        scored = []
        for source_id, source_info in self.available_sources.items():
            score = 0.0
            
            # Check category relevance
            for category in source_info['categories']:
                if self._claim_matches_category(topic.lower(), category):
                    score += 0.3
            
            # Check keyword relevance
            if any(keyword in topic.lower() for keyword in self._get_source_keywords(source_id)):
                score += 0.4
            
            # Add base confidence
            score += source_info['confidence'] * 0.3
            
            if score > 0.2:
                scored.append((score, source_id, source_info))
        
        # Sort and format recommendations
        scored.sort(key=lambda x: x[0], reverse=True)
        
        for score, source_id, source_info in scored[:8]:  # Top 8 recommendations
            recommendations.append({
                'source': source_info['name'],
                'description': source_info['description'],
                'relevance_score': f"{score*100:.0f}%",
                'best_for': source_info['best_for'],
                'auth_required': source_info['auth_required'],
                'rate_limit': source_info['rate_limit'],
                'confidence': f"{source_info['confidence']*100:.0f}%"
            })
        
        return recommendations
    
    def get_all_sources_summary(self) -> Dict[str, Any]:
        """Get a summary of all available validation sources"""
        summary = {
            'total_sources': len(self.available_sources),
            'by_category': {},
            'by_confidence': {},
            'auth_required': 0,
            'no_auth': 0,
            'sources': []
        }
        
        # Count by category
        for source_info in self.available_sources.values():
            for category in source_info['categories']:
                summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Count by confidence level
            conf_level = 'high' if source_info['confidence'] > 0.9 else 'medium' if source_info['confidence'] > 0.8 else 'standard'
            summary['by_confidence'][conf_level] = summary['by_confidence'].get(conf_level, 0) + 1
            
            # Count auth requirements
            if source_info['auth_required']:
                summary['auth_required'] += 1
            else:
                summary['no_auth'] += 1
            
            # Add to sources list
            summary['sources'].append({
                'name': source_info['name'],
                'description': source_info['description'],
                'categories': source_info['categories'],
                'confidence': f"{source_info['confidence']*100:.0f}%",
                'auth_required': source_info['auth_required']
            })
        
        return summary
    
    def validate_claim_simple(self, claim: str) -> str:
        """
        Ultra-simple validation that returns just a text summary.
        Perfect for quick agent use.
        
        Args:
            claim: Statement to validate
            
        Returns:
            Simple text summary of validation status
        """
        result = self.quick_validate(claim)
        
        if result['available_sources'] == 0:
            return f"❌ No validation sources available for: {claim}"
        
        confidence_emoji = "✅" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "⚠️"
        
        return f"""{confidence_emoji} VALIDATION: {result['confidence']*100:.0f}% confidence
Sources: {result['available_sources']} available ({', '.join(s['name'] for s in result['supporting_sources'][:3])})
Categories: {', '.join(result['categories_covered'][:5])}
Status: {'High confidence' if result['confidence'] > 0.8 else 'Moderate confidence' if result['confidence'] > 0.6 else 'Low confidence - needs more validation'}"""


# Convenience functions for easy import
def validate(claim: str) -> str:
    """Quick validation function - returns simple text summary"""
    validator = FrictionlessValidator()
    return validator.validate_claim_simple(claim)

def get_sources_for(topic: str) -> List[Dict]:
    """Get recommended sources for a topic"""
    validator = FrictionlessValidator()
    return validator.get_source_recommendations(topic)

def list_all_sources() -> Dict:
    """List all available validation sources"""
    validator = FrictionlessValidator()
    return validator.get_all_sources_summary()


def main():
    """Demo the frictionless validation system"""
    print("🔍 FRICTIONLESS VALIDATION SYSTEM")
    print("=" * 60)
    
    validator = FrictionlessValidator()
    
    # Show available sources
    summary = validator.get_all_sources_summary()
    print(f"\n📊 AVAILABLE SOURCES: {summary['total_sources']}")
    print(f"• No authentication: {summary['no_auth']}")
    print(f"• Authentication required: {summary['auth_required']}")
    print(f"• Categories covered: {', '.join(summary['by_category'].keys())}")
    
    # Test validation
    test_claims = [
        "AlphaFold can predict protein structures with high accuracy",
        "CRISPR-Cas9 can edit genes in human cells",
        "Neural networks use backpropagation for training",
        "Graphene has exceptional electrical conductivity",
        "The brain contains approximately 86 billion neurons"
    ]
    
    print(f"\n🧪 TESTING VALIDATION:")
    print("-" * 40)
    
    for claim in test_claims:
        print(f"\nClaim: {claim}")
        result = validator.validate_claim_simple(claim)
        print(result)
    
    # Show source recommendations
    print(f"\n💡 SOURCE RECOMMENDATIONS:")
    print("-" * 40)
    
    topics = ["protein structure", "machine learning", "neuroscience", "materials science"]
    
    for topic in topics:
        print(f"\nTopic: {topic}")
        recommendations = validator.get_source_recommendations(topic)
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec['source']}: {rec['best_for']} (confidence: {rec['confidence']})")
    
    print(f"\n✅ Frictionless validation system ready!")
    print("Use validate(claim) for quick validation")
    print("Use get_sources_for(topic) for source recommendations")


if __name__ == "__main__":
    main()
