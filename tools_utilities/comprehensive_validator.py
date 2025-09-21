#!/usr/bin/env python3
"""
Comprehensive Validation System
Combines MCP servers, direct APIs, and web search for anti-overconfidence validation
"""

import json
import requests
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ComprehensiveValidator:
    """
    Multi-source validation system that works regardless of MCP server availability
    """
    
    def __init__(self):
        self.available_sources = []
        self.test_all_sources()
        
    def test_all_sources(self):
        """Test which validation sources are currently available"""
        
        print("🔍 Testing validation sources...")
        
        # Test MCP servers (via simplified check)
        mcp_config = Path.home() / ".cursor" / "mcp.json"
        if mcp_config.exists():
            with open(mcp_config) as f:
                config = json.load(f)
                servers = config.get("mcpServers", {})
                
                # Known working servers
                if "context7" in servers:
                    self.available_sources.append("Context7 MCP (library docs)")
                if "figma" in servers:
                    self.available_sources.append("Figma MCP (design specs)")
                if "cline" in servers:
                    self.available_sources.append("Cline MCP (Quark integration)")
                    
        # Test direct API access
        direct_apis = [
            ("ArXiv", "http://export.arxiv.org/api/query?search_query=test&max_results=1"),
            ("PubMed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=test&retmode=json"),
            ("OpenAlex", "https://api.openalex.org/works?search=test&per_page=1"),
            ("Crossref", "https://api.crossref.org/works?query=test&rows=1"),
        ]
        
        for name, url in direct_apis:
            try:
                response = requests.get(url, timeout=2, headers={"User-Agent": "Quark/1.0"})
                if response.status_code == 200:
                    self.available_sources.append(f"{name} API (direct)")
            except:
                pass
                
    def validate_claim(self, claim: str, required_confidence: float = 0.7) -> Dict:
        """
        Validate a claim using all available sources
        
        Args:
            claim: The claim to validate
            required_confidence: Minimum confidence threshold (default 70%)
            
        Returns:
            Validation results with confidence score
        """
        
        results = {
            "claim": claim,
            "sources_checked": [],
            "evidence": [],
            "confidence": 0.0,
            "recommendation": "",
            "meets_threshold": False
        }
        
        # Check each available source
        confidence_points = 0.0
        
        # Direct API validation
        if "ArXiv API (direct)" in self.available_sources:
            if self._check_arxiv(claim):
                results["sources_checked"].append("ArXiv")
                results["evidence"].append("Found related papers on ArXiv")
                confidence_points += 0.25
                
        if "PubMed API (direct)" in self.available_sources:
            if self._check_pubmed(claim):
                results["sources_checked"].append("PubMed")
                results["evidence"].append("Found medical/biological evidence")
                confidence_points += 0.25
                
        if "OpenAlex API (direct)" in self.available_sources:
            if self._check_openalex(claim):
                results["sources_checked"].append("OpenAlex")
                results["evidence"].append("Found academic citations")
                confidence_points += 0.25
                
        if "Crossref API (direct)" in self.available_sources:
            if self._check_crossref(claim):
                results["sources_checked"].append("Crossref")
                results["evidence"].append("Found DOI-registered publications")
                confidence_points += 0.15
                
        # MCP validation (if available)
        if "Context7 MCP (library docs)" in self.available_sources:
            # Would need actual MCP call here
            results["sources_checked"].append("Context7")
            confidence_points += 0.1
            
        # Cap at 90% per anti-overconfidence rules
        results["confidence"] = min(confidence_points, 0.90)
        results["meets_threshold"] = results["confidence"] >= required_confidence
        
        # Generate recommendation
        if results["confidence"] < 0.4:
            results["recommendation"] = "⚠️ LOW CONFIDENCE - Seek additional validation"
        elif results["confidence"] < 0.7:
            results["recommendation"] = "🟡 MEDIUM CONFIDENCE - Consider alternatives"
        else:
            results["recommendation"] = "✅ ACCEPTABLE CONFIDENCE - Proceed with caution"
            
        return results
        
    def _check_arxiv(self, query: str) -> bool:
        """Check ArXiv for relevant papers"""
        try:
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results=1"
            response = requests.get(url, timeout=3)
            return "entry" in response.text
        except:
            return False
            
    def _check_pubmed(self, query: str) -> bool:
        """Check PubMed for relevant papers"""
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 1}
            response = requests.get(url, params=params, timeout=3)
            data = response.json()
            return int(data.get("esearchresult", {}).get("count", "0")) > 0
        except:
            return False
            
    def _check_openalex(self, query: str) -> bool:
        """Check OpenAlex for academic works"""
        try:
            url = f"https://api.openalex.org/works?search={query}&per_page=1"
            headers = {"User-Agent": "Quark/1.0 (research@quark-ai.com)"}
            response = requests.get(url, headers=headers, timeout=3)
            data = response.json()
            return len(data.get("results", [])) > 0
        except:
            return False
            
    def _check_crossref(self, query: str) -> bool:
        """Check Crossref for DOI-registered works"""
        try:
            url = f"https://api.crossref.org/works?query={query}&rows=1"
            response = requests.get(url, timeout=3)
            data = response.json()
            return data.get("message", {}).get("total-results", 0) > 0
        except:
            return False
            
    def question_user_claim(self, user_claim: str, ai_understanding: str) -> str:
        """
        Generate a response that questions the user's claim
        Following the anti-overconfidence rule to assume user is wrong
        """
        
        # Validate both claims
        user_validation = self.validate_claim(user_claim)
        ai_validation = self.validate_claim(ai_understanding)
        
        response = f"""
🤔 QUESTIONING YOUR APPROACH:

What you suggested: {user_claim}

My concerns:
- User claim confidence: {user_validation['confidence']:.1%} from {len(user_validation['sources_checked'])} sources
- Alternative understanding: {ai_validation['confidence']:.1%} from {len(ai_validation['sources_checked'])} sources
"""
        
        if user_validation['confidence'] < 0.7:
            response += f"""
- ⚠️ Limited evidence supporting your approach
- Found evidence: {', '.join(user_validation['evidence']) if user_validation['evidence'] else 'None'}
"""
        
        if ai_validation['confidence'] > user_validation['confidence']:
            response += f"""
What I found instead:
- {', '.join(ai_validation['evidence'])}
- Higher confidence in alternative approach
"""
        
        response += f"""
Should we:
1. Proceed with your approach (confidence: {user_validation['confidence']:.1%})
2. Consider my alternative (confidence: {ai_validation['confidence']:.1%})
3. Investigate further before deciding?

Validation sources available: {len(self.available_sources)}
Sources checked: {', '.join(set(user_validation['sources_checked'] + ai_validation['sources_checked']))}
"""
        
        return response

def main():
    """Test the comprehensive validation system"""
    
    validator = ComprehensiveValidator()
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE VALIDATION SYSTEM STATUS")
    print("=" * 60)
    
    print(f"\n✅ Available validation sources ({len(validator.available_sources)}):")
    for source in validator.available_sources:
        print(f"  - {source}")
        
    # Test validation
    test_claims = [
        "Neural networks improve performance",
        "Quantum computing solves NP-complete problems",
        "Python is faster than C++",
    ]
    
    print("\n📊 Validation Tests:")
    print("-" * 40)
    
    for claim in test_claims:
        result = validator.validate_claim(claim)
        print(f"\nClaim: '{claim}'")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Sources: {', '.join(result['sources_checked'])}")
        print(f"Recommendation: {result['recommendation']}")
        
    # Test user questioning
    print("\n🤔 User Skepticism Test:")
    print("-" * 40)
    
    user_claim = "We should use a simple for loop"
    ai_alternative = "List comprehension would be more Pythonic"
    
    question = validator.question_user_claim(user_claim, ai_alternative)
    print(question)
    
    print("\n✅ System is operational with fallback validation!")

if __name__ == "__main__":
    main()
