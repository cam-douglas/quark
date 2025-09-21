#!/usr/bin/env python3
"""
Test anti-overconfidence validation with open access sources
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from confidence_validator import ConfidenceValidator, ValidationCategory

def test_anti_overconfidence():
    """Test the anti-overconfidence validation system"""
    print("\n" + "="*60)
    print("ANTI-OVERCONFIDENCE VALIDATION TEST")
    print("="*60)
    
    validator = ConfidenceValidator()
    
    # Test Case 1: Scientific claim without user statement
    claim1 = "CRISPR-Cas9 can precisely edit DNA sequences with 100% accuracy"
    print(f"\n📝 Testing claim: '{claim1}'")
    
    result1 = validator.perform_anti_overconfidence_validation(claim1)
    print(f"\nValidation Result:")
    print(f"  Final Confidence: {result1['final_confidence']:.1%}")
    print(f"  Validated: {result1['validated']}")
    print(f"  Uncertainty Level: {result1['uncertainty_level']}")
    print(f"  Sources Consulted: {len(result1['sources_consulted'])}")
    
    print("\nSources Selected:")
    for source in result1['sources_consulted'][:5]:
        print(f"  • {source['name']} ({source['authority_level']})")
    
    print("\nUncertainty Triggers:")
    for trigger in result1['uncertainty_triggers']:
        print(f"  ⚠️ {trigger}")
    
    print("\nValidation Gaps:")
    for gap in result1['validation_gaps']:
        print(f"  ❌ {gap}")
    
    # Test Case 2: User statement with absolute claims
    user_statement = "AlphaFold always predicts protein structures perfectly"
    print(f"\n📝 Testing user statement: '{user_statement}'")
    
    result2 = validator.perform_anti_overconfidence_validation(
        claim="AlphaFold protein structure prediction",
        user_statement=user_statement
    )
    
    print(f"\nValidation Result:")
    print(f"  Final Confidence: {result2['final_confidence']:.1%}")
    print(f"  User Correction Needed: {result2['user_correction_needed']}")
    print(f"  Uncertainty Level: {result2['uncertainty_level']}")
    
    # Print the anti-overconfidence report
    print("\n" + "="*60)
    print("ANTI-OVERCONFIDENCE REPORT:")
    print("="*60)
    print(result2['anti_overconfidence_report'])
    
    # Test Case 3: Check open access sources are being used
    print("\n" + "="*60)
    print("OPEN ACCESS SOURCES CHECK:")
    print("="*60)
    
    # Count all open access sources
    open_access_count = sum(1 for name in validator.resources.keys() if 'openaccess' in name)
    print(f"✅ Total open access sources loaded: {open_access_count}")
    
    # Show some examples - both by name prefix and source type
    print("\nExample Open Access Sources (by key):")
    count = 0
    for name, resource in validator.resources.items():
        if 'openaccess' in name and count < 5:
            print(f"  • {resource.name}: {resource.config.get('url', 'N/A')}")
            print(f"    Type: {resource.config.get('source_type', 'unknown')}")
            print(f"    Status: {resource.config.get('test_status', 'unknown')}")
            count += 1
    
    # Also show resources that are actually preprint/academic sources
    print("\nAll Literature Sources Available:")
    literature_sources = [r for r in validator.resources.values() 
                         if ValidationCategory.SCIENTIFIC_LITERATURE in r.categories]
    print(f"Total literature sources: {len(literature_sources)}")
    for resource in literature_sources[:10]:
        prefix = "🔓 OPEN ACCESS" if 'openaccess' in resource.name.lower() else "📚"
        print(f"  {prefix} {resource.name}")

if __name__ == "__main__":
    test_anti_overconfidence()
