#!/usr/bin/env python3
"""
Test script to verify enhanced validation system integration
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from confidence_validator import ConfidenceValidator, ValidationCategory


def test_resource_loading():
    """Test that all resources are loaded from credentials"""
    print("\n" + "="*60)
    print("TESTING RESOURCE LOADING")
    print("="*60)
    
    validator = ConfidenceValidator()
    
    print(f"\n✅ Credentials loaded from: {validator.credentials_path}")
    print(f"✅ Total resources initialized: {len(validator.resources)}")
    
    # Count by type
    mcp_count = sum(1 for r in validator.resources.values() 
                    if r.resource_type.value == 'mcp_server')
    api_count = sum(1 for r in validator.resources.values() 
                   if r.resource_type.value == 'api')
    
    print(f"  - MCP Servers: {mcp_count}")
    print(f"  - APIs: {api_count}")
    
    return len(validator.resources) > 0


def test_category_detection():
    """Test validation category detection"""
    print("\n" + "="*60)
    print("TESTING CATEGORY DETECTION")
    print("="*60)
    
    validator = ConfidenceValidator()
    
    test_cases = [
        ("AlphaFold predicts protein structures", 
         [ValidationCategory.PROTEIN_STRUCTURE]),
        ("Machine learning model training on datasets",
         [ValidationCategory.MACHINE_LEARNING]),
        ("DNA sequence alignment using BLAST",
         [ValidationCategory.BIOLOGICAL_SEQUENCE, ValidationCategory.GENOMIC_DATA]),
        ("Chemical compound SMILES notation",
         [ValidationCategory.CHEMICAL_COMPOUND]),
        ("arXiv paper on quantum computing",
         [ValidationCategory.ARXIV_PAPER, ValidationCategory.SCIENTIFIC_LITERATURE])
    ]
    
    all_passed = True
    for text, expected_categories in test_cases:
        detected = validator.detect_validation_needs(text)
        
        # Check if expected categories are in detected
        matched = all(cat in detected for cat in expected_categories)
        
        status = "✅" if matched else "❌"
        print(f"\n{status} Text: '{text[:50]}...'")
        print(f"   Expected: {[c.value for c in expected_categories]}")
        print(f"   Detected: {[c.value for c in detected]}")
        
        if not matched:
            all_passed = False
    
    return all_passed


def test_resource_selection():
    """Test resource selection for validation"""
    print("\n" + "="*60)
    print("TESTING RESOURCE SELECTION")
    print("="*60)
    
    validator = ConfidenceValidator()
    
    # Test protein validation
    categories = [ValidationCategory.PROTEIN_STRUCTURE]
    resources = validator.select_best_resources(categories, max_resources=3)
    
    print(f"\nProtein structure validation:")
    print(f"  Selected {len(resources)} resources:")
    for resource in resources:
        print(f"    - {resource.name} ({resource.resource_type.value})")
    
    # Test ML validation
    categories = [ValidationCategory.MACHINE_LEARNING]
    resources = validator.select_best_resources(categories, max_resources=3)
    
    print(f"\nMachine learning validation:")
    print(f"  Selected {len(resources)} resources:")
    for resource in resources:
        print(f"    - {resource.name} ({resource.resource_type.value})")
    
    return len(resources) > 0


def test_validation_plan_generation():
    """Test generation of validation plans"""
    print("\n" + "="*60)
    print("TESTING VALIDATION PLAN GENERATION")
    print("="*60)
    
    validator = ConfidenceValidator()
    
    context = "The BRCA1 gene encodes a protein involved in DNA repair"
    
    # Get validation plan
    plan = validator.perform_enhanced_validation(context)
    
    print(f"\nContext: '{context}'")
    print(f"\nValidation Plan Generated:")
    print(f"  Categories detected: {plan['categories']}")
    print(f"  Resources selected: {len(plan['resources_selected'])}")
    
    for resource in plan['resources_selected']:
        print(f"    - {resource['name']} ({resource['type']})")
    
    print(f"\n  Validation instructions:")
    for i, instruction in enumerate(plan['validation_instructions'], 1):
        print(f"    {i}. {instruction}")
    
    print(f"\n  Confidence adjustment: +{plan['confidence_adjustment']:.0%}")
    
    return len(plan['resources_selected']) > 0


def test_checklist_generation():
    """Test validation checklist generation"""
    print("\n" + "="*60)
    print("TESTING CHECKLIST GENERATION")
    print("="*60)
    
    validator = ConfidenceValidator()
    checklist = validator.generate_validation_checklist()
    
    print("\nGenerated checklist preview:")
    for line in checklist[:10]:
        print(line)
    print("...")
    print(f"\nTotal checklist items: {len(checklist)}")
    
    return len(checklist) > 0


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ENHANCED VALIDATION SYSTEM TEST SUITE")
    print("="*60)
    
    tests = [
        ("Resource Loading", test_resource_loading),
        ("Category Detection", test_category_detection),
        ("Resource Selection", test_resource_selection),
        ("Validation Plan Generation", test_validation_plan_generation),
        ("Checklist Generation", test_checklist_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Enhanced validation system is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
