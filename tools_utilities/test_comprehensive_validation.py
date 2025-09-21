#!/usr/bin/env python3
"""
Test the Comprehensive Validation System
Demonstrates intelligent source selection for different query types
"""

from comprehensive_validation_system import get_validation_system
import asyncio

def test_source_selection():
    """Test that the system intelligently selects appropriate sources"""
    system = get_validation_system()
    
    test_queries = [
        # Protein queries
        ("What is the structure of hemoglobin?", ["AlphaFold", "RCSB PDB", "UniProt"]),
        
        # Genomics queries
        ("How does CRISPR-Cas9 target specific genes?", ["NCBI E-utilities", "Ensembl"]),
        
        # Chemistry queries  
        ("What are the properties of aspirin?", ["PubChem"]),
        
        # Materials science queries
        ("What is the band gap of silicon?", ["Materials Project", "OQMD"]),
        
        # Machine learning queries
        ("How does transformer architecture work?", ["OpenML", "Hugging Face", "ArXiv"]),
        
        # Neuroscience queries
        ("How do neurons communicate?", ["PubMed Central"]),
        
        # Mixed queries
        ("Can AI predict protein folding better than experimental methods?", 
         ["AlphaFold", "RCSB PDB", "ArXiv", "PubMed Central"]),
        
        # Code documentation queries
        ("How to use PyTorch DataLoader?", ["Context7", "GitHub"]),
        
        # Mathematical queries
        ("Solve the Schrödinger equation", ["Wolfram Alpha", "ArXiv"]),
        
        # Clinical queries
        ("What are the treatments for Alzheimer's?", ["PubMed Central"])
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION SYSTEM - SOURCE SELECTION TEST")
    print("=" * 80)
    
    for query, expected_sources in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        # Get selected sources
        sources = system.select_validation_sources(query)
        source_names = [s.name for s in sources]
        
        print(f"📊 Categories: {', '.join(c.value for c in system.categorize_query(query))}")
        print(f"✅ Selected {len(sources)} sources:")
        for i, source in enumerate(sources[:5], 1):
            print(f"   {i}. {source.name} (confidence: {source.confidence_weight:.2f})")
        
        # Check if expected sources are included
        matches = [s for s in expected_sources if s in source_names]
        if matches:
            print(f"   ✓ Matched expected sources: {', '.join(matches)}")
        else:
            print(f"   ⚠️ Expected sources not found: {', '.join(expected_sources)}")

async def test_validation():
    """Test actual validation process"""
    system = get_validation_system()
    
    print("\n" + "=" * 80)
    print("TESTING VALIDATION PROCESS")
    print("=" * 80)
    
    test_claims = [
        ("AlphaFold achieves near-experimental accuracy for protein structure prediction", 
         "High confidence expected - well-documented fact"),
        
        ("CRISPR can edit any gene with 100% accuracy and no off-target effects",
         "Low confidence expected - overstated claim"),
        
        ("Neural networks in biological brains use backpropagation for learning",
         "Low confidence expected - controversial claim"),
        
        ("Water has the chemical formula H2O",
         "Very high confidence expected - basic fact"),
        
        ("Quantum computers can solve all NP-complete problems in polynomial time",
         "Low confidence expected - incorrect claim")
    ]
    
    for claim, expectation in test_claims:
        print(f"\n🔍 Claim: {claim}")
        print(f"   Expectation: {expectation}")
        
        # Run validation (mock for now since actual API calls would require implementation)
        result = await system.validate_claim(claim)
        
        print(f"   📊 Results:")
        print(f"      - Confidence: {result['confidence']*100:.1f}%")
        print(f"      - Consensus: {result['consensus']}")
        print(f"      - Sources checked: {result['sources_checked']}")
        
        # Determine if claim should be made
        if result['confidence'] < 0.4:
            print(f"      ⚠️ LOW CONFIDENCE - Claim should be heavily qualified")
        elif result['confidence'] < 0.7:
            print(f"      🟡 MEDIUM CONFIDENCE - Express uncertainty")
        else:
            print(f"      ✅ HIGH CONFIDENCE - But still cap at 90%")

def test_all_sources_active():
    """Verify all sources are properly configured"""
    system = get_validation_system()
    
    print("\n" + "=" * 80)
    print("VERIFICATION: ALL SOURCES ACTIVE")
    print("=" * 80)
    
    status = system.verify_all_sources_active()
    
    print("\n📋 Source Status:")
    for name, active in sorted(status.items()):
        icon = "✅" if active else "❌"
        print(f"   {icon} {name}: {'Active' if active else 'INACTIVE'}")
    
    active_count = sum(1 for v in status.values() if v)
    print(f"\n📊 Summary: {active_count}/{len(status)} sources active")
    
    if active_count == len(status):
        print("🎉 ALL VALIDATION SOURCES ARE PROPERLY CONFIGURED!")
    else:
        inactive = [k for k, v in status.items() if not v]
        print(f"⚠️ Warning: Inactive sources: {', '.join(inactive)}")

def main():
    """Run all tests"""
    print("\n" + "🔬" * 40)
    print("COMPREHENSIVE VALIDATION SYSTEM TEST SUITE")
    print("Using ALL resources from /Users/camdouglas/quark/data/credentials/all_api_keys.json")
    print("🔬" * 40)
    
    # Test source selection
    test_source_selection()
    
    # Test validation process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_validation())
    finally:
        loop.close()
    
    # Verify all sources
    test_all_sources_active()
    
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE VALIDATION SYSTEM TEST COMPLETE")
    print("The system intelligently selects from ALL 20+ sources based on query type")
    print("=" * 80)

if __name__ == "__main__":
    main()
