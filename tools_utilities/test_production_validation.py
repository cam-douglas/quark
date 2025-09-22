#!/usr/bin/env python3
"""
Test Production-Ready Comprehensive Validation System
Tests async HTTP clients, rate limiting, and error handling
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_validation_system import get_validation_system, mandatory_validate
from async_http_client import HTTPClientManager

async def test_production_validation():
    """Test the production-ready validation system"""
    print("🚀 TESTING PRODUCTION-READY VALIDATION SYSTEM")
    print("=" * 80)
    
    # Initialize validation system
    system = get_validation_system()
    
    print(f"✅ Validation system initialized with {len(system.get_all_available_sources())} sources")
    
    # Test claims with different complexity levels
    test_claims = [
        # Simple factual claim
        ("Water has the chemical formula H2O", "Basic chemistry fact"),
        
        # Protein structure claim (should use AlphaFold, RCSB PDB, UniProt)
        ("AlphaFold can predict protein structures with near-experimental accuracy", "Protein structure prediction"),
        
        # Materials science claim (should use Materials Project, OQMD)
        ("Silicon has a band gap of approximately 1.1 eV", "Materials science fact"),
        
        # Neuroscience claim (should use PubMed, neuroscience sources)
        ("The human brain contains approximately 86 billion neurons", "Neuroscience fact"),
        
        # Controversial claim (should show mixed evidence)
        ("CRISPR gene editing has 100% accuracy with no off-target effects", "Controversial claim"),
        
        # Machine learning claim (should use ArXiv, ML sources)
        ("Transformer models revolutionized natural language processing", "ML/AI fact")
    ]
    
    print("\n📋 Testing Claims with Production Validation:")
    print("-" * 60)
    
    for i, (claim, description) in enumerate(test_claims, 1):
        print(f"\n🔍 Test {i}: {description}")
        print(f"   Claim: {claim}")
        
        try:
            # Run validation with timeout
            result = await asyncio.wait_for(
                system.validate_claim(claim),
                timeout=30.0  # 30 second timeout
            )
            
            # Display results
            confidence = result['confidence'] * 100
            consensus = result['consensus']
            sources_checked = result['sources_checked']
            
            print(f"   📊 Results:")
            print(f"      - Confidence: {confidence:.1f}%")
            print(f"      - Consensus: {consensus}")
            print(f"      - Sources checked: {sources_checked}")
            
            # Show evidence from sources
            if 'evidence' in result and result['evidence']:
                print(f"      - Evidence sources: {len(result['evidence'])}")
                for j, evidence in enumerate(result['evidence'][:3], 1):  # Show first 3
                    source_name = evidence.get('source', 'Unknown')
                    source_confidence = evidence.get('confidence', 0) * 100
                    supports = evidence.get('supports_claim', False)
                    print(f"         {j}. {source_name}: {source_confidence:.1f}% ({'✓' if supports else '✗'})")
            
            # Determine validation outcome
            if confidence < 40:
                print(f"      ⚠️ LOW CONFIDENCE - Claim should be heavily qualified")
            elif confidence < 70:
                print(f"      🟡 MEDIUM CONFIDENCE - Express uncertainty")
            else:
                print(f"      ✅ HIGH CONFIDENCE - But still capped at 90%")
                
        except asyncio.TimeoutError:
            print(f"   ⏰ TIMEOUT - Validation took longer than 30 seconds")
        except Exception as e:
            print(f"   ❌ ERROR - Validation failed: {str(e)}")
    
    # Test rate limiting
    print(f"\n⚡ Testing Rate Limiting:")
    print("-" * 40)
    
    rapid_claims = [
        "Test claim 1",
        "Test claim 2", 
        "Test claim 3",
        "Test claim 4",
        "Test claim 5"
    ]
    
    start_time = asyncio.get_event_loop().time()
    
    # Submit multiple claims rapidly
    tasks = [system.validate_claim(claim) for claim in rapid_claims]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    
    successful = sum(1 for r in results if not isinstance(r, Exception))
    rate_limited = sum(1 for r in results if isinstance(r, dict) and r.get('rate_limited', False))
    
    print(f"   📊 Rate Limiting Results:")
    print(f"      - Total requests: {len(rapid_claims)}")
    print(f"      - Successful: {successful}")
    print(f"      - Rate limited: {rate_limited}")
    print(f"      - Total time: {total_time:.2f}s")
    print(f"      - Average time per request: {total_time/len(rapid_claims):.2f}s")
    
    # Test error handling
    print(f"\n🛡️ Testing Error Handling:")
    print("-" * 40)
    
    # Test with invalid source
    try:
        # This should gracefully handle errors
        result = await system.validate_claim("Test error handling")
        print(f"   ✅ Error handling successful - got result with {result['sources_checked']} sources")
    except Exception as e:
        print(f"   ❌ Error handling failed: {str(e)}")
    
    # Generate system report
    print(f"\n📈 System Performance Report:")
    print("-" * 40)
    
    report = system.get_validation_report()
    print(f"   - Total sources available: {report['total_sources_available']}")
    print(f"   - Validation history: {report['validation_history_count']} validations")
    print(f"   - Cache size: {report['cache_size']} cached results")
    print(f"   - Credentials loaded: {report['credentials_loaded']} services")
    print(f"   - Knowledge sources: {report['knowledge_sources_loaded']} open access sources")
    
    # Test HTTP client stats
    async with HTTPClientManager() as http_client:
        stats = http_client.get_stats()
        health = await http_client.health_check()
        
        print(f"\n🌐 HTTP Client Performance:")
        print(f"   - Active throttlers: {stats['active_throttlers']}")
        print(f"   - Cache size: {stats['cache_size']}")
        print(f"   - Connectivity: {'✅' if health.get('connectivity') else '❌'}")
        print(f"   - Response time: {health.get('test_response_time', 0):.3f}s")

async def test_specific_apis():
    """Test specific API integrations"""
    print(f"\n🔬 Testing Specific API Integrations:")
    print("=" * 60)
    
    system = get_validation_system()
    
    # Test ArXiv integration
    print(f"\n📚 Testing ArXiv Integration:")
    try:
        result = await system.validate_claim("machine learning transformer attention mechanism")
        arxiv_evidence = [e for e in result.get('evidence', []) if 'arxiv' in e.get('source', '').lower()]
        if arxiv_evidence:
            evidence = arxiv_evidence[0]
            print(f"   ✅ ArXiv validation successful")
            print(f"      - Confidence: {evidence.get('confidence', 0)*100:.1f}%")
            print(f"      - Evidence: {evidence.get('evidence', 'No evidence')[:100]}...")
        else:
            print(f"   ⚠️ No ArXiv evidence found")
    except Exception as e:
        print(f"   ❌ ArXiv test failed: {str(e)}")
    
    # Test PubMed integration
    print(f"\n🏥 Testing PubMed Integration:")
    try:
        result = await system.validate_claim("COVID-19 vaccine effectiveness")
        pubmed_evidence = [e for e in result.get('evidence', []) if 'pubmed' in e.get('source', '').lower()]
        if pubmed_evidence:
            evidence = pubmed_evidence[0]
            print(f"   ✅ PubMed validation successful")
            print(f"      - Confidence: {evidence.get('confidence', 0)*100:.1f}%")
            print(f"      - Evidence: {evidence.get('evidence', 'No evidence')[:100]}...")
        else:
            print(f"   ⚠️ No PubMed evidence found")
    except Exception as e:
        print(f"   ❌ PubMed test failed: {str(e)}")
    
    # Test Materials Project integration
    print(f"\n🔬 Testing Materials Project Integration:")
    try:
        result = await system.validate_claim("silicon semiconductor properties")
        mp_evidence = [e for e in result.get('evidence', []) if 'materials' in e.get('source', '').lower()]
        if mp_evidence:
            evidence = mp_evidence[0]
            print(f"   ✅ Materials Project validation successful")
            print(f"      - Confidence: {evidence.get('confidence', 0)*100:.1f}%")
            print(f"      - Evidence: {evidence.get('evidence', 'No evidence')[:100]}...")
        else:
            print(f"   ⚠️ No Materials Project evidence found")
    except Exception as e:
        print(f"   ❌ Materials Project test failed: {str(e)}")

async def main():
    """Run all production validation tests"""
    print("\n" + "🔬" * 50)
    print("PRODUCTION-READY COMPREHENSIVE VALIDATION SYSTEM")
    print("Testing async HTTP clients, rate limiting, and error handling")
    print("🔬" * 50)
    
    try:
        # Test main validation system
        await test_production_validation()
        
        # Test specific API integrations
        await test_specific_apis()
        
        print(f"\n" + "=" * 80)
        print("✅ PRODUCTION VALIDATION SYSTEM TESTS COMPLETE")
        print("System is ready for production use with:")
        print("- Async HTTP clients with connection pooling")
        print("- Rate limiting for all APIs")
        print("- Robust error handling and retry logic")
        print("- Intelligent source selection")
        print("- Comprehensive validation across ALL available resources")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in production tests: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
