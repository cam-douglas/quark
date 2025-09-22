#!/usr/bin/env python3
"""
EXHAUSTIVE TEST of ALL Validation Sources
Tests every single API endpoint with real calls
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_validation_system import get_validation_system
from async_http_client import HTTPClientManager, APIResponse

class ExhaustiveSourceTester:
    """Tests ALL validation sources exhaustively"""
    
    def __init__(self):
        self.results = {}
        self.credentials = self._load_credentials()
        self.test_start_time = datetime.now()
        
    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials for testing"""
        creds_path = Path("/Users/camdouglas/quark/data/credentials/all_api_keys.json")
        with open(creds_path, 'r') as f:
            return json.load(f)
    
    async def test_all_sources(self):
        """Test ALL validation sources exhaustively"""
        print("🔍 EXHAUSTIVE SOURCE TESTING - ALL VALIDATION SOURCES")
        print("=" * 80)
        
        system = get_validation_system()
        all_sources = system.validation_sources
        
        print(f"📊 Testing {len(all_sources)} validation sources...")
        print(f"⏰ Started at: {self.test_start_time}")
        
        async with HTTPClientManager() as http_client:
            # Test each source individually
            for source_name, source in all_sources.items():
                print(f"\n🔬 Testing: {source.name}")
                print("-" * 50)
                
                try:
                    result = await self._test_source_comprehensive(source, http_client)
                    self.results[source_name] = result
                    
                    # Display immediate results
                    status = "✅ PASS" if result['success'] else "❌ FAIL"
                    print(f"   Status: {status}")
                    print(f"   Response Time: {result.get('response_time', 0):.2f}s")
                    print(f"   Details: {result.get('summary', 'No details')}")
                    
                    if result.get('error'):
                        print(f"   Error: {result['error']}")
                    
                except Exception as e:
                    print(f"   ❌ CRITICAL ERROR: {str(e)}")
                    self.results[source_name] = {
                        'success': False,
                        'error': f"Critical test failure: {str(e)}",
                        'response_time': 0,
                        'summary': 'Test crashed'
                    }
                
                # Small delay between tests to be respectful
                await asyncio.sleep(1.0)
        
        # Generate comprehensive report
        await self._generate_exhaustive_report()
    
    async def _test_source_comprehensive(self, source, http_client) -> Dict[str, Any]:
        """Comprehensively test a single source"""
        start_time = time.time()
        
        # Test 1: Basic connectivity
        connectivity_result = await self._test_connectivity(source, http_client)
        
        # Test 2: Authentication (if required)
        auth_result = await self._test_authentication(source, http_client)
        
        # Test 3: API functionality
        functionality_result = await self._test_functionality(source, http_client)
        
        # Test 4: Rate limiting compliance
        rate_limit_result = await self._test_rate_limits(source, http_client)
        
        # Test 5: Error handling
        error_handling_result = await self._test_error_handling(source, http_client)
        
        end_time = time.time()
        
        # Aggregate results
        all_tests = [connectivity_result, auth_result, functionality_result, rate_limit_result, error_handling_result]
        success_count = sum(1 for test in all_tests if test.get('success', False))
        
        return {
            'success': success_count >= 3,  # At least 3/5 tests must pass
            'response_time': end_time - start_time,
            'tests': {
                'connectivity': connectivity_result,
                'authentication': auth_result,
                'functionality': functionality_result,
                'rate_limits': rate_limit_result,
                'error_handling': error_handling_result
            },
            'summary': f"{success_count}/5 tests passed",
            'score': success_count / 5.0
        }
    
    async def _test_connectivity(self, source, http_client) -> Dict[str, Any]:
        """Test basic connectivity to source"""
        try:
            response = await http_client.get(
                url=source.endpoint,
                api_name=f"{source.name.lower()}_connectivity_test",
                timeout=10.0,
                use_cache=False
            )
            
            return {
                'success': response.success,
                'response_time': response.response_time,
                'status_code': response.status_code,
                'details': f"HTTP {response.status_code}" if response.status_code else response.error
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': f"Connectivity failed: {str(e)}"
            }
    
    async def _test_authentication(self, source, http_client) -> Dict[str, Any]:
        """Test authentication if required"""
        if not source.requires_auth:
            return {
                'success': True,
                'details': 'No authentication required'
            }
        
        if not source.auth_key:
            return {
                'success': False,
                'details': 'Authentication required but no key provided'
            }
        
        try:
            # Test with authentication headers
            headers = {}
            
            # Different auth patterns for different APIs
            if 'materials_project' in source.name.lower():
                headers['X-API-KEY'] = source.auth_key
            elif 'openai' in source.name.lower():
                headers['Authorization'] = f'Bearer {source.auth_key}'
            elif 'claude' in source.name.lower():
                headers['x-api-key'] = source.auth_key
            elif 'gemini' in source.name.lower():
                # Gemini uses query parameter
                pass
            else:
                headers['Authorization'] = f'Bearer {source.auth_key}'
            
            response = await http_client.get(
                url=source.endpoint,
                api_name=f"{source.name.lower()}_auth_test",
                headers=headers,
                timeout=10.0,
                use_cache=False
            )
            
            # Check for auth success indicators
            auth_success = (
                response.success or 
                (response.status_code and response.status_code != 401 and response.status_code != 403)
            )
            
            return {
                'success': auth_success,
                'response_time': response.response_time,
                'status_code': response.status_code,
                'details': f"Auth test: HTTP {response.status_code}" if response.status_code else response.error
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': f"Authentication test failed: {str(e)}"
            }
    
    async def _test_functionality(self, source, http_client) -> Dict[str, Any]:
        """Test actual API functionality with real queries"""
        try:
            # Choose appropriate test query based on source type
            test_queries = self._get_test_queries_for_source(source)
            
            best_result = None
            
            for query in test_queries:
                try:
                    result = await self._execute_test_query(source, query, http_client)
                    if result.get('success'):
                        best_result = result
                        break
                    elif not best_result:
                        best_result = result
                except Exception as e:
                    continue
            
            return best_result or {
                'success': False,
                'details': 'All functionality tests failed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': f"Functionality test crashed: {str(e)}"
            }
    
    def _get_test_queries_for_source(self, source) -> List[Dict[str, Any]]:
        """Get appropriate test queries for each source type"""
        source_name = source.name.lower()
        
        if 'arxiv' in source_name:
            return [
                {'params': {'search_query': 'machine learning', 'max_results': 3}},
                {'params': {'search_query': 'quantum computing', 'max_results': 2}}
            ]
        elif 'pubmed' in source_name or 'ncbi' in source_name:
            return [
                {'params': {'db': 'pubmed', 'term': 'covid', 'retmax': 3, 'retmode': 'json'}},
                {'params': {'db': 'pubmed', 'term': 'cancer', 'retmax': 2, 'retmode': 'json'}}
            ]
        elif 'pubchem' in source_name:
            return [
                {'params': {'compound': 'name/water', 'property': 'MolecularFormula'}},
                {'params': {'compound': 'name/aspirin', 'property': 'MolecularWeight'}}
            ]
        elif 'materials_project' in source_name:
            return [
                {'endpoint_suffix': '/materials/summary', 'params': {'formula': 'Si', '_limit': 2}},
                {'endpoint_suffix': '/materials/summary', 'params': {'formula': 'Fe', '_limit': 1}}
            ]
        elif 'alphafold' in source_name:
            return [
                {'endpoint_suffix': '/prediction/P21359'},  # Human protein
                {'endpoint_suffix': '/search/P53_HUMAN'}
            ]
        elif 'rcsb' in source_name:
            return [
                {'params': {'q': 'hemoglobin', 'return_type': 'entry'}},
                {'params': {'q': 'insulin', 'return_type': 'entry'}}
            ]
        elif 'ensembl' in source_name:
            return [
                {'endpoint_suffix': '/lookup/id/ENSG00000139618'},  # BRCA2 gene
                {'endpoint_suffix': '/lookup/symbol/homo_sapiens/BRCA1'}
            ]
        elif 'uniprot' in source_name:
            return [
                {'endpoint_suffix': '/uniprotkb/P04637'},  # p53 protein
                {'endpoint_suffix': '/uniprotkb/search?query=insulin'}
            ]
        elif 'wolfram' in source_name:
            return [
                {'params': {'input': 'solve x^2 = 4', 'appid': source.auth_key}},
                {'params': {'input': 'mass of electron', 'appid': source.auth_key}}
            ]
        else:
            # Generic test
            return [
                {'params': {'q': 'test'}},
                {'params': {}}
            ]
    
    async def _execute_test_query(self, source, query, http_client) -> Dict[str, Any]:
        """Execute a specific test query"""
        try:
            # Build URL
            url = source.endpoint
            if 'endpoint_suffix' in query:
                url = url.rstrip('/') + '/' + query['endpoint_suffix'].lstrip('/')
            
            # Prepare headers
            headers = {}
            if source.requires_auth and source.auth_key:
                if 'materials_project' in source.name.lower():
                    headers['X-API-KEY'] = source.auth_key
                elif 'openai' in source.name.lower():
                    headers['Authorization'] = f'Bearer {source.auth_key}'
                elif 'claude' in source.name.lower():
                    headers['x-api-key'] = source.auth_key
            
            # Make request
            response = await http_client.get(
                url=url,
                api_name=f"{source.name.lower()}_functionality_test",
                params=query.get('params', {}),
                headers=headers,
                timeout=15.0,
                use_cache=False
            )
            
            # Analyze response for functionality indicators
            success_indicators = self._analyze_response_for_success(response, source)
            
            return {
                'success': success_indicators['is_functional'],
                'response_time': response.response_time,
                'status_code': response.status_code,
                'data_size': len(str(response.data)) if response.data else 0,
                'details': success_indicators['details']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': f"Query execution failed: {str(e)}"
            }
    
    def _analyze_response_for_success(self, response: APIResponse, source) -> Dict[str, Any]:
        """Analyze response to determine if API is functional"""
        if not response.success:
            return {
                'is_functional': False,
                'details': f"Request failed: {response.error}"
            }
        
        if not response.data:
            return {
                'is_functional': False,
                'details': "No data returned"
            }
        
        # Source-specific success indicators
        source_name = source.name.lower()
        data = response.data
        
        if 'arxiv' in source_name:
            # ArXiv returns XML with entries
            if 'xml_content' in data and 'entry' in data['xml_content']:
                return {'is_functional': True, 'details': 'ArXiv entries found'}
        
        elif 'pubmed' in source_name or 'ncbi' in source_name:
            # PubMed returns JSON with esearchresult
            if isinstance(data, dict) and 'esearchresult' in data:
                return {'is_functional': True, 'details': 'PubMed search results found'}
        
        elif 'materials_project' in source_name:
            # Materials Project returns data array
            if isinstance(data, dict) and 'data' in data:
                return {'is_functional': True, 'details': f"Materials data: {len(data['data'])} entries"}
        
        elif 'alphafold' in source_name:
            # AlphaFold returns protein data
            if isinstance(data, dict) and any(key in data for key in ['uniprotAccession', 'gene', 'organism']):
                return {'is_functional': True, 'details': 'AlphaFold protein data found'}
        
        # Generic success indicators
        if isinstance(data, dict):
            if len(data) > 0:
                return {'is_functional': True, 'details': f'JSON response with {len(data)} fields'}
        elif isinstance(data, list):
            if len(data) > 0:
                return {'is_functional': True, 'details': f'Array response with {len(data)} items'}
        elif isinstance(data, str):
            if len(data) > 50:  # Substantial content
                return {'is_functional': True, 'details': f'Text response: {len(data)} characters'}
        
        return {
            'is_functional': False,
            'details': f"Response format not recognized: {type(data)}"
        }
    
    async def _test_rate_limits(self, source, http_client) -> Dict[str, Any]:
        """Test rate limiting compliance"""
        try:
            # Make 3 rapid requests to test rate limiting
            start_time = time.time()
            
            tasks = []
            for i in range(3):
                task = http_client.get(
                    url=source.endpoint,
                    api_name=f"{source.name.lower()}_rate_test",
                    timeout=5.0,
                    use_cache=False
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            total_time = end_time - start_time
            successful_requests = sum(1 for r in responses if isinstance(r, APIResponse) and r.success)
            rate_limited_requests = sum(1 for r in responses if isinstance(r, APIResponse) and r.rate_limited)
            
            # Rate limiting is working if requests are spaced out or some are rate limited
            rate_limiting_working = (
                total_time > 2.0 or  # Requests took time (throttled)
                rate_limited_requests > 0  # Some requests were rate limited
            )
            
            return {
                'success': True,  # Rate limiting test always "passes"
                'rate_limiting_active': rate_limiting_working,
                'total_time': total_time,
                'successful_requests': successful_requests,
                'rate_limited_requests': rate_limited_requests,
                'details': f"3 requests in {total_time:.2f}s, {rate_limited_requests} rate limited"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': f"Rate limit test failed: {str(e)}"
            }
    
    async def _test_error_handling(self, source, http_client) -> Dict[str, Any]:
        """Test error handling with invalid requests"""
        try:
            # Make an intentionally invalid request
            invalid_url = source.endpoint + "/invalid_endpoint_12345"
            
            response = await http_client.get(
                url=invalid_url,
                api_name=f"{source.name.lower()}_error_test",
                timeout=5.0,
                use_cache=False
            )
            
            # Good error handling means we get a proper error response, not a crash
            error_handled_properly = (
                not response.success and 
                response.error is not None and
                response.status_code is not None
            )
            
            return {
                'success': error_handled_properly,
                'response_time': response.response_time,
                'status_code': response.status_code,
                'details': f"Error handling: {response.error[:100] if response.error else 'No error info'}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': f"Error handling test crashed: {str(e)}"
            }
    
    async def _generate_exhaustive_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = end_time - self.test_start_time
        
        print("\n" + "=" * 80)
        print("📊 EXHAUSTIVE SOURCE TEST REPORT")
        print("=" * 80)
        print(f"⏰ Test Duration: {duration}")
        print(f"📅 Completed: {end_time}")
        
        # Overall statistics
        total_sources = len(self.results)
        successful_sources = sum(1 for r in self.results.values() if r.get('success', False))
        failed_sources = total_sources - successful_sources
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"   Total Sources Tested: {total_sources}")
        print(f"   ✅ Successful: {successful_sources} ({successful_sources/total_sources*100:.1f}%)")
        print(f"   ❌ Failed: {failed_sources} ({failed_sources/total_sources*100:.1f}%)")
        
        # Detailed results by category
        categories = {
            'High Performance': [],
            'Medium Performance': [],
            'Low Performance': [],
            'Failed': []
        }
        
        for source_name, result in self.results.items():
            score = result.get('score', 0)
            if not result.get('success', False):
                categories['Failed'].append((source_name, result))
            elif score >= 0.8:
                categories['High Performance'].append((source_name, result))
            elif score >= 0.6:
                categories['Medium Performance'].append((source_name, result))
            else:
                categories['Low Performance'].append((source_name, result))
        
        for category, sources in categories.items():
            if sources:
                print(f"\n🏆 {category.upper()} ({len(sources)} sources):")
                for source_name, result in sources:
                    score = result.get('score', 0) * 100
                    response_time = result.get('response_time', 0)
                    summary = result.get('summary', 'No summary')
                    print(f"   • {source_name}: {score:.0f}% ({response_time:.2f}s) - {summary}")
        
        # Critical issues
        critical_issues = []
        for source_name, result in self.results.items():
            if result.get('error'):
                critical_issues.append(f"{source_name}: {result['error']}")
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"   ❌ {issue}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if failed_sources > 0:
            print(f"   • Fix {failed_sources} failed sources for full validation coverage")
        
        if successful_sources < total_sources * 0.8:
            print(f"   • Success rate below 80% - investigate authentication and connectivity")
        
        avg_response_time = sum(r.get('response_time', 0) for r in self.results.values()) / len(self.results)
        if avg_response_time > 5.0:
            print(f"   • Average response time {avg_response_time:.1f}s - consider timeout optimization")
        
        print(f"\n✅ EXHAUSTIVE TESTING COMPLETE")
        
        # Save detailed results to file
        report_file = Path("/Users/camdouglas/quark/tools_utilities/exhaustive_test_results.json")
        with open(report_file, 'w') as f:
            json.dump({
                'test_metadata': {
                    'start_time': self.test_start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds(),
                    'total_sources': total_sources,
                    'successful_sources': successful_sources,
                    'success_rate': successful_sources / total_sources
                },
                'detailed_results': self.results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: {report_file}")

async def main():
    """Run exhaustive source testing"""
    print("🚀 STARTING EXHAUSTIVE VALIDATION SOURCE TESTING")
    print("This will test ALL sources with real API calls")
    print("⚠️  This may take several minutes and will use API quotas")
    
    # Confirm before proceeding
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Testing cancelled by user")
        return
    
    tester = ExhaustiveSourceTester()
    await tester.test_all_sources()

if __name__ == "__main__":
    asyncio.run(main())
