#!/usr/bin/env python3
"""
Async HTTP Client with Rate Limiting and Error Handling
For Comprehensive Validation System
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
from asyncio_throttle import Throttler
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import backoff
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration for different APIs"""
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 5
    
    # API-specific configurations
    @classmethod
    def for_api(cls, api_name: str) -> 'RateLimitConfig':
        """Get rate limit config for specific APIs"""
        configs = {
            # High-throughput APIs
            'arxiv': cls(requests_per_second=3.0, requests_per_minute=180),
            'pubchem': cls(requests_per_second=5.0, requests_per_minute=300),
            'ensembl': cls(requests_per_second=15.0, requests_per_minute=900),
            
            # Medium-throughput APIs
            'rcsb_pdb': cls(requests_per_second=2.0, requests_per_minute=120),
            'uniprot': cls(requests_per_second=2.0, requests_per_minute=120),
            'ncbi_eutils': cls(requests_per_second=10.0, requests_per_minute=600),  # With API key
            
            # Rate-limited APIs
            'materials_project': cls(requests_per_second=1.67, requests_per_minute=100),  # 100/min limit
            'openai': cls(requests_per_second=0.5, requests_per_minute=20),
            'claude': cls(requests_per_second=0.33, requests_per_minute=10),
            'gemini': cls(requests_per_second=0.5, requests_per_minute=15),
            
            # Conservative defaults for unknown APIs
            'default': cls(requests_per_second=1.0, requests_per_minute=60)
        }
        return configs.get(api_name, configs['default'])

@dataclass
class APIResponse:
    """Standardized API response wrapper"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: float = 0.0
    source: str = ""
    cached: bool = False
    rate_limited: bool = False

class AsyncHTTPClient:
    """
    Production-ready async HTTP client with rate limiting and error handling
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.throttlers: Dict[str, Throttler] = {}
        self.rate_limits: Dict[str, RateLimitConfig] = {}
        self.request_counts: Dict[str, Dict[str, int]] = {}
        self.last_reset: Dict[str, datetime] = {}
        self.cache: Dict[str, APIResponse] = {}
        self.cache_ttl = 3600  # 1 hour cache
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def start(self):
        """Initialize the HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=20,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'Quark-Validation-System/1.0 (https://github.com/quark-ai/validation)',
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Encoding': 'gzip, deflate'
                }
            )
            logger.info("✅ Async HTTP client initialized")
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("✅ Async HTTP client closed")
    
    def _get_throttler(self, api_name: str) -> Throttler:
        """Get or create throttler for API"""
        if api_name not in self.throttlers:
            config = RateLimitConfig.for_api(api_name)
            self.rate_limits[api_name] = config
            self.throttlers[api_name] = Throttler(rate_limit=config.requests_per_second)
            self.request_counts[api_name] = {'minute': 0, 'hour': 0}
            self.last_reset[api_name] = datetime.now()
            logger.info(f"✅ Created throttler for {api_name}: {config.requests_per_second} req/sec")
        
        return self.throttlers[api_name]
    
    def _check_rate_limits(self, api_name: str) -> bool:
        """Check if request is within rate limits"""
        now = datetime.now()
        config = self.rate_limits.get(api_name, RateLimitConfig.for_api('default'))
        counts = self.request_counts.get(api_name, {'minute': 0, 'hour': 0})
        last_reset = self.last_reset.get(api_name, now)
        
        # Reset counters if needed
        if now - last_reset > timedelta(minutes=1):
            counts['minute'] = 0
            if now - last_reset > timedelta(hours=1):
                counts['hour'] = 0
            self.last_reset[api_name] = now
        
        # Check limits
        if counts['minute'] >= config.requests_per_minute:
            logger.warning(f"⚠️ Rate limit exceeded for {api_name}: {counts['minute']}/{config.requests_per_minute} per minute")
            return False
        
        if counts['hour'] >= config.requests_per_hour:
            logger.warning(f"⚠️ Rate limit exceeded for {api_name}: {counts['hour']}/{config.requests_per_hour} per hour")
            return False
        
        return True
    
    def _update_request_count(self, api_name: str):
        """Update request counters"""
        if api_name not in self.request_counts:
            self.request_counts[api_name] = {'minute': 0, 'hour': 0}
        
        self.request_counts[api_name]['minute'] += 1
        self.request_counts[api_name]['hour'] += 1
    
    def _get_cache_key(self, url: str, params: Dict[str, Any], headers: Dict[str, str]) -> str:
        """Generate cache key for request"""
        cache_data = {
            'url': url,
            'params': sorted(params.items()) if params else [],
            'headers': sorted(headers.items()) if headers else []
        }
        return str(hash(str(cache_data)))
    
    def _is_cache_valid(self, response: APIResponse) -> bool:
        """Check if cached response is still valid"""
        if not response.cached:
            return False
        
        # Simple TTL-based cache validation
        # In production, you might want more sophisticated cache invalidation
        return True  # For now, assume cache is valid within TTL
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(
        self,
        method: str,
        url: str,
        api_name: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> APIResponse:
        """Make HTTP request with retry logic"""
        start_time = time.time()
        
        try:
            # Ensure session is started
            if not self.session:
                await self.start()
            
            # Prepare headers
            request_headers = {}
            if headers:
                request_headers.update(headers)
            
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=timeout or 30)
            ) as response:
                response_time = time.time() - start_time
                
                # Handle different content types
                content_type = response.headers.get('content-type', '').lower()
                
                if 'application/json' in content_type:
                    data = await response.json()
                elif 'application/xml' in content_type or 'text/xml' in content_type:
                    text = await response.text()
                    data = {'xml_content': text}
                else:
                    text = await response.text()
                    data = {'text_content': text}
                
                # Check for API-specific error patterns
                if response.status >= 400:
                    error_msg = f"HTTP {response.status}: {data if isinstance(data, str) else data.get('error', 'Unknown error')}"
                    return APIResponse(
                        success=False,
                        error=error_msg,
                        status_code=response.status,
                        response_time=response_time,
                        source=api_name
                    )
                
                return APIResponse(
                    success=True,
                    data=data,
                    status_code=response.status,
                    response_time=response_time,
                    source=api_name
                )
                
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return APIResponse(
                success=False,
                error=f"Request timeout after {response_time:.2f}s",
                response_time=response_time,
                source=api_name
            )
        
        except aiohttp.ClientError as e:
            response_time = time.time() - start_time
            return APIResponse(
                success=False,
                error=f"Client error: {str(e)}",
                response_time=response_time,
                source=api_name
            )
        
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"❌ Unexpected error in {api_name} request: {e}")
            return APIResponse(
                success=False,
                error=f"Unexpected error: {str(e)}",
                response_time=response_time,
                source=api_name
            )
    
    async def get(
        self,
        url: str,
        api_name: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True
    ) -> APIResponse:
        """Make GET request with rate limiting and caching"""
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(url, params or {}, headers or {})
            if cache_key in self.cache:
                cached_response = self.cache[cache_key]
                if self._is_cache_valid(cached_response):
                    cached_response.cached = True
                    logger.debug(f"✅ Cache hit for {api_name}")
                    return cached_response
        
        # Check rate limits
        if not self._check_rate_limits(api_name):
            return APIResponse(
                success=False,
                error=f"Rate limit exceeded for {api_name}",
                source=api_name,
                rate_limited=True
            )
        
        # Get throttler and wait if needed
        throttler = self._get_throttler(api_name)
        async with throttler:
            # Update request count
            self._update_request_count(api_name)
            
            # Make request
            response = await self._make_request(
                method='GET',
                url=url,
                api_name=api_name,
                params=params,
                headers=headers,
                timeout=timeout
            )
            
            # Cache successful responses
            if use_cache and response.success:
                cache_key = self._get_cache_key(url, params or {}, headers or {})
                self.cache[cache_key] = response
            
            return response
    
    async def post(
        self,
        url: str,
        api_name: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> APIResponse:
        """Make POST request with rate limiting"""
        
        # Check rate limits
        if not self._check_rate_limits(api_name):
            return APIResponse(
                success=False,
                error=f"Rate limit exceeded for {api_name}",
                source=api_name,
                rate_limited=True
            )
        
        # Get throttler and wait if needed
        throttler = self._get_throttler(api_name)
        async with throttler:
            # Update request count
            self._update_request_count(api_name)
            
            # Make request
            return await self._make_request(
                method='POST',
                url=url,
                api_name=api_name,
                params=params,
                headers=headers,
                json_data=json_data,
                timeout=timeout
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = {
            'active_throttlers': len(self.throttlers),
            'cache_size': len(self.cache),
            'request_counts': self.request_counts.copy(),
            'rate_limits': {
                name: {
                    'requests_per_second': config.requests_per_second,
                    'requests_per_minute': config.requests_per_minute
                }
                for name, config in self.rate_limits.items()
            }
        }
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the client"""
        health = {
            'session_active': self.session is not None,
            'throttlers_count': len(self.throttlers),
            'cache_size': len(self.cache),
            'timestamp': datetime.now().isoformat()
        }
        
        # Test a simple request
        try:
            test_response = await self.get(
                url='https://httpbin.org/status/200',
                api_name='health_check',
                timeout=5.0,
                use_cache=False
            )
            health['connectivity'] = test_response.success
            health['test_response_time'] = test_response.response_time
        except Exception as e:
            health['connectivity'] = False
            health['connectivity_error'] = str(e)
        
        return health

# Singleton instance
_http_client = None

async def get_http_client() -> AsyncHTTPClient:
    """Get or create the singleton HTTP client"""
    global _http_client
    if _http_client is None:
        _http_client = AsyncHTTPClient()
        await _http_client.start()
    return _http_client

async def close_http_client():
    """Close the singleton HTTP client"""
    global _http_client
    if _http_client:
        await _http_client.close()
        _http_client = None

# Context manager for automatic cleanup
class HTTPClientManager:
    """Context manager for HTTP client lifecycle"""
    
    def __init__(self):
        self.client = None
    
    async def __aenter__(self) -> AsyncHTTPClient:
        self.client = AsyncHTTPClient()
        await self.client.start()
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()

if __name__ == "__main__":
    async def test_client():
        """Test the async HTTP client"""
        async with HTTPClientManager() as client:
            print("🔍 Testing Async HTTP Client")
            
            # Test rate limiting
            print("\n📊 Testing rate limits...")
            for i in range(5):
                response = await client.get(
                    url='https://httpbin.org/delay/1',
                    api_name='test_api'
                )
                print(f"Request {i+1}: Success={response.success}, Time={response.response_time:.2f}s")
            
            # Test caching
            print("\n💾 Testing caching...")
            url = 'https://httpbin.org/json'
            
            # First request
            response1 = await client.get(url, 'test_cache')
            print(f"First request: Time={response1.response_time:.2f}s, Cached={response1.cached}")
            
            # Second request (should be cached)
            response2 = await client.get(url, 'test_cache')
            print(f"Second request: Time={response2.response_time:.2f}s, Cached={response2.cached}")
            
            # Health check
            print("\n🏥 Health check...")
            health = await client.health_check()
            print(f"Health: {health}")
            
            # Stats
            print("\n📈 Client stats...")
            stats = client.get_stats()
            print(f"Stats: {stats}")
    
    asyncio.run(test_client())
