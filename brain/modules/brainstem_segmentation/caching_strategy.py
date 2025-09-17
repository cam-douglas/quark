"""
Caching and scaling strategy for brainstem segmentation services.

Implements intelligent caching, batch inference optimization, and
storage format optimization for high-throughput production deployment.
"""
from __future__ import annotations

import hashlib
import pickle
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class CachingConfig:
    """Configuration for caching and scaling strategy."""
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600  # 1 hour TTL
    
    # Batch processing
    max_batch_size: int = 8
    batch_timeout_ms: int = 1000  # 1 second
    
    # Storage optimization
    compression_level: int = 6  # zlib compression
    use_float16: bool = True    # Reduce precision for storage
    
    # Memory management
    max_cache_size_mb: int = 1024  # 1GB cache limit
    cleanup_threshold: float = 0.8  # Cleanup at 80% full


class SegmentationCache:
    """Intelligent caching system for segmentation results."""
    
    def __init__(self, config: CachingConfig):
        self.config = config
        self.redis_client = None
        self.local_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "size_mb": 0}
        
        self._setup_redis()
    
    def _setup_redis(self) -> None:
        """Setup Redis connection with fallback to local cache."""
        if not REDIS_AVAILABLE:
            logger.info("Redis not installed, using local cache only")
            self.redis_client = None
            return
            
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=False  # Keep binary data
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable, using local cache: {e}")
            self.redis_client = None
    
    def _compute_cache_key(self, volume: np.ndarray, morphogen_data: Optional[np.ndarray] = None) -> str:
        """Compute cache key for volume and morphogen data."""
        
        # Hash volume data
        volume_hash = hashlib.sha256(volume.tobytes()).hexdigest()[:16]
        
        # Hash morphogen data if present
        if morphogen_data is not None:
            morphogen_hash = hashlib.sha256(morphogen_data.tobytes()).hexdigest()[:16]
            cache_key = f"seg:{volume_hash}:{morphogen_hash}"
        else:
            cache_key = f"seg:{volume_hash}"
        
        return cache_key
    
    def get_cached_segmentation(
        self, 
        volume: np.ndarray, 
        morphogen_data: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Retrieve cached segmentation if available."""
        
        cache_key = self._compute_cache_key(volume, morphogen_data)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    segmentation = self._deserialize_segmentation(cached_data)
                    self.cache_stats["hits"] += 1
                    logger.debug(f"Cache hit: {cache_key}")
                    return segmentation
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Try local cache
        if cache_key in self.local_cache:
            self.cache_stats["hits"] += 1
            logger.debug(f"Local cache hit: {cache_key}")
            return self.local_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        return None
    
    def cache_segmentation(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray,
        morphogen_data: Optional[np.ndarray] = None
    ) -> None:
        """Cache segmentation result."""
        
        cache_key = self._compute_cache_key(volume, morphogen_data)
        serialized_data = self._serialize_segmentation(segmentation)
        
        # Store in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, self.config.cache_ttl, serialized_data)
                logger.debug(f"Cached to Redis: {cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache store error: {e}")
        
        # Store in local cache (with size management)
        self._store_local_cache(cache_key, segmentation)
    
    def _serialize_segmentation(self, segmentation: np.ndarray) -> bytes:
        """Serialize segmentation for storage."""
        
        # Convert to smaller dtype if enabled
        if self.config.use_float16:
            data = segmentation.astype(np.float16)
        else:
            data = segmentation
        
        # Compress with pickle + zlib
        import zlib
        pickled_data = pickle.dumps(data)
        compressed_data = zlib.compress(pickled_data, level=self.config.compression_level)
        
        return compressed_data
    
    def _deserialize_segmentation(self, data: bytes) -> np.ndarray:
        """Deserialize segmentation from storage."""
        import zlib
        
        decompressed_data = zlib.decompress(data)
        segmentation = pickle.loads(decompressed_data)
        
        # Convert back to int32 if needed
        if segmentation.dtype == np.float16:
            segmentation = segmentation.astype(np.int32)
        
        return segmentation
    
    def _store_local_cache(self, cache_key: str, segmentation: np.ndarray) -> None:
        """Store in local cache with size management."""
        
        # Check cache size
        current_size_mb = self._estimate_cache_size_mb()
        
        if current_size_mb > self.config.max_cache_size_mb * self.config.cleanup_threshold:
            self._cleanup_local_cache()
        
        self.local_cache[cache_key] = segmentation
        self.cache_stats["size_mb"] = self._estimate_cache_size_mb()
    
    def _estimate_cache_size_mb(self) -> float:
        """Estimate local cache size in MB."""
        total_bytes = 0
        for segmentation in self.local_cache.values():
            total_bytes += segmentation.nbytes
        
        return total_bytes / (1024 * 1024)
    
    def _cleanup_local_cache(self) -> None:
        """Remove old entries from local cache."""
        # Simple cleanup: remove half the entries
        keys_to_remove = list(self.local_cache.keys())[:len(self.local_cache) // 2]
        
        for key in keys_to_remove:
            del self.local_cache[key]
        
        logger.info(f"Cleaned up local cache: {len(keys_to_remove)} entries removed")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(total_requests, 1)
        
        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_size_mb": self.cache_stats["size_mb"],
            "redis_available": self.redis_client is not None
        }


class BatchInferenceManager:
    """Manages batch inference for improved throughput."""
    
    def __init__(self, config: CachingConfig):
        self.config = config
        self.pending_requests = []
        self.request_futures = {}
        self.batch_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Start batch processing loop
        self.batch_task = asyncio.create_task(self._batch_processing_loop())
    
    async def submit_inference_request(
        self,
        volume: np.ndarray,
        morphogen_data: Optional[np.ndarray] = None,
        request_id: str = None
    ) -> np.ndarray:
        """Submit inference request for batch processing."""
        
        if request_id is None:
            request_id = str(time.time())
        
        # Create future for this request
        future = asyncio.Future()
        
        with self.batch_lock:
            self.pending_requests.append({
                "request_id": request_id,
                "volume": volume,
                "morphogen_data": morphogen_data,
                "future": future
            })
            self.request_futures[request_id] = future
        
        # Wait for result
        return await future
    
    async def _batch_processing_loop(self) -> None:
        """Background loop for batch processing."""
        
        while True:
            await asyncio.sleep(self.config.batch_timeout_ms / 1000)
            
            if not self.pending_requests:
                continue
            
            with self.batch_lock:
                # Get batch of requests
                batch = self.pending_requests[:self.config.max_batch_size]
                self.pending_requests = self.pending_requests[self.config.max_batch_size:]
            
            if batch:
                # Process batch in thread pool
                self.executor.submit(self._process_batch, batch)
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of inference requests."""
        
        logger.info(f"Processing batch of {len(batch)} requests")
        
        try:
            # Extract volumes for batch processing
            volumes = [req["volume"] for req in batch]
            morphogen_data_list = [req["morphogen_data"] for req in batch]
            
            # Run batch inference
            results = self._run_batch_inference(volumes, morphogen_data_list)
            
            # Return results to futures
            for req, result in zip(batch, results):
                future = req["future"]
                if not future.cancelled():
                    future.set_result(result)
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Set exception for all futures
            for req in batch:
                future = req["future"]
                if not future.cancelled():
                    future.set_exception(e)
    
    def _run_batch_inference(
        self,
        volumes: List[np.ndarray],
        morphogen_data_list: List[Optional[np.ndarray]]
    ) -> List[np.ndarray]:
        """Run inference on batch of volumes."""
        
        from brain.modules.brainstem_segmentation.inference_engine import auto_segment_brainstem
        
        results = []
        for volume, morphogen_data in zip(volumes, morphogen_data_list):
            segmentation = auto_segment_brainstem(volume, morphogen_data)
            results.append(segmentation)
        
        return results


def run_caching_demo() -> Dict[str, Any]:
    """Demonstrate caching and scaling framework."""
    
    print("🚀 Caching & Scaling Demo")
    print("=" * 40)
    
    config = CachingConfig()
    
    # Test caching system
    cache = SegmentationCache(config)
    
    # Create test data
    test_volume = np.random.rand(64, 64, 64).astype(np.float32)
    test_segmentation = np.random.randint(0, 16, (64, 64, 64)).astype(np.int32)
    
    # Test cache miss
    cached_result = cache.get_cached_segmentation(test_volume)
    assert cached_result is None, "Should be cache miss"
    
    # Store in cache
    cache.cache_segmentation(test_volume, test_segmentation)
    
    # Test cache hit
    cached_result = cache.get_cached_segmentation(test_volume)
    assert cached_result is not None, "Should be cache hit"
    assert np.array_equal(cached_result, test_segmentation), "Cached data should match"
    
    # Get cache stats
    stats = cache.get_cache_stats()
    
    print("📊 Cache Performance:")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache size: {stats['cache_size_mb']:.1f} MB")
    print(f"  Redis available: {stats['redis_available']}")
    
    return {
        "cache_stats": stats,
        "test_passed": True,
        "config": config
    }


if __name__ == "__main__":
    # Run demonstration
    demo_results = run_caching_demo()
    
    # Save results
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "caching_strategy_demo.json"
    with open(results_path, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n📄 Demo results saved: {results_path}")
    print("✅ Caching & scaling framework ready for production")
