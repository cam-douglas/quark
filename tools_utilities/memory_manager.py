#!/usr/bin/env python3
"""
Memory Management System for Validation System
Handles memory optimization, garbage collection, and resource monitoring
"""

import gc
import psutil
import asyncio
import weakref
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    timestamp: datetime
    process_memory_mb: float
    system_memory_percent: float
    cache_size: int
    active_connections: int
    gc_collections: Dict[int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'process_memory_mb': self.process_memory_mb,
            'system_memory_percent': self.system_memory_percent,
            'cache_size': self.cache_size,
            'active_connections': self.active_connections,
            'gc_collections': self.gc_collections
        }

class MemoryManager:
    """
    Production-ready memory management for validation system
    """
    
    def __init__(self, 
                 max_memory_mb: int = 1024,  # 1GB default limit
                 cache_cleanup_threshold: float = 0.8,  # Cleanup at 80% memory usage
                 monitoring_interval: int = 30):  # Monitor every 30 seconds
        
        self.max_memory_mb = max_memory_mb
        self.cache_cleanup_threshold = cache_cleanup_threshold
        self.monitoring_interval = monitoring_interval
        
        # Memory tracking
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 100  # Keep last 100 measurements
        
        # Cache management
        self.managed_caches: Dict[str, weakref.ref] = {}
        self.cache_cleanup_callbacks: Dict[str, Callable] = {}
        
        # Connection tracking
        self.active_connections = 0
        self.connection_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Alerts
        self.alert_callbacks: List[Callable[[str, MemoryStats], None]] = []
        
        logger.info(f"✅ Memory Manager initialized: {max_memory_mb}MB limit")
    
    def get_current_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return MemoryStats(
            timestamp=datetime.now(),
            process_memory_mb=memory_info.rss / 1024 / 1024,  # Convert to MB
            system_memory_percent=psutil.virtual_memory().percent,
            cache_size=sum(len(cache()) for cache in self.managed_caches.values() if cache() is not None),
            active_connections=self.active_connections,
            gc_collections={i: gc.get_count()[i] for i in range(3)}
        )
    
    def register_cache(self, name: str, cache_obj: Any, cleanup_callback: Optional[Callable] = None):
        """Register a cache for memory management"""
        self.managed_caches[name] = weakref.ref(cache_obj)
        if cleanup_callback:
            self.cache_cleanup_callbacks[name] = cleanup_callback
        logger.info(f"✅ Registered cache: {name}")
    
    def register_connection_callback(self, callback: Callable):
        """Register callback for connection count updates"""
        self.connection_callbacks.append(callback)
    
    def update_connection_count(self, count: int):
        """Update active connection count"""
        self.active_connections = count
        for callback in self.connection_callbacks:
            try:
                callback(count)
            except Exception as e:
                logger.error(f"❌ Connection callback error: {e}")
    
    def register_alert_callback(self, callback: Callable[[str, MemoryStats], None]):
        """Register callback for memory alerts"""
        self.alert_callbacks.append(callback)
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        stats = self.get_current_memory_stats()
        
        # Check process memory limit
        if stats.process_memory_mb > self.max_memory_mb:
            return True
        
        # Check system memory pressure
        if stats.system_memory_percent > 90:
            return True
        
        return False
    
    async def cleanup_caches(self, aggressive: bool = False) -> int:
        """Clean up managed caches"""
        cleaned_items = 0
        
        for cache_name, cache_ref in list(self.managed_caches.items()):
            cache = cache_ref()
            if cache is None:
                # Cache was garbage collected
                del self.managed_caches[cache_name]
                continue
            
            # Use custom cleanup callback if available
            if cache_name in self.cache_cleanup_callbacks:
                try:
                    items_cleaned = self.cache_cleanup_callbacks[cache_name](aggressive)
                    cleaned_items += items_cleaned
                    logger.info(f"✅ Cleaned {items_cleaned} items from {cache_name}")
                except Exception as e:
                    logger.error(f"❌ Cache cleanup error for {cache_name}: {e}")
            
            # Generic cleanup for dict-like caches
            elif hasattr(cache, 'clear'):
                if aggressive:
                    size_before = len(cache) if hasattr(cache, '__len__') else 0
                    cache.clear()
                    cleaned_items += size_before
                    logger.info(f"✅ Cleared cache {cache_name}: {size_before} items")
                else:
                    # Partial cleanup - remove oldest 50% if it's a dict
                    if hasattr(cache, 'items') and hasattr(cache, '__delitem__'):
                        items = list(cache.items())
                        items_to_remove = len(items) // 2
                        for key, _ in items[:items_to_remove]:
                            try:
                                del cache[key]
                                cleaned_items += 1
                            except KeyError:
                                pass
                        logger.info(f"✅ Partial cleanup {cache_name}: {items_to_remove} items")
        
        return cleaned_items
    
    def force_garbage_collection(self) -> Dict[int, int]:
        """Force garbage collection and return collection counts"""
        before_counts = gc.get_count()
        
        # Force collection of all generations
        collected = {}
        for generation in range(3):
            collected[generation] = gc.collect(generation)
        
        after_counts = gc.get_count()
        
        logger.info(f"✅ Garbage collection: {collected} objects collected")
        return collected
    
    async def handle_memory_pressure(self) -> bool:
        """Handle memory pressure situation"""
        logger.warning("⚠️ Memory pressure detected - starting cleanup")
        
        # Step 1: Partial cache cleanup
        cleaned_items = await self.cleanup_caches(aggressive=False)
        
        # Step 2: Force garbage collection
        gc_collected = self.force_garbage_collection()
        
        # Step 3: Check if pressure is resolved
        if not self.check_memory_pressure():
            logger.info(f"✅ Memory pressure resolved: {cleaned_items} cache items, {sum(gc_collected.values())} GC objects")
            return True
        
        # Step 4: Aggressive cleanup
        logger.warning("⚠️ Memory pressure persists - aggressive cleanup")
        cleaned_items += await self.cleanup_caches(aggressive=True)
        
        # Step 5: Final check
        pressure_resolved = not self.check_memory_pressure()
        
        if pressure_resolved:
            logger.info(f"✅ Memory pressure resolved after aggressive cleanup")
        else:
            logger.error(f"❌ Memory pressure persists after cleanup - consider increasing limits")
        
        return pressure_resolved
    
    async def start_monitoring(self):
        """Start memory monitoring task"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"✅ Memory monitoring started (interval: {self.monitoring_interval}s)")
    
    async def stop_monitoring(self):
        """Stop memory monitoring task"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("✅ Memory monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                # Collect memory stats
                stats = self.get_current_memory_stats()
                self.memory_history.append(stats)
                
                # Trim history if too long
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history = self.memory_history[-self.max_history_size:]
                
                # Check for memory pressure
                memory_usage_ratio = stats.process_memory_mb / self.max_memory_mb
                
                if memory_usage_ratio > self.cache_cleanup_threshold:
                    # Send alerts
                    for callback in self.alert_callbacks:
                        try:
                            callback("memory_pressure", stats)
                        except Exception as e:
                            logger.error(f"❌ Alert callback error: {e}")
                    
                    # Handle memory pressure
                    await self.handle_memory_pressure()
                
                # Log periodic stats
                if len(self.memory_history) % 10 == 0:  # Every 10 measurements
                    logger.info(f"📊 Memory: {stats.process_memory_mb:.1f}MB ({memory_usage_ratio*100:.1f}%), "
                              f"Cache: {stats.cache_size} items, Connections: {stats.active_connections}")
                
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("✅ Memory monitoring loop cancelled")
        except Exception as e:
            logger.error(f"❌ Memory monitoring error: {e}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report"""
        if not self.memory_history:
            return {"error": "No memory history available"}
        
        current_stats = self.memory_history[-1]
        
        # Calculate trends
        if len(self.memory_history) >= 2:
            previous_stats = self.memory_history[-2]
            memory_trend = current_stats.process_memory_mb - previous_stats.process_memory_mb
            cache_trend = current_stats.cache_size - previous_stats.cache_size
        else:
            memory_trend = 0
            cache_trend = 0
        
        # Calculate averages over last 10 measurements
        recent_history = self.memory_history[-10:]
        avg_memory = sum(s.process_memory_mb for s in recent_history) / len(recent_history)
        avg_cache_size = sum(s.cache_size for s in recent_history) / len(recent_history)
        
        return {
            'current': current_stats.to_dict(),
            'trends': {
                'memory_change_mb': memory_trend,
                'cache_size_change': cache_trend
            },
            'averages': {
                'memory_mb': avg_memory,
                'cache_size': avg_cache_size
            },
            'limits': {
                'max_memory_mb': self.max_memory_mb,
                'cleanup_threshold': self.cache_cleanup_threshold,
                'usage_ratio': current_stats.process_memory_mb / self.max_memory_mb
            },
            'managed_caches': list(self.managed_caches.keys()),
            'history_size': len(self.memory_history)
        }
    
    def save_memory_report(self, filepath: Optional[Path] = None):
        """Save memory report to file"""
        if filepath is None:
            filepath = Path("/Users/camdouglas/quark/tools_utilities/memory_report.json")
        
        report = self.get_memory_report()
        report['generated_at'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Memory report saved to: {filepath}")
    
    def optimize_for_production(self):
        """Apply production optimizations"""
        # Optimize garbage collection
        gc.set_threshold(700, 10, 10)  # More aggressive GC
        
        # Enable garbage collection debugging in development
        # gc.set_debug(gc.DEBUG_STATS)
        
        logger.info("✅ Applied production memory optimizations")

# Singleton memory manager
_memory_manager: Optional[MemoryManager] = None

def get_memory_manager() -> MemoryManager:
    """Get or create singleton memory manager"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def setup_validation_memory_management(validation_system, http_client):
    """Set up memory management for validation system components"""
    memory_manager = get_memory_manager()
    
    # Register validation system cache
    if hasattr(validation_system, 'validation_cache'):
        def cleanup_validation_cache(aggressive: bool = False) -> int:
            cache = validation_system.validation_cache
            if aggressive:
                size = len(cache)
                cache.clear()
                return size
            else:
                # Remove oldest 50%
                items = list(cache.items())
                items_to_remove = len(items) // 2
                for key, _ in items[:items_to_remove]:
                    cache.pop(key, None)
                return items_to_remove
        
        memory_manager.register_cache(
            "validation_cache", 
            validation_system.validation_cache,
            cleanup_validation_cache
        )
    
    # Register HTTP client cache
    if hasattr(http_client, 'cache'):
        def cleanup_http_cache(aggressive: bool = False) -> int:
            cache = http_client.cache
            if aggressive:
                size = len(cache)
                cache.clear()
                return size
            else:
                # Remove oldest 50%
                items = list(cache.items())
                items_to_remove = len(items) // 2
                for key, _ in items[:items_to_remove]:
                    cache.pop(key, None)
                return items_to_remove
        
        memory_manager.register_cache(
            "http_cache",
            http_client.cache,
            cleanup_http_cache
        )
    
    # Set up connection monitoring
    def update_connections():
        if hasattr(http_client, 'session') and http_client.session:
            connector = http_client.session.connector
            if hasattr(connector, '_conns'):
                total_connections = sum(len(conns) for conns in connector._conns.values())
                memory_manager.update_connection_count(total_connections)
    
    memory_manager.register_connection_callback(update_connections)
    
    # Set up memory alerts
    def memory_alert_handler(alert_type: str, stats: MemoryStats):
        if alert_type == "memory_pressure":
            logger.warning(f"🚨 MEMORY ALERT: {stats.process_memory_mb:.1f}MB used, "
                         f"{stats.system_memory_percent:.1f}% system memory")
    
    memory_manager.register_alert_callback(memory_alert_handler)
    
    logger.info("✅ Memory management configured for validation system")
    return memory_manager

if __name__ == "__main__":
    async def test_memory_manager():
        """Test memory management system"""
        print("🧠 Testing Memory Management System")
        
        # Create memory manager
        memory_manager = MemoryManager(max_memory_mb=512)  # 512MB limit for testing
        
        # Create test cache
        test_cache = {}
        for i in range(1000):
            test_cache[f"key_{i}"] = f"value_{i}" * 100  # Create some memory usage
        
        memory_manager.register_cache("test_cache", test_cache)
        
        # Start monitoring
        await memory_manager.start_monitoring()
        
        # Get initial stats
        stats = memory_manager.get_current_memory_stats()
        print(f"📊 Initial memory: {stats.process_memory_mb:.1f}MB")
        
        # Simulate memory pressure
        print("🔥 Simulating memory pressure...")
        large_data = {}
        for i in range(10000):
            large_data[f"large_key_{i}"] = "x" * 1000
        
        # Wait for monitoring to detect and handle pressure
        await asyncio.sleep(35)  # Wait for monitoring cycle
        
        # Check final stats
        final_stats = memory_manager.get_current_memory_stats()
        print(f"📊 Final memory: {final_stats.process_memory_mb:.1f}MB")
        
        # Generate report
        report = memory_manager.get_memory_report()
        print(f"📄 Memory report: {json.dumps(report, indent=2)}")
        
        # Stop monitoring
        await memory_manager.stop_monitoring()
        
        print("✅ Memory management test complete")
    
    asyncio.run(test_memory_manager())
