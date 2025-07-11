"""
Advanced Caching System for Coda Performance Optimization.

Provides intelligent caching with automatic optimization, compression,
and memory management for maximum performance.
"""

import asyncio
import gzip
import logging
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live only
    HYBRID = "hybrid"  # Combination of LRU and frequency


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    compressed: bool = False
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl

    def get_age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def get_idle_seconds(self) -> float:
        """Get idle time since last access."""
        return (datetime.now() - self.last_accessed).total_seconds()


class OptimizedCache:
    """
    High-performance cache with intelligent optimization.

    Features:
    - Multiple eviction policies (LRU, LFU, TTL, Hybrid)
    - Automatic compression for large values
    - Memory usage monitoring and optimization
    - Thread-safe operations
    - Performance metrics and analytics
    """

    def __init__(
        self,
        max_memory_mb: float = 100.0,
        max_entries: int = 10000,
        default_ttl: Optional[float] = None,
        policy: CachePolicy = CachePolicy.HYBRID,
        compression_threshold: int = 1024,
        cleanup_interval: float = 60.0,
        name: str = "cache",
    ):
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.policy = policy
        self.compression_threshold = compression_threshold
        self.cleanup_interval = cleanup_interval
        self.name = name

        # Storage
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()

        # Frequency tracking for LFU
        self.frequency_counter: Dict[str, int] = {}

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired_removals": 0,
            "total_memory_bytes": 0,
            "peak_memory_bytes": 0,
            "peak_entries": 0,
            "compression_saves_bytes": 0,
            "total_operations": 0,
        }

        # Background cleanup
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info(f"OptimizedCache '{name}' initialized with {max_memory_mb}MB limit")

    async def start(self):
        """Start background cleanup task."""
        if not self.running:
            self.running = True
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop background cleanup task."""
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            self.stats["total_operations"] += 1

            if key not in self.entries:
                self.stats["misses"] += 1
                return None

            entry = self.entries[key]

            # Check if expired
            if entry.is_expired():
                del self.entries[key]
                self.frequency_counter.pop(key, None)
                self.stats["expired_removals"] += 1
                self.stats["misses"] += 1
                self._update_memory_usage()
                return None

            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.frequency_counter[key] = self.frequency_counter.get(key, 0) + 1

            # Move to end for LRU
            if self.policy in [CachePolicy.LRU, CachePolicy.HYBRID]:
                self.entries.move_to_end(key)

            self.stats["hits"] += 1

            # Decompress if needed
            value = entry.value
            if entry.compressed:
                try:
                    value = pickle.loads(gzip.decompress(value))
                except Exception as e:
                    logger.error(f"Failed to decompress cache entry {key}: {e}")
                    return None

            return value

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            self.stats["total_operations"] += 1

            # Serialize and optionally compress value
            try:
                serialized_value = pickle.dumps(value)
                compressed = False

                if len(serialized_value) > self.compression_threshold:
                    compressed_value = gzip.compress(serialized_value)
                    if len(compressed_value) < len(serialized_value):
                        self.stats["compression_saves_bytes"] += len(serialized_value) - len(
                            compressed_value
                        )
                        serialized_value = compressed_value
                        compressed = True

            except Exception as e:
                logger.error(f"Failed to serialize cache value for key {key}: {e}")
                return False

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=serialized_value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl=ttl or self.default_ttl,
                compressed=compressed,
                size_bytes=len(serialized_value),
            )

            # Remove existing entry if present
            if key in self.entries:
                old_entry = self.entries[key]
                self.stats["total_memory_bytes"] -= old_entry.size_bytes

            # Check if we need to make space
            await self._ensure_space(entry.size_bytes)

            # Add new entry
            self.entries[key] = entry
            self.frequency_counter[key] = self.frequency_counter.get(key, 0) + 1
            self.stats["total_memory_bytes"] += entry.size_bytes

            # Update peak statistics
            self.stats["peak_memory_bytes"] = max(
                self.stats["peak_memory_bytes"], self.stats["total_memory_bytes"]
            )
            self.stats["peak_entries"] = max(self.stats["peak_entries"], len(self.entries))

            return True

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key in self.entries:
                entry = self.entries[key]
                self.stats["total_memory_bytes"] -= entry.size_bytes
                del self.entries[key]
                self.frequency_counter.pop(key, None)
                return True
            return False

    async def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.entries.clear()
            self.frequency_counter.clear()
            self.stats["total_memory_bytes"] = 0

    async def cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            expired_keys = []

            for key, entry in self.entries.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self.entries[key]
                self.stats["total_memory_bytes"] -= entry.size_bytes
                del self.entries[key]
                self.frequency_counter.pop(key, None)
                self.stats["expired_removals"] += 1

    async def optimize(self):
        """Optimize cache performance."""
        with self.lock:
            # Remove expired entries
            await self.cleanup_expired()

            # Optimize based on hit rates and access patterns
            if self.get_hit_rate() < 0.5:
                # Low hit rate - adjust eviction policy
                await self._optimize_eviction_policy()

            # Compress more aggressively if memory usage is high
            if self.get_memory_usage_percent() > 80:
                await self._optimize_compression()

    async def _ensure_space(self, needed_bytes: int):
        """Ensure there's enough space for new entry."""
        # Check memory limit
        while (
            self.stats["total_memory_bytes"] + needed_bytes > self.max_memory_bytes
            or len(self.entries) >= self.max_entries
        ):

            if not self.entries:
                break

            # Evict based on policy
            evicted_key = self._select_eviction_candidate()
            if evicted_key:
                entry = self.entries[evicted_key]
                self.stats["total_memory_bytes"] -= entry.size_bytes
                del self.entries[evicted_key]
                self.frequency_counter.pop(evicted_key, None)
                self.stats["evictions"] += 1
            else:
                break

    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on policy."""
        if not self.entries:
            return None

        if self.policy == CachePolicy.LRU:
            # Remove least recently used (first in OrderedDict)
            return next(iter(self.entries))

        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            min_freq = min(self.frequency_counter.values())
            for key, freq in self.frequency_counter.items():
                if freq == min_freq and key in self.entries:
                    return key

        elif self.policy == CachePolicy.TTL:
            # Remove oldest entry
            oldest_key = None
            oldest_time = datetime.now()
            for key, entry in self.entries.items():
                if entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    oldest_key = key
            return oldest_key

        elif self.policy == CachePolicy.HYBRID:
            # Hybrid approach: consider both recency and frequency
            scores = {}
            now = datetime.now()

            for key, entry in self.entries.items():
                # Calculate score based on recency and frequency
                recency_score = 1.0 / (1.0 + entry.get_idle_seconds())
                frequency_score = self.frequency_counter.get(key, 1)

                # Combine scores (lower is worse)
                scores[key] = recency_score * frequency_score

            # Return key with lowest score
            return min(scores.keys(), key=lambda k: scores[k])

        # Fallback to LRU
        return next(iter(self.entries))

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.running:
            try:
                await self.cleanup_expired()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(5)

    async def _optimize_eviction_policy(self):
        """Optimize eviction policy based on access patterns."""
        # This could implement dynamic policy adjustment

    async def _optimize_compression(self):
        """Optimize compression settings."""
        # This could implement dynamic compression threshold adjustment

    def _update_memory_usage(self):
        """Update memory usage statistics."""
        total_bytes = sum(entry.size_bytes for entry in self.entries.values())
        self.stats["total_memory_bytes"] = total_bytes

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0

    def get_memory_usage_percent(self) -> float:
        """Get memory usage as percentage of limit."""
        return (self.stats["total_memory_bytes"] / self.max_memory_bytes) * 100

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_ops = self.stats["total_operations"]

        return {
            "name": self.name,
            "policy": self.policy.value,
            "entries": len(self.entries),
            "max_entries": self.max_entries,
            "memory_usage_mb": self.stats["total_memory_bytes"] / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "memory_usage_percent": self.get_memory_usage_percent(),
            "hit_rate": self.get_hit_rate(),
            "total_operations": total_ops,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "expired_removals": self.stats["expired_removals"],
            "compression_saves_mb": self.stats["compression_saves_bytes"] / (1024 * 1024),
            "peak_memory_mb": self.stats["peak_memory_bytes"] / (1024 * 1024),
            "peak_entries": self.stats["peak_entries"],
        }


# Global cache instances
_global_caches: Dict[str, OptimizedCache] = {}


def get_cache(name: str, **kwargs) -> OptimizedCache:
    """Get or create a named cache instance."""
    if name not in _global_caches:
        _global_caches[name] = OptimizedCache(name=name, **kwargs)
    return _global_caches[name]


async def clear_all_caches():
    """Clear all global cache instances."""
    for cache in _global_caches.values():
        await cache.clear()


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all global caches."""
    return {name: cache.get_stats() for name, cache in _global_caches.items()}
