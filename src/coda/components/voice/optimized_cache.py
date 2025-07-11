"""
High-performance LRU cache implementation for voice processing.

This module provides optimized caching with proper eviction policies,
size limits, and performance monitoring.
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")


class CachePolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    HYBRID = "hybrid"  # LRU + TTL


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""

    value: T
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    ttl: Optional[float] = None
    size_bytes: int = 0

    def __post_init__(self):
        """Initialize timestamps."""
        current_time = time.time()
        if self.created_at == 0.0:
            self.created_at = current_time
        if self.last_accessed == 0.0:
            self.last_accessed = current_time

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class OptimizedLRUCache(Generic[K, T]):
    """High-performance LRU cache with advanced features."""

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        default_ttl: Optional[float] = None,
        policy: CachePolicy = CachePolicy.LRU,
        cleanup_interval: float = 60.0,
    ):
        """Initialize optimized cache."""
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.policy = policy
        self.cleanup_interval = cleanup_interval

        # Storage
        self.cache: OrderedDict[K, CacheEntry[T]] = OrderedDict()
        self.lock = threading.RLock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
            "total_size_bytes": 0,
            "peak_size": 0,
            "peak_memory_bytes": 0,
        }

        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

        logger.info(
            f"OptimizedLRUCache initialized: max_size={max_size}, "
            f"max_memory={max_memory_mb}MB, policy={policy.value}"
        )

    def get(self, key: K, default: Optional[T] = None) -> Optional[T]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return default

            entry = self.cache[key]

            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                self._update_memory_stats()
                return default

            # Update access metadata
            entry.touch()

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            self.stats["hits"] += 1
            return entry.value

    def put(
        self, key: K, value: T, ttl: Optional[float] = None, size_hint: Optional[int] = None
    ) -> bool:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            size_bytes = size_hint or self._estimate_size(value)

            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False

            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats["total_size_bytes"] -= old_entry.size_bytes
                del self.cache[key]

            # Create new entry
            entry = CacheEntry(value=value, ttl=ttl or self.default_ttl, size_bytes=size_bytes)

            # Ensure we have space
            self._ensure_space(size_bytes)

            # Add to cache
            self.cache[key] = entry
            self.stats["total_size_bytes"] += size_bytes

            # Update statistics
            self._update_stats()

            return True

    def delete(self, key: K) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats["total_size_bytes"] -= entry.size_bytes
                del self.cache[key]
                self._update_memory_stats()
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats["total_size_bytes"] = 0

    def get_or_compute(
        self, key: K, compute_func: Callable[[], T], ttl: Optional[float] = None
    ) -> T:
        """Get value or compute if not present."""
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        computed_value = compute_func()
        self.put(key, computed_value, ttl)
        return computed_value

    def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure cache has space for new entry."""
        # Check size limit
        while len(self.cache) >= self.max_size:
            self._evict_one()

        # Check memory limit
        while self.stats["total_size_bytes"] + needed_bytes > self.max_memory_bytes and self.cache:
            self._evict_one()

    def _evict_one(self) -> None:
        """Evict one entry based on policy."""
        if not self.cache:
            return

        if self.policy == CachePolicy.LRU:
            # Remove least recently used (first item)
            key, entry = self.cache.popitem(last=False)
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            min_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            entry = self.cache.pop(min_key)
        else:  # TTL or HYBRID
            # Remove oldest entry
            key, entry = self.cache.popitem(last=False)

        self.stats["total_size_bytes"] -= entry.size_bytes
        self.stats["evictions"] += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            if hasattr(value, "__sizeof__"):
                return value.__sizeof__()
            elif isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple, dict)):
                return sum(self._estimate_size(item) for item in value)
            else:
                # Rough estimate
                return 64  # Default size
        except Exception:
            return 64  # Fallback size

    def _update_stats(self) -> None:
        """Update cache statistics."""
        current_size = len(self.cache)
        if current_size > self.stats["peak_size"]:
            self.stats["peak_size"] = current_size

        if self.stats["total_size_bytes"] > self.stats["peak_memory_bytes"]:
            self.stats["peak_memory_bytes"] = self.stats["total_size_bytes"]

    def _update_memory_stats(self) -> None:
        """Update memory statistics after deletion."""
        # Recalculate total size to handle any inconsistencies
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        self.stats["total_size_bytes"] = total_size

    def _cleanup_worker(self) -> None:
        """Background cleanup worker."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self.cache[key]
                self.stats["total_size_bytes"] -= entry.size_bytes
                del self.cache[key]
                self.stats["expired"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / max(1, total_requests)

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": self.stats["total_size_bytes"] / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "expired": self.stats["expired"],
                "peak_size": self.stats["peak_size"],
                "peak_memory_mb": self.stats["peak_memory_bytes"] / (1024 * 1024),
            }


class VoiceResponseCache:
    """Specialized cache for voice responses."""

    def __init__(self, max_size: int = 500, max_memory_mb: float = 50.0):
        """Initialize voice response cache."""
        self.cache = OptimizedLRUCache[str, Any](
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            default_ttl=300.0,  # 5 minutes
            policy=CachePolicy.HYBRID,
        )

    def generate_key(self, voice_message: Any, context: Optional[Dict] = None) -> str:
        """Generate cache key for voice message."""
        # Create hash from message content and relevant context
        key_data = {
            "text": getattr(voice_message, "text_content", ""),
            "conversation_id": getattr(voice_message, "conversation_id", ""),
            "speaker": getattr(voice_message, "speaker", ""),
        }

        # Add relevant context
        if context:
            key_data["context_hash"] = self._hash_context(context)

        # Generate hash
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _hash_context(self, context: Dict) -> str:
        """Generate hash for context data."""
        # Only include stable context elements
        stable_context = {}

        if "personality" in context:
            # Include personality traits but not dynamic state
            personality = context["personality"]
            if isinstance(personality, dict) and "traits" in personality:
                stable_context["personality_traits"] = personality["traits"]

        if "memory" in context:
            # Include recent topics but not full conversation history
            memory = context["memory"]
            if isinstance(memory, dict) and "recent_topics" in memory:
                stable_context["recent_topics"] = memory["recent_topics"][:5]  # Last 5 topics

        return hashlib.md5(str(sorted(stable_context.items())).encode()).hexdigest()

    def get_response(self, voice_message: Any, context: Optional[Dict] = None) -> Optional[Any]:
        """Get cached response."""
        key = self.generate_key(voice_message, context)
        return self.cache.get(key)

    def cache_response(
        self,
        voice_message: Any,
        response: Any,
        context: Optional[Dict] = None,
        ttl: Optional[float] = None,
    ) -> bool:
        """Cache voice response."""
        key = self.generate_key(voice_message, context)
        return self.cache.put(key, response, ttl)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# Global cache instances
_response_cache: Optional[VoiceResponseCache] = None
_context_cache: Optional[OptimizedLRUCache] = None


def get_response_cache() -> VoiceResponseCache:
    """Get global response cache."""
    global _response_cache
    if _response_cache is None:
        _response_cache = VoiceResponseCache()
    return _response_cache


def get_context_cache() -> OptimizedLRUCache:
    """Get global context cache."""
    global _context_cache
    if _context_cache is None:
        _context_cache = OptimizedLRUCache(
            max_size=200, max_memory_mb=20.0, default_ttl=600.0  # 10 minutes
        )
    return _context_cache
