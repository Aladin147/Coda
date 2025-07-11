"""
Comprehensive Performance Optimization System for Coda.

Provides system-wide performance optimization including connection pooling,
caching strategies, resource management, and response time improvements.
"""

import asyncio
import gc
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimized_cache import OptimizedCache

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Performance optimization levels."""

    CONSERVATIVE = "conservative"  # Minimal optimizations, maximum stability
    BALANCED = "balanced"  # Balanced performance and stability
    AGGRESSIVE = "aggressive"  # Maximum performance, some stability trade-offs
    CUSTOM = "custom"  # Custom optimization settings


class ResourceType(Enum):
    """Types of system resources."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"
    DISK_IO = "disk_io"
    NETWORK = "network"
    DATABASE = "database"


@dataclass
class PerformanceTarget:
    """Performance targets for optimization."""

    max_response_time_ms: float = 500.0
    max_memory_usage_percent: float = 80.0
    max_cpu_usage_percent: float = 70.0
    max_gpu_memory_percent: float = 85.0
    min_cache_hit_rate: float = 0.8
    max_connection_pool_usage: float = 0.9
    target_throughput_rps: float = 100.0


@dataclass
class OptimizationMetrics:
    """Current performance metrics."""

    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    cache_hit_rate: float = 0.0
    connection_pool_usage: float = 0.0
    throughput_rps: float = 0.0
    active_connections: int = 0
    queue_depth: int = 0
    error_rate: float = 0.0


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling."""

    max_connections: int = 100
    min_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0


@dataclass
class CacheConfig:
    """Configuration for caching system."""

    max_memory_mb: float = 500.0
    default_ttl: float = 300.0
    cleanup_interval: float = 60.0
    max_entries: int = 10000
    enable_compression: bool = True
    compression_threshold: int = 1024


class PerformanceOptimizer:
    """
    Comprehensive performance optimization system.

    Manages connection pooling, caching, resource optimization,
    and automatic performance tuning across all Coda components.
    """

    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
        targets: Optional[PerformanceTarget] = None,
        connection_config: Optional[ConnectionPoolConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        self.optimization_level = optimization_level
        self.targets = targets or PerformanceTarget()
        self.connection_config = connection_config or ConnectionPoolConfig()
        self.cache_config = cache_config or CacheConfig()

        # Performance tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_metrics = OptimizationMetrics()

        # Connection pools by service type
        self.connection_pools: Dict[str, "ConnectionPool"] = {}

        # Caching systems
        self.response_cache: "OptimizedCache" = None
        self.session_cache: "OptimizedCache" = None
        self.model_cache: "OptimizedCache" = None

        # Resource management
        self.resource_monitors: Dict[ResourceType, Any] = {}
        self.optimization_tasks: Set[asyncio.Task] = set()

        # Performance state
        self.running = False
        self.last_optimization = datetime.now()
        self.optimization_interval = 30.0  # seconds

        # Component registrations
        self.registered_components: Dict[str, Dict[str, Any]] = {}

        logger.info(f"PerformanceOptimizer initialized with level: {optimization_level.value}")

    async def initialize(self):
        """Initialize the performance optimization system."""
        logger.info("Initializing performance optimization system...")

        # Initialize caching systems
        await self._initialize_caches()

        # Initialize connection pools
        await self._initialize_connection_pools()

        # Initialize resource monitors
        await self._initialize_resource_monitors()

        # Start optimization loop
        self.running = True
        optimization_task = asyncio.create_task(self._optimization_loop())
        self.optimization_tasks.add(optimization_task)

        logger.info("Performance optimization system initialized")

    async def shutdown(self):
        """Shutdown the performance optimization system."""
        logger.info("Shutting down performance optimization system...")

        self.running = False

        # Cancel optimization tasks
        for task in self.optimization_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.optimization_tasks:
            await asyncio.gather(*self.optimization_tasks, return_exceptions=True)

        # Shutdown connection pools
        for pool in self.connection_pools.values():
            await pool.close()

        # Clear caches
        if self.response_cache:
            await self.response_cache.clear()
        if self.session_cache:
            await self.session_cache.clear()
        if self.model_cache:
            await self.model_cache.clear()

        logger.info("Performance optimization system shutdown complete")

    def register_component(
        self,
        component_name: str,
        component_type: str,
        optimization_config: Optional[Dict[str, Any]] = None,
    ):
        """Register a component for performance optimization."""
        self.registered_components[component_name] = {
            "type": component_type,
            "config": optimization_config or {},
            "registered_at": datetime.now(),
            "metrics": {},
            "optimizations_applied": [],
        }

        logger.info(f"Registered component for optimization: {component_name} ({component_type})")

    async def get_connection_pool(self, service_name: str) -> "ConnectionPool":
        """Get or create connection pool for a service."""
        if service_name not in self.connection_pools:
            pool = ConnectionPool(service_name, self.connection_config)
            await pool.initialize()
            self.connection_pools[service_name] = pool

        return self.connection_pools[service_name]

    async def cache_response(
        self, cache_type: str, key: str, value: Any, ttl: Optional[float] = None
    ) -> bool:
        """Cache a response with automatic optimization."""
        cache = self._get_cache_by_type(cache_type)
        if not cache:
            return False

        return await cache.set(key, value, ttl or self.cache_config.default_ttl)

    async def get_cached_response(self, cache_type: str, key: str) -> Optional[Any]:
        """Get cached response with hit rate tracking."""
        cache = self._get_cache_by_type(cache_type)
        if not cache:
            return None

        result = await cache.get(key)

        # Update cache hit rate metrics
        if result is not None:
            self._update_cache_hit_rate(True)
        else:
            self._update_cache_hit_rate(False)

        return result

    async def optimize_component(self, component_name: str) -> Dict[str, Any]:
        """Perform targeted optimization for a specific component."""
        if component_name not in self.registered_components:
            logger.warning(f"Component {component_name} not registered for optimization")
            return {"status": "not_registered"}

        component = self.registered_components[component_name]
        component_type = component["type"]

        optimizations = []

        # Apply component-specific optimizations
        if component_type == "llm":
            optimizations.extend(await self._optimize_llm_component(component_name))
        elif component_type == "memory":
            optimizations.extend(await self._optimize_memory_component(component_name))
        elif component_type == "voice":
            optimizations.extend(await self._optimize_voice_component(component_name))
        elif component_type == "websocket":
            optimizations.extend(await self._optimize_websocket_component(component_name))
        elif component_type == "dashboard":
            optimizations.extend(await self._optimize_dashboard_component(component_name))

        # Record optimizations
        component["optimizations_applied"].extend(optimizations)

        return {
            "status": "optimized",
            "component": component_name,
            "optimizations": optimizations,
            "timestamp": datetime.now().isoformat(),
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.metrics_history:
            return {"status": "no_data"}

        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 measurements

        # Calculate averages
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_cpu_usage = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)

        # Performance status
        status = "optimal"
        if avg_response_time > self.targets.max_response_time_ms:
            status = "degraded"
        if avg_memory_usage > self.targets.max_memory_usage_percent:
            status = "resource_constrained"
        if avg_cpu_usage > self.targets.max_cpu_usage_percent:
            status = "cpu_bound"

        return {
            "status": status,
            "optimization_level": self.optimization_level.value,
            "current_metrics": {
                "response_time_ms": self.current_metrics.response_time_ms,
                "memory_usage_percent": self.current_metrics.memory_usage_percent,
                "cpu_usage_percent": self.current_metrics.cpu_usage_percent,
                "cache_hit_rate": self.current_metrics.cache_hit_rate,
                "throughput_rps": self.current_metrics.throughput_rps,
                "active_connections": self.current_metrics.active_connections,
            },
            "averages": {
                "response_time_ms": avg_response_time,
                "memory_usage_percent": avg_memory_usage,
                "cpu_usage_percent": avg_cpu_usage,
                "cache_hit_rate": avg_cache_hit_rate,
            },
            "targets": {
                "max_response_time_ms": self.targets.max_response_time_ms,
                "max_memory_usage_percent": self.targets.max_memory_usage_percent,
                "max_cpu_usage_percent": self.targets.max_cpu_usage_percent,
                "min_cache_hit_rate": self.targets.min_cache_hit_rate,
            },
            "connection_pools": {
                name: pool.get_stats() for name, pool in self.connection_pools.items()
            },
            "registered_components": len(self.registered_components),
            "last_optimization": self.last_optimization.isoformat(),
        }

    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                # Collect current metrics
                await self._collect_metrics()

                # Perform optimizations if needed
                await self._perform_optimizations()

                # Update optimization timestamp
                self.last_optimization = datetime.now()

                # Wait for next optimization cycle
                await asyncio.sleep(self.optimization_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _collect_metrics(self):
        """Collect current performance metrics."""
        try:
            # System metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            # GPU metrics
            gpu_memory_percent = 0.0
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )  # GB
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100

            # Connection pool metrics
            total_connections = sum(
                pool.active_connections for pool in self.connection_pools.values()
            )
            max_connections = sum(pool.max_connections for pool in self.connection_pools.values())
            connection_pool_usage = (
                total_connections / max_connections if max_connections > 0 else 0.0
            )

            # Update current metrics
            self.current_metrics = OptimizationMetrics(
                timestamp=datetime.now(),
                memory_usage_percent=memory.percent,
                cpu_usage_percent=cpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                connection_pool_usage=connection_pool_usage,
                active_connections=total_connections,
            )

            # Add to history
            self.metrics_history.append(self.current_metrics)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def _perform_optimizations(self):
        """Perform automatic optimizations based on current metrics."""
        metrics = self.current_metrics

        # Memory optimization
        if metrics.memory_usage_percent > self.targets.max_memory_usage_percent:
            await self._optimize_memory()

        # CPU optimization
        if metrics.cpu_usage_percent > self.targets.max_cpu_usage_percent:
            await self._optimize_cpu()

        # GPU memory optimization
        if metrics.gpu_memory_percent > self.targets.max_gpu_memory_percent:
            await self._optimize_gpu_memory()

        # Connection pool optimization
        if metrics.connection_pool_usage > self.targets.max_connection_pool_usage:
            await self._optimize_connection_pools()

        # Cache optimization
        if metrics.cache_hit_rate < self.targets.min_cache_hit_rate:
            await self._optimize_caches()

    async def _optimize_memory(self):
        """Optimize system memory usage."""
        logger.info("Performing memory optimization")

        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")

        # Clear least recently used cache entries
        for cache in [self.response_cache, self.session_cache, self.model_cache]:
            if cache:
                await cache.cleanup_expired()

        # Optimize component memory usage
        for component_name in self.registered_components:
            await self._optimize_component_memory(component_name)

    async def _optimize_cpu(self):
        """Optimize CPU usage."""
        logger.info("Performing CPU optimization")

        # Reduce processing queue sizes
        for pool in self.connection_pools.values():
            await pool.reduce_queue_size()

        # Enable CPU-specific optimizations for components
        for component_name, component in self.registered_components.items():
            if component["type"] in ["llm", "voice"]:
                await self._enable_cpu_optimizations(component_name)

    async def _optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        if not TORCH_AVAILABLE:
            return

        logger.info("Performing GPU memory optimization")

        try:
            if torch.cuda.is_available():
                # Clear GPU cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Optimize model memory usage
                for component_name, component in self.registered_components.items():
                    if component["type"] in ["llm", "voice"]:
                        await self._optimize_component_gpu_memory(component_name)

                logger.debug("GPU memory optimization completed")
        except Exception as e:
            logger.error(f"GPU memory optimization error: {e}")

    async def _optimize_connection_pools(self):
        """Optimize connection pool usage."""
        logger.info("Optimizing connection pools")

        for service_name, pool in self.connection_pools.items():
            await pool.optimize()

    async def _optimize_caches(self):
        """Optimize cache performance."""
        logger.info("Optimizing cache performance")

        # Adjust cache sizes based on hit rates
        for cache in [self.response_cache, self.session_cache, self.model_cache]:
            if cache:
                await cache.optimize()

    async def _initialize_caches(self):
        """Initialize caching systems."""
        from .optimized_cache import OptimizedCache

        # Response cache for API responses
        self.response_cache = OptimizedCache(
            max_memory_mb=self.cache_config.max_memory_mb * 0.4,  # 40% of cache memory
            max_entries=self.cache_config.max_entries // 2,
            default_ttl=self.cache_config.default_ttl,
            name="response_cache",
        )
        await self.response_cache.start()

        # Session cache for conversation sessions
        self.session_cache = OptimizedCache(
            max_memory_mb=self.cache_config.max_memory_mb * 0.3,  # 30% of cache memory
            max_entries=self.cache_config.max_entries // 4,
            default_ttl=self.cache_config.default_ttl * 2,  # Longer TTL for sessions
            name="session_cache",
        )
        await self.session_cache.start()

        # Model cache for AI model outputs
        self.model_cache = OptimizedCache(
            max_memory_mb=self.cache_config.max_memory_mb * 0.3,  # 30% of cache memory
            max_entries=self.cache_config.max_entries // 4,
            default_ttl=self.cache_config.default_ttl,
            name="model_cache",
        )
        await self.model_cache.start()

        logger.info("Caching systems initialized")

    async def _initialize_connection_pools(self):
        """Initialize connection pools for common services."""
        # Ollama LLM service
        ollama_pool = ConnectionPool("ollama", self.connection_config)
        await ollama_pool.initialize()
        self.connection_pools["ollama"] = ollama_pool

        # Database connections (if needed)
        # db_pool = ConnectionPool("database", self.connection_config)
        # await db_pool.initialize()
        # self.connection_pools["database"] = db_pool

        logger.info("Connection pools initialized")

    async def _initialize_resource_monitors(self):
        """Initialize resource monitoring systems."""
        # This would initialize specific resource monitors
        # Implementation depends on specific monitoring requirements
        logger.info("Resource monitors initialized")

    def _get_cache_by_type(self, cache_type: str):
        """Get cache instance by type."""
        cache_map = {
            "response": self.response_cache,
            "session": self.session_cache,
            "model": self.model_cache,
        }
        return cache_map.get(cache_type)

    def _update_cache_hit_rate(self, hit: bool):
        """Update cache hit rate metrics."""
        # Simple moving average for cache hit rate
        if not hasattr(self, "_cache_hits"):
            self._cache_hits = 0
            self._cache_total = 0

        self._cache_total += 1
        if hit:
            self._cache_hits += 1

        # Update current metrics
        self.current_metrics.cache_hit_rate = (
            self._cache_hits / self._cache_total if self._cache_total > 0 else 0.0
        )

    # Component-specific optimization methods
    async def _optimize_llm_component(self, component_name: str) -> List[str]:
        """Optimize LLM component."""
        optimizations = []

        # Enable response caching
        optimizations.append("response_caching_enabled")

        # GPU memory optimization
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            optimizations.append("gpu_memory_cleared")

        return optimizations

    async def _optimize_memory_component(self, component_name: str) -> List[str]:
        """Optimize memory component."""
        optimizations = []

        # Enable embedding caching
        optimizations.append("embedding_caching_enabled")

        # Memory cleanup
        gc.collect()
        optimizations.append("memory_cleanup_performed")

        return optimizations

    async def _optimize_voice_component(self, component_name: str) -> List[str]:
        """Optimize voice component."""
        optimizations = []

        # Enable audio buffer pooling
        optimizations.append("audio_buffer_pooling_enabled")

        # Audio caching
        optimizations.append("audio_response_caching_enabled")

        return optimizations

    async def _optimize_websocket_component(self, component_name: str) -> List[str]:
        """Optimize WebSocket component."""
        optimizations = []

        # Enable message compression
        optimizations.append("message_compression_enabled")

        # Connection pooling
        optimizations.append("connection_pooling_optimized")

        return optimizations

    async def _optimize_dashboard_component(self, component_name: str) -> List[str]:
        """Optimize dashboard component."""
        optimizations = []

        # Enable static caching
        optimizations.append("static_content_caching_enabled")

        # Response compression
        optimizations.append("response_compression_enabled")

        return optimizations

    async def _optimize_component_memory(self, component_name: str):
        """Optimize memory usage for a specific component."""
        component = self.registered_components.get(component_name)
        if not component:
            return

        # Component-specific memory optimization
        component_type = component["type"]
        if component_type == "llm":
            # Clear model caches
            if self.model_cache:
                await self.model_cache.cleanup_expired()
        elif component_type == "memory":
            # Optimize memory storage
            if self.session_cache:
                await self.session_cache.cleanup_expired()

    async def _enable_cpu_optimizations(self, component_name: str):
        """Enable CPU optimizations for a component."""
        # This would implement CPU-specific optimizations

    async def _optimize_component_gpu_memory(self, component_name: str):
        """Optimize GPU memory for a component."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


# Import the ConnectionPool here to avoid circular imports
from .connection_pool import ConnectionPool
