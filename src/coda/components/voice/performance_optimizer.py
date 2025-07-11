"""
Voice - LLM Performance Optimization

This module provides performance optimization for real - time voice processing with LLMs,
focusing on latency reduction, throughput optimization, and resource management.
"""

import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .context_integration import ContextConfig, VoiceContextManager
from .llm_integration import VoiceLLMConfig
from .models import VoiceMessage, VoiceResponse

logger = logging.getLogger("coda.voice.performance_optimizer")


class OptimizationLevel(str, Enum):
    """Performance optimization levels."""

    MINIMAL = "minimal"  # Basic optimizations
    BALANCED = "balanced"  # Balance between quality and speed
    AGGRESSIVE = "aggressive"  # Maximum speed optimizations
    QUALITY = "quality"  # Prioritize quality over speed


@dataclass
class PerformanceMetrics:
    """Performance metrics for voice processing."""

    # Latency metrics (milliseconds)
    total_latency_ms: float = 0.0
    context_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    audio_latency_ms: float = 0.0

    # Throughput metrics
    messages_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Resource metrics
    vram_usage_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_gb: float = 0.0

    # Quality metrics
    response_quality_score: float = 0.0
    context_relevance_score: float = 0.0

    # System metrics
    queue_depth: int = 0
    active_connections: int = 0
    cache_hit_rate: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""

    # Optimization level
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED

    # Latency targets (milliseconds)
    target_total_latency_ms: float = 500.0
    target_llm_latency_ms: float = 300.0
    target_context_latency_ms: float = 50.0

    # Throughput targets
    target_messages_per_second: float = 10.0
    target_tokens_per_second: float = 50.0

    # Resource limits
    max_vram_usage_gb: float = 20.0
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_gb: float = 8.0

    # Queue management
    max_queue_depth: int = 10
    queue_timeout_seconds: float = 5.0

    # Caching
    enable_response_caching: bool = True
    enable_context_caching: bool = True
    cache_ttl_seconds: int = 300

    # Parallel processing
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4

    # Model optimization
    enable_model_quantization: bool = False
    enable_kv_caching: bool = True
    enable_speculative_decoding: bool = False


class PerformanceOptimizer:
    """
    Optimizes voice - LLM processing for real - time performance.

    Features:
    - Latency optimization
    - Throughput optimization
    - Resource management
    - Adaptive optimization
    - Performance monitoring
    """

    def __init__(self, config: OptimizationConfig):
        """Initialize the performance optimizer."""
        self.config = config

        # Performance tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_metrics = PerformanceMetrics()

        # Processing queue
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_depth)
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # Caching
        self.response_cache: Dict[str, Tuple[VoiceResponse, float]] = {}
        self.context_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

        # Thread pool for CPU - intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_worker_threads)

        # Optimization state
        self.optimization_active = False
        self.adaptive_adjustments: Dict[str, Any] = {}

        logger.info(f"PerformanceOptimizer initialized with {config.optimization_level} level")

    async def optimize_processing(
        self,
        voice_message: VoiceMessage,
        context_manager: VoiceContextManager,
        conversation_state: Optional[Any] = None,
    ) -> VoiceResponse:
        """
        Process voice message with performance optimizations.

        Args:
            voice_message: The voice message to process
            context_manager: Voice context manager
            conversation_state: Current conversation state

        Returns:
            Optimized voice response
        """
        start_time = time.time()

        try:
            # Check cache first
            if self.config.enable_response_caching:
                cached_response = self._check_response_cache(voice_message)
                if cached_response:
                    logger.debug("Using cached response")
                    return cached_response

            # Queue management
            if self.processing_queue.qsize() >= self.config.max_queue_depth:
                logger.warning("Processing queue full, dropping oldest request")
                try:
                    self.processing_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            # Add to processing queue
            await asyncio.wait_for(
                self.processing_queue.put(voice_message), timeout=self.config.queue_timeout_seconds
            )

            # Process with optimizations
            if self.config.enable_parallel_processing:
                response = await self._process_parallel(
                    voice_message, context_manager, conversation_state
                )
            else:
                response = await self._process_sequential(
                    voice_message, context_manager, conversation_state
                )

            # Cache response
            if self.config.enable_response_caching:
                self._cache_response(voice_message, response)

            # Update metrics
            total_latency = (time.time() - start_time) * 1000
            self._update_metrics(total_latency, response)

            # Adaptive optimization
            if self.optimization_active:
                await self._adaptive_optimize()

            return response

        except Exception as e:
            logger.error(f"Optimized processing failed: {e}")
            # Fallback to basic processing
            return await context_manager.process_with_context(voice_message, conversation_state)

        finally:
            # Remove from queue
            try:
                self.processing_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

    async def _process_parallel(
        self,
        voice_message: VoiceMessage,
        context_manager: VoiceContextManager,
        conversation_state: Optional[Any],
    ) -> VoiceResponse:
        """Process with parallel optimizations."""

        # Start context building in parallel with other preparations
        context_task = asyncio.create_task(
            context_manager.context_injector.build_context(voice_message, conversation_state)
        )

        # Prepare other components in parallel
        preparation_tasks = []

        if self.config.optimization_level == OptimizationLevel.AGGRESSIVE:
            # Pre - warm model if needed
            preparation_tasks.append(self._prewarm_model(context_manager))

        # Wait for context and preparations
        context = await context_task
        if preparation_tasks:
            await asyncio.gather(*preparation_tasks, return_exceptions=True)

        # Process with context
        return await context_manager.process_with_context(voice_message, conversation_state)

    async def _process_sequential(
        self,
        voice_message: VoiceMessage,
        context_manager: VoiceContextManager,
        conversation_state: Optional[Any],
    ) -> VoiceResponse:
        """Process with sequential optimizations."""
        return await context_manager.process_with_context(voice_message, conversation_state)

    async def _prewarm_model(self, context_manager: VoiceContextManager) -> None:
        """Pre - warm the model for faster inference."""
        try:
            if hasattr(context_manager, "current_processor") and context_manager.current_processor:
                # This would trigger model loading / warming
                pass
        except Exception as e:
            logger.debug(f"Model pre - warming failed: {e}")

    def _check_response_cache(self, voice_message: VoiceMessage) -> Optional[VoiceResponse]:
        """Check if response is cached."""
        cache_key = self._generate_cache_key(voice_message)

        if cache_key in self.response_cache:
            response, timestamp = self.response_cache[cache_key]

            # Check if cache is still valid
            if time.time() - timestamp < self.config.cache_ttl_seconds:
                self.current_metrics.cache_hit_rate += 1
                return response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]

        return None

    def _cache_response(self, voice_message: VoiceMessage, response: VoiceResponse) -> None:
        """Cache the response."""
        cache_key = self._generate_cache_key(voice_message)
        self.response_cache[cache_key] = (response, time.time())

        # Clean up old cache entries
        self._cleanup_cache()

    def _generate_cache_key(self, voice_message: VoiceMessage) -> str:
        """Generate cache key for voice message."""
        # Simple cache key based on text content
        text_content = voice_message.text_content or ""
        return f"{voice_message.conversation_id}:{hash(text_content)}"

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []

        for key, (_, timestamp) in self.response_cache.items():
            if current_time - timestamp > self.config.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.response_cache[key]

    def _update_metrics(self, total_latency: float, response: VoiceResponse) -> None:
        """Update performance metrics."""
        self.current_metrics.total_latency_ms = total_latency
        self.current_metrics.llm_latency_ms = getattr(response, "llm_latency_ms", 0.0) or 0.0
        self.current_metrics.queue_depth = self.processing_queue.qsize()

        # Calculate throughput
        if len(self.metrics_history) > 0:
            time_window = 60.0  # 1 minute window
            recent_metrics = [
                m for m in self.metrics_history if time.time() - m.get("timestamp", 0) < time_window
            ]
            if recent_metrics:
                self.current_metrics.messages_per_second = len(recent_metrics) / time_window

        # Add to history
        self.metrics_history.append(
            {
                "timestamp": time.time(),
                "total_latency_ms": total_latency,
                "llm_latency_ms": self.current_metrics.llm_latency_ms,
                "queue_depth": self.current_metrics.queue_depth,
            }
        )

    async def _adaptive_optimize(self) -> None:
        """Perform adaptive optimization based on current metrics."""
        try:
            # Check if we're meeting latency targets
            if self.current_metrics.total_latency_ms > self.config.target_total_latency_ms:
                await self._optimize_for_latency()

            # Check if we're meeting throughput targets
            if self.current_metrics.messages_per_second < self.config.target_messages_per_second:
                await self._optimize_for_throughput()

            # Check resource usage
            if self.current_metrics.vram_usage_gb > self.config.max_vram_usage_gb:
                await self._optimize_for_memory()

        except Exception as e:
            logger.error(f"Adaptive optimization failed: {e}")

    async def _optimize_for_latency(self) -> None:
        """Optimize for lower latency."""
        logger.info("Optimizing for latency")

        # Reduce context length
        if "context_length" not in self.adaptive_adjustments:
            self.adaptive_adjustments["context_length"] = 0.8

        # Enable more aggressive caching
        self.config.enable_response_caching = True
        self.config.enable_context_caching = True

    async def _optimize_for_throughput(self) -> None:
        """Optimize for higher throughput."""
        logger.info("Optimizing for throughput")

        # Increase parallel processing
        self.config.enable_parallel_processing = True

        # Increase queue size
        if self.config.max_queue_depth < 20:
            self.config.max_queue_depth += 2

    async def _optimize_for_memory(self) -> None:
        """Optimize for lower memory usage."""
        logger.info("Optimizing for memory usage")

        # Clear caches
        self.response_cache.clear()
        self.context_cache.clear()

        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.current_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}

        recent_latencies = [m["total_latency_ms"] for m in self.metrics_history[-100:]]

        return {
            "current_metrics": {
                "total_latency_ms": self.current_metrics.total_latency_ms,
                "messages_per_second": self.current_metrics.messages_per_second,
                "queue_depth": self.current_metrics.queue_depth,
                "cache_hit_rate": self.current_metrics.cache_hit_rate,
            },
            "performance_stats": {
                "avg_latency_ms": (
                    sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
                ),
                "min_latency_ms": min(recent_latencies) if recent_latencies else 0,
                "max_latency_ms": max(recent_latencies) if recent_latencies else 0,
                "p95_latency_ms": (
                    sorted(recent_latencies)[int(len(recent_latencies) * 0.95)]
                    if recent_latencies
                    else 0
                ),
            },
            "optimization_status": {
                "level": self.config.optimization_level.value,
                "adaptive_active": self.optimization_active,
                "adjustments": self.adaptive_adjustments,
            },
            "cache_stats": {
                "response_cache_size": len(self.response_cache),
                "context_cache_size": len(self.context_cache),
            },
        }

    def enable_adaptive_optimization(self) -> None:
        """Enable adaptive optimization."""
        self.optimization_active = True
        logger.info("Adaptive optimization enabled")

    def disable_adaptive_optimization(self) -> None:
        """Disable adaptive optimization."""
        self.optimization_active = False
        logger.info("Adaptive optimization disabled")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cancel active tasks
            for task in self.active_tasks.values():
                task.cancel()

            # Clear caches
            self.response_cache.clear()
            self.context_cache.clear()

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            logger.info("PerformanceOptimizer cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
