"""
WebSocket performance tracking integration.

This module provides performance monitoring and metrics collection
for the WebSocket interface.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from .server import CodaWebSocketServer
from .events import EventType

logger = logging.getLogger("coda.websocket.performance")


class PerformanceTracker:
    """
    Performance tracker for timing operations and collecting metrics.
    """

    def __init__(self):
        """Initialize the performance tracker."""
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}

    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()

    def end_timing(self, operation: str) -> float:
        """End timing an operation and return the duration."""
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation: {operation}")
            return 0.0

        duration = time.time() - self.start_times[operation]
        self.timings[operation] = duration
        del self.start_times[operation]
        return duration

    @asynccontextmanager
    async def time_operation(self, operation: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timings[operation] = duration

    def increment_counter(self, counter: str, amount: int = 1) -> None:
        """Increment a counter."""
        self.counters[counter] = self.counters.get(counter, 0) + amount

    def get_timing(self, operation: str) -> Optional[float]:
        """Get the timing for an operation."""
        return self.timings.get(operation)

    def get_counter(self, counter: str) -> int:
        """Get the value of a counter."""
        return self.counters.get(counter, 0)

    def get_all_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return self.timings.copy()

    def get_all_counters(self) -> Dict[str, int]:
        """Get all counter values."""
        return self.counters.copy()

    def reset(self) -> None:
        """Reset all timings and counters."""
        self.timings.clear()
        self.start_times.clear()
        self.counters.clear()


class SystemMetricsCollector:
    """
    Collector for system performance metrics.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self.process = psutil.Process()
        self.start_time = time.time()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return 0.0

    def get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage in MB (if available)."""
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                return memory_allocated
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
        return None

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all system metrics."""
        return {
            "memory_mb": self.get_memory_usage(),
            "cpu_percent": self.get_cpu_usage(),
            "gpu_vram_mb": self.get_gpu_memory(),
            "uptime_seconds": self.get_uptime(),
        }


class WebSocketPerfIntegration:
    """
    Integration between performance tracking and the WebSocket server.
    
    This class provides methods to connect performance monitoring to the WebSocket server,
    allowing clients to receive real-time performance metrics.
    """

    def __init__(self, server: CodaWebSocketServer, metrics_interval: float = 5.0):
        """
        Initialize the performance integration.

        Args:
            server: The WebSocket server to use
            metrics_interval: Interval between metrics broadcasts in seconds
        """
        self.server = server
        self.metrics_interval = metrics_interval
        self.performance_tracker = PerformanceTracker()
        self.metrics_collector = SystemMetricsCollector()
        self.metrics_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("WebSocket performance integration initialized")

    async def start(self) -> None:
        """Start the performance monitoring."""
        if self.running:
            return

        self.running = True
        self.metrics_task = asyncio.create_task(self._metrics_loop())
        logger.info("Performance monitoring started")

    async def stop(self) -> None:
        """Stop the performance monitoring."""
        if not self.running:
            return

        self.running = False
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _metrics_loop(self) -> None:
        """Main metrics collection and broadcasting loop."""
        while self.running:
            try:
                # Collect system metrics
                metrics = self.metrics_collector.get_all_metrics()
                
                # Broadcast system metrics
                await self.server.broadcast_system_metrics(**metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retrying

    async def track_operation(self, component: str, operation: str, duration_ms: float, 
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track an operation's performance.

        Args:
            component: The component performing the operation
            operation: The operation being performed
            duration_ms: Duration in milliseconds
            metadata: Optional metadata about the operation
        """
        # Record in performance tracker
        self.performance_tracker.timings[f"{component}.{operation}"] = duration_ms / 1000.0
        
        # Broadcast latency trace event
        await self.server.broadcast_event(
            EventType.LATENCY_TRACE,
            {
                "component": component,
                "operation": operation,
                "duration_ms": duration_ms,
                "metadata": metadata,
            }
        )

    async def track_component_timings(self, component: str, timings: Dict[str, float]) -> None:
        """
        Track multiple timings for a component.

        Args:
            component: The component name
            timings: Dictionary of operation -> duration_ms
        """
        # Record in performance tracker
        for operation, duration_ms in timings.items():
            self.performance_tracker.timings[f"{component}.{operation}"] = duration_ms / 1000.0

        # Broadcast component timing event
        await self.server.broadcast_event(
            EventType.COMPONENT_TIMING,
            {"component": component, "timings": timings}
        )

    @asynccontextmanager
    async def time_operation(self, component: str, operation: str, 
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for timing operations with automatic WebSocket reporting.

        Args:
            component: The component performing the operation
            operation: The operation being performed
            metadata: Optional metadata about the operation
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            await self.track_operation(component, operation, duration_ms, metadata)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        return {
            "system_metrics": self.metrics_collector.get_all_metrics(),
            "operation_timings": self.performance_tracker.get_all_timings(),
            "counters": self.performance_tracker.get_all_counters(),
            "server_stats": self.server.get_stats(),
        }

    async def broadcast_performance_summary(self) -> None:
        """Broadcast a performance summary to all clients."""
        summary = self.get_performance_summary()
        await self.server.broadcast_system_info({
            "event": "performance_summary",
            "data": summary,
        })

    def increment_counter(self, counter: str, amount: int = 1) -> None:
        """Increment a performance counter."""
        self.performance_tracker.increment_counter(counter, amount)

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.performance_tracker.reset()
        logger.info("Performance metrics reset")


# Convenience decorators for performance tracking

def track_async_operation(perf_integration: WebSocketPerfIntegration, component: str, operation: str):
    """Decorator for tracking async operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with perf_integration.time_operation(component, operation):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def track_sync_operation(perf_integration: WebSocketPerfIntegration, component: str, operation: str):
    """Decorator for tracking synchronous operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start_time) * 1000
                # Schedule the async tracking
                asyncio.create_task(
                    perf_integration.track_operation(component, operation, duration_ms)
                )
        return wrapper
    return decorator
