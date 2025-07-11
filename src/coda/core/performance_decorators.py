"""
Performance decorators and context managers for Coda.

This module provides easy-to-use decorators and context managers
for performance monitoring and optimization.
"""

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

from .performance_monitor import get_performance_monitor

logger = logging.getLogger("coda.core.performance_decorators")

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class PerformanceContext:
    """Context for performance measurement."""

    component_name: str
    operation_name: str
    start_time: float
    custom_metrics: Dict[str, float]

    def add_metric(self, name: str, value: float) -> None:
        """Add custom metric to context."""
        self.custom_metrics[name] = value

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time


@contextmanager
def performance_context(component_name: str, operation_name: str = "operation"):
    """
    Context manager for performance monitoring.

    Usage:
        with performance_context("llm", "generate_response") as ctx:
            # Your code here
            result = do_something()
            ctx.add_metric("tokens_generated", len(result))
    """
    monitor = get_performance_monitor()

    context = PerformanceContext(
        component_name=component_name,
        operation_name=operation_name,
        start_time=time.time(),
        custom_metrics={},
    )

    success = True
    try:
        yield context
    except Exception as e:
        success = False
        logger.error(f"Performance context error in {component_name}.{operation_name}: {e}")
        raise
    finally:
        elapsed_time = context.get_elapsed_time()
        monitor.record_component_operation(
            component_name=component_name,
            execution_time=elapsed_time,
            success=success,
            custom_metrics=context.custom_metrics,
        )


@asynccontextmanager
async def async_performance_context(component_name: str, operation_name: str = "operation"):
    """
    Async context manager for performance monitoring.

    Usage:
        async with async_performance_context("memory", "store_conversation") as ctx:
            # Your async code here
            await store_data()
            ctx.add_metric("items_stored", count)
    """
    monitor = get_performance_monitor()

    context = PerformanceContext(
        component_name=component_name,
        operation_name=operation_name,
        start_time=time.time(),
        custom_metrics={},
    )

    success = True
    try:
        yield context
    except Exception as e:
        success = False
        logger.error(f"Async performance context error in {component_name}.{operation_name}: {e}")
        raise
    finally:
        elapsed_time = context.get_elapsed_time()
        monitor.record_component_operation(
            component_name=component_name,
            execution_time=elapsed_time,
            success=success,
            custom_metrics=context.custom_metrics,
        )


def performance_monitor(component_name: str, operation_name: Optional[str] = None):
    """
    Decorator for automatic performance monitoring.

    Usage:
        @performance_monitor("llm")
        def generate_response(self, prompt):
            return self.model.generate(prompt)

        @performance_monitor("memory", "store")
        async def store_conversation(self, conversation):
            await self.storage.store(conversation)
    """

    def decorator(func: F) -> F:
        op_name = operation_name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with async_performance_context(component_name, op_name) as ctx:
                    result = await func(*args, **kwargs)

                    # Add result-based metrics if possible
                    if hasattr(result, "__len__"):
                        try:
                            ctx.add_metric("result_size", len(result))
                        except (TypeError, AttributeError):
                            pass

                    return result

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with performance_context(component_name, op_name) as ctx:
                    result = func(*args, **kwargs)

                    # Add result-based metrics if possible
                    if hasattr(result, "__len__"):
                        try:
                            ctx.add_metric("result_size", len(result))
                        except (TypeError, AttributeError):
                            pass

                    return result

            return sync_wrapper

    return decorator


def memory_monitor(threshold_mb: float = 100.0):
    """
    Decorator for memory usage monitoring.

    Usage:
        @memory_monitor(threshold_mb=500.0)
        def process_large_data(self, data):
            return self.process(data)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB

            try:
                result = func(*args, **kwargs)

                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_delta = memory_after - memory_before

                if memory_delta > threshold_mb:
                    logger.warning(
                        f"High memory usage in {func.__name__}: "
                        f"{memory_delta:.1f}MB (threshold: {threshold_mb}MB)"
                    )

                # Record memory metrics
                monitor = get_performance_monitor()
                monitor.record_component_operation(
                    component_name=func.__module__.split(".")[-1],
                    execution_time=0.0,  # Not measuring time here
                    success=True,
                    custom_metrics={
                        "memory_delta_mb": memory_delta,
                        "memory_before_mb": memory_before,
                        "memory_after_mb": memory_after,
                    },
                )

                return result

            except Exception as e:
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_delta = memory_after - memory_before

                monitor = get_performance_monitor()
                monitor.record_component_operation(
                    component_name=func.__module__.split(".")[-1],
                    execution_time=0.0,
                    success=False,
                    custom_metrics={
                        "memory_delta_mb": memory_delta,
                        "memory_before_mb": memory_before,
                        "memory_after_mb": memory_after,
                    },
                )
                raise

        return wrapper

    return decorator


def cpu_monitor(threshold_percent: float = 80.0):
    """
    Decorator for CPU usage monitoring.

    Usage:
        @cpu_monitor(threshold_percent=90.0)
        def cpu_intensive_task(self, data):
            return self.process(data)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import os

            import psutil

            process = psutil.Process(os.getpid())
            cpu_before = process.cpu_percent()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                execution_time = time.time() - start_time
                cpu_after = process.cpu_percent()

                # Calculate average CPU usage during execution
                avg_cpu = (cpu_before + cpu_after) / 2

                if avg_cpu > threshold_percent:
                    logger.warning(
                        f"High CPU usage in {func.__name__}: "
                        f"{avg_cpu:.1f}% (threshold: {threshold_percent}%)"
                    )

                # Record CPU metrics
                monitor = get_performance_monitor()
                monitor.record_component_operation(
                    component_name=func.__module__.split(".")[-1],
                    execution_time=execution_time,
                    success=True,
                    custom_metrics={
                        "cpu_usage_percent": avg_cpu,
                        "cpu_before_percent": cpu_before,
                        "cpu_after_percent": cpu_after,
                    },
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                cpu_after = process.cpu_percent()
                avg_cpu = (cpu_before + cpu_after) / 2

                monitor = get_performance_monitor()
                monitor.record_component_operation(
                    component_name=func.__module__.split(".")[-1],
                    execution_time=execution_time,
                    success=False,
                    custom_metrics={
                        "cpu_usage_percent": avg_cpu,
                        "cpu_before_percent": cpu_before,
                        "cpu_after_percent": cpu_after,
                    },
                )
                raise

        return wrapper

    return decorator


def gpu_monitor():
    """
    Decorator for GPU usage monitoring.

    Usage:
        @gpu_monitor()
        def gpu_inference(self, data):
            return self.model(data)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import torch

                gpu_available = torch.cuda.is_available()
            except ImportError:
                gpu_available = False

            if not gpu_available:
                return func(*args, **kwargs)

            # Record GPU memory before
            gpu_memory_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                execution_time = time.time() - start_time
                gpu_memory_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                gpu_memory_delta = gpu_memory_after - gpu_memory_before

                # Record GPU metrics
                monitor = get_performance_monitor()
                monitor.record_component_operation(
                    component_name=func.__module__.split(".")[-1],
                    execution_time=execution_time,
                    success=True,
                    custom_metrics={
                        "gpu_memory_delta_mb": gpu_memory_delta,
                        "gpu_memory_before_mb": gpu_memory_before,
                        "gpu_memory_after_mb": gpu_memory_after,
                    },
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                gpu_memory_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                gpu_memory_delta = gpu_memory_after - gpu_memory_before

                monitor = get_performance_monitor()
                monitor.record_component_operation(
                    component_name=func.__module__.split(".")[-1],
                    execution_time=execution_time,
                    success=False,
                    custom_metrics={
                        "gpu_memory_delta_mb": gpu_memory_delta,
                        "gpu_memory_before_mb": gpu_memory_before,
                        "gpu_memory_after_mb": gpu_memory_after,
                    },
                )
                raise

        return wrapper

    return decorator


def comprehensive_monitor(
    component_name: str,
    operation_name: Optional[str] = None,
    memory_threshold_mb: float = 100.0,
    cpu_threshold_percent: float = 80.0,
):
    """
    Comprehensive monitoring decorator that combines all monitoring types.

    Usage:
        @comprehensive_monitor("llm", "generate", memory_threshold_mb=500.0)
        def generate_response(self, prompt):
            return self.model.generate(prompt)
    """

    def decorator(func: F) -> F:
        # Apply all monitoring decorators
        monitored_func = performance_monitor(component_name, operation_name)(func)
        monitored_func = memory_monitor(memory_threshold_mb)(monitored_func)
        monitored_func = cpu_monitor(cpu_threshold_percent)(monitored_func)
        monitored_func = gpu_monitor()(monitored_func)

        return monitored_func

    return decorator


class PerformanceBenchmark:
    """
    Utility class for performance benchmarking.

    Usage:
        benchmark = PerformanceBenchmark("llm_inference")

        for i in range(100):
            with benchmark.measure():
                result = model.generate(prompt)

        stats = benchmark.get_statistics()
    """

    def __init__(self, name: str):
        self.name = name
        self.measurements: list = []
        self.start_time: Optional[float] = None

    @contextmanager
    def measure(self):
        """Context manager for individual measurements."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.measurements.append(elapsed)

    def get_statistics(self) -> Dict[str, float]:
        """Get benchmark statistics."""
        if not self.measurements:
            return {}

        measurements = self.measurements
        return {
            "count": len(measurements),
            "total_time": sum(measurements),
            "average_time": sum(measurements) / len(measurements),
            "min_time": min(measurements),
            "max_time": max(measurements),
            "median_time": sorted(measurements)[len(measurements) // 2],
            "throughput_ops_per_sec": (
                len(measurements) / sum(measurements) if sum(measurements) > 0 else 0
            ),
        }

    def reset(self) -> None:
        """Reset benchmark measurements."""
        self.measurements.clear()

    def report(self) -> str:
        """Generate benchmark report."""
        stats = self.get_statistics()
        if not stats:
            return f"Benchmark '{self.name}': No measurements"

        return (
            f"Benchmark '{self.name}':\n"
            f"  Count: {stats['count']}\n"
            f"  Average: {stats['average_time']:.4f}s\n"
            f"  Min: {stats['min_time']:.4f}s\n"
            f"  Max: {stats['max_time']:.4f}s\n"
            f"  Median: {stats['median_time']:.4f}s\n"
            f"  Throughput: {stats['throughput_ops_per_sec']:.2f} ops/sec"
        )
