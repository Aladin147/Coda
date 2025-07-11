"""
Advanced performance profiler for voice processing system.

This module provides detailed performance monitoring, bottleneck detection,
and optimization recommendations.
"""

import gc
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

import psutil
import torch

logger = logging.getLogger(__name__)


class ProfilerLevel(Enum):
    """Profiler detail levels."""

    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    component: str = "unknown"
    operation: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileSession:
    """Profiling session data."""

    session_id: str
    start_time: float
    end_time: Optional[float] = None
    metrics: List[PerformanceMetric] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get session duration."""
        end = self.end_time or time.time()
        return end - self.start_time


class PerformanceProfiler:
    """Advanced performance profiler with bottleneck detection."""

    def __init__(
        self,
        level: ProfilerLevel = ProfilerLevel.DETAILED,
        max_history: int = 10000,
        enable_gpu_monitoring: bool = True,
    ):
        """Initialize performance profiler."""
        self.level = level
        self.max_history = max_history
        self.enable_gpu_monitoring = enable_gpu_monitoring

        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_history)
        self.component_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Active sessions
        self.active_sessions: Dict[str, ProfileSession] = {}
        self.session_counter = 0

        # Performance thresholds
        self.thresholds = {
            "latency_ms": 500.0,
            "memory_mb": 1000.0,
            "cpu_percent": 80.0,
            "gpu_percent": 90.0,
            "gpu_memory_mb": 8000.0,
        }

        # Thread safety
        self.lock = threading.RLock()

        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        logger.info(
            f"PerformanceProfiler initialized: level={level.value}, "
            f"gpu_monitoring={enable_gpu_monitoring}"
        )

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background system monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker, args=(interval,), daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Background monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Background monitoring stopped")

    @contextmanager
    def profile_operation(
        self, component: str, operation: str, session_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Context manager for profiling operations."""
        start_time = time.time()
        operation_id = f"{component}_{operation}_{int(start_time * 1000)}"

        # Record start metrics
        if self.level in [ProfilerLevel.DETAILED, ProfilerLevel.COMPREHENSIVE]:
            self._record_system_metrics(component, f"{operation}_start")

        try:
            yield operation_id
        finally:
            # Record end metrics
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Record operation metrics
            self.record_metric(
                name="operation_duration",
                value=duration_ms,
                unit="ms",
                component=component,
                operation=operation,
            )

            if self.level in [ProfilerLevel.DETAILED, ProfilerLevel.COMPREHENSIVE]:
                self._record_system_metrics(component, f"{operation}_end")

            # Add to session if specified
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id].metrics.append(
                    PerformanceMetric(
                        name="operation_duration",
                        value=duration_ms,
                        unit="ms",
                        component=component,
                        operation=operation,
                    )
                )

    def start_session(self, session_name: str = None) -> str:
        """Start a profiling session."""
        with self.lock:
            self.session_counter += 1
            session_id = session_name or f"session_{self.session_counter}"

            session = ProfileSession(session_id=session_id, start_time=time.time())

            self.active_sessions[session_id] = session
            logger.debug(f"Started profiling session: {session_id}")
            return session_id

    def end_session(self, session_id: str) -> ProfileSession:
        """End a profiling session and return results."""
        with self.lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self.active_sessions[session_id]
            session.end_time = time.time()

            # Analyze session for bottlenecks
            self._analyze_session_bottlenecks(session)

            # Remove from active sessions
            del self.active_sessions[session_id]

            logger.debug(
                f"Ended profiling session: {session_id}, " f"duration: {session.duration:.2f}s"
            )
            return session

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        component: str = "unknown",
        operation: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            component=component,
            operation=operation,
            metadata=metadata or {},
        )

        with self.lock:
            self.metrics_history.append(metric)
            self.component_metrics[component].append(metric)

        # Check for threshold violations
        self._check_thresholds(metric)

    def _record_system_metrics(self, component: str, operation: str) -> None:
        """Record system-level metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            self.record_metric("cpu_usage", cpu_percent, "percent", component, operation)
            self.record_metric("memory_usage", memory.used / (1024**2), "MB", component, operation)
            self.record_metric("memory_percent", memory.percent, "percent", component, operation)

            # GPU metrics if enabled
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**2)
                gpu_memory_cached = torch.cuda.memory_reserved() / (1024**2)

                self.record_metric("gpu_memory", gpu_memory, "MB", component, operation)
                self.record_metric(
                    "gpu_memory_cached", gpu_memory_cached, "MB", component, operation
                )

                # GPU utilization (if nvidia-ml-py is available)
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.record_metric(
                        "gpu_utilization", gpu_util.gpu, "percent", component, operation
                    )
                except ImportError:
                    pass  # nvidia-ml-py not available

        except Exception as e:
            logger.debug(f"Failed to record system metrics: {e}")

    def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric exceeds thresholds."""
        threshold_key = None

        if metric.name == "operation_duration" and metric.unit == "ms":
            threshold_key = "latency_ms"
        elif metric.name == "memory_usage" and metric.unit == "MB":
            threshold_key = "memory_mb"
        elif metric.name == "cpu_usage" and metric.unit == "percent":
            threshold_key = "cpu_percent"
        elif metric.name == "gpu_utilization" and metric.unit == "percent":
            threshold_key = "gpu_percent"
        elif metric.name == "gpu_memory" and metric.unit == "MB":
            threshold_key = "gpu_memory_mb"

        if threshold_key and metric.value > self.thresholds[threshold_key]:
            logger.warning(
                f"Performance threshold exceeded: {metric.name}={metric.value}{metric.unit} "
                f"(threshold: {self.thresholds[threshold_key]}) "
                f"in {metric.component}.{metric.operation}"
            )

    def _analyze_session_bottlenecks(self, session: ProfileSession) -> None:
        """Analyze session for performance bottlenecks."""
        if not session.metrics:
            return

        # Group metrics by component and operation
        component_times = defaultdict(float)
        operation_times = defaultdict(float)

        for metric in session.metrics:
            if metric.name == "operation_duration":
                component_times[metric.component] += metric.value
                operation_times[f"{metric.component}.{metric.operation}"] += metric.value

        # Find bottlenecks (operations taking >20% of total time)
        total_time = sum(component_times.values())
        if total_time > 0:
            for operation, time_ms in operation_times.items():
                if time_ms / total_time > 0.2:  # 20% threshold
                    session.bottlenecks.append(
                        f"{operation}: {time_ms:.1f}ms ({time_ms/total_time*100:.1f}%)"
                    )

        # Generate recommendations
        self._generate_recommendations(session, component_times, operation_times)

    def _generate_recommendations(
        self,
        session: ProfileSession,
        component_times: Dict[str, float],
        operation_times: Dict[str, float],
    ) -> None:
        """Generate optimization recommendations."""
        recommendations = []

        # Check for slow components
        total_time = sum(component_times.values())
        for component, time_ms in component_times.items():
            if time_ms > 1000:  # >1 second
                recommendations.append(f"Optimize {component} component (taking {time_ms:.1f}ms)")

        # Check for memory issues
        memory_metrics = [m for m in session.metrics if m.name == "memory_usage"]
        if memory_metrics:
            max_memory = max(m.value for m in memory_metrics)
            if max_memory > 2000:  # >2GB
                recommendations.append(f"High memory usage detected ({max_memory:.1f}MB)")

        # Check for GPU utilization
        gpu_metrics = [m for m in session.metrics if m.name == "gpu_utilization"]
        if gpu_metrics:
            avg_gpu = sum(m.value for m in gpu_metrics) / len(gpu_metrics)
            if avg_gpu < 50:  # <50% utilization
                recommendations.append("Low GPU utilization - consider optimizing GPU usage")

        session.recommendations = recommendations

    def _monitoring_worker(self, interval: float) -> None:
        """Background monitoring worker."""
        while self.monitoring_active:
            try:
                self._record_system_metrics("system", "background_monitor")
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring worker error: {e}")
                time.sleep(interval)

    def get_performance_summary(
        self, component: Optional[str] = None, last_n_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get performance summary."""
        with self.lock:
            # Filter metrics
            metrics = list(self.metrics_history)

            if component:
                metrics = [m for m in metrics if m.component == component]

            if last_n_minutes:
                cutoff_time = time.time() - (last_n_minutes * 60)
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if not metrics:
                return {"error": "No metrics found"}

            # Calculate summary statistics
            latency_metrics = [m for m in metrics if m.name == "operation_duration"]
            memory_metrics = [m for m in metrics if m.name == "memory_usage"]
            cpu_metrics = [m for m in metrics if m.name == "cpu_usage"]

            summary = {
                "total_metrics": len(metrics),
                "time_range": {
                    "start": min(m.timestamp for m in metrics),
                    "end": max(m.timestamp for m in metrics),
                },
            }

            if latency_metrics:
                latencies = [m.value for m in latency_metrics]
                summary["latency"] = {
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
                }

            if memory_metrics:
                memory_values = [m.value for m in memory_metrics]
                summary["memory"] = {
                    "avg_mb": sum(memory_values) / len(memory_values),
                    "peak_mb": max(memory_values),
                }

            if cpu_metrics:
                cpu_values = [m.value for m in cpu_metrics]
                summary["cpu"] = {
                    "avg_percent": sum(cpu_values) / len(cpu_values),
                    "peak_percent": max(cpu_values),
                }

            return summary

    def cleanup(self) -> None:
        """Cleanup profiler resources."""
        self.stop_monitoring()
        with self.lock:
            self.metrics_history.clear()
            self.component_metrics.clear()
            self.active_sessions.clear()


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def cleanup_performance_profiler() -> None:
    """Cleanup global profiler."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.cleanup()
        _global_profiler = None
