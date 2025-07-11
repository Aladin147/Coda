"""
Core Performance Monitor for Coda.

This module provides comprehensive performance monitoring, CPU tracking,
memory optimization, and system-wide performance profiling.
"""

import gc
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import RTX 5090 optimizer
try:
    from .rtx5090_optimizer import get_rtx5090_optimizer
    RTX5090_OPTIMIZER_AVAILABLE = True
except ImportError:
    RTX5090_OPTIMIZER_AVAILABLE = False

try:
    import pynvml

    pynvml.nvmlInit()
    NVIDIA_ML_AVAILABLE = True
except (ImportError, Exception):
    NVIDIA_ML_AVAILABLE = False

logger = logging.getLogger("coda.core.performance_monitor")


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""

    timestamp: float = field(default_factory=time.time)

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    cpu_freq_mhz: float = 0.0
    load_average: List[float] = field(default_factory=list)

    # Memory metrics
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    memory_percent: float = 0.0

    # GPU metrics (if available)
    gpu_count: int = 0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_temperature_c: float = 0.0

    # Process metrics
    process_memory_gb: float = 0.0
    process_cpu_percent: float = 0.0
    thread_count: int = 0

    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

    # Disk metrics
    disk_usage_percent: float = 0.0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0


@dataclass
class ComponentMetrics:
    """Performance metrics for individual components."""

    component_name: str
    timestamp: float = field(default_factory=time.time)

    # Execution metrics
    operation_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float("inf")
    max_execution_time: float = 0.0

    # Error metrics
    error_count: int = 0
    error_rate: float = 0.0

    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_time_seconds: float = 0.0

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations."""

    alert_id: str
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'warning', 'critical'
    timestamp: float = field(default_factory=time.time)
    message: str = ""


class PerformanceThresholds:
    """Performance threshold configuration."""

    def __init__(self):
        self.thresholds = {
            # System thresholds
            "cpu_percent": {"warning": 70.0, "critical": 90.0},
            "memory_percent": {"warning": 80.0, "critical": 95.0},
            "gpu_utilization_percent": {"warning": 85.0, "critical": 95.0},
            "gpu_memory_percent": {"warning": 80.0, "critical": 90.0},
            # Component thresholds
            "average_execution_time": {"warning": 1.0, "critical": 5.0},
            "execution_time": {"warning": 1.0, "critical": 5.0},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            # Network thresholds
            "network_latency_ms": {"warning": 100.0, "critical": 500.0},
            # Disk thresholds
            "disk_usage_percent": {"warning": 85.0, "critical": 95.0},
        }

    def check_threshold(self, metric_name: str, value: float) -> Optional[str]:
        """Check if value exceeds threshold."""
        if metric_name not in self.thresholds:
            return None

        thresholds = self.thresholds[metric_name]

        if value >= thresholds["critical"]:
            return "critical"
        elif value >= thresholds["warning"]:
            return "warning"

        return None

    def set_threshold(self, metric_name: str, warning: float, critical: float):
        """Set custom threshold."""
        self.thresholds[metric_name] = {"warning": warning, "critical": critical}


class CodaPerformanceMonitor:
    """
    Comprehensive performance monitor for Coda system.

    Provides CPU tracking, memory optimization, performance profiling,
    and real-time monitoring capabilities.
    """

    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 3600,  # 1 hour at 1s intervals
        enable_gpu_monitoring: bool = True,
        enable_component_profiling: bool = True,
    ):
        """Initialize performance monitor."""
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and TORCH_AVAILABLE
        self.enable_component_profiling = enable_component_profiling

        # Metrics storage
        self.system_metrics_history: deque = deque(maxlen=history_size)
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.component_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Performance thresholds and alerts
        self.thresholds = PerformanceThresholds()
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        # Process reference
        self.process = psutil.Process()

        # Optimization state
        self.optimization_enabled = True
        self.last_optimization = 0.0
        self.optimization_interval = 60.0  # 1 minute

        logger.info("CodaPerformanceMonitor initialized")

    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker, daemon=True, name="PerformanceMonitor"
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")

    def _monitoring_worker(self) -> None:
        """Background monitoring worker thread."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)

                # Check thresholds and generate alerts
                self._check_thresholds(metrics)

                # Perform optimization if needed
                if (
                    self.optimization_enabled
                    and time.time() - self.last_optimization > self.optimization_interval
                ):
                    self._perform_optimization()
                    self.last_optimization = time.time()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics()

        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=None)
            metrics.cpu_count = psutil.cpu_count()

            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics.cpu_freq_mhz = cpu_freq.current

            # Load average (Unix-like systems)
            try:
                metrics.load_average = list(psutil.getloadavg())
            except AttributeError:
                pass  # Not available on Windows

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_total_gb = memory.total / (1024**3)
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_available_gb = memory.available / (1024**3)
            metrics.memory_percent = memory.percent

            # Process metrics
            metrics.process_memory_gb = self.process.memory_info().rss / (1024**3)
            metrics.process_cpu_percent = self.process.cpu_percent()
            metrics.thread_count = self.process.num_threads()

            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.network_bytes_sent = net_io.bytes_sent
                metrics.network_bytes_recv = net_io.bytes_recv

            # Disk metrics
            disk_usage = psutil.disk_usage("/")
            metrics.disk_usage_percent = (disk_usage.used / disk_usage.total) * 100

            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.disk_read_bytes = disk_io.read_bytes
                metrics.disk_write_bytes = disk_io.write_bytes

            # GPU metrics
            if self.enable_gpu_monitoring:
                self._collect_gpu_metrics(metrics)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

        return metrics

    def _collect_gpu_metrics(self, metrics: SystemMetrics) -> None:
        """Collect GPU metrics."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                metrics.gpu_count = torch.cuda.device_count()

                # PyTorch GPU memory
                if metrics.gpu_count > 0:
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    metrics.gpu_memory_used_gb = gpu_memory_allocated

                    # Get total GPU memory from nvidia-ml-py if available
                    if NVIDIA_ML_AVAILABLE:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics.gpu_memory_total_gb = mem_info.total / (1024**3)

                        # GPU utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics.gpu_utilization_percent = util.gpu

                        # GPU temperature
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        metrics.gpu_temperature_c = temp

        except Exception as e:
            logger.debug(f"Error collecting GPU metrics: {e}")

    def _check_thresholds(self, metrics: SystemMetrics) -> None:
        """Check metrics against thresholds and generate alerts."""
        checks = [
            ("cpu_percent", metrics.cpu_percent),
            ("memory_percent", metrics.memory_percent),
            ("gpu_utilization_percent", metrics.gpu_utilization_percent),
            ("disk_usage_percent", metrics.disk_usage_percent),
        ]

        for metric_name, value in checks:
            if value > 0:  # Only check valid values
                severity = self.thresholds.check_threshold(metric_name, value)
                if severity:
                    self._generate_alert("system", metric_name, value, severity)

    def _generate_alert(
        self, component: str, metric_name: str, current_value: float, severity: str
    ) -> None:
        """Generate performance alert."""
        alert_id = f"{component}_{metric_name}_{severity}"

        # Check if alert already exists
        if alert_id in self.active_alerts:
            # Update existing alert
            self.active_alerts[alert_id].current_value = current_value
            self.active_alerts[alert_id].timestamp = time.time()
        else:
            # Create new alert
            threshold_value = self.thresholds.thresholds[metric_name][severity]
            alert = PerformanceAlert(
                alert_id=alert_id,
                component=component,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                severity=severity,
                message=f"{component} {metric_name} is {current_value:.1f} (threshold: {threshold_value})",
            )

            self.active_alerts[alert_id] = alert

            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

            logger.warning(f"Performance alert: {alert.message}")

    def _perform_optimization(self) -> None:
        """Perform automatic performance optimization."""
        try:
            # Get current metrics
            if not self.system_metrics_history:
                return

            current_metrics = self.system_metrics_history[-1]

            # Memory optimization
            if current_metrics.memory_percent > 80:
                self._optimize_memory()

            # GPU memory optimization
            if current_metrics.gpu_memory_used_gb > 0 and current_metrics.gpu_memory_total_gb > 0:
                gpu_usage_percent = (
                    current_metrics.gpu_memory_used_gb / current_metrics.gpu_memory_total_gb
                ) * 100
                if gpu_usage_percent > 80:
                    self._optimize_gpu_memory()

            logger.debug("Performance optimization completed")

        except Exception as e:
            logger.error(f"Performance optimization error: {e}")

    def _optimize_memory(self) -> None:
        """Optimize system memory usage."""
        logger.info("Performing memory optimization")

        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")

        # Clear component caches if available
        # This would be implemented by individual components

    def _optimize_gpu_memory(self) -> None:
        """Optimize GPU memory usage."""
        if not TORCH_AVAILABLE:
            return

        logger.info("Performing GPU memory optimization")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU memory cache cleared")
        except Exception as e:
            logger.error(f"GPU memory optimization error: {e}")

    def record_component_operation(
        self,
        component_name: str,
        execution_time: float,
        success: bool = True,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record component operation metrics."""
        if not self.enable_component_profiling:
            return

        if component_name not in self.component_metrics:
            self.component_metrics[component_name] = ComponentMetrics(component_name)

        metrics = self.component_metrics[component_name]

        # Update execution metrics
        metrics.operation_count += 1
        metrics.total_execution_time += execution_time
        metrics.average_execution_time = metrics.total_execution_time / metrics.operation_count
        metrics.min_execution_time = min(metrics.min_execution_time, execution_time)
        metrics.max_execution_time = max(metrics.max_execution_time, execution_time)

        # Update error metrics
        if not success:
            metrics.error_count += 1
        metrics.error_rate = metrics.error_count / metrics.operation_count

        # Update custom metrics
        if custom_metrics:
            metrics.custom_metrics.update(custom_metrics)

        # Store in history
        self.component_history[component_name].append(
            {
                "timestamp": time.time(),
                "execution_time": execution_time,
                "success": success,
                "custom_metrics": custom_metrics or {},
            }
        )

        # Check component thresholds
        if execution_time > 0:
            severity = self.thresholds.check_threshold("average_execution_time", execution_time)
            if severity:
                self._generate_alert(component_name, "execution_time", execution_time, severity)

        if metrics.error_rate > 0:
            severity = self.thresholds.check_threshold("error_rate", metrics.error_rate)
            if severity:
                self._generate_alert(component_name, "error_rate", metrics.error_rate, severity)

    def get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics."""
        if self.system_metrics_history:
            return self.system_metrics_history[-1]
        return None

    def get_component_metrics(self, component_name: str) -> Optional[ComponentMetrics]:
        """Get metrics for specific component."""
        return self.component_metrics.get(component_name)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_metrics = self.get_system_metrics()

        summary = {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "system_metrics": current_metrics.__dict__ if current_metrics else None,
            "component_count": len(self.component_metrics),
            "active_alerts": len(self.active_alerts),
            "alerts": [alert.__dict__ for alert in self.active_alerts.values()],
            "components": {
                name: metrics.__dict__ for name, metrics in self.component_metrics.items()
            },
        }

        return summary

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)

    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self.active_alerts.clear()

    def export_metrics(
        self, filepath: Path, include_history: bool = True, time_range_hours: Optional[float] = None
    ) -> None:
        """Export metrics to JSON file."""
        try:
            data = {"export_timestamp": time.time(), "summary": self.get_performance_summary()}

            if include_history:
                # Filter history by time range if specified
                if time_range_hours:
                    cutoff_time = time.time() - (time_range_hours * 3600)
                    system_history = [
                        m.__dict__
                        for m in self.system_metrics_history
                        if m.timestamp >= cutoff_time
                    ]
                else:
                    system_history = [m.__dict__ for m in self.system_metrics_history]

                data["system_history"] = system_history
                data["component_history"] = {
                    name: list(history) for name, history in self.component_history.items()
                }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Performance metrics exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


# Global performance monitor instance
_global_performance_monitor: Optional[CodaPerformanceMonitor] = None


def get_performance_monitor() -> CodaPerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = CodaPerformanceMonitor()
    return _global_performance_monitor


def start_performance_monitoring() -> None:
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring() -> None:
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()
