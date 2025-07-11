"""
Real-time monitoring and analytics for WebSocket voice processing.

This module provides comprehensive monitoring capabilities for WebSocket
connections, performance metrics, and real-time analytics dashboard.
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .performance_profiler import get_performance_profiler
from .websocket_events import EventType, VoiceEventBroadcaster
from .websocket_handler import VoiceWebSocketHandler

logger = logging.getLogger(__name__)


@dataclass
class ConnectionMetrics:
    """Metrics for a WebSocket connection."""

    client_id: str
    connected_at: float
    last_activity: float
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    latency_samples: List[float] = None

    def __post_init__(self):
        if self.latency_samples is None:
            self.latency_samples = []

    @property
    def connection_duration(self) -> float:
        """Get connection duration in seconds."""
        return time.time() - self.connected_at

    @property
    def average_latency(self) -> float:
        """Get average latency in milliseconds."""
        if not self.latency_samples:
            return 0.0
        return statistics.mean(self.latency_samples)

    @property
    def is_active(self) -> bool:
        """Check if connection is recently active."""
        return time.time() - self.last_activity < 60.0  # 1 minute threshold


@dataclass
class SystemMetrics:
    """System-wide metrics."""

    timestamp: float
    active_connections: int
    total_connections: int
    messages_per_second: float
    bytes_per_second: float
    average_latency: float
    error_rate: float
    cpu_usage: float
    memory_usage: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class WebSocketMonitor:
    """
    Real-time monitoring system for WebSocket voice processing.

    Provides comprehensive monitoring of WebSocket connections,
    performance metrics, and real-time analytics with alerting
    and dashboard capabilities.

    Features:
    - Real-time connection monitoring
    - Performance metrics collection
    - Latency tracking and analysis
    - Error rate monitoring
    - System health checks
    - Alert generation
    - Metrics dashboard

    Example:
        >>> monitor = WebSocketMonitor(websocket_handler, event_broadcaster)
        >>> await monitor.start_monitoring()
        >>> metrics = monitor.get_real_time_metrics()
    """

    def __init__(
        self,
        websocket_handler: VoiceWebSocketHandler,
        event_broadcaster: VoiceEventBroadcaster,
        metrics_interval: float = 5.0,
        history_size: int = 1000,
    ):
        """
        Initialize WebSocket monitor.

        Args:
            websocket_handler: WebSocket handler to monitor
            event_broadcaster: Event broadcaster for alerts
            metrics_interval: Metrics collection interval in seconds
            history_size: Number of historical metrics to keep
        """
        self.websocket_handler = websocket_handler
        self.event_broadcaster = event_broadcaster
        self.metrics_interval = metrics_interval
        self.history_size = history_size

        # Connection metrics
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}

        # System metrics history
        self.metrics_history: deque = deque(maxlen=history_size)

        # Performance profiler
        self.profiler = get_performance_profiler()

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Alert thresholds
        self.alert_thresholds = {
            "max_latency_ms": 1000.0,
            "max_error_rate": 0.05,  # 5%
            "max_cpu_usage": 80.0,
            "max_memory_usage": 85.0,
            "min_connection_health": 0.9,
        }

        # Counters for rate calculations
        self.last_metrics_time = time.time()
        self.last_message_count = 0
        self.last_byte_count = 0

        logger.info("WebSocketMonitor initialized")

    async def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("WebSocket monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("WebSocket monitoring stopped")

    def track_connection(self, client_id: str) -> None:
        """Start tracking a new connection."""
        self.connection_metrics[client_id] = ConnectionMetrics(
            client_id=client_id, connected_at=time.time(), last_activity=time.time()
        )

    def untrack_connection(self, client_id: str) -> None:
        """Stop tracking a connection."""
        if client_id in self.connection_metrics:
            del self.connection_metrics[client_id]

    def record_message_sent(self, client_id: str, message_size: int) -> None:
        """Record a message sent to client."""
        if client_id in self.connection_metrics:
            metrics = self.connection_metrics[client_id]
            metrics.messages_sent += 1
            metrics.bytes_sent += message_size
            metrics.last_activity = time.time()

    def record_message_received(self, client_id: str, message_size: int) -> None:
        """Record a message received from client."""
        if client_id in self.connection_metrics:
            metrics = self.connection_metrics[client_id]
            metrics.messages_received += 1
            metrics.bytes_received += message_size
            metrics.last_activity = time.time()

    def record_latency(self, client_id: str, latency_ms: float) -> None:
        """Record latency measurement for client."""
        if client_id in self.connection_metrics:
            metrics = self.connection_metrics[client_id]
            metrics.latency_samples.append(latency_ms)

            # Keep only recent samples
            if len(metrics.latency_samples) > 100:
                metrics.latency_samples.pop(0)

    def record_error(self, client_id: str) -> None:
        """Record an error for client."""
        if client_id in self.connection_metrics:
            self.connection_metrics[client_id].errors += 1

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()

                # Add to history
                self.metrics_history.append(system_metrics)

                # Check for alerts
                await self._check_alerts(system_metrics)

                # Broadcast metrics to subscribers
                await self._broadcast_metrics(system_metrics)

                await asyncio.sleep(self.metrics_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.metrics_interval)

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        current_time = time.time()

        # Connection metrics
        active_connections = len([m for m in self.connection_metrics.values() if m.is_active])
        total_connections = len(self.connection_metrics)

        # Calculate rates
        time_delta = current_time - self.last_metrics_time

        current_message_count = sum(
            m.messages_sent + m.messages_received for m in self.connection_metrics.values()
        )
        current_byte_count = sum(
            m.bytes_sent + m.bytes_received for m in self.connection_metrics.values()
        )

        messages_per_second = (current_message_count - self.last_message_count) / time_delta
        bytes_per_second = (current_byte_count - self.last_byte_count) / time_delta

        self.last_metrics_time = current_time
        self.last_message_count = current_message_count
        self.last_byte_count = current_byte_count

        # Latency metrics
        all_latencies = []
        for metrics in self.connection_metrics.values():
            all_latencies.extend(metrics.latency_samples)

        average_latency = statistics.mean(all_latencies) if all_latencies else 0.0

        # Error rate
        total_messages = current_message_count
        total_errors = sum(m.errors for m in self.connection_metrics.values())
        error_rate = total_errors / max(1, total_messages)

        # System resource usage
        cpu_usage, memory_usage = await self._get_system_resources()

        return SystemMetrics(
            timestamp=current_time,
            active_connections=active_connections,
            total_connections=total_connections,
            messages_per_second=messages_per_second,
            bytes_per_second=bytes_per_second,
            average_latency=average_latency,
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
        )

    async def _get_system_resources(self) -> Tuple[float, float]:
        """Get system CPU and memory usage."""
        try:
            import psutil

            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            return cpu_usage, memory_usage
        except ImportError:
            # Fallback if psutil not available
            return 0.0, 0.0

    async def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check metrics against alert thresholds."""
        alerts = []

        # High latency alert
        if metrics.average_latency > self.alert_thresholds["max_latency_ms"]:
            alerts.append(
                {
                    "type": "high_latency",
                    "message": f"High average latency: {metrics.average_latency:.1f}ms",
                    "severity": "warning",
                    "value": metrics.average_latency,
                    "threshold": self.alert_thresholds["max_latency_ms"],
                }
            )

        # High error rate alert
        if metrics.error_rate > self.alert_thresholds["max_error_rate"]:
            alerts.append(
                {
                    "type": "high_error_rate",
                    "message": f"High error rate: {metrics.error_rate:.2%}",
                    "severity": "critical",
                    "value": metrics.error_rate,
                    "threshold": self.alert_thresholds["max_error_rate"],
                }
            )

        # High CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds["max_cpu_usage"]:
            alerts.append(
                {
                    "type": "high_cpu_usage",
                    "message": f"High CPU usage: {metrics.cpu_usage:.1f}%",
                    "severity": "warning",
                    "value": metrics.cpu_usage,
                    "threshold": self.alert_thresholds["max_cpu_usage"],
                }
            )

        # High memory usage alert
        if metrics.memory_usage > self.alert_thresholds["max_memory_usage"]:
            alerts.append(
                {
                    "type": "high_memory_usage",
                    "message": f"High memory usage: {metrics.memory_usage:.1f}%",
                    "severity": "critical",
                    "value": metrics.memory_usage,
                    "threshold": self.alert_thresholds["max_memory_usage"],
                }
            )

        # Broadcast alerts
        for alert in alerts:
            await self.event_broadcaster.broadcast_event(EventType.PERFORMANCE_ALERT, alert)

            logger.warning(f"Performance alert: {alert['message']}")

    async def _broadcast_metrics(self, metrics: SystemMetrics) -> None:
        """Broadcast metrics to subscribers."""
        await self.event_broadcaster.broadcast_event(
            EventType.SYSTEM_STATUS,
            {
                "metrics": metrics.to_dict(),
                "connection_details": {
                    client_id: {
                        "duration": conn.connection_duration,
                        "messages": conn.messages_sent + conn.messages_received,
                        "latency": conn.average_latency,
                        "errors": conn.errors,
                    }
                    for client_id, conn in self.connection_metrics.items()
                },
            },
        )

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        latest_metrics = self.metrics_history[-1]

        return {
            "current": latest_metrics.to_dict(),
            "connections": {
                client_id: {
                    "connected_at": conn.connected_at,
                    "duration": conn.connection_duration,
                    "messages_sent": conn.messages_sent,
                    "messages_received": conn.messages_received,
                    "bytes_sent": conn.bytes_sent,
                    "bytes_received": conn.bytes_received,
                    "average_latency": conn.average_latency,
                    "errors": conn.errors,
                    "is_active": conn.is_active,
                }
                for client_id, conn in self.connection_metrics.items()
            },
            "summary": {
                "total_connections": len(self.connection_metrics),
                "active_connections": len(
                    [c for c in self.connection_metrics.values() if c.is_active]
                ),
                "total_messages": sum(
                    c.messages_sent + c.messages_received for c in self.connection_metrics.values()
                ),
                "total_errors": sum(c.errors for c in self.connection_metrics.values()),
            },
        }

    def get_historical_metrics(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get historical metrics for the specified time period."""
        cutoff_time = time.time() - (minutes * 60)

        return [
            metrics.to_dict()
            for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]

    def get_connection_details(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a specific connection."""
        if client_id not in self.connection_metrics:
            return None

        conn = self.connection_metrics[client_id]

        return {
            "client_id": client_id,
            "connected_at": conn.connected_at,
            "last_activity": conn.last_activity,
            "duration": conn.connection_duration,
            "messages_sent": conn.messages_sent,
            "messages_received": conn.messages_received,
            "bytes_sent": conn.bytes_sent,
            "bytes_received": conn.bytes_received,
            "average_latency": conn.average_latency,
            "latency_samples": conn.latency_samples[-20:],  # Last 20 samples
            "errors": conn.errors,
            "is_active": conn.is_active,
        }

    def set_alert_threshold(self, metric: str, value: float) -> bool:
        """Set alert threshold for a metric."""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = value
            logger.info(f"Updated alert threshold for {metric}: {value}")
            return True
        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the last hour."""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        # Get last hour of metrics
        one_hour_ago = time.time() - 3600
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= one_hour_ago]

        if not recent_metrics:
            return {"error": "No recent metrics available"}

        # Calculate summary statistics
        latencies = [m.average_latency for m in recent_metrics if m.average_latency > 0]
        error_rates = [m.error_rate for m in recent_metrics]
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]

        return {
            "time_period": "last_hour",
            "latency": {
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "avg": statistics.mean(latencies) if latencies else 0,
                "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
            },
            "error_rate": {
                "min": min(error_rates),
                "max": max(error_rates),
                "avg": statistics.mean(error_rates),
            },
            "cpu_usage": {
                "min": min(cpu_usage),
                "max": max(cpu_usage),
                "avg": statistics.mean(cpu_usage),
            },
            "memory_usage": {
                "min": min(memory_usage),
                "max": max(memory_usage),
                "avg": statistics.mean(memory_usage),
            },
            "connections": {
                "peak": max(m.active_connections for m in recent_metrics),
                "avg": statistics.mean([m.active_connections for m in recent_metrics]),
            },
        }
