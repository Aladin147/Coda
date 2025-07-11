"""
Comprehensive optimization suite for the voice processing system.

This module extends the existing performance optimizer with comprehensive
system-wide optimization including automatic tuning and adaptive configuration.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

from .models import AudioConfig, MoshiConfig, VoiceConfig
from .performance_optimizer import (
    OptimizationLevel,
    PerformanceMetrics,
    PerformanceOptimizer,
)
from .performance_profiler import get_performance_profiler

logger = logging.getLogger(__name__)


@dataclass
class SystemOptimizationMetrics:
    """Extended system-wide optimization metrics."""

    # Core performance metrics
    base_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # System resource metrics
    system_cpu_percent: float = 0.0
    system_memory_percent: float = 0.0
    disk_io_mb_per_sec: float = 0.0
    network_io_mb_per_sec: float = 0.0

    # Voice processing specific metrics
    conversation_count: int = 0
    active_websocket_connections: int = 0
    audio_buffer_utilization: float = 0.0
    cache_efficiency: float = 0.0

    # Quality metrics
    audio_quality_score: float = 0.0
    response_accuracy_score: float = 0.0
    user_satisfaction_score: float = 0.0

    # Error and reliability metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    recovery_time_ms: float = 0.0

    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration."""

    name: str
    description: str
    target_metrics: Dict[str, float]
    optimization_actions: List[str]
    priority: int = 1  # 1 = highest, 5 = lowest
    enabled: bool = True

    # Conditions for applying this strategy
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Expected improvements
    expected_improvements: Dict[str, str] = field(default_factory=dict)


class ComprehensiveOptimizer:
    """
    Comprehensive system optimization engine.

    Features:
    - Multi-dimensional performance optimization
    - Adaptive strategy selection
    - Real-time system monitoring
    - Predictive optimization
    - Quality-performance trade-off management
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        """Initialize comprehensive optimizer."""
        self.config = config or VoiceConfig()
        self.base_optimizer = PerformanceOptimizer()
        self.profiler = get_performance_profiler()

        # Metrics and monitoring
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_metrics: Optional[SystemOptimizationMetrics] = None

        # Optimization strategies
        self.optimization_strategies = self._initialize_strategies()
        self.active_strategies: List[str] = []

        # Adaptive configuration
        self.adaptive_config = {
            "optimization_level": OptimizationLevel.BALANCED,
            "auto_tune_enabled": True,
            "quality_threshold": 0.8,
            "performance_threshold": 0.7,
            "resource_limit_cpu": 80.0,
            "resource_limit_memory": 1000.0,
            "latency_target_ms": 200.0,
            "throughput_target_rps": 10.0,
        }

        # Optimization state
        self.is_running = False
        self.optimization_task: Optional[asyncio.Task] = None
        self.optimization_interval = 30.0  # seconds

        # Callbacks and notifications
        self.optimization_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []

        logger.info("ComprehensiveOptimizer initialized")

    async def start_optimization(self) -> None:
        """Start comprehensive optimization engine."""
        if self.is_running:
            return

        self.is_running = True

        # Start base optimizer
        await self.base_optimizer.start_optimization()

        # Start comprehensive optimization loop
        self.optimization_task = asyncio.create_task(self._optimization_loop())

        logger.info("Comprehensive optimization started")

    async def stop_optimization(self) -> None:
        """Stop optimization engine."""
        self.is_running = False

        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass

        # Stop base optimizer
        await self.base_optimizer.stop_optimization()

        logger.info("Comprehensive optimization stopped")

    def collect_system_metrics(self) -> SystemOptimizationMetrics:
        """Collect comprehensive system metrics."""
        # Get base performance metrics
        base_metrics = self.base_optimizer.get_current_metrics()

        # Get system resource metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()

        # Get voice system specific metrics
        profiler_stats = self.profiler.get_performance_summary()

        # Calculate derived metrics
        cache_efficiency = self._calculate_cache_efficiency(profiler_stats)
        audio_quality = self._estimate_audio_quality(profiler_stats)

        metrics = SystemOptimizationMetrics(
            base_metrics=base_metrics,
            system_cpu_percent=cpu_percent,
            system_memory_percent=memory.percent,
            disk_io_mb_per_sec=self._calculate_io_rate(disk_io, "disk"),
            network_io_mb_per_sec=self._calculate_io_rate(network_io, "network"),
            conversation_count=profiler_stats.get("active_conversations", 0),
            active_websocket_connections=profiler_stats.get("websocket_connections", 0),
            audio_buffer_utilization=profiler_stats.get("buffer_utilization", 0.0),
            cache_efficiency=cache_efficiency,
            audio_quality_score=audio_quality,
            response_accuracy_score=profiler_stats.get("accuracy_score", 0.0),
            error_rate=profiler_stats.get("error_rate", 0.0),
            timeout_rate=profiler_stats.get("timeout_rate", 0.0),
            recovery_time_ms=profiler_stats.get("recovery_time_ms", 0.0),
        )

        return metrics

    def analyze_optimization_opportunities(
        self, metrics: SystemOptimizationMetrics
    ) -> List[OptimizationStrategy]:
        """Analyze current metrics and identify optimization opportunities."""
        opportunities = []

        # Check each optimization strategy
        for strategy in self.optimization_strategies.values():
            if not strategy.enabled:
                continue

            # Check if strategy conditions are met
            if self._strategy_conditions_met(strategy, metrics):
                opportunities.append(strategy)

        # Sort by priority
        opportunities.sort(key=lambda s: s.priority)

        return opportunities

    async def apply_optimization_strategy(self, strategy: OptimizationStrategy) -> bool:
        """Apply an optimization strategy."""
        try:
            logger.info(f"Applying optimization strategy: {strategy.name}")

            success = True
            for action in strategy.optimization_actions:
                action_success = await self._execute_optimization_action(action)
                if not action_success:
                    success = False
                    logger.warning(f"Optimization action failed: {action}")

            if success:
                self.active_strategies.append(strategy.name)
                logger.info(f"Successfully applied strategy: {strategy.name}")

            return success

        except Exception as e:
            logger.error(f"Failed to apply optimization strategy {strategy.name}: {e}")
            return False

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        if not self.current_metrics:
            return {"error": "No metrics available"}

        # Calculate performance scores
        performance_score = self._calculate_performance_score(self.current_metrics)
        quality_score = self._calculate_quality_score(self.current_metrics)
        efficiency_score = self._calculate_efficiency_score(self.current_metrics)

        # Get optimization opportunities
        opportunities = self.analyze_optimization_opportunities(self.current_metrics)

        # Calculate trends
        trends = self._calculate_performance_trends()

        return {
            "timestamp": time.time(),
            "performance_scores": {
                "overall_performance": performance_score,
                "quality_score": quality_score,
                "efficiency_score": efficiency_score,
                "composite_score": (performance_score + quality_score + efficiency_score) / 3,
            },
            "current_metrics": {
                "latency_ms": self.current_metrics.base_metrics.total_latency_ms,
                "throughput_rps": self.current_metrics.base_metrics.messages_per_second,
                "cpu_usage": self.current_metrics.system_cpu_percent,
                "memory_usage": self.current_metrics.system_memory_percent,
                "error_rate": self.current_metrics.error_rate,
                "cache_efficiency": self.current_metrics.cache_efficiency,
                "audio_quality": self.current_metrics.audio_quality_score,
            },
            "optimization_opportunities": [
                {
                    "name": opp.name,
                    "description": opp.description,
                    "priority": opp.priority,
                    "expected_improvements": opp.expected_improvements,
                }
                for opp in opportunities[:5]  # Top 5 opportunities
            ],
            "active_strategies": self.active_strategies,
            "trends": trends,
            "adaptive_config": self.adaptive_config,
            "recommendations": self._generate_recommendations(),
        }

    def configure_optimization(self, **kwargs) -> None:
        """Configure optimization parameters."""
        for key, value in kwargs.items():
            if key in self.adaptive_config:
                self.adaptive_config[key] = value
                logger.info(f"Updated optimization config: {key} = {value}")
            else:
                logger.warning(f"Unknown optimization config key: {key}")

    # Private methods

    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = self.collect_system_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)

                # Analyze optimization opportunities
                opportunities = self.analyze_optimization_opportunities(metrics)

                # Apply high-priority optimizations if auto-tune is enabled
                if self.adaptive_config["auto_tune_enabled"]:
                    for opportunity in opportunities[:2]:  # Apply top 2 opportunities
                        if opportunity.priority <= 2:  # Only high priority
                            await self.apply_optimization_strategy(opportunity)

                # Check for alerts
                await self._check_performance_alerts(metrics)

                # Notify callbacks
                for callback in self.optimization_callbacks:
                    try:
                        callback(metrics, opportunities)
                    except Exception as e:
                        logger.error(f"Optimization callback error: {e}")

                await asyncio.sleep(self.optimization_interval)

            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(self.optimization_interval)

    def _initialize_strategies(self) -> Dict[str, OptimizationStrategy]:
        """Initialize optimization strategies."""
        strategies = {}

        # Latency optimization strategy
        strategies["reduce_latency"] = OptimizationStrategy(
            name="reduce_latency",
            description="Reduce processing latency through various optimizations",
            target_metrics={"latency_ms": 150.0},
            optimization_actions=[
                "optimize_audio_chunk_size",
                "enable_response_caching",
                "optimize_llm_context_size",
            ],
            priority=1,
            conditions={"latency_ms": ">200"},
            expected_improvements={
                "latency_reduction": "20-40%",
                "user_experience": "Significantly improved",
            },
        )

        # Throughput optimization strategy
        strategies["increase_throughput"] = OptimizationStrategy(
            name="increase_throughput",
            description="Increase system throughput and concurrent capacity",
            target_metrics={"throughput_rps": 20.0},
            optimization_actions=[
                "increase_buffer_pool_size",
                "optimize_websocket_handling",
                "enable_request_batching",
            ],
            priority=2,
            conditions={"throughput_rps": "<10"},
            expected_improvements={
                "throughput_increase": "50-100%",
                "concurrent_capacity": "2x improvement",
            },
        )

        # Resource optimization strategy
        strategies["optimize_resources"] = OptimizationStrategy(
            name="optimize_resources",
            description="Optimize CPU and memory usage",
            target_metrics={"cpu_usage": 70.0, "memory_usage": 800.0},
            optimization_actions=[
                "optimize_memory_usage",
                "reduce_cpu_intensive_operations",
                "cleanup_unused_resources",
            ],
            priority=1,
            conditions={"cpu_usage": ">80", "memory_usage": ">1000"},
            expected_improvements={
                "resource_efficiency": "15-25% improvement",
                "system_stability": "Enhanced",
            },
        )

        # Quality optimization strategy
        strategies["enhance_quality"] = OptimizationStrategy(
            name="enhance_quality",
            description="Enhance audio and response quality",
            target_metrics={"audio_quality": 0.9, "accuracy": 0.95},
            optimization_actions=[
                "enhance_audio_processing",
                "optimize_model_parameters",
                "improve_context_handling",
            ],
            priority=3,
            conditions={"audio_quality": "<0.8", "accuracy": "<0.9"},
            expected_improvements={
                "audio_quality": "10-20% improvement",
                "response_accuracy": "5-15% improvement",
            },
        )

        return strategies

    def _strategy_conditions_met(
        self, strategy: OptimizationStrategy, metrics: SystemOptimizationMetrics
    ) -> bool:
        """Check if strategy conditions are met."""
        for condition, threshold in strategy.conditions.items():
            metric_value = self._get_metric_value(metrics, condition)

            if metric_value is None:
                continue

            # Parse threshold condition
            if threshold.startswith(">"):
                if metric_value <= float(threshold[1:]):
                    return False
            elif threshold.startswith("<"):
                if metric_value >= float(threshold[1:]):
                    return False
            elif threshold.startswith(">="):
                if metric_value < float(threshold[2:]):
                    return False
            elif threshold.startswith("<="):
                if metric_value > float(threshold[2:]):
                    return False
            else:
                if metric_value != float(threshold):
                    return False

        return True

    def _get_metric_value(
        self, metrics: SystemOptimizationMetrics, metric_name: str
    ) -> Optional[float]:
        """Get metric value by name."""
        metric_map = {
            "latency_ms": metrics.base_metrics.total_latency_ms,
            "throughput_rps": metrics.base_metrics.messages_per_second,
            "cpu_usage": metrics.system_cpu_percent,
            "memory_usage": metrics.system_memory_percent,
            "error_rate": metrics.error_rate,
            "audio_quality": metrics.audio_quality_score,
            "accuracy": metrics.response_accuracy_score,
            "cache_efficiency": metrics.cache_efficiency,
        }

        return metric_map.get(metric_name)

    async def _execute_optimization_action(self, action: str) -> bool:
        """Execute a specific optimization action."""
        try:
            if action == "optimize_audio_chunk_size":
                return await self._optimize_audio_chunk_size()
            elif action == "enable_response_caching":
                return await self._enable_response_caching()
            elif action == "optimize_llm_context_size":
                return await self._optimize_llm_context_size()
            elif action == "increase_buffer_pool_size":
                return await self._increase_buffer_pool_size()
            elif action == "optimize_websocket_handling":
                return await self._optimize_websocket_handling()
            elif action == "enable_request_batching":
                return await self._enable_request_batching()
            elif action == "optimize_memory_usage":
                return await self._optimize_memory_usage()
            elif action == "reduce_cpu_intensive_operations":
                return await self._reduce_cpu_intensive_operations()
            elif action == "cleanup_unused_resources":
                return await self._cleanup_unused_resources()
            elif action == "enhance_audio_processing":
                return await self._enhance_audio_processing()
            elif action == "optimize_model_parameters":
                return await self._optimize_model_parameters()
            elif action == "improve_context_handling":
                return await self._improve_context_handling()
            else:
                logger.warning(f"Unknown optimization action: {action}")
                return False

        except Exception as e:
            logger.error(f"Error executing optimization action {action}: {e}")
            return False

    # Optimization action implementations (simplified)

    async def _optimize_audio_chunk_size(self) -> bool:
        """Optimize audio chunk size for better latency."""
        logger.info("Optimizing audio chunk size")
        # Implementation would adjust audio processing parameters
        return True

    async def _enable_response_caching(self) -> bool:
        """Enable or optimize response caching."""
        logger.info("Enabling response caching")
        # Implementation would configure caching system
        return True

    async def _optimize_llm_context_size(self) -> bool:
        """Optimize LLM context size."""
        logger.info("Optimizing LLM context size")
        # Implementation would adjust LLM parameters
        return True

    async def _increase_buffer_pool_size(self) -> bool:
        """Increase buffer pool size."""
        logger.info("Increasing buffer pool size")
        # Implementation would adjust buffer pool configuration
        return True

    async def _optimize_websocket_handling(self) -> bool:
        """Optimize WebSocket handling."""
        logger.info("Optimizing WebSocket handling")
        # Implementation would adjust WebSocket parameters
        return True

    async def _enable_request_batching(self) -> bool:
        """Enable request batching."""
        logger.info("Enabling request batching")
        # Implementation would enable batching mechanisms
        return True

    async def _optimize_memory_usage(self) -> bool:
        """Optimize memory usage."""
        logger.info("Optimizing memory usage")
        # Implementation would clean up memory and optimize allocation
        return True

    async def _reduce_cpu_intensive_operations(self) -> bool:
        """Reduce CPU intensive operations."""
        logger.info("Reducing CPU intensive operations")
        # Implementation would optimize CPU usage
        return True

    async def _cleanup_unused_resources(self) -> bool:
        """Cleanup unused resources."""
        logger.info("Cleaning up unused resources")
        # Implementation would perform resource cleanup
        return True

    async def _enhance_audio_processing(self) -> bool:
        """Enhance audio processing quality."""
        logger.info("Enhancing audio processing")
        # Implementation would improve audio processing
        return True

    async def _optimize_model_parameters(self) -> bool:
        """Optimize model parameters."""
        logger.info("Optimizing model parameters")
        # Implementation would tune model parameters
        return True

    async def _improve_context_handling(self) -> bool:
        """Improve context handling."""
        logger.info("Improving context handling")
        # Implementation would optimize context management
        return True

    def _calculate_cache_efficiency(self, profiler_stats: Dict[str, Any]) -> float:
        """Calculate cache efficiency."""
        cache_stats = profiler_stats.get("cache", {})
        hit_rate = cache_stats.get("hit_rate", 0.0)
        return hit_rate

    def _estimate_audio_quality(self, profiler_stats: Dict[str, Any]) -> float:
        """Estimate audio quality score."""
        # Simplified quality estimation based on error rates and processing metrics
        error_rate = profiler_stats.get("error_rate", 0.0)
        latency = profiler_stats.get("avg_latency_ms", 0.0)

        quality_score = 1.0 - error_rate
        if latency > 500:  # High latency affects perceived quality
            quality_score *= 0.8
        elif latency > 200:
            quality_score *= 0.9

        return max(0.0, min(1.0, quality_score))

    def _calculate_io_rate(self, io_counters, io_type: str) -> float:
        """Calculate I/O rate in MB/s."""
        # Simplified I/O rate calculation
        # In real implementation, this would track deltas over time
        return 0.0

    def _calculate_performance_score(self, metrics: SystemOptimizationMetrics) -> float:
        """Calculate overall performance score."""
        score = 100.0

        # Latency penalty
        if metrics.base_metrics.total_latency_ms > 200:
            score -= min(30, (metrics.base_metrics.total_latency_ms - 200) / 10)

        # Throughput bonus/penalty
        if metrics.base_metrics.messages_per_second < 5:
            score -= 20
        elif metrics.base_metrics.messages_per_second > 15:
            score += 10

        # Resource usage penalty
        if metrics.system_cpu_percent > 80:
            score -= 20
        if metrics.system_memory_percent > 80:
            score -= 15

        # Error rate penalty
        score -= metrics.error_rate * 100

        return max(0.0, min(100.0, score))

    def _calculate_quality_score(self, metrics: SystemOptimizationMetrics) -> float:
        """Calculate quality score."""
        return (metrics.audio_quality_score + metrics.response_accuracy_score) * 50

    def _calculate_efficiency_score(self, metrics: SystemOptimizationMetrics) -> float:
        """Calculate efficiency score."""
        cpu_efficiency = max(0, 100 - metrics.system_cpu_percent)
        memory_efficiency = max(0, 100 - metrics.system_memory_percent)
        cache_efficiency = metrics.cache_efficiency * 100

        return (cpu_efficiency + memory_efficiency + cache_efficiency) / 3

    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends."""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}

        # Simplified trend calculation
        recent = list(self.metrics_history)[-5:]
        older = list(self.metrics_history)[-10:-5]

        trends = {}

        if older:
            recent_latency = statistics.mean([m.base_metrics.total_latency_ms for m in recent])
            older_latency = statistics.mean([m.base_metrics.total_latency_ms for m in older])

            if recent_latency > older_latency * 1.1:
                trends["latency"] = "increasing"
            elif recent_latency < older_latency * 0.9:
                trends["latency"] = "decreasing"
            else:
                trends["latency"] = "stable"

        return trends

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if not self.current_metrics:
            return recommendations

        metrics = self.current_metrics

        if metrics.base_metrics.total_latency_ms > 300:
            recommendations.append("Consider reducing audio chunk size to improve latency")

        if metrics.system_cpu_percent > 85:
            recommendations.append("High CPU usage detected - consider load balancing")

        if metrics.error_rate > 0.05:
            recommendations.append("High error rate - investigate error causes")

        if metrics.cache_efficiency < 0.7:
            recommendations.append("Low cache efficiency - consider increasing cache size")

        return recommendations

    async def _check_performance_alerts(self, metrics: SystemOptimizationMetrics) -> None:
        """Check for performance alerts."""
        alerts = []

        if metrics.system_cpu_percent > 90:
            alerts.append("CRITICAL: CPU usage above 90%")

        if metrics.error_rate > 0.1:
            alerts.append("WARNING: High error rate detected")

        if metrics.base_metrics.total_latency_ms > 1000:
            alerts.append("WARNING: Very high latency detected")

        # Notify alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, metrics)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")


# Global comprehensive optimizer instance
_global_comprehensive_optimizer: Optional[ComprehensiveOptimizer] = None


def get_comprehensive_optimizer() -> ComprehensiveOptimizer:
    """Get global comprehensive optimizer instance."""
    global _global_comprehensive_optimizer
    if _global_comprehensive_optimizer is None:
        _global_comprehensive_optimizer = ComprehensiveOptimizer()
    return _global_comprehensive_optimizer
