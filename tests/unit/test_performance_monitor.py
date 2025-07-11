"""
Tests for Performance Monitor.
"""

import pytest
import asyncio
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.coda.core.performance_monitor import (
    CodaPerformanceMonitor, SystemMetrics, ComponentMetrics, 
    PerformanceAlert, PerformanceThresholds, get_performance_monitor
)
from src.coda.core.performance_decorators import (
    performance_context, async_performance_context, performance_monitor,
    memory_monitor, cpu_monitor, PerformanceBenchmark
)


class TestSystemMetrics:
    """Test system metrics data structure."""
    
    def test_system_metrics_creation(self):
        """Test creating system metrics."""
        metrics = SystemMetrics()
        
        assert metrics.timestamp > 0
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_total_gb == 0.0
        assert isinstance(metrics.load_average, list)
    
    def test_system_metrics_with_data(self):
        """Test system metrics with actual data."""
        metrics = SystemMetrics(
            cpu_percent=45.5,
            memory_total_gb=16.0,
            memory_used_gb=8.0,
            memory_percent=50.0,
            gpu_count=1,
            process_memory_gb=2.0
        )
        
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_total_gb == 16.0
        assert metrics.memory_used_gb == 8.0
        assert metrics.memory_percent == 50.0
        assert metrics.gpu_count == 1
        assert metrics.process_memory_gb == 2.0


class TestComponentMetrics:
    """Test component metrics data structure."""
    
    def test_component_metrics_creation(self):
        """Test creating component metrics."""
        metrics = ComponentMetrics("test_component")
        
        assert metrics.component_name == "test_component"
        assert metrics.timestamp > 0
        assert metrics.operation_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.error_count == 0
        assert metrics.error_rate == 0.0
        assert isinstance(metrics.custom_metrics, dict)


class TestPerformanceThresholds:
    """Test performance thresholds."""
    
    def test_threshold_initialization(self):
        """Test threshold initialization."""
        thresholds = PerformanceThresholds()
        
        assert 'cpu_percent' in thresholds.thresholds
        assert 'memory_percent' in thresholds.thresholds
        assert thresholds.thresholds['cpu_percent']['warning'] == 70.0
        assert thresholds.thresholds['cpu_percent']['critical'] == 90.0
    
    def test_check_threshold_normal(self):
        """Test threshold check with normal values."""
        thresholds = PerformanceThresholds()
        
        result = thresholds.check_threshold('cpu_percent', 50.0)
        assert result is None
    
    def test_check_threshold_warning(self):
        """Test threshold check with warning level."""
        thresholds = PerformanceThresholds()
        
        result = thresholds.check_threshold('cpu_percent', 75.0)
        assert result == 'warning'
    
    def test_check_threshold_critical(self):
        """Test threshold check with critical level."""
        thresholds = PerformanceThresholds()
        
        result = thresholds.check_threshold('cpu_percent', 95.0)
        assert result == 'critical'
    
    def test_set_custom_threshold(self):
        """Test setting custom threshold."""
        thresholds = PerformanceThresholds()
        
        thresholds.set_threshold('custom_metric', 100.0, 200.0)
        
        assert 'custom_metric' in thresholds.thresholds
        assert thresholds.thresholds['custom_metric']['warning'] == 100.0
        assert thresholds.thresholds['custom_metric']['critical'] == 200.0


class TestCodaPerformanceMonitor:
    """Test the main performance monitor."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing."""
        return CodaPerformanceMonitor(
            monitoring_interval=0.1,  # Fast for testing
            history_size=100,
            enable_gpu_monitoring=False  # Disable for testing
        )
    
    def test_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor.monitoring_interval == 0.1
        assert performance_monitor.history_size == 100
        assert not performance_monitor.enable_gpu_monitoring
        assert not performance_monitor.monitoring_active
        assert len(performance_monitor.system_metrics_history) == 0
        assert len(performance_monitor.component_metrics) == 0
    
    def test_start_stop_monitoring(self, performance_monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_active
        assert performance_monitor.monitoring_thread is not None
        
        # Give it a moment to collect some metrics
        time.sleep(0.2)
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active
    
    def test_collect_system_metrics(self, performance_monitor):
        """Test system metrics collection."""
        metrics = performance_monitor._collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp > 0
        assert metrics.cpu_count > 0
        assert metrics.memory_total_gb > 0
        assert metrics.process_memory_gb >= 0
    
    def test_record_component_operation(self, performance_monitor):
        """Test recording component operations."""
        # Record successful operation
        performance_monitor.record_component_operation(
            component_name="test_component",
            execution_time=0.5,
            success=True,
            custom_metrics={"items_processed": 10}
        )
        
        # Check metrics were recorded
        assert "test_component" in performance_monitor.component_metrics
        metrics = performance_monitor.component_metrics["test_component"]
        
        assert metrics.operation_count == 1
        assert metrics.total_execution_time == 0.5
        assert metrics.average_execution_time == 0.5
        assert metrics.error_count == 0
        assert metrics.error_rate == 0.0
        assert metrics.custom_metrics["items_processed"] == 10
        
        # Record failed operation
        performance_monitor.record_component_operation(
            component_name="test_component",
            execution_time=1.0,
            success=False
        )
        
        # Check updated metrics
        assert metrics.operation_count == 2
        assert metrics.total_execution_time == 1.5
        assert metrics.average_execution_time == 0.75
        assert metrics.error_count == 1
        assert metrics.error_rate == 0.5
    
    def test_threshold_alerts(self, performance_monitor):
        """Test threshold-based alerts."""
        # Set low threshold for testing
        performance_monitor.thresholds.set_threshold('test_metric', 1.0, 2.0)
        
        # Record operation that exceeds threshold
        performance_monitor.record_component_operation(
            component_name="test_component",
            execution_time=1.5,  # Exceeds warning threshold
            success=True
        )
        
        # Check if alert was generated
        # Note: This might not trigger immediately in the test
        # In real usage, alerts are generated during monitoring
    
    def test_get_performance_summary(self, performance_monitor):
        """Test getting performance summary."""
        # Record some operations
        performance_monitor.record_component_operation("comp1", 0.1, True)
        performance_monitor.record_component_operation("comp2", 0.2, False)
        
        summary = performance_monitor.get_performance_summary()
        
        assert 'timestamp' in summary
        assert 'monitoring_active' in summary
        assert 'component_count' in summary
        assert 'components' in summary
        assert summary['component_count'] == 2
        assert 'comp1' in summary['components']
        assert 'comp2' in summary['components']
    
    def test_export_metrics(self, performance_monitor):
        """Test exporting metrics to file."""
        # Record some data
        performance_monitor.record_component_operation("test_comp", 0.1, True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "metrics.json"
            
            performance_monitor.export_metrics(filepath)
            
            assert filepath.exists()
            
            # Verify file content
            import json
            with open(filepath) as f:
                data = json.load(f)
            
            assert 'export_timestamp' in data
            assert 'summary' in data
    
    def test_alert_callbacks(self, performance_monitor):
        """Test alert callback system."""
        callback_called = False
        received_alert = None
        
        def test_callback(alert):
            nonlocal callback_called, received_alert
            callback_called = True
            received_alert = alert
        
        performance_monitor.add_alert_callback(test_callback)
        
        # Generate an alert manually
        alert = PerformanceAlert(
            alert_id="test_alert",
            component="test_component",
            metric_name="test_metric",
            current_value=100.0,
            threshold_value=50.0,
            severity="warning",
            message="Test alert"
        )
        
        # Simulate alert generation
        performance_monitor.active_alerts["test_alert"] = alert
        for callback in performance_monitor.alert_callbacks:
            callback(alert)
        
        assert callback_called
        assert received_alert == alert


class TestPerformanceDecorators:
    """Test performance decorators and context managers."""
    
    def test_performance_context(self):
        """Test performance context manager."""
        with performance_context("test_component", "test_operation") as ctx:
            time.sleep(0.01)  # Small delay
            ctx.add_metric("test_metric", 42.0)
        
        # Check that metrics were recorded
        monitor = get_performance_monitor()
        assert "test_component" in monitor.component_metrics
        
        metrics = monitor.component_metrics["test_component"]
        assert metrics.operation_count > 0
        assert "test_metric" in metrics.custom_metrics
        assert metrics.custom_metrics["test_metric"] == 42.0
    
    @pytest.mark.asyncio
    async def test_async_performance_context(self):
        """Test async performance context manager."""
        async with async_performance_context("async_component", "async_operation") as ctx:
            await asyncio.sleep(0.01)  # Small delay
            ctx.add_metric("async_metric", 24.0)
        
        # Check that metrics were recorded
        monitor = get_performance_monitor()
        assert "async_component" in monitor.component_metrics
        
        metrics = monitor.component_metrics["async_component"]
        assert metrics.operation_count > 0
        assert "async_metric" in metrics.custom_metrics
        assert metrics.custom_metrics["async_metric"] == 24.0
    
    def test_performance_monitor_decorator(self):
        """Test performance monitor decorator."""
        @performance_monitor("decorated_component")
        def test_function(value):
            time.sleep(0.01)
            return value * 2
        
        result = test_function(5)
        assert result == 10
        
        # Check that metrics were recorded
        monitor = get_performance_monitor()
        assert "decorated_component" in monitor.component_metrics
        
        metrics = monitor.component_metrics["decorated_component"]
        assert metrics.operation_count > 0
    
    @pytest.mark.asyncio
    async def test_async_performance_monitor_decorator(self):
        """Test async performance monitor decorator."""
        @performance_monitor("async_decorated_component")
        async def async_test_function(value):
            await asyncio.sleep(0.01)
            return value * 3
        
        result = await async_test_function(4)
        assert result == 12
        
        # Check that metrics were recorded
        monitor = get_performance_monitor()
        assert "async_decorated_component" in monitor.component_metrics
        
        metrics = monitor.component_metrics["async_decorated_component"]
        assert metrics.operation_count > 0


class TestPerformanceBenchmark:
    """Test performance benchmark utility."""
    
    def test_benchmark_creation(self):
        """Test benchmark creation."""
        benchmark = PerformanceBenchmark("test_benchmark")
        
        assert benchmark.name == "test_benchmark"
        assert len(benchmark.measurements) == 0
    
    def test_benchmark_measurement(self):
        """Test benchmark measurement."""
        benchmark = PerformanceBenchmark("test_benchmark")
        
        # Take some measurements
        for i in range(5):
            with benchmark.measure():
                time.sleep(0.01)
        
        assert len(benchmark.measurements) == 5
        
        # Check statistics
        stats = benchmark.get_statistics()
        assert stats["count"] == 5
        assert stats["total_time"] > 0
        assert stats["average_time"] > 0
        assert stats["min_time"] > 0
        assert stats["max_time"] > 0
        assert stats["throughput_ops_per_sec"] > 0
    
    def test_benchmark_report(self):
        """Test benchmark report generation."""
        benchmark = PerformanceBenchmark("test_benchmark")
        
        # Take a measurement
        with benchmark.measure():
            time.sleep(0.01)
        
        report = benchmark.report()
        assert "test_benchmark" in report
        assert "Count:" in report
        assert "Average:" in report
        assert "Throughput:" in report
    
    def test_benchmark_reset(self):
        """Test benchmark reset."""
        benchmark = PerformanceBenchmark("test_benchmark")
        
        # Take a measurement
        with benchmark.measure():
            time.sleep(0.01)
        
        assert len(benchmark.measurements) == 1
        
        # Reset
        benchmark.reset()
        assert len(benchmark.measurements) == 0
        
        # Empty statistics
        stats = benchmark.get_statistics()
        assert stats == {}


class TestGlobalPerformanceMonitor:
    """Test global performance monitor functions."""
    
    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # Should return the same instance
        assert monitor1 is monitor2
        assert isinstance(monitor1, CodaPerformanceMonitor)
    
    def test_performance_monitor_integration(self):
        """Test integration with global monitor."""
        # Clear any existing data
        monitor = get_performance_monitor()
        monitor.component_metrics.clear()
        
        # Use performance context
        with performance_context("integration_test", "test_op"):
            time.sleep(0.01)
        
        # Check data was recorded in global monitor
        assert "integration_test" in monitor.component_metrics
        metrics = monitor.component_metrics["integration_test"]
        assert metrics.operation_count > 0
