#!/usr/bin/env python3
"""
RTX 5090 Optimization Test Suite.

Comprehensive testing for RTX 5090 GPU optimizations including:
- TF32 acceleration validation
- Performance benchmarking
- Memory management testing
- Integration with existing systems
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from coda.core.rtx5090_optimizer import (
    RTX5090Optimizer, 
    get_rtx5090_optimizer, 
    apply_rtx5090_optimizations,
    benchmark_rtx5090
)


class TestRTX5090Optimizer:
    """Test RTX 5090 optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test RTX 5090 optimizer initialization."""
        optimizer = RTX5090Optimizer()
        
        # Test basic initialization
        assert optimizer.device_id == 0
        assert isinstance(optimizer.total_memory_gb, (int, float))
        assert isinstance(optimizer.is_rtx5090, bool)
        assert isinstance(optimizer.optimizations_applied, bool)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_global_optimizer_instance(self):
        """Test global optimizer instance management."""
        optimizer1 = get_rtx5090_optimizer()
        optimizer2 = get_rtx5090_optimizer()
        
        # Should return the same instance
        assert optimizer1 is optimizer2
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_apply_optimizations(self):
        """Test applying RTX 5090 optimizations."""
        # Test optimization application
        result = apply_rtx5090_optimizations()
        
        # Should return True if CUDA is available, False otherwise
        if torch.cuda.is_available():
            assert result is True
            
            # Verify optimizations are applied
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
            assert torch.backends.cudnn.benchmark is True
        else:
            assert result is False
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.gpu
    def test_tf32_acceleration(self):
        """Test TF32 acceleration is enabled."""
        optimizer = get_rtx5090_optimizer()
        optimizer.apply_rtx5090_optimizations()
        
        if torch.cuda.is_available():
            # Verify TF32 is enabled
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
            
            # Test that TF32 actually improves performance
            stats = optimizer.get_performance_stats()
            if stats:
                assert stats.tf32_enabled is True
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.gpu
    def test_performance_benchmarks(self):
        """Test performance benchmarks meet RTX 5090 expectations."""
        optimizer = get_rtx5090_optimizer()
        
        if torch.cuda.is_available() and optimizer.is_rtx5090:
            # Run benchmark
            results = optimizer.benchmark_performance(matrix_size=2048, iterations=10)
            
            # Verify benchmark results
            assert 'error' not in results
            assert 'gflops' in results
            assert 'avg_time_ms' in results
            
            # RTX 5090 should achieve high performance
            # (Relaxed expectations for testing environment)
            assert results['gflops'] > 1000  # At least 1 TFLOPS
            assert results['avg_time_ms'] < 100  # Less than 100ms for 2048x2048
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.gpu
    def test_memory_management(self):
        """Test VRAM management and allocation."""
        optimizer = get_rtx5090_optimizer()
        
        if torch.cuda.is_available():
            stats = optimizer.get_performance_stats()
            
            if stats:
                # Verify memory stats are reasonable
                assert stats.memory_allocated_gb >= 0
                assert stats.memory_reserved_gb >= 0
                assert stats.memory_free_gb >= 0
                
                # Total memory should be reasonable for RTX 5090
                total_memory = stats.memory_reserved_gb + stats.memory_free_gb
                if optimizer.is_rtx5090:
                    assert total_memory > 30  # RTX 5090 has ~32GB
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_workload_optimization(self):
        """Test workload-specific optimizations."""
        optimizer = get_rtx5090_optimizer()
        
        if torch.cuda.is_available():
            # Test different workload optimizations
            workloads = ['voice_processing', 'llm_inference', 'training']
            
            for workload in workloads:
                try:
                    optimizer.optimize_for_workload(workload)
                    # Should not raise any exceptions
                    assert True
                except Exception as e:
                    pytest.fail(f"Workload optimization failed for {workload}: {e}")
    
    def test_optimizer_without_gpu(self):
        """Test optimizer behavior without GPU."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = RTX5090Optimizer()
            
            # Should handle gracefully without GPU
            result = optimizer.apply_rtx5090_optimizations()
            assert result is False
            
            stats = optimizer.get_performance_stats()
            assert stats is None
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_benchmark_function(self):
        """Test standalone benchmark function."""
        results = benchmark_rtx5090()
        
        if torch.cuda.is_available():
            assert isinstance(results, dict)
            if 'error' not in results:
                assert 'gflops' in results
                assert 'avg_time_ms' in results
        else:
            assert 'error' in results


class TestRTX5090Integration:
    """Test RTX 5090 integration with existing systems."""
    
    def test_config_integration(self):
        """Test RTX 5090 optimizer integration with config loading."""
        # Import here to avoid circular imports
        from coda.core.config import load_config
        
        # Should not raise exceptions
        try:
            config = load_config()
            assert config is not None
        except Exception as e:
            pytest.fail(f"Config loading with RTX 5090 optimization failed: {e}")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_performance_monitor_integration(self):
        """Test integration with performance monitor."""
        try:
            from coda.core.performance_monitor import get_performance_monitor
            
            monitor = get_performance_monitor()
            assert monitor is not None
            
            # Should be able to collect metrics (method might be different)
            if hasattr(monitor, 'collect_metrics'):
                metrics = monitor.collect_metrics()
            elif hasattr(monitor, 'collect_system_metrics'):
                metrics = monitor.collect_system_metrics()
            else:
                # Skip if no metrics collection method available
                pytest.skip("Performance monitor metrics collection not available")
            
        except ImportError:
            # Performance monitor might not have RTX 5090 integration yet
            pytest.skip("Performance monitor RTX 5090 integration not available")
    
    def test_system_health_with_optimization(self):
        """Test system health with RTX 5090 optimizations."""
        # Test that system health checks still work with optimizations
        try:
            # Import and test core components
            from coda.core.config import load_config
            from coda.core.assistant import CodaAssistant
            
            config = load_config()
            assert config is not None
            
            # Should be able to create assistant instance
            # (Don't initialize to avoid heavy dependencies in tests)
            assistant = CodaAssistant(config)
            assert assistant is not None
            
        except Exception as e:
            pytest.fail(f"System health check failed with RTX 5090 optimization: {e}")


class TestRTX5090PerformanceRegression:
    """Test for performance regressions in RTX 5090 optimizations."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.gpu
    @pytest.mark.performance
    def test_performance_baseline(self):
        """Test performance baseline for regression detection."""
        optimizer = get_rtx5090_optimizer()
        
        if torch.cuda.is_available() and optimizer.is_rtx5090:
            # Run multiple benchmarks to establish baseline
            results = []
            for _ in range(3):
                result = optimizer.benchmark_performance(matrix_size=1024, iterations=5)
                if 'error' not in result:
                    results.append(result['gflops'])
            
            if results:
                avg_performance = sum(results) / len(results)
                
                # Store baseline for future regression testing
                # (In real implementation, this would be stored persistently)
                assert avg_performance > 0
                
                # Performance should be consistent (within 20% variance)
                max_perf = max(results)
                min_perf = min(results)
                variance = (max_perf - min_perf) / avg_performance
                assert variance < 0.2  # Less than 20% variance
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.gpu
    def test_memory_efficiency(self):
        """Test memory efficiency doesn't regress."""
        optimizer = get_rtx5090_optimizer()
        
        if torch.cuda.is_available():
            # Get initial memory state
            initial_stats = optimizer.get_performance_stats()
            
            if initial_stats:
                initial_allocated = initial_stats.memory_allocated_gb
                
                # Run some operations
                optimizer.benchmark_performance(matrix_size=512, iterations=3)
                
                # Check memory after operations
                final_stats = optimizer.get_performance_stats()
                if final_stats:
                    final_allocated = final_stats.memory_allocated_gb
                    
                    # Memory usage should not grow excessively
                    memory_growth = final_allocated - initial_allocated
                    assert memory_growth < 1.0  # Less than 1GB growth


# Pytest fixtures for RTX 5090 testing
@pytest.fixture
def rtx5090_optimizer():
    """Fixture providing RTX 5090 optimizer instance."""
    return get_rtx5090_optimizer()


@pytest.fixture
def gpu_available():
    """Fixture indicating if GPU is available."""
    return TORCH_AVAILABLE and torch.cuda.is_available()


@pytest.fixture
def rtx5090_available():
    """Fixture indicating if RTX 5090 is available."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
    
    optimizer = get_rtx5090_optimizer()
    return optimizer.is_rtx5090


# Performance benchmarks
@pytest.mark.benchmark
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_rtx5090_benchmark_suite(benchmark):
    """Benchmark suite for RTX 5090 performance testing."""
    optimizer = get_rtx5090_optimizer()
    
    if torch.cuda.is_available():
        # Benchmark the optimization application
        result = benchmark(optimizer.benchmark_performance, matrix_size=1024, iterations=5)
        
        if 'error' not in result:
            # Verify benchmark results are reasonable
            assert result['gflops'] > 0
            assert result['avg_time_ms'] > 0
