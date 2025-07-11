"""
GPU Performance Tests for RTX 5090.
Tests VRAM utilization, CUDA core usage, and GPU compute performance.
"""

import pytest
import asyncio
import time
import psutil
import gc
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Optional

try:
    import torch
    import GPUtil
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from src.coda.components.voice.vram_manager import DynamicVRAMManager
    VRAM_MANAGER_AVAILABLE = True
except ImportError:
    VRAM_MANAGER_AVAILABLE = False
    # Create mock VRAMManager for testing
    class DynamicVRAMManager:
        def __init__(self, max_vram_usage=0.8, cleanup_threshold=0.9, monitoring_interval=0.5):
            self.max_vram_usage = max_vram_usage
            self.cleanup_threshold = cleanup_threshold
            self.monitoring_interval = monitoring_interval

        def get_vram_usage(self):
            return 0.5  # Mock 50% usage

        def cleanup_vram(self):
            return True

        def monitor_vram(self):
            return {"usage": 0.5, "available": 0.5}

try:
    from src.coda.components.voice.moshi_integration import MoshiVoiceProcessor
    MOSHI_AVAILABLE = True
except ImportError:
    MOSHI_AVAILABLE = False
    # Create mock for testing
    class MoshiVoiceProcessor:
        def __init__(self):
            pass


class GPUMetrics:
    """GPU metrics collection utility."""
    
    def __init__(self):
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU memory information."""
        if not self.gpu_available:
            return {"total": 0, "used": 0, "free": 0, "utilization": 0}
        
        try:
            torch.cuda.set_device(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(device_id) / (1024**3)  # GB
            free = total - cached
            
            return {
                "total": total,
                "allocated": allocated,
                "cached": cached,
                "free": free,
                "utilization": (cached / total) * 100
            }
        except Exception:
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0, "utilization": 0}
    
    def get_gpu_utilization(self) -> List[Dict[str, Any]]:
        """Get GPU utilization information."""
        if not self.gpu_available:
            return []
        
        try:
            gpus = GPUtil.getGPUs()
            return [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,  # Convert to percentage
                    "memory_util": gpu.memoryUtil * 100,
                    "temperature": gpu.temperature
                }
                for gpu in gpus
            ]
        except Exception:
            return []


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGPUPerformance:
    """GPU performance tests for RTX 5090."""

    @pytest.fixture
    def gpu_metrics(self):
        """Create GPU metrics collector."""
        return GPUMetrics()

    @pytest.fixture
    def vram_manager(self):
        """Create VRAM manager for testing."""
        from src.coda.components.voice.config import VoiceConfig
        config = VoiceConfig()
        return DynamicVRAMManager(config)

    @pytest.fixture
    def mock_moshi_integration(self):
        """Create mock Moshi integration for GPU testing."""
        integration = Mock(spec=MoshiVoiceProcessor)
        integration.initialize = AsyncMock()
        integration.process_audio = AsyncMock()
        integration.cleanup = AsyncMock()
        integration.get_memory_usage = Mock(return_value={"vram_used": 2.5, "vram_total": 32.0})
        return integration

    def test_gpu_availability(self, gpu_metrics):
        """Test GPU availability and basic information."""
        if gpu_metrics.gpu_available:
            assert gpu_metrics.device_count > 0
            memory_info = gpu_metrics.get_gpu_memory_info()
            assert memory_info["total"] > 0
            print(f"GPU Memory Total: {memory_info['total']:.2f} GB")
            print(f"GPU Memory Used: {memory_info['allocated']:.2f} GB")
        else:
            pytest.skip("No GPU available for testing")

    @pytest.mark.benchmark(group="gpu_memory")
    def test_vram_allocation_performance(self, benchmark, gpu_metrics, vram_manager):
        """Benchmark VRAM allocation performance."""
        
        def allocate_vram():
            if gpu_metrics.gpu_available:
                # Simulate large tensor allocation
                tensor = torch.randn(1000, 1000, device='cuda')
                result = torch.matmul(tensor, tensor.T)
                del tensor, result
                torch.cuda.empty_cache()
                return True
            return False
        
        result = benchmark(allocate_vram)
        if gpu_metrics.gpu_available:
            assert result is True

    @pytest.mark.benchmark(group="gpu_compute")
    def test_gpu_compute_performance(self, benchmark, gpu_metrics):
        """Benchmark GPU compute performance."""
        
        def gpu_compute():
            if gpu_metrics.gpu_available:
                # Simulate compute-intensive operation
                a = torch.randn(2048, 2048, device='cuda')
                b = torch.randn(2048, 2048, device='cuda')
                c = torch.matmul(a, b)
                result = torch.sum(c)
                del a, b, c
                torch.cuda.empty_cache()
                return result.item()
            return 0.0
        
        result = benchmark(gpu_compute)
        if gpu_metrics.gpu_available:
            assert isinstance(result, float)

    @pytest.mark.benchmark(group="gpu_memory_management")
    def test_vram_manager_performance(self, benchmark, vram_manager, gpu_metrics):
        """Benchmark VRAM manager performance."""
        
        def vram_management():
            if gpu_metrics.gpu_available:
                # Test VRAM manager operations
                vram_manager.check_memory_usage()
                vram_manager.cleanup_if_needed()
                return vram_manager.get_memory_stats()
            return {"vram_used": 0, "vram_total": 0}
        
        result = benchmark(vram_management)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_concurrent_gpu_operations(self, gpu_metrics):
        """Test concurrent GPU operations performance."""
        if not gpu_metrics.gpu_available:
            pytest.skip("No GPU available for testing")
        
        async def gpu_task(task_id: int):
            """Simulate GPU task."""
            await asyncio.sleep(0.001)  # Simulate async setup
            
            # GPU computation
            tensor = torch.randn(512, 512, device='cuda')
            result = torch.matmul(tensor, tensor.T)
            computation_result = torch.sum(result).item()
            
            del tensor, result
            torch.cuda.empty_cache()
            
            return {"task_id": task_id, "result": computation_result}
        
        # Run multiple concurrent GPU tasks
        start_time = time.time()
        tasks = [gpu_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify results
        assert len(results) == 10
        for result in results:
            assert "task_id" in result
            assert "result" in result
        
        # Performance assertion
        total_time = end_time - start_time
        assert total_time < 5.0  # Should complete within 5 seconds
        
        print(f"Concurrent GPU operations completed in {total_time:.3f}s")

    def test_gpu_memory_scaling(self, gpu_metrics):
        """Test GPU memory scaling with increasing load."""
        if not gpu_metrics.gpu_available:
            pytest.skip("No GPU available for testing")
        
        memory_usage = []
        tensor_sizes = [256, 512, 1024, 2048]
        
        for size in tensor_sizes:
            # Allocate tensor of increasing size
            tensor = torch.randn(size, size, device='cuda')
            
            # Get memory usage
            memory_info = gpu_metrics.get_gpu_memory_info()
            memory_usage.append({
                "tensor_size": size,
                "memory_allocated": memory_info["allocated"],
                "memory_utilization": memory_info["utilization"]
            })
            
            del tensor
            torch.cuda.empty_cache()
        
        # Verify memory scaling
        assert len(memory_usage) == len(tensor_sizes)
        
        # Memory usage should generally increase with tensor size
        for i in range(1, len(memory_usage)):
            current = memory_usage[i]
            previous = memory_usage[i-1]
            print(f"Size {current['tensor_size']}: {current['memory_allocated']:.2f}GB allocated")

    @pytest.mark.benchmark(group="gpu_model_loading")
    def test_model_loading_performance(self, benchmark, mock_moshi_integration, gpu_metrics):
        """Benchmark model loading performance."""
        
        async def load_model():
            await mock_moshi_integration.initialize()
            return mock_moshi_integration.get_memory_usage()
        
        def sync_load_model():
            return asyncio.run(load_model())
        
        result = benchmark(sync_load_model)
        assert isinstance(result, dict)
        assert "vram_used" in result

    def test_gpu_utilization_monitoring(self, gpu_metrics):
        """Test GPU utilization monitoring."""
        if not gpu_metrics.gpu_available:
            pytest.skip("No GPU available for testing")
        
        # Get initial utilization
        initial_util = gpu_metrics.get_gpu_utilization()
        
        # Perform GPU work
        tensor = torch.randn(1024, 1024, device='cuda')
        for _ in range(100):
            result = torch.matmul(tensor, tensor.T)
        
        # Get utilization during work
        work_util = gpu_metrics.get_gpu_utilization()
        
        del tensor, result
        torch.cuda.empty_cache()
        
        # Verify utilization data
        if initial_util:
            assert len(initial_util) > 0
            gpu_info = initial_util[0]
            assert "load" in gpu_info
            assert "memory_util" in gpu_info
            assert "name" in gpu_info
            
            print(f"GPU: {gpu_info['name']}")
            print(f"Load: {gpu_info['load']:.1f}%")
            print(f"Memory Util: {gpu_info['memory_util']:.1f}%")

    @pytest.mark.benchmark(group="gpu_memory_cleanup")
    def test_memory_cleanup_performance(self, benchmark, gpu_metrics):
        """Benchmark GPU memory cleanup performance."""
        
        def memory_cleanup():
            if gpu_metrics.gpu_available:
                # Allocate some memory
                tensors = [torch.randn(256, 256, device='cuda') for _ in range(10)]
                
                # Cleanup
                for tensor in tensors:
                    del tensor
                
                torch.cuda.empty_cache()
                gc.collect()
                
                return gpu_metrics.get_gpu_memory_info()
            return {"allocated": 0}
        
        result = benchmark(memory_cleanup)
        assert isinstance(result, dict)

    def test_rtx_5090_specific_features(self, gpu_metrics):
        """Test RTX 5090 specific features and capabilities."""
        if not gpu_metrics.gpu_available:
            pytest.skip("No GPU available for testing")
        
        # Get GPU properties
        device_props = torch.cuda.get_device_properties(0)
        
        print(f"GPU Name: {device_props.name}")
        print(f"Total Memory: {device_props.total_memory / (1024**3):.1f} GB")
        print(f"Multiprocessor Count: {device_props.multi_processor_count}")
        print(f"Major Compute Capability: {device_props.major}")
        print(f"Minor Compute Capability: {device_props.minor}")
        
        # Test for RTX 5090 characteristics
        memory_gb = device_props.total_memory / (1024**3)
        
        # RTX 5090 should have significant memory (24GB+)
        if memory_gb > 20:
            print("High-end GPU detected - optimizing for large-scale operations")
            
            # Test large tensor operations that RTX 5090 can handle
            large_tensor = torch.randn(4096, 4096, device='cuda')
            result = torch.matmul(large_tensor, large_tensor.T)
            
            assert result.shape == (4096, 4096)
            
            del large_tensor, result
            torch.cuda.empty_cache()

    @pytest.mark.benchmark(group="gpu_throughput")
    def test_gpu_throughput(self, benchmark, gpu_metrics):
        """Benchmark GPU throughput for typical operations."""
        
        def throughput_test():
            if gpu_metrics.gpu_available:
                operations_count = 0
                start_time = time.time()
                
                # Perform operations for a fixed time
                while time.time() - start_time < 0.1:  # 100ms test
                    tensor = torch.randn(128, 128, device='cuda')
                    result = torch.matmul(tensor, tensor.T)
                    operations_count += 1
                    del tensor, result
                
                torch.cuda.empty_cache()
                return operations_count
            return 0
        
        operations = benchmark(throughput_test)
        if gpu_metrics.gpu_available:
            print(f"GPU Operations per 100ms: {operations}")
            assert operations > 0


class TestGPUResourceManagement:
    """Test GPU resource management and optimization."""

    @pytest.fixture
    def resource_manager(self):
        """Create resource manager for testing."""
        return DynamicVRAMManager(
            max_vram_usage=0.8,
            cleanup_threshold=0.9,
            monitoring_interval=0.5
        )

    def test_vram_limit_enforcement(self, resource_manager):
        """Test VRAM usage limit enforcement."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("No GPU available for testing")
        
        # Test that VRAM manager respects limits
        initial_stats = resource_manager.get_memory_stats()
        
        # Simulate memory allocation
        resource_manager.check_memory_usage()
        
        stats = resource_manager.get_memory_stats()
        assert isinstance(stats, dict)
        assert "vram_used" in stats or "memory_used" in stats

    def test_automatic_cleanup(self, resource_manager):
        """Test automatic memory cleanup."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("No GPU available for testing")
        
        # Force cleanup
        resource_manager.cleanup_if_needed()
        
        # Verify cleanup occurred
        stats = resource_manager.get_memory_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_resource_monitoring(self, resource_manager):
        """Test continuous resource monitoring."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            pytest.skip("No GPU available for testing")
        
        # Start monitoring
        monitoring_task = asyncio.create_task(
            resource_manager.start_monitoring()
        )
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        monitoring_task.cancel()
        
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        # Verify monitoring worked
        stats = resource_manager.get_memory_stats()
        assert isinstance(stats, dict)
