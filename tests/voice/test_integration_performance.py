"""
Integration tests for performance optimization components.

Tests the integration of audio buffer pools, caching systems,
VRAM management, and performance profiling.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.coda.components.voice.audio_buffer_pool import (
    AudioBufferPool, OptimizedAudioProcessor, get_global_buffer_pool
)
from src.coda.components.voice.optimized_cache import (
    OptimizedLRUCache, VoiceResponseCache, CachePolicy
)
from src.coda.components.voice.optimized_vram_manager import (
    OptimizedVRAMManager, AllocationStrategy
)
from src.coda.components.voice.performance_profiler import (
    PerformanceProfiler, ProfilerLevel
)
from src.coda.components.voice.models import VoiceMessage, VoiceResponse


class TestPerformanceIntegration:
    """Integration tests for performance optimization components."""
    
    @pytest.fixture
    def audio_buffer_pool(self):
        """Create audio buffer pool for testing."""
        pool = AudioBufferPool(max_buffers=20, cleanup_interval=1.0)
        yield pool
        pool.clear_pool()
    
    @pytest.fixture
    def optimized_cache(self):
        """Create optimized cache for testing."""
        cache = OptimizedLRUCache(
            max_size=100,
            max_memory_mb=10.0,
            policy=CachePolicy.LRU
        )
        yield cache
        cache.clear()
    
    @pytest.fixture
    def vram_manager(self):
        """Create VRAM manager for testing."""
        manager = OptimizedVRAMManager(
            total_vram_gb=8.0,
            reserved_system_gb=1.0,
            allocation_strategy=AllocationStrategy.BEST_FIT
        )
        yield manager
        manager.cleanup()
    
    @pytest.fixture
    def performance_profiler(self):
        """Create performance profiler for testing."""
        profiler = PerformanceProfiler(
            level=ProfilerLevel.DETAILED,
            max_history=1000,
            enable_gpu_monitoring=False  # Disable for testing
        )
        yield profiler
        profiler.cleanup()
    
    def test_audio_buffer_pool_performance(self, audio_buffer_pool):
        """Test audio buffer pool performance and memory efficiency."""
        buffer_size = 48000  # 1 second at 48kHz
        num_operations = 100
        
        # Test buffer acquisition and release performance
        start_time = time.time()
        buffers = []
        
        for _ in range(num_operations):
            buffer = audio_buffer_pool.acquire_buffer(buffer_size)
            buffers.append(buffer)
        
        acquisition_time = time.time() - start_time
        
        # Release all buffers
        start_time = time.time()
        for buffer in buffers:
            audio_buffer_pool.release_buffer(buffer)
        
        release_time = time.time() - start_time
        
        # Verify performance metrics
        stats = audio_buffer_pool.get_stats()
        assert stats["total_allocations"] == num_operations
        assert stats["cache_hit_rate"] > 0.8  # Should have high cache hit rate after first few
        
        # Performance should be reasonable
        assert acquisition_time < 1.0  # Should be fast
        assert release_time < 0.5
        
        print(f"Buffer pool stats: {stats}")
        print(f"Acquisition time: {acquisition_time:.3f}s, Release time: {release_time:.3f}s")
    
    def test_optimized_audio_processor_integration(self, audio_buffer_pool):
        """Test optimized audio processor with buffer pool."""
        processor = OptimizedAudioProcessor(buffer_pool=audio_buffer_pool)
        
        # Create test audio data
        import numpy as np
        sample_rate = 16000
        duration = 1.0
        audio_samples = int(sample_rate * duration)
        audio_data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, audio_samples)) * 32767).astype(np.int16)
        audio_bytes = audio_data.tobytes()
        
        # Test processing with different operations
        operations_sets = [
            ["normalize"],
            ["normalize", "denoise"],
            ["normalize", "denoise", "compress"]
        ]
        
        for operations in operations_sets:
            start_time = time.time()
            processed_audio = processor.process_audio_optimized(audio_bytes, operations)
            processing_time = time.time() - start_time
            
            assert len(processed_audio) == len(audio_bytes)
            assert processing_time < 0.1  # Should be fast
        
        # Verify buffer pool was used efficiently
        stats = processor.get_performance_stats()
        assert "buffer_pool_stats" in stats
        assert stats["buffer_pool_stats"]["cache_hit_rate"] > 0.5
    
    def test_cache_performance_and_memory_management(self, optimized_cache):
        """Test cache performance and memory management."""
        # Test cache performance with various data sizes
        test_data = [
            ("small", "x" * 100),
            ("medium", "x" * 1000),
            ("large", "x" * 10000)
        ]
        
        # Fill cache
        for key, value in test_data * 20:  # Repeat to test eviction
            cache_key = f"{key}_{time.time()}"
            optimized_cache.put(cache_key, value)
        
        # Test cache hit performance
        start_time = time.time()
        for i in range(1000):
            key = f"small_{i % 10}"
            optimized_cache.get(key, "default")
        
        lookup_time = time.time() - start_time
        
        # Verify performance
        stats = optimized_cache.get_stats()
        assert lookup_time < 0.1  # Should be very fast
        assert stats["size"] <= optimized_cache.max_size
        assert stats["memory_usage_mb"] <= optimized_cache.max_memory_bytes / (1024 * 1024)
        
        print(f"Cache stats: {stats}")
        print(f"Lookup time for 1000 operations: {lookup_time:.3f}s")
    
    def test_voice_response_cache_integration(self):
        """Test voice response cache with realistic data."""
        cache = VoiceResponseCache(max_size=50, max_memory_mb=5.0)
        
        # Create test voice messages and responses
        voice_messages = []
        responses = []
        
        for i in range(20):
            voice_msg = VoiceMessage(
                conversation_id=f"conv_{i % 5}",  # 5 different conversations
                audio_data=b"mock_audio_data",
                text_content=f"Test message {i}",
                speaker="user"
            )
            
            response = VoiceResponse(
                conversation_id=voice_msg.conversation_id,
                text_content=f"Response to message {i}",
                processing_time=100.0 + i * 10
            )
            
            voice_messages.append(voice_msg)
            responses.append(response)
        
        # Cache responses
        for msg, resp in zip(voice_messages, responses):
            cache.cache_response(msg, resp)
        
        # Test cache hits
        cache_hits = 0
        for msg in voice_messages:
            cached_response = cache.get_response(msg)
            if cached_response is not None:
                cache_hits += 1
        
        # Should have good cache hit rate
        hit_rate = cache_hits / len(voice_messages)
        assert hit_rate > 0.8
        
        stats = cache.get_stats()
        print(f"Voice cache stats: {stats}")
        print(f"Cache hit rate: {hit_rate:.2f}")
    
    def test_vram_manager_allocation_performance(self, vram_manager):
        """Test VRAM manager allocation performance."""
        num_components = 20
        allocation_sizes = [100, 200, 500, 1000, 2000]  # MB
        
        # Test allocation performance
        start_time = time.time()
        allocated_components = []
        
        for i in range(num_components):
            component_id = f"component_{i}"
            size_mb = allocation_sizes[i % len(allocation_sizes)]
            
            success = vram_manager.allocate(component_id, size_mb, priority=5)
            if success:
                allocated_components.append(component_id)
        
        allocation_time = time.time() - start_time
        
        # Test deallocation performance
        start_time = time.time()
        for component_id in allocated_components:
            vram_manager.deallocate(component_id)
        
        deallocation_time = time.time() - start_time
        
        # Verify performance
        stats = vram_manager.get_usage_stats()
        assert allocation_time < 1.0  # Should be fast
        assert deallocation_time < 0.5
        assert stats["allocated_mb"] == 0  # All should be deallocated
        
        print(f"VRAM manager stats: {stats}")
        print(f"Allocation time: {allocation_time:.3f}s, Deallocation time: {deallocation_time:.3f}s")
    
    def test_performance_profiler_integration(self, performance_profiler):
        """Test performance profiler with realistic workload."""
        # Start monitoring
        performance_profiler.start_monitoring(interval=0.1)
        
        # Start a profiling session
        session_id = performance_profiler.start_session("integration_test")
        
        # Simulate voice processing operations
        operations = [
            ("audio_processor", "process_input"),
            ("moshi_client", "inference"),
            ("llm_integration", "generate_response"),
            ("hybrid_orchestrator", "coordinate"),
            ("audio_processor", "process_output")
        ]
        
        for component, operation in operations:
            with performance_profiler.profile_operation(component, operation, session_id):
                # Simulate work
                time.sleep(0.05 + (hash(operation) % 100) / 1000.0)  # Variable delay
                
                # Record some metrics
                performance_profiler.record_metric(
                    name="operation_duration",
                    value=50.0 + (hash(operation) % 100),
                    unit="ms",
                    component=component,
                    operation=operation
                )
        
        # End session and get results
        session = performance_profiler.end_session(session_id)
        
        # Stop monitoring
        performance_profiler.stop_monitoring()
        
        # Verify session results
        assert session.duration > 0
        assert len(session.metrics) > 0
        assert len(session.bottlenecks) >= 0  # May or may not have bottlenecks
        
        # Get performance summary
        summary = performance_profiler.get_performance_summary(last_n_minutes=1)
        assert "total_metrics" in summary
        assert summary["total_metrics"] > 0
        
        print(f"Profiling session duration: {session.duration:.3f}s")
        print(f"Metrics collected: {len(session.metrics)}")
        print(f"Bottlenecks detected: {len(session.bottlenecks)}")
        if session.recommendations:
            print(f"Recommendations: {session.recommendations}")
    
    def test_integrated_performance_optimization(self, audio_buffer_pool, optimized_cache, 
                                               vram_manager, performance_profiler):
        """Test integrated performance optimization across all components."""
        # Start profiling
        session_id = performance_profiler.start_session("integrated_test")
        
        # Simulate a complete voice processing pipeline
        with performance_profiler.profile_operation("integration", "full_pipeline", session_id):
            
            # 1. VRAM allocation
            with performance_profiler.profile_operation("vram", "allocate", session_id):
                vram_success = vram_manager.allocate("test_model", 1000.0)
                assert vram_success
            
            # 2. Audio buffer management
            with performance_profiler.profile_operation("audio", "buffer_management", session_id):
                buffer = audio_buffer_pool.acquire_buffer(48000)
                # Simulate audio processing
                import numpy as np
                buffer.data[:1000] = np.random.random(1000).astype(np.float32)
                audio_buffer_pool.release_buffer(buffer)
            
            # 3. Cache operations
            with performance_profiler.profile_operation("cache", "operations", session_id):
                # Cache some responses
                for i in range(10):
                    key = f"test_key_{i}"
                    value = f"test_value_{i}" * 100  # Some data
                    optimized_cache.put(key, value)
                
                # Retrieve cached data
                for i in range(10):
                    key = f"test_key_{i}"
                    cached_value = optimized_cache.get(key)
                    assert cached_value is not None
            
            # 4. Cleanup
            with performance_profiler.profile_operation("cleanup", "resources", session_id):
                vram_manager.deallocate("test_model")
        
        # End profiling session
        session = performance_profiler.end_session(session_id)
        
        # Verify integrated performance
        assert session.duration < 1.0  # Should complete quickly
        assert len(session.metrics) > 0
        
        # Get component statistics
        buffer_stats = audio_buffer_pool.get_stats()
        cache_stats = optimized_cache.get_stats()
        vram_stats = vram_manager.get_usage_stats()
        
        # Verify all components performed well
        assert buffer_stats["cache_hit_rate"] >= 0  # May be 0 for first use
        assert cache_stats["hit_rate"] > 0.8  # Should have good hit rate
        assert vram_stats["allocated_mb"] == 0  # Should be cleaned up
        
        print(f"Integrated test completed in {session.duration:.3f}s")
        print(f"Buffer pool efficiency: {buffer_stats['cache_hit_rate']:.2f}")
        print(f"Cache hit rate: {cache_stats['hit_rate']:.2f}")
        print(f"VRAM utilization: {vram_stats['usage_percent']:.1f}%")
    
    def test_concurrent_performance_operations(self, audio_buffer_pool, optimized_cache):
        """Test performance under concurrent operations."""
        num_threads = 5
        operations_per_thread = 20
        
        def worker_thread(thread_id):
            """Worker thread for concurrent testing."""
            for i in range(operations_per_thread):
                # Buffer operations
                buffer = audio_buffer_pool.acquire_buffer(1000 + i * 100)
                time.sleep(0.001)  # Simulate work
                audio_buffer_pool.release_buffer(buffer)
                
                # Cache operations
                key = f"thread_{thread_id}_item_{i}"
                value = f"data_{i}" * 50
                optimized_cache.put(key, value)
                
                # Retrieve from cache
                retrieved = optimized_cache.get(key)
                assert retrieved == value
        
        # Start concurrent threads
        threads = []
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Verify performance under concurrency
        buffer_stats = audio_buffer_pool.get_stats()
        cache_stats = optimized_cache.get_stats()
        
        total_operations = num_threads * operations_per_thread
        assert total_time < 5.0  # Should complete reasonably quickly
        assert buffer_stats["total_allocations"] == total_operations
        assert cache_stats["size"] > 0
        
        print(f"Concurrent test: {total_operations} operations in {total_time:.3f}s")
        print(f"Operations per second: {total_operations / total_time:.1f}")


if __name__ == "__main__":
    pytest.main([__file__])
