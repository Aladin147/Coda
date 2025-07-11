"""
Performance benchmarking tests for the voice processing system.

This module provides comprehensive performance testing including latency,
throughput, memory usage, and concurrent operation benchmarks.
"""

import pytest
import asyncio
import time
import psutil
import statistics
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.coda.components.voice.manager import VoiceManager
from src.coda.components.voice.models import VoiceConfig, AudioConfig, MoshiConfig
from src.coda.components.voice.websocket_handler import VoiceWebSocketHandler
from src.coda.components.voice.websocket_server import VoiceWebSocketServer
from src.coda.components.voice.audio_buffer_pool import AudioBufferPool
from src.coda.components.voice.optimized_cache import OptimizedLRUCache
from src.coda.components.voice.optimized_vram_manager import OptimizedVRAMManager
from src.coda.components.voice.performance_profiler import PerformanceProfiler


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self):
        self.results = {}
        self.profiler = PerformanceProfiler()
    
    def measure_time(self, func_name: str):
        """Decorator to measure execution time."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000  # Convert to ms
                if func_name not in self.results:
                    self.results[func_name] = []
                self.results[func_name].append(execution_time)
                
                return result
            return wrapper
        return decorator
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of measurements."""
        if metric_name not in self.results:
            return {}
        
        values = self.results[metric_name]
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }


class VoiceManagerPerformanceBenchmark(PerformanceBenchmark):
    """Performance tests for VoiceManager."""
    
    @pytest.fixture
    async def voice_manager(self):
        """Create optimized voice manager for testing."""
        config = VoiceConfig(
            audio=AudioConfig(
                sample_rate=16000,
                channels=1,
                format="wav"
            ),
            moshi=MoshiConfig(
                device="cpu",
                vram_allocation="1GB"
            )
        )
        
        manager = VoiceManager(config)
        
        # Mock dependencies for performance testing
        manager.vram_manager = Mock()
        manager.vram_manager.register_component.return_value = True
        manager.pipeline_manager = AsyncMock()
        manager.memory_manager = AsyncMock()
        manager.personality_manager = AsyncMock()
        manager.tool_manager = AsyncMock()
        
        await manager.initialize(config)
        return manager
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate realistic audio data for benchmarking."""
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        
        # Generate sine wave audio
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz tone
        audio_data = (audio_data * 32767).astype(np.int16)
        
        return audio_data.tobytes()
    
    @pytest.mark.asyncio
    async def test_conversation_creation_performance(self, voice_manager):
        """Benchmark conversation creation performance."""
        
        @self.measure_time("conversation_creation")
        async def create_conversation(conv_id):
            return await voice_manager.start_conversation(conv_id)
        
        # Create multiple conversations
        tasks = []
        for i in range(100):
            task = create_conversation(f"conv_{i}")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        stats = self.get_statistics("conversation_creation")
        
        # Performance assertions
        assert stats["mean"] < 50.0, f"Average conversation creation time too high: {stats['mean']}ms"
        assert stats["p95"] < 100.0, f"95th percentile too high: {stats['p95']}ms"
        assert stats["max"] < 200.0, f"Maximum time too high: {stats['max']}ms"
        
        print(f"Conversation Creation Performance: {stats}")
    
    @pytest.mark.asyncio
    async def test_voice_processing_latency(self, voice_manager, sample_audio_data):
        """Benchmark voice processing latency."""
        
        # Start conversation
        conversation_id = "perf_test_conv"
        await voice_manager.start_conversation(conversation_id)
        
        # Mock voice processing to return quickly
        mock_response = Mock()
        mock_response.conversation_id = conversation_id
        mock_response.text_content = "Test response"
        mock_response.processing_time = 50.0
        
        with patch.object(voice_manager, '_process_with_mode', return_value=mock_response):
            
            @self.measure_time("voice_processing")
            async def process_voice(audio_data):
                return await voice_manager.process_voice_input(conversation_id, audio_data)
            
            # Process multiple audio chunks
            for i in range(50):
                await process_voice(sample_audio_data)
        
        stats = self.get_statistics("voice_processing")
        
        # Performance assertions
        assert stats["mean"] < 100.0, f"Average processing time too high: {stats['mean']}ms"
        assert stats["p95"] < 200.0, f"95th percentile too high: {stats['p95']}ms"
        
        print(f"Voice Processing Performance: {stats}")
    
    @pytest.mark.asyncio
    async def test_concurrent_conversation_performance(self, voice_manager, sample_audio_data):
        """Benchmark concurrent conversation handling."""
        
        # Create multiple conversations
        conversation_ids = [f"concurrent_conv_{i}" for i in range(10)]
        for conv_id in conversation_ids:
            await voice_manager.start_conversation(conv_id)
        
        # Mock processing
        mock_response = Mock()
        mock_response.text_content = "Concurrent response"
        mock_response.processing_time = 75.0
        
        with patch.object(voice_manager, '_process_with_mode', return_value=mock_response):
            
            @self.measure_time("concurrent_processing")
            async def process_concurrent():
                tasks = []
                for conv_id in conversation_ids:
                    mock_response.conversation_id = conv_id
                    task = voice_manager.process_voice_input(conv_id, sample_audio_data)
                    tasks.append(task)
                
                return await asyncio.gather(*tasks)
            
            # Run concurrent processing multiple times
            for i in range(10):
                await process_concurrent()
        
        stats = self.get_statistics("concurrent_processing")
        
        # Performance assertions for concurrent processing
        assert stats["mean"] < 500.0, f"Concurrent processing too slow: {stats['mean']}ms"
        
        print(f"Concurrent Processing Performance: {stats}")


class WebSocketPerformanceBenchmark(PerformanceBenchmark):
    """Performance tests for WebSocket components."""
    
    @pytest.fixture
    def mock_voice_manager(self):
        """Create mock voice manager for WebSocket testing."""
        manager = AsyncMock()
        manager.start_conversation.return_value = Mock(conversation_id="test", is_active=True)
        manager.process_voice_input.return_value = Mock(
            conversation_id="test",
            text_content="Response",
            processing_time=50.0
        )
        return manager
    
    @pytest.mark.asyncio
    async def test_websocket_message_throughput(self, mock_voice_manager):
        """Benchmark WebSocket message handling throughput."""
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            host="localhost",
            port=0,
            auth_required=False
        )
        
        # Mock connections
        mock_connections = {}
        for i in range(100):
            mock_websocket = AsyncMock()
            client_id = f"client_{i}"
            connection = Mock()
            connection.client_id = client_id
            connection.websocket = mock_websocket
            connection.conversation_id = "test_conv"
            mock_connections[client_id] = connection
        
        handler.connections = mock_connections
        
        @self.measure_time("message_broadcast")
        async def broadcast_message():
            return await handler.broadcast_message(
                "status",
                {"message": "Test broadcast"},
                conversation_id="test_conv"
            )
        
        # Broadcast multiple messages
        for i in range(20):
            await broadcast_message()
        
        stats = self.get_statistics("message_broadcast")
        
        # Performance assertions
        assert stats["mean"] < 100.0, f"Message broadcast too slow: {stats['mean']}ms"
        
        print(f"WebSocket Message Broadcast Performance: {stats}")
    
    @pytest.mark.asyncio
    async def test_connection_handling_performance(self, mock_voice_manager):
        """Benchmark connection establishment and cleanup."""
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            host="localhost",
            port=0,
            auth_required=False
        )
        
        @self.measure_time("connection_setup")
        async def setup_connection(client_id):
            mock_websocket = AsyncMock()
            connection = Mock()
            connection.client_id = client_id
            connection.websocket = mock_websocket
            handler.connections[client_id] = connection
            return connection
        
        @self.measure_time("connection_cleanup")
        async def cleanup_connection(client_id):
            await handler._close_connection(client_id, "Performance test")
        
        # Test connection lifecycle
        for i in range(50):
            client_id = f"perf_client_{i}"
            await setup_connection(client_id)
            await cleanup_connection(client_id)
        
        setup_stats = self.get_statistics("connection_setup")
        cleanup_stats = self.get_statistics("connection_cleanup")
        
        # Performance assertions
        assert setup_stats["mean"] < 10.0, f"Connection setup too slow: {setup_stats['mean']}ms"
        assert cleanup_stats["mean"] < 20.0, f"Connection cleanup too slow: {cleanup_stats['mean']}ms"
        
        print(f"Connection Setup Performance: {setup_stats}")
        print(f"Connection Cleanup Performance: {cleanup_stats}")


class MemoryPerformanceBenchmark(PerformanceBenchmark):
    """Memory usage and performance tests."""
    
    def test_audio_buffer_pool_performance(self):
        """Benchmark audio buffer pool operations."""
        buffer_pool = AudioBufferPool(max_buffers=100)
        
        # Measure buffer acquisition
        start_time = time.perf_counter()
        buffers = []
        for i in range(1000):
            buffer = buffer_pool.get_buffer(1024)
            buffers.append(buffer)
        acquisition_time = (time.perf_counter() - start_time) * 1000
        
        # Measure buffer release
        start_time = time.perf_counter()
        for buffer in buffers:
            buffer_pool.return_buffer(buffer)
        release_time = (time.perf_counter() - start_time) * 1000
        
        # Performance assertions
        assert acquisition_time < 100.0, f"Buffer acquisition too slow: {acquisition_time}ms"
        assert release_time < 50.0, f"Buffer release too slow: {release_time}ms"
        
        print(f"Buffer Pool - Acquisition: {acquisition_time:.2f}ms, Release: {release_time:.2f}ms")
    
    def test_cache_performance(self):
        """Benchmark cache operations."""
        cache = OptimizedLRUCache(max_size=1000)
        
        # Measure cache writes
        start_time = time.perf_counter()
        for i in range(10000):
            cache.put(f"key_{i}", f"value_{i}")
        write_time = (time.perf_counter() - start_time) * 1000
        
        # Measure cache reads
        start_time = time.perf_counter()
        for i in range(10000):
            cache.get(f"key_{i}")
        read_time = (time.perf_counter() - start_time) * 1000
        
        # Performance assertions
        assert write_time < 500.0, f"Cache writes too slow: {write_time}ms"
        assert read_time < 200.0, f"Cache reads too slow: {read_time}ms"
        
        print(f"Cache Performance - Writes: {write_time:.2f}ms, Reads: {read_time:.2f}ms")
    
    def test_memory_usage_monitoring(self):
        """Monitor memory usage during operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large data structures
        large_cache = OptimizedLRUCache(max_size=10000)
        buffer_pool = AudioBufferPool(max_buffers=1000)
        
        # Fill with data
        for i in range(5000):
            large_cache.put(f"key_{i}", b"x" * 1024)  # 1KB values
            buffer_pool.get_buffer(2048)  # 2KB buffers
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        large_cache.clear()
        buffer_pool.clear_pool()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_recovered = peak_memory - final_memory
        
        print(f"Memory Usage - Initial: {initial_memory:.1f}MB, "
              f"Peak: {peak_memory:.1f}MB, Final: {final_memory:.1f}MB")
        print(f"Memory Increase: {memory_increase:.1f}MB, "
              f"Memory Recovered: {memory_recovered:.1f}MB")
        
        # Memory usage assertions
        assert memory_increase < 100.0, f"Memory usage too high: {memory_increase}MB"
        assert memory_recovered > memory_increase * 0.8, "Memory not properly released"


class ConcurrencyPerformanceBenchmark(PerformanceBenchmark):
    """Concurrency and load testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_voice_processing(self):
        """Test performance under concurrent voice processing load."""
        
        # Create multiple voice managers
        managers = []
        for i in range(5):
            config = VoiceConfig(
                audio=AudioConfig(),
                moshi=MoshiConfig(device="cpu")
            )
            manager = VoiceManager(config)
            
            # Mock dependencies
            manager.vram_manager = Mock()
            manager.vram_manager.register_component.return_value = True
            manager.pipeline_manager = AsyncMock()
            manager.memory_manager = AsyncMock()
            manager.personality_manager = AsyncMock()
            
            await manager.initialize(config)
            managers.append(manager)
        
        # Mock processing
        mock_response = Mock()
        mock_response.text_content = "Concurrent response"
        mock_response.processing_time = 100.0
        
        @self.measure_time("concurrent_load")
        async def concurrent_processing():
            tasks = []
            for i, manager in enumerate(managers):
                conv_id = f"load_conv_{i}"
                mock_response.conversation_id = conv_id
                
                await manager.start_conversation(conv_id)
                
                with patch.object(manager, '_process_with_mode', return_value=mock_response):
                    task = manager.process_voice_input(conv_id, b"test_audio")
                    tasks.append(task)
            
            return await asyncio.gather(*tasks)
        
        # Run concurrent load test
        for i in range(10):
            await concurrent_processing()
        
        stats = self.get_statistics("concurrent_load")
        
        # Performance assertions
        assert stats["mean"] < 1000.0, f"Concurrent load too slow: {stats['mean']}ms"
        
        print(f"Concurrent Load Performance: {stats}")
    
    @pytest.mark.asyncio
    async def test_websocket_concurrent_connections(self):
        """Test WebSocket performance with many concurrent connections."""
        mock_voice_manager = AsyncMock()
        
        server = VoiceWebSocketServer(
            voice_config=None,
            host="localhost",
            port=0,
            max_connections=200,
            auth_required=False
        )
        
        # Mock server components
        server.voice_manager = mock_voice_manager
        server.websocket_handler = Mock()
        server.websocket_handler.connections = {}
        
        @self.measure_time("connection_simulation")
        async def simulate_connections(num_connections):
            # Simulate multiple connections
            for i in range(num_connections):
                client_id = f"load_client_{i}"
                connection = Mock()
                connection.client_id = client_id
                connection.conversation_id = f"load_conv_{i}"
                server.websocket_handler.connections[client_id] = connection
        
        # Test different connection loads
        for num_connections in [50, 100, 150]:
            await simulate_connections(num_connections)
            
            # Verify performance doesn't degrade significantly
            stats = self.get_statistics("connection_simulation")
            if stats:
                assert stats["mean"] < 100.0, f"Connection simulation too slow: {stats['mean']}ms"
        
        print(f"WebSocket Concurrent Connections: {len(server.websocket_handler.connections)}")


@pytest.mark.performance
class TestSystemPerformance:
    """System-wide performance tests."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency(self):
        """Test complete end-to-end system latency."""
        # This would test the complete pipeline from audio input to response
        # Including WebSocket, voice processing, and response generation
        
        config = VoiceConfig(
            audio=AudioConfig(sample_rate=16000),
            moshi=MoshiConfig(device="cpu")
        )
        
        # Create complete system
        voice_manager = VoiceManager(config)
        
        # Mock all components for performance testing
        voice_manager.vram_manager = Mock()
        voice_manager.vram_manager.register_component.return_value = True
        voice_manager.pipeline_manager = AsyncMock()
        voice_manager.memory_manager = AsyncMock()
        voice_manager.personality_manager = AsyncMock()
        
        await voice_manager.initialize(config)
        
        websocket_handler = VoiceWebSocketHandler(
            voice_manager=voice_manager,
            host="localhost",
            port=0,
            auth_required=False
        )
        
        # Measure end-to-end latency
        start_time = time.perf_counter()
        
        # Simulate complete workflow
        conversation_id = "e2e_test"
        await voice_manager.start_conversation(conversation_id)
        
        # Mock voice processing
        mock_response = Mock()
        mock_response.conversation_id = conversation_id
        mock_response.text_content = "End-to-end response"
        mock_response.processing_time = 150.0
        
        with patch.object(voice_manager, '_process_with_mode', return_value=mock_response):
            response = await voice_manager.process_voice_input(
                conversation_id, b"test_audio_data"
            )
        
        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Performance assertions
        assert total_latency < 500.0, f"End-to-end latency too high: {total_latency}ms"
        assert response.text_content == "End-to-end response"
        
        print(f"End-to-End Latency: {total_latency:.2f}ms")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])
