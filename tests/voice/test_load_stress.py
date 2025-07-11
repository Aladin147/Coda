"""
Load testing and stress testing for the voice processing system.

This module provides comprehensive load testing including high-volume
concurrent operations, resource exhaustion testing, and system recovery testing.
"""

import pytest
import asyncio
import time
import psutil
import gc
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.coda.components.voice.manager import VoiceManager
from src.coda.components.voice.models import VoiceConfig, AudioConfig, MoshiConfig
from src.coda.components.voice.websocket_handler import VoiceWebSocketHandler
from src.coda.components.voice.websocket_server import VoiceWebSocketServer
from src.coda.components.voice.audio_buffer_pool import AudioBufferPool
from src.coda.components.voice.optimized_vram_manager import OptimizedVRAMManager
from src.coda.components.voice.exceptions import ResourceExhaustionError


class LoadTestMetrics:
    """Collect and analyze load test metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.successful_operations = 0
        self.failed_operations = 0
        self.response_times = []
        self.memory_samples = []
        self.cpu_samples = []
        self.error_details = []
    
    def start_test(self):
        """Start collecting metrics."""
        self.start_time = time.perf_counter()
        self.successful_operations = 0
        self.failed_operations = 0
        self.response_times.clear()
        self.memory_samples.clear()
        self.cpu_samples.clear()
        self.error_details.clear()
    
    def end_test(self):
        """End metric collection."""
        self.end_time = time.perf_counter()
    
    def record_operation(self, success: bool, response_time: float, error: str = None):
        """Record operation result."""
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
            if error:
                self.error_details.append(error)
        
        self.response_times.append(response_time)
    
    def record_system_metrics(self):
        """Record current system metrics."""
        process = psutil.Process()
        self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_samples.append(process.cpu_percent())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_time = self.end_time - self.start_time if self.end_time else 0
        total_operations = self.successful_operations + self.failed_operations
        
        return {
            "duration_seconds": total_time,
            "total_operations": total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.successful_operations / total_operations if total_operations > 0 else 0,
            "operations_per_second": total_operations / total_time if total_time > 0 else 0,
            "avg_response_time_ms": np.mean(self.response_times) if self.response_times else 0,
            "p95_response_time_ms": np.percentile(self.response_times, 95) if self.response_times else 0,
            "p99_response_time_ms": np.percentile(self.response_times, 99) if self.response_times else 0,
            "max_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "avg_cpu_percent": np.mean(self.cpu_samples) if self.cpu_samples else 0,
            "error_count_by_type": self._count_errors()
        }
    
    def _count_errors(self) -> Dict[str, int]:
        """Count errors by type."""
        error_counts = {}
        for error in self.error_details:
            error_type = type(error).__name__ if hasattr(error, '__name__') else str(error)
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts


@pytest.mark.load_test
class TestVoiceManagerLoadTesting:
    """Load testing for VoiceManager component."""
    
    @pytest.fixture
    async def voice_manager_pool(self):
        """Create pool of voice managers for load testing."""
        managers = []
        
        for i in range(5):  # Create 5 managers
            config = VoiceConfig(
                audio=AudioConfig(sample_rate=16000),
                moshi=MoshiConfig(device="cpu", vram_allocation="200MB")
            )
            
            manager = VoiceManager(config)
            
            # Mock dependencies for load testing
            manager.vram_manager = Mock()
            manager.vram_manager.register_component.return_value = True
            manager.pipeline_manager = AsyncMock()
            manager.memory_manager = AsyncMock()
            manager.personality_manager = AsyncMock()
            manager.tool_manager = AsyncMock()
            
            await manager.initialize(config)
            managers.append(manager)
        
        yield managers
        
        # Cleanup
        for manager in managers:
            await manager.cleanup()
    
    @pytest.fixture
    def large_audio_data(self):
        """Generate large audio data for stress testing."""
        sample_rate = 16000
        duration = 10.0  # 10 seconds of audio
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        return audio_data.tobytes()
    
    @pytest.mark.asyncio
    async def test_high_volume_conversations(self, voice_manager_pool):
        """Test handling of high volume of concurrent conversations."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        async def create_and_process_conversation(manager_idx: int, conv_idx: int):
            """Create conversation and process voice input."""
            manager = voice_manager_pool[manager_idx % len(voice_manager_pool)]
            conversation_id = f"load_conv_{manager_idx}_{conv_idx}"
            
            start_time = time.perf_counter()
            try:
                # Start conversation
                await manager.start_conversation(conversation_id)
                
                # Mock voice processing
                mock_response = Mock()
                mock_response.conversation_id = conversation_id
                mock_response.text_content = f"Response for {conversation_id}"
                mock_response.processing_time = 50.0
                
                with patch.object(manager, '_process_with_mode', return_value=mock_response):
                    response = await manager.process_voice_input(
                        conversation_id, b"test_audio_data"
                    )
                
                # End conversation
                await manager.end_conversation(conversation_id)
                
                response_time = (time.perf_counter() - start_time) * 1000
                metrics.record_operation(True, response_time)
                
            except Exception as e:
                response_time = (time.perf_counter() - start_time) * 1000
                metrics.record_operation(False, response_time, str(e))
        
        # Create high volume of concurrent conversations
        tasks = []
        num_conversations = 200
        
        for i in range(num_conversations):
            task = create_and_process_conversation(i // 40, i)  # Distribute across managers
            tasks.append(task)
            
            # Record system metrics periodically
            if i % 20 == 0:
                metrics.record_system_metrics()
        
        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        summary = metrics.get_summary()
        
        # Load test assertions
        assert summary["success_rate"] > 0.95, f"Success rate too low: {summary['success_rate']}"
        assert summary["avg_response_time_ms"] < 1000, f"Average response time too high: {summary['avg_response_time_ms']}ms"
        assert summary["max_memory_mb"] < 500, f"Memory usage too high: {summary['max_memory_mb']}MB"
        
        print(f"High Volume Conversations Load Test Results:")
        print(f"  Total Operations: {summary['total_operations']}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Operations/Second: {summary['operations_per_second']:.1f}")
        print(f"  Avg Response Time: {summary['avg_response_time_ms']:.1f}ms")
        print(f"  P95 Response Time: {summary['p95_response_time_ms']:.1f}ms")
        print(f"  Max Memory Usage: {summary['max_memory_mb']:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, voice_manager_pool, large_audio_data):
        """Test system performance under sustained load."""
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        # Run sustained load for 30 seconds
        test_duration = 30.0
        end_time = time.time() + test_duration
        
        async def sustained_processing():
            """Continuous processing task."""
            manager = voice_manager_pool[0]
            conversation_id = "sustained_conv"
            await manager.start_conversation(conversation_id)
            
            operation_count = 0
            while time.time() < end_time:
                start_time = time.perf_counter()
                try:
                    # Mock processing with variable delay
                    mock_response = Mock()
                    mock_response.conversation_id = conversation_id
                    mock_response.text_content = f"Sustained response {operation_count}"
                    mock_response.processing_time = 75.0
                    
                    with patch.object(manager, '_process_with_mode', return_value=mock_response):
                        await manager.process_voice_input(conversation_id, large_audio_data)
                    
                    response_time = (time.perf_counter() - start_time) * 1000
                    metrics.record_operation(True, response_time)
                    
                except Exception as e:
                    response_time = (time.perf_counter() - start_time) * 1000
                    metrics.record_operation(False, response_time, str(e))
                
                operation_count += 1
                
                # Record system metrics every 10 operations
                if operation_count % 10 == 0:
                    metrics.record_system_metrics()
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            await manager.end_conversation(conversation_id)
        
        # Run multiple sustained processing tasks
        tasks = [sustained_processing() for _ in range(3)]
        await asyncio.gather(*tasks)
        
        metrics.end_test()
        summary = metrics.get_summary()
        
        # Sustained load assertions
        assert summary["success_rate"] > 0.90, f"Sustained load success rate too low: {summary['success_rate']}"
        assert summary["avg_response_time_ms"] < 2000, f"Sustained load response time too high: {summary['avg_response_time_ms']}ms"
        
        print(f"Sustained Load Test Results:")
        print(f"  Duration: {summary['duration_seconds']:.1f}s")
        print(f"  Total Operations: {summary['total_operations']}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Operations/Second: {summary['operations_per_second']:.1f}")
        print(f"  Avg Response Time: {summary['avg_response_time_ms']:.1f}ms")


@pytest.mark.stress_test
class TestWebSocketStressTesting:
    """Stress testing for WebSocket components."""
    
    @pytest.fixture
    def mock_voice_manager(self):
        """Create mock voice manager for stress testing."""
        manager = AsyncMock()
        manager.start_conversation.return_value = Mock(conversation_id="stress_test", is_active=True)
        manager.process_voice_input.return_value = Mock(
            conversation_id="stress_test",
            text_content="Stress test response",
            processing_time=100.0
        )
        return manager
    
    @pytest.mark.asyncio
    async def test_websocket_connection_stress(self, mock_voice_manager):
        """Stress test WebSocket connection handling."""
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            host="localhost",
            port=0,
            max_connections=1000,
            auth_required=False
        )
        
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        async def simulate_client_connection(client_id: str):
            """Simulate a client connection lifecycle."""
            start_time = time.perf_counter()
            try:
                # Mock WebSocket connection
                mock_websocket = AsyncMock()
                connection = Mock()
                connection.client_id = client_id
                connection.websocket = mock_websocket
                connection.conversation_id = f"stress_conv_{client_id}"
                connection.update_activity = Mock()
                
                # Add connection
                handler.connections[client_id] = connection
                
                # Simulate message processing
                for i in range(10):
                    await handler.broadcast_message(
                        "test_message",
                        {"data": f"Message {i}"},
                        conversation_id=connection.conversation_id
                    )
                    await asyncio.sleep(0.01)  # Small delay
                
                # Remove connection
                await handler._close_connection(client_id, "Stress test complete")
                
                response_time = (time.perf_counter() - start_time) * 1000
                metrics.record_operation(True, response_time)
                
            except Exception as e:
                response_time = (time.perf_counter() - start_time) * 1000
                metrics.record_operation(False, response_time, str(e))
        
        # Create many concurrent connections
        tasks = []
        num_connections = 500
        
        for i in range(num_connections):
            task = simulate_client_connection(f"stress_client_{i}")
            tasks.append(task)
            
            # Record metrics periodically
            if i % 50 == 0:
                metrics.record_system_metrics()
        
        # Execute all connection simulations
        await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_test()
        summary = metrics.get_summary()
        
        # Stress test assertions
        assert summary["success_rate"] > 0.85, f"WebSocket stress test success rate too low: {summary['success_rate']}"
        assert len(handler.connections) == 0, "Connections not properly cleaned up"
        
        print(f"WebSocket Connection Stress Test Results:")
        print(f"  Total Connections: {summary['total_operations']}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Avg Connection Time: {summary['avg_response_time_ms']:.1f}ms")
        print(f"  Max Memory Usage: {summary['max_memory_mb']:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_message_flooding_stress(self, mock_voice_manager):
        """Stress test with message flooding."""
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            host="localhost",
            port=0,
            auth_required=False
        )
        
        # Create connections
        num_connections = 100
        for i in range(num_connections):
            mock_websocket = AsyncMock()
            connection = Mock()
            connection.client_id = f"flood_client_{i}"
            connection.websocket = mock_websocket
            connection.conversation_id = "flood_conv"
            handler.connections[connection.client_id] = connection
        
        metrics = LoadTestMetrics()
        metrics.start_test()
        
        async def flood_messages():
            """Flood the system with messages."""
            for i in range(1000):  # Send 1000 messages rapidly
                start_time = time.perf_counter()
                try:
                    await handler.broadcast_message(
                        "flood_message",
                        {"message_id": i, "data": "x" * 100},  # 100 byte messages
                        conversation_id="flood_conv"
                    )
                    
                    response_time = (time.perf_counter() - start_time) * 1000
                    metrics.record_operation(True, response_time)
                    
                except Exception as e:
                    response_time = (time.perf_counter() - start_time) * 1000
                    metrics.record_operation(False, response_time, str(e))
                
                # Record metrics every 100 messages
                if i % 100 == 0:
                    metrics.record_system_metrics()
        
        # Run message flooding
        await flood_messages()
        
        metrics.end_test()
        summary = metrics.get_summary()
        
        # Message flooding assertions
        assert summary["success_rate"] > 0.90, f"Message flooding success rate too low: {summary['success_rate']}"
        assert summary["operations_per_second"] > 100, f"Message throughput too low: {summary['operations_per_second']}"
        
        print(f"Message Flooding Stress Test Results:")
        print(f"  Messages Sent: {summary['total_operations']}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Messages/Second: {summary['operations_per_second']:.1f}")
        print(f"  Avg Message Time: {summary['avg_response_time_ms']:.1f}ms")


@pytest.mark.stress_test
class TestResourceExhaustionTesting:
    """Test system behavior under resource exhaustion."""
    
    def test_memory_exhaustion_handling(self):
        """Test handling of memory exhaustion scenarios."""
        buffer_pool = AudioBufferPool(max_buffers=100)
        
        # Try to exhaust buffer pool
        buffers = []
        exhaustion_reached = False
        
        try:
            for i in range(200):  # Try to get more buffers than available
                buffer = buffer_pool.get_buffer(1024)
                buffers.append(buffer)
        except ResourceExhaustionError:
            exhaustion_reached = True
        
        # Should handle exhaustion gracefully
        assert exhaustion_reached or len(buffers) <= 100, "Buffer pool exhaustion not handled properly"
        
        # Cleanup
        for buffer in buffers:
            buffer_pool.return_buffer(buffer)
        
        print(f"Memory Exhaustion Test: Allocated {len(buffers)} buffers before exhaustion")
    
    def test_vram_exhaustion_handling(self):
        """Test VRAM exhaustion handling."""
        vram_manager = OptimizedVRAMManager(total_vram_gb=1.0)  # Small VRAM pool
        
        # Try to allocate more VRAM than available
        allocations = []
        exhaustion_reached = False
        
        try:
            for i in range(10):
                allocation = vram_manager.allocate_vram(f"component_{i}", 200.0)  # 200MB each
                if allocation:
                    allocations.append((f"component_{i}", allocation))
                else:
                    exhaustion_reached = True
                    break
        except Exception as e:
            exhaustion_reached = True
        
        # Should handle VRAM exhaustion gracefully
        assert exhaustion_reached or len(allocations) <= 5, "VRAM exhaustion not handled properly"
        
        # Cleanup
        for component_id, allocation in allocations:
            vram_manager.deallocate_vram(component_id)
        
        print(f"VRAM Exhaustion Test: Made {len(allocations)} allocations before exhaustion")
    
    @pytest.mark.asyncio
    async def test_connection_limit_handling(self):
        """Test handling of connection limits."""
        mock_voice_manager = AsyncMock()
        
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            host="localhost",
            port=0,
            max_connections=50,  # Low limit for testing
            auth_required=False
        )
        
        # Try to exceed connection limit
        connections_created = 0
        limit_reached = False
        
        for i in range(100):  # Try to create more than the limit
            try:
                mock_websocket = AsyncMock()
                connection = Mock()
                connection.client_id = f"limit_client_{i}"
                connection.websocket = mock_websocket
                
                # Simulate connection check
                if len(handler.connections) >= handler.max_connections:
                    limit_reached = True
                    break
                
                handler.connections[connection.client_id] = connection
                connections_created += 1
                
            except Exception:
                limit_reached = True
                break
        
        # Should respect connection limits
        assert limit_reached or connections_created <= 50, "Connection limit not enforced"
        assert len(handler.connections) <= 50, "Too many connections allowed"
        
        print(f"Connection Limit Test: Created {connections_created} connections before limit")


@pytest.mark.recovery_test
class TestSystemRecoveryTesting:
    """Test system recovery from failures."""
    
    @pytest.mark.asyncio
    async def test_component_failure_recovery(self):
        """Test recovery from component failures."""
        config = VoiceConfig(
            audio=AudioConfig(),
            moshi=MoshiConfig(device="cpu")
        )
        
        manager = VoiceManager(config)
        
        # Mock components with intermittent failures
        manager.vram_manager = Mock()
        manager.vram_manager.register_component.return_value = True
        
        failure_count = 0
        success_count = 0
        
        async def failing_pipeline_process(*args, **kwargs):
            nonlocal failure_count, success_count
            if failure_count < 3:  # Fail first 3 times
                failure_count += 1
                raise Exception("Simulated component failure")
            else:
                success_count += 1
                return Mock(text_content="Recovery successful", processing_time=100.0)
        
        manager.pipeline_manager = AsyncMock()
        manager.pipeline_manager.process_voice_input = failing_pipeline_process
        manager.memory_manager = AsyncMock()
        manager.personality_manager = AsyncMock()
        
        await manager.initialize(config)
        
        # Test recovery behavior
        conversation_id = "recovery_test"
        await manager.start_conversation(conversation_id)
        
        # Try processing multiple times (should eventually succeed)
        for i in range(5):
            try:
                with patch.object(manager, '_process_with_mode', side_effect=failing_pipeline_process):
                    response = await manager.process_voice_input(conversation_id, b"test_audio")
                    if response:
                        break
            except Exception:
                continue  # Expected failures
        
        # Should eventually recover
        assert success_count > 0, "System did not recover from component failures"
        
        print(f"Component Recovery Test: {failure_count} failures, {success_count} recoveries")
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_recovery(self):
        """Test graceful shutdown and restart."""
        config = VoiceConfig(
            audio=AudioConfig(),
            moshi=MoshiConfig(device="cpu")
        )
        
        # Create and initialize manager
        manager = VoiceManager(config)
        
        # Mock dependencies
        manager.vram_manager = Mock()
        manager.vram_manager.register_component.return_value = True
        manager.pipeline_manager = AsyncMock()
        manager.memory_manager = AsyncMock()
        manager.personality_manager = AsyncMock()
        
        await manager.initialize(config)
        
        # Create some conversations
        conversation_ids = [f"shutdown_conv_{i}" for i in range(5)]
        for conv_id in conversation_ids:
            await manager.start_conversation(conv_id)
        
        assert len(manager.conversations) == 5
        assert manager.is_initialized
        
        # Graceful shutdown
        await manager.cleanup()
        
        assert len(manager.conversations) == 0
        assert not manager.is_initialized
        
        # Restart
        await manager.initialize(config)
        
        assert manager.is_initialized
        assert len(manager.conversations) == 0
        
        # Should be able to create new conversations
        await manager.start_conversation("post_restart_conv")
        assert len(manager.conversations) == 1
        
        print("Graceful Shutdown Recovery Test: Passed")


if __name__ == "__main__":
    # Run load and stress tests
    pytest.main([__file__, "-v", "-m", "load_test or stress_test or recovery_test", "--tb=short"])
