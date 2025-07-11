"""
WebSocket Load Testing.
Tests concurrent connections, message throughput, and streaming performance under load.
"""

import pytest
import asyncio
import time
import json
import uuid
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Optional

from src.coda.components.voice.websocket_handler import VoiceWebSocketHandler
from src.coda.interfaces.websocket.server import CodaWebSocketServer
from src.coda.components.voice.models import VoiceStreamChunk


class WebSocketLoadMetrics:
    """WebSocket load testing metrics collector."""
    
    def __init__(self):
        self.connection_times = []
        self.message_latencies = []
        self.throughput_data = []
        self.error_counts = {"connection": 0, "message": 0, "timeout": 0}
    
    def record_connection_time(self, time_ms: float):
        """Record connection establishment time."""
        self.connection_times.append(time_ms)
    
    def record_message_latency(self, latency_ms: float):
        """Record message round-trip latency."""
        self.message_latencies.append(latency_ms)
    
    def record_throughput(self, messages_per_second: float):
        """Record message throughput."""
        self.throughput_data.append(messages_per_second)
    
    def record_error(self, error_type: str):
        """Record error occurrence."""
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load testing statistics."""
        import numpy as np
        return {
            "avg_connection_time": np.mean(self.connection_times) if self.connection_times else 0,
            "max_connection_time": np.max(self.connection_times) if self.connection_times else 0,
            "avg_message_latency": np.mean(self.message_latencies) if self.message_latencies else 0,
            "max_message_latency": np.max(self.message_latencies) if self.message_latencies else 0,
            "avg_throughput": np.mean(self.throughput_data) if self.throughput_data else 0,
            "max_throughput": np.max(self.throughput_data) if self.throughput_data else 0,
            "total_connections": len(self.connection_times),
            "total_messages": len(self.message_latencies),
            "error_counts": self.error_counts.copy()
        }


class MockWebSocket:
    """Mock WebSocket connection for load testing."""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.closed = False
        self.messages_sent = []
        self.messages_received = []
        self.remote_address = ("127.0.0.1", 12345 + hash(connection_id) % 1000)
    
    async def send(self, message: str):
        """Mock send message."""
        await asyncio.sleep(0.001)  # Simulate network delay
        self.messages_sent.append(message)
    
    async def recv(self):
        """Mock receive message."""
        await asyncio.sleep(0.001)  # Simulate network delay
        if self.messages_received:
            return self.messages_received.pop(0)
        raise asyncio.TimeoutError("No message available")
    
    async def close(self):
        """Mock close connection."""
        self.closed = True
    
    async def ping(self):
        """Mock ping."""
        await asyncio.sleep(0.001)
        return True


class TestWebSocketLoad:
    """WebSocket load testing."""

    @pytest.fixture
    def load_metrics(self):
        """Create load testing metrics collector."""
        return WebSocketLoadMetrics()

    @pytest.fixture
    def mock_voice_manager(self):
        """Create mock voice manager for load testing."""
        manager = Mock()
        
        # Mock voice processing with variable latency
        async def mock_process(message):
            # Simulate variable processing time
            processing_time = 0.02 + (hash(message.message_id) % 100) / 10000  # 20-30ms
            await asyncio.sleep(processing_time)
            
            return {
                "response_id": str(uuid.uuid4()),
                "text_content": f"Processed: {message.text_content[:50]}...",
                "processing_time_ms": processing_time * 1000
            }
        
        manager.process_voice_input = AsyncMock(side_effect=mock_process)
        manager.is_initialized = True
        
        return manager

    @pytest.fixture
    def websocket_handler(self, mock_voice_manager):
        """Create WebSocket handler for load testing."""
        return VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            max_connections=1000,  # High limit for load testing
            ping_interval=30.0
        )

    @pytest.mark.asyncio
    async def test_connection_scaling(self, websocket_handler, load_metrics):
        """Test WebSocket connection scaling."""
        
        connection_counts = [10, 50, 100, 200, 500]
        
        for count in connection_counts:
            connections = []
            
            # Create connections
            start_time = time.time()
            
            for i in range(count):
                mock_ws = MockWebSocket(f"conn-{i}")
                
                connection_start = time.time()
                await websocket_handler.handle_connection(mock_ws)
                connection_end = time.time()
                
                connection_time = (connection_end - connection_start) * 1000
                load_metrics.record_connection_time(connection_time)
                
                connections.append(mock_ws)
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            print(f"Connected {count} clients in {total_time:.2f}ms")
            print(f"Active connections: {len(websocket_handler.active_connections)}")
            
            # Verify connections
            assert len(websocket_handler.active_connections) == count
            
            # Performance assertions
            avg_connection_time = total_time / count
            assert avg_connection_time < 50  # Under 50ms per connection
            
            # Cleanup connections
            for conn in connections:
                await websocket_handler._cleanup_connection(conn)
            
            assert len(websocket_handler.active_connections) == 0

    @pytest.mark.asyncio
    async def test_message_throughput(self, websocket_handler, load_metrics):
        """Test message throughput under load."""
        
        # Setup connections
        connection_count = 50
        connections = []
        
        for i in range(connection_count):
            mock_ws = MockWebSocket(f"throughput-{i}")
            await websocket_handler.handle_connection(mock_ws)
            connections.append(mock_ws)
        
        # Test different message rates
        message_rates = [10, 50, 100, 200]  # messages per second
        
        for rate in message_rates:
            messages_sent = 0
            test_duration = 2.0  # 2 seconds
            message_interval = 1.0 / rate
            
            start_time = time.time()
            
            # Send messages at specified rate
            while time.time() - start_time < test_duration:
                for conn in connections:
                    message = {
                        "type": "voice_input",
                        "conversation_id": f"throughput-test-{messages_sent}",
                        "message_id": str(uuid.uuid4()),
                        "text_content": f"Throughput test message {messages_sent}"
                    }
                    
                    message_start = time.time()
                    await websocket_handler._handle_message(conn, json.dumps(message))
                    message_end = time.time()
                    
                    latency = (message_end - message_start) * 1000
                    load_metrics.record_message_latency(latency)
                    
                    messages_sent += 1
                    
                    # Control message rate
                    await asyncio.sleep(message_interval / connection_count)
            
            end_time = time.time()
            actual_duration = end_time - start_time
            actual_rate = messages_sent / actual_duration
            
            load_metrics.record_throughput(actual_rate)
            
            print(f"Target rate: {rate} msg/s, Actual rate: {actual_rate:.1f} msg/s")
            print(f"Messages sent: {messages_sent} in {actual_duration:.2f}s")
            
            # Performance assertions
            assert actual_rate > rate * 0.8  # At least 80% of target rate
        
        # Cleanup
        for conn in connections:
            await websocket_handler._cleanup_connection(conn)

    @pytest.mark.asyncio
    async def test_concurrent_streaming(self, websocket_handler, load_metrics):
        """Test concurrent streaming performance."""
        
        # Setup connections for streaming
        stream_count = 20
        connections = []
        
        for i in range(stream_count):
            mock_ws = MockWebSocket(f"stream-{i}")
            await websocket_handler.handle_connection(mock_ws)
            connections.append(mock_ws)
        
        # Create streaming chunks
        async def create_stream_chunks(stream_id: str):
            chunks = []
            for i in range(10):  # 10 chunks per stream
                chunk = VoiceStreamChunk(
                    conversation_id=f"stream-{stream_id}",
                    chunk_index=i,
                    text_content=f"Stream {stream_id} chunk {i}",
                    audio_data=b"fake_audio_data",
                    timestamp=time.time(),
                    is_complete=(i == 9),
                    chunk_type="audio"
                )
                chunks.append(chunk)
            return chunks
        
        # Start concurrent streaming
        start_time = time.time()
        
        streaming_tasks = []
        for i, conn in enumerate(connections):
            chunks = await create_stream_chunks(str(i))
            
            async def stream_to_connection(connection, chunk_list):
                for chunk in chunk_list:
                    await websocket_handler.broadcast_stream_chunk(chunk)
                    await asyncio.sleep(0.01)  # 10ms between chunks
            
            task = asyncio.create_task(stream_to_connection(conn, chunks))
            streaming_tasks.append(task)
        
        # Wait for all streams to complete
        await asyncio.gather(*streaming_tasks)
        
        end_time = time.time()
        total_streaming_time = (end_time - start_time) * 1000
        
        print(f"Concurrent streaming: {stream_count} streams in {total_streaming_time:.2f}ms")
        
        # Performance assertions
        assert total_streaming_time < 5000  # Under 5 seconds
        
        # Cleanup
        for conn in connections:
            await websocket_handler._cleanup_connection(conn)

    @pytest.mark.asyncio
    async def test_connection_stability_under_load(self, websocket_handler, load_metrics):
        """Test connection stability under sustained load."""
        
        # Create long-lived connections
        connection_count = 100
        connections = []
        
        for i in range(connection_count):
            mock_ws = MockWebSocket(f"stable-{i}")
            await websocket_handler.handle_connection(mock_ws)
            connections.append(mock_ws)
        
        # Send periodic messages over time
        test_duration = 5.0  # 5 seconds
        message_interval = 0.1  # Every 100ms
        
        start_time = time.time()
        messages_sent = 0
        connection_errors = 0
        
        while time.time() - start_time < test_duration:
            # Send message to random connection
            import random
            conn = random.choice(connections)
            
            try:
                message = {
                    "type": "voice_input",
                    "conversation_id": f"stability-{messages_sent}",
                    "message_id": str(uuid.uuid4()),
                    "text_content": f"Stability test message {messages_sent}"
                }
                
                await websocket_handler._handle_message(conn, json.dumps(message))
                messages_sent += 1
                
            except Exception as e:
                connection_errors += 1
                load_metrics.record_error("message")
            
            await asyncio.sleep(message_interval)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        print(f"Stability test: {messages_sent} messages, {connection_errors} errors")
        print(f"Error rate: {(connection_errors / messages_sent * 100):.2f}%")
        print(f"Active connections: {len(websocket_handler.active_connections)}")
        
        # Stability assertions
        error_rate = connection_errors / messages_sent if messages_sent > 0 else 0
        assert error_rate < 0.05  # Less than 5% error rate
        assert len(websocket_handler.active_connections) == connection_count  # No dropped connections
        
        # Cleanup
        for conn in connections:
            await websocket_handler._cleanup_connection(conn)

    @pytest.mark.benchmark(group="websocket_performance")
    def test_websocket_handler_performance(self, benchmark, websocket_handler):
        """Benchmark WebSocket handler performance."""
        
        def handle_message():
            mock_ws = MockWebSocket("benchmark")
            message = {
                "type": "voice_input",
                "conversation_id": "benchmark-test",
                "message_id": str(uuid.uuid4()),
                "text_content": "Benchmark test message"
            }
            
            return asyncio.run(websocket_handler._handle_message(mock_ws, json.dumps(message)))
        
        result = benchmark(handle_message)
        # Message handling should complete successfully

    @pytest.mark.asyncio
    async def test_websocket_memory_usage(self, websocket_handler):
        """Test WebSocket memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many connections
        connections = []
        for i in range(500):
            mock_ws = MockWebSocket(f"memory-{i}")
            await websocket_handler.handle_connection(mock_ws)
            connections.append(mock_ws)
        
        # Send messages through all connections
        for i, conn in enumerate(connections):
            message = {
                "type": "voice_input",
                "conversation_id": f"memory-test-{i}",
                "message_id": str(uuid.uuid4()),
                "text_content": f"Memory test message {i}"
            }
            await websocket_handler._handle_message(conn, json.dumps(message))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        print(f"Memory per connection: {memory_increase / len(connections):.3f}MB")
        
        # Memory usage should be reasonable
        assert memory_increase < 1000  # Under 1GB increase
        assert memory_increase / len(connections) < 2  # Under 2MB per connection
        
        # Cleanup
        for conn in connections:
            await websocket_handler._cleanup_connection(conn)

    @pytest.mark.asyncio
    async def test_websocket_error_recovery(self, websocket_handler, load_metrics):
        """Test WebSocket error recovery under load."""
        
        # Create connections
        connections = []
        for i in range(50):
            mock_ws = MockWebSocket(f"error-{i}")
            await websocket_handler.handle_connection(mock_ws)
            connections.append(mock_ws)
        
        # Simulate various error conditions
        error_scenarios = [
            ("invalid_json", "{ invalid json }"),
            ("missing_type", '{"conversation_id": "test"}'),
            ("invalid_type", '{"type": "invalid_type"}'),
            ("malformed_message", '{"type": "voice_input"}'),  # Missing required fields
        ]
        
        total_errors = 0
        
        for scenario_name, invalid_message in error_scenarios:
            scenario_errors = 0
            
            # Send invalid messages to all connections
            for conn in connections:
                try:
                    await websocket_handler._handle_message(conn, invalid_message)
                except Exception:
                    scenario_errors += 1
                    load_metrics.record_error("message")
            
            total_errors += scenario_errors
            print(f"Scenario '{scenario_name}': {scenario_errors} errors")
        
        # Verify system is still operational after errors
        assert len(websocket_handler.active_connections) == len(connections)
        
        # Send valid message to verify recovery
        valid_message = {
            "type": "voice_input",
            "conversation_id": "recovery-test",
            "message_id": str(uuid.uuid4()),
            "text_content": "Recovery test message"
        }
        
        recovery_success = 0
        for conn in connections:
            try:
                await websocket_handler._handle_message(conn, json.dumps(valid_message))
                recovery_success += 1
            except Exception:
                pass
        
        print(f"Recovery: {recovery_success}/{len(connections)} connections operational")
        
        # Most connections should recover
        recovery_rate = recovery_success / len(connections)
        assert recovery_rate > 0.9  # At least 90% recovery rate
        
        # Cleanup
        for conn in connections:
            await websocket_handler._cleanup_connection(conn)

    def test_websocket_performance_metrics(self, websocket_handler, load_metrics):
        """Test WebSocket performance metrics collection."""
        
        async def metrics_test():
            # Simulate load testing scenario
            connections = []
            
            # Connection phase
            for i in range(20):
                start_time = time.time()
                mock_ws = MockWebSocket(f"metrics-{i}")
                await websocket_handler.handle_connection(mock_ws)
                end_time = time.time()
                
                load_metrics.record_connection_time((end_time - start_time) * 1000)
                connections.append(mock_ws)
            
            # Message phase
            for i, conn in enumerate(connections):
                message = {
                    "type": "voice_input",
                    "conversation_id": f"metrics-{i}",
                    "message_id": str(uuid.uuid4()),
                    "text_content": f"Metrics test message {i}"
                }
                
                start_time = time.time()
                await websocket_handler._handle_message(conn, json.dumps(message))
                end_time = time.time()
                
                load_metrics.record_message_latency((end_time - start_time) * 1000)
            
            # Throughput calculation
            throughput = len(connections) / 2.0  # Assume 2 second test
            load_metrics.record_throughput(throughput)
            
            # Cleanup
            for conn in connections:
                await websocket_handler._cleanup_connection(conn)
        
        asyncio.run(metrics_test())
        
        # Get and verify metrics
        stats = load_metrics.get_stats()
        
        print("WebSocket Load Test Metrics:")
        print(f"  Average connection time: {stats['avg_connection_time']:.2f}ms")
        print(f"  Average message latency: {stats['avg_message_latency']:.2f}ms")
        print(f"  Average throughput: {stats['avg_throughput']:.2f} msg/s")
        print(f"  Total connections: {stats['total_connections']}")
        print(f"  Total messages: {stats['total_messages']}")
        
        # Verify metrics were collected
        assert stats['total_connections'] > 0
        assert stats['total_messages'] > 0
        assert stats['avg_connection_time'] > 0
        assert stats['avg_message_latency'] > 0
