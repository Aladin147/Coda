"""
Comprehensive tests for WebSocket voice processing components.

This module provides thorough testing of all WebSocket-related functionality
including connection handling, audio streaming, event broadcasting, and monitoring.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import websockets
from websockets.exceptions import ConnectionClosed

from src.coda.components.voice.websocket_handler import (
    VoiceWebSocketHandler, MessageType, WebSocketMessage, ClientConnection
)
from src.coda.components.voice.websocket_events import (
    VoiceEventBroadcaster, EventType, VoiceEvent
)
from src.coda.components.voice.websocket_audio_streaming import (
    AudioStreamProcessor, AudioStreamConfig, AudioFormat, StreamingMode
)
from src.coda.components.voice.websocket_monitoring import (
    WebSocketMonitor, ConnectionMetrics, SystemMetrics
)
from src.coda.components.voice.models import VoiceMessage, VoiceResponse


class TestWebSocketMessageHandling:
    """Test WebSocket message creation and handling."""
    
    def test_websocket_message_creation(self):
        """Test WebSocket message creation and serialization."""
        message = WebSocketMessage(
            type=MessageType.VOICE_START,
            data={"mode": "adaptive", "config": {"sample_rate": 16000}},
            conversation_id="test-conv"
        )
        
        assert message.type == MessageType.VOICE_START
        assert message.data["mode"] == "adaptive"
        assert message.conversation_id == "test-conv"
        assert message.message_id is not None
        assert message.timestamp is not None
    
    def test_websocket_message_serialization(self):
        """Test message JSON serialization and deserialization."""
        original_message = WebSocketMessage(
            type=MessageType.CONVERSATION_START,
            data={"conversation_id": "test-123"},
            conversation_id="test-123"
        )
        
        # Serialize to JSON
        json_str = original_message.to_json()
        assert isinstance(json_str, str)
        
        # Deserialize from JSON
        parsed_message = WebSocketMessage.from_json(json_str)
        
        assert parsed_message.type == original_message.type
        assert parsed_message.data == original_message.data
        assert parsed_message.conversation_id == original_message.conversation_id
        assert parsed_message.message_id == original_message.message_id
    
    def test_websocket_message_invalid_json(self):
        """Test handling of invalid JSON in message parsing."""
        invalid_json = '{"type": "invalid", "data": {'  # Incomplete JSON
        
        with pytest.raises(json.JSONDecodeError):
            WebSocketMessage.from_json(invalid_json)
    
    def test_client_connection_tracking(self):
        """Test client connection information tracking."""
        mock_websocket = AsyncMock()
        
        connection = ClientConnection(
            websocket=mock_websocket,
            client_id="test-client-123",
            user_id="user-456"
        )
        
        assert connection.client_id == "test-client-123"
        assert connection.user_id == "user-456"
        assert not connection.is_authenticated
        assert connection.connection_duration > 0
        
        # Test activity update
        old_activity = connection.last_activity
        time.sleep(0.01)
        connection.update_activity()
        assert connection.last_activity > old_activity


class TestWebSocketHandler:
    """Test WebSocket handler functionality."""
    
    @pytest.fixture
    def mock_voice_manager(self):
        """Create mock voice manager for testing."""
        manager = AsyncMock()
        manager.start_conversation.return_value = Mock(
            conversation_id="test-conv",
            is_active=True
        )
        manager.process_voice_input.return_value = VoiceResponse(
            conversation_id="test-conv",
            text_content="Test response",
            processing_time=100.0
        )
        return manager
    
    @pytest.fixture
    def websocket_handler(self, mock_voice_manager):
        """Create WebSocket handler for testing."""
        return VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            host="localhost",
            port=0,  # Random port for testing
            auth_required=False
        )
    
    @pytest.mark.asyncio
    async def test_websocket_handler_initialization(self, websocket_handler):
        """Test WebSocket handler initialization."""
        assert websocket_handler.host == "localhost"
        assert websocket_handler.port == 0
        assert not websocket_handler.auth_required
        assert len(websocket_handler.connections) == 0
        assert not websocket_handler.is_running
    
    @pytest.mark.asyncio
    async def test_message_broadcasting(self, websocket_handler):
        """Test message broadcasting to multiple clients."""
        # Add mock connections
        mock_connections = {}
        for i in range(3):
            mock_websocket = AsyncMock()
            client_id = f"client_{i}"
            connection = ClientConnection(
                websocket=mock_websocket,
                client_id=client_id,
                conversation_id="test-conv"
            )
            mock_connections[client_id] = connection
        
        websocket_handler.connections = mock_connections
        
        # Broadcast message
        sent_count = await websocket_handler.broadcast_message(
            MessageType.STATUS,
            {"message": "Test broadcast", "timestamp": time.time()},
            conversation_id="test-conv"
        )
        
        assert sent_count == 3
        
        # Verify all connections received the message
        for connection in mock_connections.values():
            connection.websocket.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connection_cleanup(self, websocket_handler):
        """Test proper connection cleanup."""
        # Add mock connection
        mock_websocket = AsyncMock()
        client_id = "test-client"
        connection = ClientConnection(
            websocket=mock_websocket,
            client_id=client_id,
            conversation_id="test-conv"
        )
        
        websocket_handler.connections[client_id] = connection
        
        # Close connection
        await websocket_handler._close_connection(client_id, "Test cleanup")
        
        # Verify cleanup
        assert client_id not in websocket_handler.connections
        mock_websocket.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authentication_handling(self, websocket_handler):
        """Test authentication message handling."""
        mock_websocket = AsyncMock()
        connection = ClientConnection(
            websocket=mock_websocket,
            client_id="test-client"
        )
        
        # Test successful authentication
        auth_message = WebSocketMessage(
            type=MessageType.AUTH,
            data={"user_id": "test-user", "token": "valid-token"}
        )
        
        with patch.object(websocket_handler, '_validate_auth', return_value=True):
            await websocket_handler._handle_auth(connection, auth_message)
            
            assert connection.is_authenticated
            assert connection.user_id == "test-user"
    
    @pytest.mark.asyncio
    async def test_voice_processing_integration(self, websocket_handler):
        """Test voice processing through WebSocket."""
        mock_websocket = AsyncMock()
        connection = ClientConnection(
            websocket=mock_websocket,
            client_id="test-client",
            conversation_id="test-conv"
        )
        
        # Test voice chunk processing
        voice_chunk_message = WebSocketMessage(
            type=MessageType.VOICE_CHUNK,
            data={
                "audio_data": "deadbeef",  # Hex audio data
                "text_content": "Hello world",
                "is_final": True
            }
        )
        
        await websocket_handler._handle_voice_chunk(connection, voice_chunk_message)
        
        # Verify voice manager was called
        websocket_handler.voice_manager.process_voice_input.assert_called_once()


class TestEventBroadcasting:
    """Test event broadcasting functionality."""
    
    @pytest.fixture
    def mock_websocket_handler(self):
        """Create mock WebSocket handler."""
        handler = Mock()
        handler.connections = {
            "client_1": Mock(conversation_id="conv_1", user_id="user_1"),
            "client_2": Mock(conversation_id="conv_1", user_id="user_2"),
            "client_3": Mock(conversation_id="conv_2", user_id="user_1")
        }
        handler._send_message = AsyncMock()
        return handler
    
    @pytest.fixture
    def event_broadcaster(self, mock_websocket_handler):
        """Create event broadcaster for testing."""
        return VoiceEventBroadcaster(mock_websocket_handler)
    
    def test_voice_event_creation(self):
        """Test voice event creation and validation."""
        event = VoiceEvent(
            event_type=EventType.PROCESSING_STARTED,
            data={"mode": "hybrid", "estimated_time": 2.5},
            conversation_id="test-conv",
            user_id="test-user"
        )
        
        assert event.event_type == EventType.PROCESSING_STARTED
        assert event.data["mode"] == "hybrid"
        assert event.conversation_id == "test-conv"
        assert event.user_id == "test-user"
        assert event.timestamp > 0
        assert event.event_id is not None
    
    @pytest.mark.asyncio
    async def test_event_broadcasting_by_conversation(self, event_broadcaster):
        """Test event broadcasting filtered by conversation."""
        # Mock successful message sending
        event_broadcaster._send_event_to_client = AsyncMock(return_value=True)
        
        sent_count = await event_broadcaster.broadcast_event(
            EventType.PROCESSING_STARTED,
            {"mode": "hybrid"},
            conversation_id="conv_1"
        )
        
        # Should send to clients in conv_1
        assert sent_count >= 0
        assert event_broadcaster._send_event_to_client.called
    
    @pytest.mark.asyncio
    async def test_event_broadcasting_by_user(self, event_broadcaster):
        """Test event broadcasting filtered by user."""
        event_broadcaster._send_event_to_client = AsyncMock(return_value=True)
        
        sent_count = await event_broadcaster.broadcast_event(
            EventType.SYSTEM_STATUS,
            {"message": "System update"},
            user_id="user_1"
        )
        
        assert sent_count >= 0
        assert event_broadcaster._send_event_to_client.called
    
    @pytest.mark.asyncio
    async def test_client_subscription_management(self, event_broadcaster):
        """Test client event subscription and unsubscription."""
        client_id = "test_client"
        event_types = [EventType.PROCESSING_STARTED, EventType.PROCESSING_COMPLETED]
        
        # Subscribe client
        await event_broadcaster.subscribe_client(client_id, event_types)
        
        # Verify subscription
        for event_type in event_types:
            assert client_id in event_broadcaster.subscribers[event_type]
        
        # Unsubscribe from specific event
        await event_broadcaster.unsubscribe_client(client_id, [EventType.PROCESSING_STARTED])
        assert client_id not in event_broadcaster.subscribers[EventType.PROCESSING_STARTED]
        assert client_id in event_broadcaster.subscribers[EventType.PROCESSING_COMPLETED]
        
        # Unsubscribe from all events
        await event_broadcaster.unsubscribe_client(client_id)
        assert client_id not in event_broadcaster.subscribers[EventType.PROCESSING_COMPLETED]
    
    @pytest.mark.asyncio
    async def test_event_queuing_for_offline_clients(self, event_broadcaster):
        """Test event queuing for offline clients."""
        client_id = "offline_client"
        
        # Subscribe client but don't add to connections (offline)
        await event_broadcaster.subscribe_client(client_id, [EventType.PROCESSING_STARTED])
        
        # Broadcast event
        await event_broadcaster.broadcast_event(
            EventType.PROCESSING_STARTED,
            {"mode": "adaptive"}
        )
        
        # Verify event was queued
        assert client_id in event_broadcaster.client_queues
        assert len(event_broadcaster.client_queues[client_id]) > 0
    
    def test_event_history_management(self, event_broadcaster):
        """Test event history storage and retrieval."""
        # Add events to history
        for i in range(10):
            event = VoiceEvent(
                event_type=EventType.PROCESSING_STARTED,
                data={"iteration": i},
                conversation_id="test-conv"
            )
            event_broadcaster._add_to_history(event)
        
        # Get full history
        history = event_broadcaster.get_event_history()
        assert len(history) == 10
        
        # Get filtered history
        filtered_history = event_broadcaster.get_event_history(
            conversation_id="test-conv",
            event_types=[EventType.PROCESSING_STARTED],
            limit=5
        )
        assert len(filtered_history) == 5
        assert all(event.conversation_id == "test-conv" for event in filtered_history)


class TestAudioStreaming:
    """Test audio streaming functionality."""
    
    @pytest.fixture
    def mock_websocket_handler(self):
        """Create mock WebSocket handler."""
        handler = Mock()
        handler.connections = {
            "client_1": Mock(conversation_id="conv_1")
        }
        handler._send_message = AsyncMock()
        return handler
    
    @pytest.fixture
    def audio_processor(self, mock_websocket_handler):
        """Create audio stream processor."""
        return AudioStreamProcessor(mock_websocket_handler)
    
    def test_audio_stream_config_validation(self):
        """Test audio stream configuration validation."""
        config = AudioStreamConfig(
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            chunk_size_ms=100,
            streaming_mode=StreamingMode.VOICE_ACTIVITY,
            enable_vad=True,
            vad_threshold=0.5
        )
        
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.format == AudioFormat.WAV
        assert config.chunk_size_samples == 1600  # 100ms at 16kHz
        assert config.streaming_mode == StreamingMode.VOICE_ACTIVITY
        assert config.enable_vad
        assert config.vad_threshold == 0.5
    
    @pytest.mark.asyncio
    async def test_audio_stream_lifecycle(self, audio_processor):
        """Test complete audio stream lifecycle."""
        client_id = "test_client"
        config = AudioStreamConfig(
            sample_rate=16000,
            chunk_size_ms=100
        )
        
        # Start stream
        success = await audio_processor.start_audio_stream(client_id, config)
        assert success
        assert client_id in audio_processor.active_streams
        
        # Verify stream state
        stream_state = audio_processor.active_streams[client_id]
        assert stream_state["config"] == config
        assert stream_state["chunk_counter"] == 0
        assert len(stream_state["audio_buffer"]) == 0
        
        # Stop stream
        success = await audio_processor.stop_audio_stream(client_id)
        assert success
        assert client_id not in audio_processor.active_streams
    
    @pytest.mark.asyncio
    async def test_audio_chunk_processing(self, audio_processor):
        """Test audio chunk processing and buffering."""
        client_id = "test_client"
        config = AudioStreamConfig()
        
        # Start stream
        await audio_processor.start_audio_stream(client_id, config)
        
        # Process audio chunks
        audio_data = b'\x00\x01' * 1000  # Mock audio data
        
        for i in range(3):
            success = await audio_processor.process_audio_chunk(
                client_id,
                audio_data,
                {"chunk_id": f"chunk_{i}", "is_final": (i == 2)}
            )
            assert success
        
        # Verify stream state was updated
        stream_state = audio_processor.active_streams[client_id]
        assert stream_state["chunk_counter"] == 3
        assert stream_state["last_activity"] > 0
    
    def test_audio_format_conversion(self, audio_processor):
        """Test audio format conversion utilities."""
        import numpy as np
        
        # Test PCM conversion
        config = AudioStreamConfig(format=AudioFormat.PCM)
        audio_data = b'\x00\x01\x02\x03' * 100
        
        # Convert to numpy array
        audio_array = audio_processor._convert_audio_data(audio_data, config)
        assert isinstance(audio_array, np.ndarray)
        assert audio_array.dtype == np.float32
        
        # Convert back to bytes
        audio_bytes = audio_processor._convert_to_bytes(audio_array, config)
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
    
    def test_voice_activity_detection(self, audio_processor):
        """Test voice activity detection algorithm."""
        import numpy as np
        
        # Test with silence (should return False)
        silence = np.zeros(1000, dtype=np.float32)
        vad_result = audio_processor._detect_voice_activity(silence, 0.01)
        assert not vad_result
        
        # Test with voice signal (should return True)
        voice_signal = np.random.random(1000).astype(np.float32) * 0.5
        vad_result = audio_processor._detect_voice_activity(voice_signal, 0.01)
        assert vad_result
        
        # Test with empty array
        empty_array = np.array([], dtype=np.float32)
        vad_result = audio_processor._detect_voice_activity(empty_array, 0.01)
        assert not vad_result
    
    def test_audio_enhancement_pipeline(self, audio_processor):
        """Test audio enhancement processing."""
        import numpy as np
        
        config = AudioStreamConfig(
            enable_noise_reduction=True,
            enable_echo_cancellation=True
        )
        
        # Create test audio data
        audio_data = np.random.random(1000).astype(np.float32) * 0.1
        
        # Apply enhancements
        enhanced_audio = asyncio.run(
            audio_processor._enhance_audio(audio_data, config)
        )
        
        assert isinstance(enhanced_audio, np.ndarray)
        assert enhanced_audio.shape == audio_data.shape
        assert enhanced_audio.dtype == np.float32


class TestWebSocketMonitoring:
    """Test WebSocket monitoring and metrics."""
    
    @pytest.fixture
    def mock_websocket_handler(self):
        """Create mock WebSocket handler."""
        handler = Mock()
        handler.connections = {}
        return handler
    
    @pytest.fixture
    def mock_event_broadcaster(self):
        """Create mock event broadcaster."""
        broadcaster = Mock()
        broadcaster.broadcast_event = AsyncMock()
        return broadcaster
    
    @pytest.fixture
    def monitor(self, mock_websocket_handler, mock_event_broadcaster):
        """Create WebSocket monitor."""
        return WebSocketMonitor(mock_websocket_handler, mock_event_broadcaster)
    
    def test_connection_metrics_tracking(self, monitor):
        """Test connection metrics tracking and calculation."""
        client_id = "test_client"
        
        # Track new connection
        monitor.track_connection(client_id)
        assert client_id in monitor.connection_metrics
        
        metrics = monitor.connection_metrics[client_id]
        assert metrics.client_id == client_id
        assert metrics.messages_sent == 0
        assert metrics.messages_received == 0
        assert metrics.connection_duration > 0
        
        # Record activities
        monitor.record_message_sent(client_id, 150)
        monitor.record_message_received(client_id, 200)
        monitor.record_latency(client_id, 75.5)
        monitor.record_error(client_id)
        
        # Verify metrics
        assert metrics.messages_sent == 1
        assert metrics.messages_received == 1
        assert metrics.bytes_sent == 150
        assert metrics.bytes_received == 200
        assert 75.5 in metrics.latency_samples
        assert metrics.errors == 1
        
        # Test average latency calculation
        monitor.record_latency(client_id, 25.5)
        assert metrics.average_latency == 50.5  # (75.5 + 25.5) / 2
    
    def test_system_metrics_collection(self, monitor):
        """Test system-wide metrics collection."""
        # Add some test connections
        for i in range(3):
            client_id = f"client_{i}"
            monitor.track_connection(client_id)
            monitor.record_message_sent(client_id, 100)
            monitor.record_latency(client_id, 50.0 + i * 10)
        
        # Collect system metrics
        system_metrics = asyncio.run(monitor._collect_system_metrics())
        
        assert isinstance(system_metrics, SystemMetrics)
        assert system_metrics.active_connections >= 0
        assert system_metrics.total_connections == 3
        assert system_metrics.average_latency > 0
        assert system_metrics.timestamp > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitor):
        """Test monitoring start and stop."""
        # Start monitoring
        await monitor.start_monitoring()
        assert monitor.is_monitoring
        assert monitor.monitoring_task is not None
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert not monitor.is_monitoring
    
    def test_alert_threshold_management(self, monitor):
        """Test alert threshold configuration."""
        # Test setting valid threshold
        success = monitor.set_alert_threshold("max_latency_ms", 500.0)
        assert success
        assert monitor.alert_thresholds["max_latency_ms"] == 500.0
        
        # Test setting invalid threshold
        success = monitor.set_alert_threshold("invalid_metric", 100.0)
        assert not success
    
    def test_performance_summary_generation(self, monitor):
        """Test performance summary calculation."""
        # Add test data
        for i in range(5):
            client_id = f"client_{i}"
            monitor.track_connection(client_id)
            monitor.record_latency(client_id, 100.0 + i * 20)
        
        # Add some metrics to history
        for i in range(10):
            metrics = SystemMetrics(
                timestamp=time.time() - (10 - i) * 60,  # 10 minutes ago to now
                active_connections=3,
                total_connections=5,
                messages_per_second=10.0,
                bytes_per_second=1000.0,
                average_latency=100.0 + i * 5,
                error_rate=0.01,
                cpu_usage=50.0 + i,
                memory_usage=60.0 + i
            )
            monitor.metrics_history.append(metrics)
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        
        assert "latency" in summary
        assert "error_rate" in summary
        assert "cpu_usage" in summary
        assert "memory_usage" in summary
        assert "connections" in summary


if __name__ == "__main__":
    pytest.main([__file__])
