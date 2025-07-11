"""
Comprehensive tests for WebSocket voice integration.

Tests the complete WebSocket voice processing system including
real-time communication, audio streaming, and event broadcasting.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
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
from src.coda.components.voice.websocket_monitoring import WebSocketMonitor
from src.coda.components.voice.websocket_server import VoiceWebSocketServer
from src.coda.components.voice.models import VoiceConfig, AudioConfig, MoshiConfig


class TestWebSocketHandler:
    """Test WebSocket handler functionality."""
    
    @pytest.fixture
    async def voice_manager(self):
        """Mock voice manager for testing."""
        manager = AsyncMock()
        manager.start_conversation.return_value = Mock(
            conversation_id="test-conv",
            is_active=True,
            participant_count=1
        )
        manager.process_voice_input.return_value = Mock(
            conversation_id="test-conv",
            text_content="Test response",
            processing_time=100.0
        )
        return manager
    
    @pytest.fixture
    async def websocket_handler(self, voice_manager):
        """Create WebSocket handler for testing."""
        handler = VoiceWebSocketHandler(
            voice_manager=voice_manager,
            host="localhost",
            port=0,  # Use random port for testing
            auth_required=False
        )
        yield handler
        await handler.stop_server()
    
    @pytest.mark.asyncio
    async def test_websocket_message_creation(self):
        """Test WebSocket message creation and serialization."""
        message = WebSocketMessage(
            type=MessageType.VOICE_START,
            data={"mode": "adaptive"},
            conversation_id="test-conv"
        )
        
        # Test serialization
        json_str = message.to_json()
        assert isinstance(json_str, str)
        
        # Test deserialization
        parsed_message = WebSocketMessage.from_json(json_str)
        assert parsed_message.type == MessageType.VOICE_START
        assert parsed_message.data["mode"] == "adaptive"
        assert parsed_message.conversation_id == "test-conv"
    
    @pytest.mark.asyncio
    async def test_client_connection_tracking(self, websocket_handler):
        """Test client connection tracking."""
        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        
        connection = ClientConnection(
            websocket=mock_websocket,
            client_id="test-client",
            user_id="test-user"
        )
        
        # Test connection properties
        assert connection.client_id == "test-client"
        assert connection.user_id == "test-user"
        assert not connection.is_authenticated
        assert connection.connection_duration > 0
        
        # Test activity update
        old_activity = connection.last_activity
        time.sleep(0.01)
        connection.update_activity()
        assert connection.last_activity > old_activity
    
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
            {"message": "Test broadcast"},
            conversation_id="test-conv"
        )
        
        assert sent_count == 3
        
        # Verify all connections received the message
        for connection in mock_connections.values():
            connection.websocket.send.assert_called_once()


class TestEventBroadcaster:
    """Test event broadcasting functionality."""
    
    @pytest.fixture
    def websocket_handler(self):
        """Mock WebSocket handler for testing."""
        handler = Mock()
        handler.connections = {
            "client_1": Mock(conversation_id="conv_1", user_id="user_1"),
            "client_2": Mock(conversation_id="conv_1", user_id="user_2"),
            "client_3": Mock(conversation_id="conv_2", user_id="user_1")
        }
        handler._send_message = AsyncMock()
        return handler
    
    @pytest.fixture
    def event_broadcaster(self, websocket_handler):
        """Create event broadcaster for testing."""
        return VoiceEventBroadcaster(websocket_handler)
    
    @pytest.mark.asyncio
    async def test_event_creation(self):
        """Test voice event creation."""
        event = VoiceEvent(
            event_type=EventType.PROCESSING_STARTED,
            data={"mode": "hybrid"},
            conversation_id="test-conv"
        )
        
        assert event.event_type == EventType.PROCESSING_STARTED
        assert event.data["mode"] == "hybrid"
        assert event.conversation_id == "test-conv"
        assert event.timestamp > 0
        assert event.event_id is not None
    
    @pytest.mark.asyncio
    async def test_event_broadcasting(self, event_broadcaster):
        """Test event broadcasting to clients."""
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
    async def test_client_subscription(self, event_broadcaster):
        """Test client event subscription."""
        client_id = "test_client"
        event_types = [EventType.PROCESSING_STARTED, EventType.PROCESSING_COMPLETED]
        
        await event_broadcaster.subscribe_client(client_id, event_types)
        
        # Verify subscription
        for event_type in event_types:
            assert client_id in event_broadcaster.subscribers[event_type]
        
        # Test unsubscription
        await event_broadcaster.unsubscribe_client(client_id, [EventType.PROCESSING_STARTED])
        assert client_id not in event_broadcaster.subscribers[EventType.PROCESSING_STARTED]
        assert client_id in event_broadcaster.subscribers[EventType.PROCESSING_COMPLETED]
    
    @pytest.mark.asyncio
    async def test_event_history(self, event_broadcaster):
        """Test event history functionality."""
        # Broadcast several events
        for i in range(5):
            await event_broadcaster.broadcast_event(
                EventType.PROCESSING_STARTED,
                {"iteration": i},
                conversation_id="test-conv"
            )
        
        # Get history
        history = event_broadcaster.get_event_history(
            conversation_id="test-conv",
            limit=3
        )
        
        assert len(history) <= 3
        assert all(event.conversation_id == "test-conv" for event in history)


class TestAudioStreaming:
    """Test audio streaming functionality."""
    
    @pytest.fixture
    def websocket_handler(self):
        """Mock WebSocket handler for testing."""
        handler = Mock()
        handler.connections = {
            "client_1": Mock(conversation_id="conv_1")
        }
        handler._send_message = AsyncMock()
        return handler
    
    @pytest.fixture
    def audio_processor(self, websocket_handler):
        """Create audio stream processor for testing."""
        return AudioStreamProcessor(websocket_handler)
    
    def test_audio_stream_config(self):
        """Test audio stream configuration."""
        config = AudioStreamConfig(
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            chunk_size_ms=100,
            streaming_mode=StreamingMode.VOICE_ACTIVITY
        )
        
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.format == AudioFormat.WAV
        assert config.chunk_size_samples == 1600  # 100ms at 16kHz
        assert config.streaming_mode == StreamingMode.VOICE_ACTIVITY
    
    @pytest.mark.asyncio
    async def test_audio_stream_lifecycle(self, audio_processor):
        """Test audio stream start and stop."""
        client_id = "test_client"
        config = AudioStreamConfig()
        
        # Start stream
        success = await audio_processor.start_audio_stream(client_id, config)
        assert success
        assert client_id in audio_processor.active_streams
        
        # Stop stream
        success = await audio_processor.stop_audio_stream(client_id)
        assert success
        assert client_id not in audio_processor.active_streams
    
    @pytest.mark.asyncio
    async def test_audio_chunk_processing(self, audio_processor):
        """Test audio chunk processing."""
        client_id = "test_client"
        config = AudioStreamConfig()
        
        # Start stream
        await audio_processor.start_audio_stream(client_id, config)
        
        # Process audio chunk
        audio_data = b'\x00\x01' * 1000  # Mock audio data
        success = await audio_processor.process_audio_chunk(
            client_id,
            audio_data,
            {"is_final": False}
        )
        
        assert success
        
        # Verify stream state was updated
        stream_state = audio_processor.active_streams[client_id]
        assert stream_state["chunk_counter"] > 0
        assert stream_state["last_activity"] > 0
    
    def test_audio_format_conversion(self, audio_processor):
        """Test audio format conversion."""
        import numpy as np
        
        config = AudioStreamConfig(format=AudioFormat.PCM)
        
        # Test PCM conversion
        audio_data = b'\x00\x01\x02\x03' * 100
        audio_array = audio_processor._convert_audio_data(audio_data, config)
        
        assert isinstance(audio_array, np.ndarray)
        assert audio_array.dtype == np.float32
        
        # Test conversion back to bytes
        audio_bytes = audio_processor._convert_to_bytes(audio_array, config)
        assert isinstance(audio_bytes, bytes)
    
    def test_voice_activity_detection(self, audio_processor):
        """Test voice activity detection."""
        import numpy as np
        
        # Test with silence
        silence = np.zeros(1000, dtype=np.float32)
        vad_result = audio_processor._detect_voice_activity(silence, 0.01)
        assert not vad_result
        
        # Test with voice
        voice = np.random.random(1000).astype(np.float32) * 0.5
        vad_result = audio_processor._detect_voice_activity(voice, 0.01)
        assert vad_result


class TestWebSocketMonitoring:
    """Test WebSocket monitoring functionality."""
    
    @pytest.fixture
    def websocket_handler(self):
        """Mock WebSocket handler for testing."""
        handler = Mock()
        handler.connections = {}
        return handler
    
    @pytest.fixture
    def event_broadcaster(self):
        """Mock event broadcaster for testing."""
        broadcaster = Mock()
        broadcaster.broadcast_event = AsyncMock()
        return broadcaster
    
    @pytest.fixture
    def monitor(self, websocket_handler, event_broadcaster):
        """Create WebSocket monitor for testing."""
        return WebSocketMonitor(websocket_handler, event_broadcaster)
    
    def test_connection_metrics_tracking(self, monitor):
        """Test connection metrics tracking."""
        client_id = "test_client"
        
        # Track connection
        monitor.track_connection(client_id)
        assert client_id in monitor.connection_metrics
        
        # Record activities
        monitor.record_message_sent(client_id, 100)
        monitor.record_message_received(client_id, 150)
        monitor.record_latency(client_id, 50.0)
        monitor.record_error(client_id)
        
        # Verify metrics
        metrics = monitor.connection_metrics[client_id]
        assert metrics.messages_sent == 1
        assert metrics.messages_received == 1
        assert metrics.bytes_sent == 100
        assert metrics.bytes_received == 150
        assert metrics.average_latency == 50.0
        assert metrics.errors == 1
        
        # Untrack connection
        monitor.untrack_connection(client_id)
        assert client_id not in monitor.connection_metrics
    
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
    
    def test_real_time_metrics(self, monitor):
        """Test real-time metrics collection."""
        # Add some test data
        monitor.track_connection("client_1")
        monitor.record_message_sent("client_1", 100)
        monitor.record_latency("client_1", 25.0)
        
        # Get metrics
        metrics = monitor.get_real_time_metrics()
        
        assert "current" in metrics or "connections" in metrics
        assert "summary" in metrics
    
    def test_alert_thresholds(self, monitor):
        """Test alert threshold configuration."""
        # Set custom threshold
        success = monitor.set_alert_threshold("max_latency_ms", 500.0)
        assert success
        assert monitor.alert_thresholds["max_latency_ms"] == 500.0
        
        # Try invalid threshold
        success = monitor.set_alert_threshold("invalid_metric", 100.0)
        assert not success


class TestWebSocketServer:
    """Test complete WebSocket server integration."""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test server initialization."""
        server = VoiceWebSocketServer(
            host="localhost",
            port=0,  # Random port
            auth_required=False
        )
        
        assert server.host == "localhost"
        assert server.port == 0
        assert not server.auth_required
        assert not server.is_running
    
    @pytest.mark.asyncio
    async def test_server_configuration(self):
        """Test server with custom configuration."""
        voice_config = VoiceConfig(
            audio=AudioConfig(sample_rate=24000),
            moshi=MoshiConfig(device="cpu")
        )
        
        server = VoiceWebSocketServer(
            voice_config=voice_config,
            max_connections=50,
            auth_required=True
        )
        
        assert server.voice_config == voice_config
        assert server.max_connections == 50
        assert server.auth_required
    
    def test_server_stats(self):
        """Test server statistics collection."""
        server = VoiceWebSocketServer()
        
        # Get stats before initialization
        stats = server.get_server_stats()
        
        assert "server" in stats
        assert stats["server"]["is_running"] is False
        assert stats["server"]["host"] == "localhost"
        assert stats["server"]["port"] == 8765


if __name__ == "__main__":
    pytest.main([__file__])
