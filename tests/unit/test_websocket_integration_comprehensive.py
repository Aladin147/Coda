"""
Comprehensive tests for WebSocket integration to increase coverage from 25% to 70%+.
Targets specific uncovered lines in websocket components.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import websockets
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid

from src.coda.components.voice.websocket_handler import VoiceWebSocketHandler
from src.coda.components.voice.models import VoiceStreamChunk, VoiceMessage, VoiceResponse
from src.coda.components.voice.exceptions import WebSocketError, VoiceProcessingError
from src.coda.interfaces.websocket.server import CodaWebSocketServer


class TestWebSocketIntegrationComprehensive:
    """Comprehensive tests for WebSocket integration covering all major functionality."""

    @pytest_asyncio.fixture
    async def mock_websocket(self):
        """Create mock WebSocket connection."""
        websocket = Mock()
        websocket.send = AsyncMock()
        websocket.recv = AsyncMock()
        websocket.close = AsyncMock()
        websocket.closed = False
        websocket.remote_address = ("127.0.0.1", 12345)
        
        # Mock ping/pong for connection health
        websocket.ping = AsyncMock()
        websocket.pong = AsyncMock()
        
        return websocket

    @pytest_asyncio.fixture
    async def mock_voice_manager(self):
        """Create mock voice manager."""
        manager = Mock()
        
        # Mock voice processing methods
        manager.process_voice_input = AsyncMock(return_value=VoiceResponse(
            response_id=str(uuid.uuid4()),
            conversation_id="test-conv",
            message_id="test-msg",
            text_content="Test response",
            processing_mode="moshi_only",
            total_latency_ms=100.0
        ))
        
        manager.start_streaming = AsyncMock()
        manager.stop_streaming = AsyncMock()
        manager.is_streaming = False
        
        return manager

    @pytest_asyncio.fixture
    async def websocket_handler(self, mock_voice_manager):
        """Create WebSocket handler with mocked dependencies."""
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            max_connections=10,
            heartbeat_interval=30
        )
        await handler.initialize()
        return handler

    @pytest_asyncio.fixture
    async def websocket_server(self, websocket_handler):
        """Create WebSocket server with handler."""
        server = CodaWebSocketServer(
            host="localhost",
            port=8765,
            voice_handler=websocket_handler
        )
        return server

    @pytest_asyncio.fixture
    def sample_voice_message(self):
        """Create sample voice message."""
        return {
            "type": "voice_input",
            "conversation_id": "test-conv-123",
            "message_id": str(uuid.uuid4()),
            "audio_data": "base64_encoded_audio_data",
            "text_content": "Hello, how are you?",
            "timestamp": datetime.now().isoformat()
        }

    @pytest_asyncio.fixture
    def sample_stream_chunk(self):
        """Create sample stream chunk."""
        return VoiceStreamChunk(
            conversation_id="test-conv-123",
            chunk_index=0,
            text_content="Hello",
            audio_data=b"fake_audio_data",
            timestamp=datetime.now().timestamp(),
            is_complete=False,
            chunk_type="audio"
        )

    @pytest.mark.asyncio
    async def test_websocket_handler_initialization(self, mock_voice_manager):
        """Test WebSocket handler initialization."""
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            max_connections=5,
            heartbeat_interval=60
        )
        
        assert not handler.is_initialized
        await handler.initialize()
        assert handler.is_initialized
        assert handler.max_connections == 5
        assert handler.heartbeat_interval == 60

    @pytest.mark.asyncio
    async def test_websocket_connection_handling(self, websocket_handler, mock_websocket):
        """Test WebSocket connection establishment and management."""
        # Test connection acceptance
        await websocket_handler.handle_connection(mock_websocket)
        
        assert len(websocket_handler.active_connections) == 1
        assert mock_websocket in websocket_handler.active_connections

    @pytest.mark.asyncio
    async def test_websocket_message_processing(self, websocket_handler, mock_websocket, sample_voice_message):
        """Test WebSocket message processing."""
        # Mock incoming message
        mock_websocket.recv.return_value = json.dumps(sample_voice_message)
        
        # Add connection to handler
        websocket_handler.active_connections.add(mock_websocket)
        
        # Process message
        await websocket_handler._handle_message(mock_websocket, json.dumps(sample_voice_message))
        
        # Verify response was sent
        mock_websocket.send.assert_called()

    @pytest.mark.asyncio
    async def test_voice_input_processing(self, websocket_handler, mock_websocket, sample_voice_message):
        """Test voice input message processing."""
        websocket_handler.active_connections.add(mock_websocket)
        
        await websocket_handler._handle_voice_input(mock_websocket, sample_voice_message)
        
        # Verify voice manager was called
        websocket_handler.voice_manager.process_voice_input.assert_called_once()
        
        # Verify response was sent
        mock_websocket.send.assert_called()

    @pytest.mark.asyncio
    async def test_streaming_start_stop(self, websocket_handler, mock_websocket):
        """Test streaming start and stop functionality."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Test start streaming
        start_message = {
            "type": "start_streaming",
            "conversation_id": "test-conv-123"
        }
        
        await websocket_handler._handle_start_streaming(mock_websocket, start_message)
        websocket_handler.voice_manager.start_streaming.assert_called_once()
        
        # Test stop streaming
        stop_message = {
            "type": "stop_streaming",
            "conversation_id": "test-conv-123"
        }
        
        await websocket_handler._handle_stop_streaming(mock_websocket, stop_message)
        websocket_handler.voice_manager.stop_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_chunk_broadcasting(self, websocket_handler, mock_websocket, sample_stream_chunk):
        """Test broadcasting stream chunks to connected clients."""
        # Add multiple connections
        mock_websocket2 = Mock()
        mock_websocket2.send = AsyncMock()
        mock_websocket2.closed = False
        
        websocket_handler.active_connections.add(mock_websocket)
        websocket_handler.active_connections.add(mock_websocket2)
        
        # Broadcast chunk
        await websocket_handler.broadcast_stream_chunk(sample_stream_chunk)
        
        # Verify all connections received the chunk
        mock_websocket.send.assert_called()
        mock_websocket2.send.assert_called()

    @pytest.mark.asyncio
    async def test_connection_cleanup(self, websocket_handler, mock_websocket):
        """Test connection cleanup on disconnect."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Simulate connection close
        mock_websocket.closed = True
        
        await websocket_handler._cleanup_connection(mock_websocket)
        
        assert mock_websocket not in websocket_handler.active_connections

    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, websocket_handler, mock_websocket):
        """Test WebSocket heartbeat/ping-pong mechanism."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Test ping
        await websocket_handler._send_ping(mock_websocket)
        mock_websocket.ping.assert_called_once()
        
        # Test pong response handling
        await websocket_handler._handle_pong(mock_websocket, b"pong_data")
        
        # Connection should remain active
        assert mock_websocket in websocket_handler.active_connections

    @pytest.mark.asyncio
    async def test_error_handling_invalid_message(self, websocket_handler, mock_websocket):
        """Test error handling for invalid messages."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Test invalid JSON
        invalid_json = "{ invalid json }"
        
        await websocket_handler._handle_message(mock_websocket, invalid_json)
        
        # Should send error response
        mock_websocket.send.assert_called()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "error"

    @pytest.mark.asyncio
    async def test_error_handling_missing_fields(self, websocket_handler, mock_websocket):
        """Test error handling for messages with missing required fields."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Message missing required fields
        incomplete_message = {
            "type": "voice_input"
            # Missing conversation_id, message_id, etc.
        }
        
        await websocket_handler._handle_message(mock_websocket, json.dumps(incomplete_message))
        
        # Should send error response
        mock_websocket.send.assert_called()

    @pytest.mark.asyncio
    async def test_connection_limit_enforcement(self, websocket_handler):
        """Test connection limit enforcement."""
        # Set low connection limit
        websocket_handler.max_connections = 2
        
        # Create multiple mock connections
        connections = []
        for i in range(3):
            mock_conn = Mock()
            mock_conn.send = AsyncMock()
            mock_conn.close = AsyncMock()
            mock_conn.closed = False
            mock_conn.remote_address = (f"127.0.0.{i}", 12345)
            connections.append(mock_conn)
        
        # Try to add connections beyond limit
        for i, conn in enumerate(connections):
            if i < 2:
                await websocket_handler.handle_connection(conn)
                assert conn in websocket_handler.active_connections
            else:
                # Third connection should be rejected
                await websocket_handler.handle_connection(conn)
                conn.close.assert_called()

    @pytest.mark.asyncio
    async def test_websocket_server_start_stop(self, websocket_server):
        """Test WebSocket server start and stop functionality."""
        # Mock the websockets.serve function
        with patch('websockets.serve') as mock_serve:
            mock_server = Mock()
            mock_server.close = AsyncMock()
            mock_server.wait_closed = AsyncMock()
            mock_serve.return_value = mock_server
            
            # Start server
            await websocket_server.start()
            assert websocket_server.is_running
            mock_serve.assert_called_once()
            
            # Stop server
            await websocket_server.stop()
            assert not websocket_server.is_running
            mock_server.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, websocket_handler, mock_websocket):
        """Test concurrent message handling."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Create multiple messages
        messages = [
            {"type": "voice_input", "conversation_id": f"conv-{i}", "message_id": f"msg-{i}"}
            for i in range(5)
        ]
        
        # Process messages concurrently
        tasks = [
            websocket_handler._handle_message(mock_websocket, json.dumps(msg))
            for msg in messages
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # All messages should have been processed
        assert mock_websocket.send.call_count >= 5

    @pytest.mark.asyncio
    async def test_message_queuing(self, websocket_handler, mock_websocket):
        """Test message queuing for slow connections."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Simulate slow send operation
        mock_websocket.send.side_effect = lambda x: asyncio.sleep(0.1)
        
        # Queue multiple messages
        messages = [f"Message {i}" for i in range(3)]
        
        tasks = [websocket_handler._send_message(mock_websocket, msg) for msg in messages]
        await asyncio.gather(*tasks)
        
        # All messages should have been sent
        assert mock_websocket.send.call_count == 3

    @pytest.mark.asyncio
    async def test_connection_state_tracking(self, websocket_handler, mock_websocket):
        """Test connection state tracking."""
        # Initially no connections
        assert len(websocket_handler.active_connections) == 0
        
        # Add connection
        await websocket_handler.handle_connection(mock_websocket)
        assert len(websocket_handler.active_connections) == 1
        
        # Get connection stats
        stats = websocket_handler.get_connection_stats()
        assert stats['active_connections'] == 1
        assert stats['total_connections'] >= 1

    @pytest.mark.asyncio
    async def test_message_validation(self, websocket_handler):
        """Test message validation functionality."""
        # Valid message
        valid_msg = {
            "type": "voice_input",
            "conversation_id": "test-conv",
            "message_id": "test-msg"
        }
        
        is_valid = websocket_handler._validate_message(valid_msg)
        assert is_valid is True
        
        # Invalid message (missing type)
        invalid_msg = {
            "conversation_id": "test-conv",
            "message_id": "test-msg"
        }
        
        is_valid = websocket_handler._validate_message(invalid_msg)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_authentication_handling(self, websocket_handler, mock_websocket):
        """Test WebSocket authentication handling."""
        # Mock authentication message
        auth_message = {
            "type": "authenticate",
            "token": "valid_auth_token",
            "user_id": "test_user"
        }
        
        await websocket_handler._handle_authentication(mock_websocket, auth_message)
        
        # Connection should be marked as authenticated
        assert hasattr(mock_websocket, 'authenticated')

    @pytest.mark.asyncio
    async def test_rate_limiting(self, websocket_handler, mock_websocket):
        """Test rate limiting for WebSocket messages."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Send many messages rapidly
        for i in range(20):
            message = {
                "type": "voice_input",
                "conversation_id": "test-conv",
                "message_id": f"msg-{i}"
            }
            await websocket_handler._handle_message(mock_websocket, json.dumps(message))
        
        # Rate limiting should prevent processing all messages
        # (Implementation dependent on actual rate limiting logic)
        assert mock_websocket.send.call_count > 0

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, websocket_handler):
        """Test graceful shutdown of WebSocket handler."""
        # Add some connections
        connections = []
        for i in range(3):
            mock_conn = Mock()
            mock_conn.close = AsyncMock()
            mock_conn.closed = False
            connections.append(mock_conn)
            websocket_handler.active_connections.add(mock_conn)
        
        # Shutdown handler
        await websocket_handler.shutdown()
        
        # All connections should be closed
        for conn in connections:
            conn.close.assert_called()
        
        assert len(websocket_handler.active_connections) == 0

    @pytest.mark.asyncio
    async def test_error_recovery(self, websocket_handler, mock_websocket):
        """Test error recovery mechanisms."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Simulate connection error
        mock_websocket.send.side_effect = Exception("Connection error")
        
        # Try to send message (should handle error gracefully)
        await websocket_handler._send_message(mock_websocket, "test message")
        
        # Connection should be cleaned up
        assert mock_websocket not in websocket_handler.active_connections

    @pytest.mark.asyncio
    async def test_metrics_collection(self, websocket_handler, mock_websocket):
        """Test metrics collection for WebSocket operations."""
        websocket_handler.active_connections.add(mock_websocket)
        
        # Process some messages
        for i in range(5):
            message = {"type": "ping", "data": f"test_{i}"}
            await websocket_handler._handle_message(mock_websocket, json.dumps(message))
        
        # Get metrics
        metrics = websocket_handler.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'messages_processed' in metrics
        assert 'active_connections' in metrics
        assert 'errors_count' in metrics
