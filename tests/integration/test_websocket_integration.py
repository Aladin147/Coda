"""
Integration tests for WebSocket communication and real-time features.
Tests WebSocket server, client connections, and message handling.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import uuid
import websockets
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.coda.interfaces.websocket.server import CodaWebSocketServer
from src.coda.components.voice.websocket_handler import VoiceWebSocketHandler
from src.coda.components.voice.models import VoiceMessage, VoiceStreamChunk


class TestWebSocketIntegration:
    """Integration tests for WebSocket communication."""

    @pytest_asyncio.fixture
    async def mock_voice_manager(self):
        """Create mock voice manager for WebSocket tests."""
        manager = Mock()
        
        # Mock voice processing
        manager.process_voice_input = AsyncMock(return_value={
            "response_id": str(uuid.uuid4()),
            "text_content": "WebSocket response",
            "audio_data": "base64_encoded_audio"
        })
        
        # Mock streaming
        async def mock_stream():
            for i in range(3):
                yield VoiceStreamChunk(
                    conversation_id="ws-test-conv",
                    chunk_index=i,
                    text_content=f"Chunk {i}",
                    audio_data=b"fake_audio",
                    timestamp=1234567890.0,
                    is_complete=(i == 2),
                    chunk_type="audio"
                )
                await asyncio.sleep(0.01)
        
        manager.stream_response = AsyncMock(return_value=mock_stream())
        manager.is_initialized = True
        
        return manager

    @pytest_asyncio.fixture
    async def websocket_handler(self, mock_voice_manager):
        """Create WebSocket handler with mocked voice manager."""
        handler = VoiceWebSocketHandler(
            voice_manager=mock_voice_manager,
            max_connections=10,
            ping_interval=30.0
        )
        return handler

    @pytest_asyncio.fixture
    async def websocket_server(self, websocket_handler):
        """Create WebSocket server for testing."""
        server = CodaWebSocketServer(
            host="localhost",
            port=0,  # Use random available port
            voice_handler=websocket_handler
        )
        return server

    @pytest_asyncio.fixture
    async def running_server(self, websocket_server):
        """Start WebSocket server for testing."""
        with patch('websockets.serve') as mock_serve:
            mock_server = Mock()
            mock_server.close = AsyncMock()
            mock_server.wait_closed = AsyncMock()
            mock_serve.return_value = mock_server
            
            await websocket_server.start()
            yield websocket_server
            await websocket_server.stop()

    @pytest.mark.asyncio
    async def test_websocket_server_startup_shutdown(self, websocket_server):
        """Test WebSocket server startup and shutdown."""
        with patch('websockets.serve') as mock_serve:
            mock_server = Mock()
            mock_server.close = AsyncMock()
            mock_server.wait_closed = AsyncMock()
            mock_serve.return_value = mock_server
            
            # Test startup
            await websocket_server.start()
            assert websocket_server.is_running
            mock_serve.assert_called_once()
            
            # Test shutdown
            await websocket_server.stop()
            assert not websocket_server.is_running
            mock_server.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_connection_handling(self, websocket_handler):
        """Test WebSocket connection establishment and management."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.recv = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.closed = False
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        
        # Test connection handling
        await websocket_handler.handle_connection(mock_websocket)
        
        # Verify connection was added
        assert mock_websocket in websocket_handler.active_connections

    @pytest.mark.asyncio
    async def test_websocket_message_processing(self, websocket_handler):
        """Test WebSocket message processing."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.closed = False
        
        # Add connection to handler
        websocket_handler.active_connections.add(mock_websocket)
        
        # Test voice input message
        voice_message = {
            "type": "voice_input",
            "conversation_id": "ws-test-conv",
            "message_id": str(uuid.uuid4()),
            "text_content": "Hello WebSocket",
            "audio_data": "base64_encoded_audio"
        }
        
        # Process message
        await websocket_handler._handle_message(mock_websocket, json.dumps(voice_message))
        
        # Verify response was sent
        mock_websocket.send.assert_called()

    @pytest.mark.asyncio
    async def test_websocket_streaming(self, websocket_handler):
        """Test WebSocket streaming functionality."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.closed = False
        
        # Add connection to handler
        websocket_handler.active_connections.add(mock_websocket)
        
        # Test streaming start
        start_message = {
            "type": "start_streaming",
            "conversation_id": "ws-test-conv"
        }
        
        await websocket_handler._handle_message(mock_websocket, json.dumps(start_message))
        
        # Verify streaming was initiated
        websocket_handler.voice_manager.stream_response.assert_called()

    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, websocket_handler):
        """Test WebSocket error handling."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.closed = False
        
        # Add connection to handler
        websocket_handler.active_connections.add(mock_websocket)
        
        # Test invalid JSON message
        invalid_message = "{ invalid json }"
        
        await websocket_handler._handle_message(mock_websocket, invalid_message)
        
        # Verify error response was sent
        mock_websocket.send.assert_called()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "error"

    @pytest.mark.asyncio
    async def test_websocket_connection_limits(self, websocket_handler):
        """Test WebSocket connection limit enforcement."""
        # Set low connection limit
        websocket_handler.max_connections = 2
        
        # Create mock connections
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
    async def test_websocket_heartbeat(self, websocket_handler):
        """Test WebSocket heartbeat mechanism."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.ping = AsyncMock()
        mock_websocket.closed = False
        
        # Add connection to handler
        websocket_handler.active_connections.add(mock_websocket)
        
        # Test ping
        await websocket_handler._send_ping(mock_websocket)
        mock_websocket.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_broadcast(self, websocket_handler):
        """Test broadcasting messages to all connected clients."""
        # Create multiple mock connections
        connections = []
        for i in range(3):
            mock_conn = Mock()
            mock_conn.send = AsyncMock()
            mock_conn.closed = False
            connections.append(mock_conn)
            websocket_handler.active_connections.add(mock_conn)
        
        # Create test chunk
        chunk = VoiceStreamChunk(
            conversation_id="broadcast-test",
            chunk_index=0,
            text_content="Broadcast message",
            audio_data=b"fake_audio",
            timestamp=1234567890.0,
            is_complete=False,
            chunk_type="audio"
        )
        
        # Broadcast chunk
        await websocket_handler.broadcast_stream_chunk(chunk)
        
        # Verify all connections received the chunk
        for conn in connections:
            conn.send.assert_called()

    @pytest.mark.asyncio
    async def test_websocket_authentication(self, websocket_handler):
        """Test WebSocket authentication handling."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.closed = False
        
        # Test authentication message
        auth_message = {
            "type": "authenticate",
            "token": "valid_auth_token",
            "user_id": "test_user"
        }
        
        await websocket_handler._handle_authentication(mock_websocket, auth_message)
        
        # Verify authentication was processed
        assert hasattr(mock_websocket, 'authenticated')

    @pytest.mark.asyncio
    async def test_websocket_concurrent_connections(self, websocket_handler):
        """Test handling multiple concurrent WebSocket connections."""
        # Create multiple connections
        connections = []
        for i in range(5):
            mock_conn = Mock()
            mock_conn.send = AsyncMock()
            mock_conn.recv = AsyncMock(return_value=json.dumps({
                "type": "voice_input",
                "conversation_id": f"concurrent-{i}",
                "message_id": str(uuid.uuid4()),
                "text_content": f"Message {i}"
            }))
            mock_conn.closed = False
            mock_conn.remote_address = (f"127.0.0.{i}", 12345)
            connections.append(mock_conn)
        
        # Handle connections concurrently
        tasks = [
            websocket_handler.handle_connection(conn)
            for conn in connections
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all connections were handled
        assert len(websocket_handler.active_connections) == 5

    @pytest.mark.asyncio
    async def test_websocket_cleanup_on_disconnect(self, websocket_handler):
        """Test cleanup when WebSocket connections disconnect."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.closed = False
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        
        # Add connection
        await websocket_handler.handle_connection(mock_websocket)
        assert mock_websocket in websocket_handler.active_connections
        
        # Simulate disconnect
        mock_websocket.closed = True
        await websocket_handler._cleanup_connection(mock_websocket)
        
        # Verify cleanup
        assert mock_websocket not in websocket_handler.active_connections

    @pytest.mark.asyncio
    async def test_websocket_performance_metrics(self, websocket_handler):
        """Test WebSocket performance metrics collection."""
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.closed = False
        
        # Add connection and process messages
        websocket_handler.active_connections.add(mock_websocket)
        
        for i in range(5):
            message = {
                "type": "voice_input",
                "conversation_id": "metrics-test",
                "message_id": str(uuid.uuid4()),
                "text_content": f"Message {i}"
            }
            await websocket_handler._handle_message(mock_websocket, json.dumps(message))
        
        # Get metrics
        metrics = websocket_handler.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'messages_processed' in metrics
        assert 'active_connections' in metrics
        assert metrics['active_connections'] == 1

    @pytest.mark.asyncio
    async def test_websocket_graceful_shutdown(self, websocket_handler):
        """Test graceful shutdown of WebSocket handler."""
        # Add multiple connections
        connections = []
        for i in range(3):
            mock_conn = Mock()
            mock_conn.close = AsyncMock()
            mock_conn.closed = False
            connections.append(mock_conn)
            websocket_handler.active_connections.add(mock_conn)
        
        # Shutdown handler
        await websocket_handler.shutdown()
        
        # Verify all connections were closed
        for conn in connections:
            conn.close.assert_called()
        
        assert len(websocket_handler.active_connections) == 0
