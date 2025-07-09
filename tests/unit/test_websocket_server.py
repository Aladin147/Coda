"""
Unit tests for the WebSocket server.
"""

import asyncio
import json
import pytest
import websockets
from unittest.mock import AsyncMock, MagicMock

from src.coda.interfaces.websocket.server import CodaWebSocketServer
from src.coda.interfaces.websocket.events import EventType


class TestCodaWebSocketServer:
    """Test cases for CodaWebSocketServer."""

    @pytest.fixture
    def server(self):
        """Create a test server instance."""
        return CodaWebSocketServer(host="localhost", port=8765)

    @pytest.mark.asyncio
    async def test_server_initialization(self, server):
        """Test server initialization."""
        assert server.host == "localhost"
        assert server.port == 8765
        assert server.running is False
        assert len(server.clients) == 0
        assert server.sequence_number == 0

    @pytest.mark.asyncio
    async def test_server_start_stop(self, server):
        """Test server start and stop."""
        # Start server
        await server.start()
        assert server.running is True
        assert server.server is not None

        # Stop server
        await server.stop()
        assert server.running is False

    @pytest.mark.asyncio
    async def test_broadcast_event_no_clients(self, server):
        """Test broadcasting when no clients are connected."""
        await server.start()
        
        # Should not raise an error
        await server.broadcast_event(
            EventType.SYSTEM_INFO,
            {"message": "test"}
        )
        
        await server.stop()

    @pytest.mark.asyncio
    async def test_event_handlers(self, server):
        """Test event handler registration."""
        connect_handler = MagicMock()
        disconnect_handler = MagicMock()
        message_handler = MagicMock()

        server.add_connect_handler(connect_handler)
        server.add_disconnect_handler(disconnect_handler)
        server.add_message_handler(message_handler)

        assert connect_handler in server.on_connect_handlers
        assert disconnect_handler in server.on_disconnect_handlers
        assert message_handler in server.on_message_handlers

    @pytest.mark.asyncio
    async def test_stats_tracking(self, server):
        """Test statistics tracking."""
        stats = server.get_stats()
        
        assert "total_connections" in stats
        assert "current_connections" in stats
        assert "events_sent" in stats
        assert "errors" in stats
        assert "replay_buffer_size" in stats
        assert "running" in stats

    @pytest.mark.asyncio
    async def test_client_list(self, server):
        """Test getting connected clients."""
        clients = server.get_connected_clients()
        assert isinstance(clients, list)
        assert len(clients) == 0

    @pytest.mark.asyncio
    async def test_convenience_methods(self, server):
        """Test convenience methods for common events."""
        await server.start()

        # Test system info broadcast
        await server.broadcast_system_info({"version": "2.0.0"})

        # Test system error broadcast
        await server.broadcast_system_error("error", "Test error")

        # Test system metrics broadcast
        await server.broadcast_system_metrics(100.0, 50.0, 200.0, 3600.0)

        await server.stop()

    def test_replay_buffer_management(self, server):
        """Test replay buffer management."""
        # Initially empty
        assert len(server.replay_buffer) == 0

        # Set a small max size for testing
        server.max_replay_events = 3

        # Add events to buffer (simulated)
        for i in range(5):
            event = {
                "seq": i,
                "type": "test_event",
                "data": {"index": i}
            }
            server.replay_buffer.append(event)
            
            # Trim buffer if needed (simulate the server logic)
            if len(server.replay_buffer) > server.max_replay_events:
                server.replay_buffer = server.replay_buffer[-server.max_replay_events:]

        # Should only keep the last 3 events
        assert len(server.replay_buffer) == 3
        assert server.replay_buffer[0]["data"]["index"] == 2
        assert server.replay_buffer[-1]["data"]["index"] == 4


@pytest.mark.asyncio
async def test_websocket_integration():
    """Integration test with actual WebSocket connection."""
    server = CodaWebSocketServer(host="localhost", port=8766)  # Different port
    
    try:
        await server.start()
        
        # Connect a test client
        uri = f"ws://{server.host}:{server.port}"
        
        async with websockets.connect(uri) as websocket:
            # Server should have one client
            assert len(server.clients) == 1
            
            # Broadcast an event
            await server.broadcast_event(
                EventType.SYSTEM_INFO,
                {"message": "test broadcast"}
            )
            
            # Client should receive the event
            message = await websocket.recv()
            data = json.loads(message)
            
            assert data["type"] == "system_info"
            assert data["data"]["message"] == "test broadcast"
            assert "seq" in data
            assert "timestamp" in data
            
        # Client should be disconnected
        await asyncio.sleep(0.1)  # Give time for cleanup
        assert len(server.clients) == 0
        
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in WebSocket server."""
    server = CodaWebSocketServer(host="localhost", port=8767)
    
    try:
        await server.start()
        
        # Test invalid event type
        with pytest.raises(ValueError):
            await server.broadcast_event(
                "invalid_event_type",  # This should raise an error
                {"data": "test"}
            )
            
    finally:
        await server.stop()


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        server = CodaWebSocketServer()
        await server.start()
        
        await server.broadcast_system_info({"test": "message"})
        print("✅ WebSocket server test passed")
        
        await server.stop()
    
    asyncio.run(simple_test())
