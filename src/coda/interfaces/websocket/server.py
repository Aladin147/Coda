"""
WebSocket server for Coda.

This module provides a modern, async WebSocket server that allows clients to connect
to Coda and receive real-time events about its operation.

Features:
- Async/await based architecture
- Type-safe event system with Pydantic models
- Client connection management with automatic cleanup
- Event replay buffer for new clients
- Performance monitoring integration
- Graceful shutdown handling
- Error recovery and logging
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed, WebSocketException

from .events import EventType, create_event

logger = logging.getLogger("coda.websocket.server")


class CodaWebSocketServer:
    """
    Modern WebSocket server for Coda.

    This server allows clients to connect and receive real-time events about
    Coda's operation, including STT, LLM, TTS, memory, and tool events.

    Features:
    - Async event broadcasting
    - Client lifecycle management
    - Event replay for new connections
    - Performance monitoring
    - Graceful error handling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        max_replay_events: int = 50,
        ping_interval: int = 20,
        ping_timeout: int = 10,
    ):
        """
        Initialize the WebSocket server.

        Args:
            host: Host to bind to
            port: Port to bind to
            max_replay_events: Maximum events to keep in replay buffer
            ping_interval: Ping interval in seconds
            ping_timeout: Ping timeout in seconds
        """
        self.host = host
        self.port = port
        self.max_replay_events = max_replay_events
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # Server state
        self.clients: Dict[str, ServerConnection] = {}
        self.server: Optional[websockets.WebSocketServer] = None
        self.sequence_number = 0
        self.replay_buffer: List[Dict[str, Any]] = []
        self.running = False

        # Audio streaming state
        self.audio_sessions: Dict[str, Dict[str, Any]] = {}  # client_id -> session_info
        self.voice_manager = None  # Will be set by integration layer

        # Event handlers
        self.on_connect_handlers: List[Callable[[str], None]] = []
        self.on_disconnect_handlers: List[Callable[[str], None]] = []
        self.on_message_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

        # Performance tracking
        self.stats = {
            "total_connections": 0,
            "current_connections": 0,
            "events_sent": 0,
            "errors": 0,
            "audio_chunks_processed": 0,
        }

        logger.info(f"WebSocket server initialized at ws://{host}:{port}")

    def set_voice_manager(self, voice_manager) -> None:
        """Set the voice manager for audio processing."""
        self.voice_manager = voice_manager
        logger.info("Voice manager set for WebSocket audio streaming")

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self.running:
            logger.warning("WebSocket server already running")
            return

        try:
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
            )

            self.running = True
            logger.info(f"WebSocket server started at ws://{self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the WebSocket server gracefully."""
        if not self.running:
            return

        logger.info("Stopping WebSocket server...")
        self.running = False

        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[self._disconnect_client(client_id) for client_id in list(self.clients.keys())],
                return_exceptions=True,
            )

        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info("WebSocket server stopped")

    @asynccontextmanager
    async def run_context(self):
        """Context manager for running the server."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def _handle_client(self, websocket: ServerConnection, path: str) -> None:
        """
        Handle a client connection.

        Args:
            websocket: The WebSocket connection
            path: The connection path
        """
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        self.stats["total_connections"] += 1
        self.stats["current_connections"] += 1

        # Notify connection handlers
        for handler in self.on_connect_handlers:
            try:
                handler(client_id)
            except Exception as e:
                logger.error(f"Error in connect handler: {e}")

        logger.info(f"Client {client_id} connected from {websocket.remote_address}")

        try:
            # Send replay buffer if available
            if self.replay_buffer:
                await self._send_to_client(
                    websocket, {"type": "replay", "events": self.replay_buffer}
                )
                logger.debug(
                    f"Sent replay buffer ({len(self.replay_buffer)} events) to client {client_id}"
                )

            # Handle incoming messages
            async for message in websocket:
                try:
                    # Check if message is binary (audio data)
                    if isinstance(message, bytes):
                        await self._handle_binary_message(client_id, message)
                    else:
                        # Handle text message (JSON)
                        data = json.loads(message)
                        await self._handle_client_message(client_id, data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from client {client_id}: {e}")
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error handling message from client {client_id}: {e}")
                    self.stats["errors"] += 1

        except ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except WebSocketException as e:
            logger.warning(f"WebSocket error for client {client_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for client {client_id}: {e}")
            self.stats["errors"] += 1
        finally:
            await self._cleanup_client(client_id)

    async def _cleanup_client(self, client_id: str) -> None:
        """Clean up a disconnected client."""
        if client_id in self.clients:
            del self.clients[client_id]
            self.stats["current_connections"] -= 1

        # Clean up audio session if exists
        if client_id in self.audio_sessions:
            session_info = self.audio_sessions[client_id]
            session_id = session_info.get('session_id')
            logger.info(
                f"Cleaning up audio session for disconnected client {client_id}, "
                f"session {session_id}"
            )
            del self.audio_sessions[client_id]

        # Notify disconnect handlers
        for handler in self.on_disconnect_handlers:
            try:
                handler(client_id)
            except Exception as e:
                logger.error(f"Error in disconnect handler: {e}")

    async def _disconnect_client(self, client_id: str) -> None:
        """Forcefully disconnect a client."""
        if client_id in self.clients:
            websocket = self.clients[client_id]
            try:
                await websocket.close()
            except Exception as e:
                logger.debug(f"Error closing connection for client {client_id}: {e}")
            finally:
                await self._cleanup_client(client_id)

    async def _handle_client_message(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle a message from a client."""
        try:
            message_type = data.get("type", "unknown")
            logger.debug(f"Received message from client {client_id}: {message_type}")

            # Handle chat messages specially
            if message_type == "chat_message":
                await self._handle_chat_message(client_id, data)
                return

            # Handle audio session management
            if message_type == "audio_session_start":
                await self._handle_audio_session_start(client_id, data)
                return
            elif message_type == "audio_session_end":
                await self._handle_audio_session_end(client_id, data)
                return

            # Notify message handlers
            for handler in self.on_message_handlers:
                try:
                    handler(client_id, data)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")

        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def _handle_binary_message(self, client_id: str, binary_data: bytes) -> None:
        """Handle binary audio data from client."""
        try:
            # Check if client has an active audio session
            if client_id not in self.audio_sessions:
                logger.warning(
                    f"Received audio data from client {client_id} without active session"
                )
                return

            session_info = self.audio_sessions[client_id]

            # Process audio chunk
            if self.voice_manager:
                try:
                    # Get session ID for voice processing
                    session_id = session_info.get("session_id")

                    # Process audio through voice manager
                    result = await self.voice_manager.process_voice_input(
                        conversation_id=session_id, audio_data=binary_data
                    )

                    # Send response back to client
                    if result and "response" in result:
                        await self._send_audio_response(client_id, result)

                    self.stats["audio_chunks_processed"] += 1

                except Exception as e:
                    logger.error(f"Voice processing error for client {client_id}: {e}")
                    await self._send_audio_error(client_id, str(e))
            else:
                logger.warning("Voice manager not available for audio processing")
                await self._send_audio_error(client_id, "Voice processing not available")

        except Exception as e:
            logger.error(f"Error handling binary message from client {client_id}: {e}")
            self.stats["errors"] += 1

    async def _handle_audio_session_start(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle audio session start request."""
        try:
            session_id = data.get("session_id")
            audio_config = data.get("config", {})

            if not session_id:
                await self._send_error_to_client(client_id, "Session ID required for audio session")
                return

            # Create audio session
            self.audio_sessions[client_id] = {
                "session_id": session_id,
                "config": audio_config,
                "started_at": time.time(),
                "chunks_received": 0,
                "chunks_sent": 0,
            }

            # Send confirmation
            await self._send_to_client(
                self.clients[client_id],
                {
                    "type": "audio_session_started",
                    "session_id": session_id,
                    "client_id": client_id,
                    "timestamp": time.time(),
                },
            )

            logger.info(f"Started audio session for client {client_id}, session {session_id}")

        except Exception as e:
            logger.error(f"Error starting audio session for client {client_id}: {e}")
            await self._send_error_to_client(client_id, f"Failed to start audio session: {e}")

    async def _handle_audio_session_end(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle audio session end request."""
        try:
            if client_id in self.audio_sessions:
                session_info = self.audio_sessions[client_id]
                session_id = session_info["session_id"]

                # Clean up session
                del self.audio_sessions[client_id]

                # Send confirmation
                await self._send_to_client(
                    self.clients[client_id],
                    {
                        "type": "audio_session_ended",
                        "session_id": session_id,
                        "client_id": client_id,
                        "chunks_processed": session_info.get("chunks_received", 0),
                        "timestamp": time.time(),
                    },
                )

                logger.info(f"Ended audio session for client {client_id}, session {session_id}")
            else:
                logger.warning(f"No active audio session found for client {client_id}")

        except Exception as e:
            logger.error(f"Error ending audio session for client {client_id}: {e}")

    async def _send_audio_response(self, client_id: str, voice_result: Dict[str, Any]) -> None:
        """Send audio response back to client."""
        try:
            if client_id not in self.clients:
                logger.warning(f"Cannot send audio response to disconnected client {client_id}")
                return

            response_data = {
                "type": "audio_response",
                "session_id": voice_result.get("session_id"),
                "response": voice_result.get("response", {}),
                "processing_time_ms": voice_result.get("processing_time_ms", 0),
                "timestamp": time.time(),
            }

            # Send text response
            await self._send_to_client(self.clients[client_id], response_data)

            # Send binary audio data if available
            audio_data = voice_result.get("response", {}).get("audio_data")
            if audio_data and isinstance(audio_data, bytes):
                await self.clients[client_id].send(audio_data)

                # Update session stats
                if client_id in self.audio_sessions:
                    self.audio_sessions[client_id]["chunks_sent"] += 1

            logger.debug(f"Sent audio response to client {client_id}")

        except Exception as e:
            logger.error(f"Error sending audio response to client {client_id}: {e}")

    async def _send_audio_error(self, client_id: str, error_message: str) -> None:
        """Send audio processing error to client."""
        try:
            if client_id not in self.clients:
                return

            error_data = {"type": "audio_error", "error": error_message, "timestamp": time.time()}

            await self._send_to_client(self.clients[client_id], error_data)

        except Exception as e:
            logger.error(f"Error sending audio error to client {client_id}: {e}")

    async def _send_error_to_client(self, client_id: str, error_message: str) -> None:
        """Send error message to specific client."""
        try:
            if client_id not in self.clients:
                return

            error_data = {"type": "error", "error": error_message, "timestamp": time.time()}

            await self._send_to_client(self.clients[client_id], error_data)

        except Exception as e:
            logger.error(f"Error sending error to client {client_id}: {e}")

    async def _handle_chat_message(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle chat messages from clients."""
        try:
            # Get the client websocket
            websocket = self.clients.get(client_id)
            if not websocket:
                logger.warning(f"Client {client_id} not found for chat message")
                return

            # Forward to integration layer if available
            if hasattr(self, "integration_handler") and self.integration_handler:
                await self.integration_handler.handle_chat_message(websocket, data)
            else:
                # Send fallback response
                await self._send_to_client(
                    websocket,
                    {
                        "type": "chat_response",
                        "data": {
                            "response": {
                                "content": (
                                    "Chat functionality not available - "
                                    "integration layer not connected."
                                )
                            },
                            "session_id": data.get("data", {}).get(
                                "session_id", "fallback_session"
                            ),
                            "error": True,
                        },
                    },
                )

        except Exception as e:
            logger.error(f"Error handling chat message from client {client_id}: {e}")

    def set_integration_handler(self, handler) -> None:
        """Set the integration handler for processing chat messages."""
        self.integration_handler = handler
        logger.info("Integration handler set for WebSocket server")

    async def _send_to_client(self, websocket: ServerConnection, data: Dict[str, Any]) -> None:
        """Send data to a specific client."""
        try:
            message = json.dumps(data)
            await websocket.send(message)
            self.stats["events_sent"] += 1
        except ConnectionClosed:
            # Client disconnected, will be cleaned up by connection handler
            pass
        except Exception as e:
            logger.error(f"Error sending data to client: {e}")
            self.stats["errors"] += 1

    async def _send_error(self, websocket: ServerConnection, error_message: str) -> None:
        """Send an error message to a client."""
        error_data = {"type": "error", "message": error_message, "timestamp": time.time()}
        await self._send_to_client(websocket, error_data)

    # Public API methods

    async def broadcast_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        high_priority: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Broadcast an event to all connected clients.

        Args:
            event_type: The type of event
            data: The event data
            high_priority: Whether this is a high-priority event (for replay buffer)
            session_id: Optional session ID for the event
        """
        if not self.running:
            logger.debug(f"Event {event_type} dropped (server not running)")
            return

        if not self.clients:
            logger.debug(f"Event {event_type} dropped (no clients connected)")
            return

        # Create the event
        self.sequence_number += 1
        try:
            event = create_event(
                event_type=event_type, seq=self.sequence_number, session_id=session_id, **data
            )
            event_dict = event.model_dump()
        except Exception as e:
            logger.error(f"Error creating event {event_type}: {e}")
            return

        # Add to replay buffer if high priority
        if high_priority:
            self.replay_buffer.append(event_dict)
            # Trim buffer if needed
            if len(self.replay_buffer) > self.max_replay_events:
                self.replay_buffer = self.replay_buffer[-self.max_replay_events:]

        # Broadcast to all clients
        if self.clients:
            await asyncio.gather(
                *[
                    self._send_to_client(websocket, event_dict)
                    for websocket in self.clients.values()
                ],
                return_exceptions=True,
            )

        logger.debug(f"Broadcasted event {event_type} to {len(self.clients)} clients")

    async def send_to_client(
        self,
        client_id: str,
        event_type: EventType,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Send an event to a specific client.

        Args:
            client_id: The client ID to send to
            event_type: The type of event
            data: The event data
            session_id: Optional session ID for the event

        Returns:
            True if sent successfully, False otherwise
        """
        if client_id not in self.clients:
            logger.warning(f"Client {client_id} not found")
            return False

        # Create the event
        self.sequence_number += 1
        try:
            event = create_event(
                event_type=event_type, seq=self.sequence_number, session_id=session_id, **data
            )
            event_dict = event.model_dump()
        except Exception as e:
            logger.error(f"Error creating event {event_type}: {e}")
            return False

        # Send to specific client
        websocket = self.clients[client_id]
        await self._send_to_client(websocket, event_dict)
        return True

    def add_connect_handler(self, handler: Callable[[str], None]) -> None:
        """Add a handler for client connections."""
        self.on_connect_handlers.append(handler)

    def add_disconnect_handler(self, handler: Callable[[str], None]) -> None:
        """Add a handler for client disconnections."""
        self.on_disconnect_handlers.append(handler)

    def add_message_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a handler for client messages."""
        self.on_message_handlers.append(handler)

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            **self.stats,
            "replay_buffer_size": len(self.replay_buffer),
            "running": self.running,
        }

    def get_connected_clients(self) -> List[str]:
        """Get list of connected client IDs."""
        return list(self.clients.keys())

    async def wait_for_clients(self, min_clients: int = 1, timeout: float = 30.0) -> bool:
        """
        Wait for a minimum number of clients to connect.

        Args:
            min_clients: Minimum number of clients to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            True if minimum clients connected, False if timeout
        """
        start_time = time.time()

        while len(self.clients) < min_clients:
            if time.time() - start_time > timeout:
                return False
            await asyncio.sleep(0.1)

        return True

    # Convenience methods for common events

    async def broadcast_system_info(self, info: Dict[str, Any]) -> None:
        """Broadcast system information."""
        await self.broadcast_event(EventType.SYSTEM_INFO, {"data": info}, high_priority=True)

    async def broadcast_system_error(
        self, level: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast system error."""
        await self.broadcast_event(
            EventType.SYSTEM_ERROR,
            {"level": level, "message": message, "details": details},
            high_priority=True,
        )

    async def broadcast_system_metrics(
        self,
        memory_mb: float,
        cpu_percent: float,
        gpu_vram_mb: Optional[float] = None,
        uptime_seconds: float = 0,
    ) -> None:
        """Broadcast system metrics."""
        await self.broadcast_event(
            EventType.SYSTEM_METRICS,
            {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "gpu_vram_mb": gpu_vram_mb,
                "uptime_seconds": uptime_seconds,
            },
        )
