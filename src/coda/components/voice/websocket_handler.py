"""
WebSocket handler for real-time voice streaming.

This module provides WebSocket-based real-time voice communication,
handling bidirectional audio streaming, connection management, and
real-time voice processing coordination.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed, WebSocketException

from .exceptions import (
    ErrorCodes,
    ValidationError,
    VoiceProcessingError,
    VoiceTimeoutError,
    WebSocketError,
)
from .manager import VoiceManager
from .models import (
    ConversationState,
    VoiceMessage,
    VoiceProcessingMode,
    VoiceResponse,
    VoiceStreamChunk,
)
from .performance_profiler import get_performance_profiler

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types for voice communication."""

    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"

    # Authentication
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"

    # Voice processing
    VOICE_START = "voice_start"
    VOICE_CHUNK = "voice_chunk"
    VOICE_END = "voice_end"
    VOICE_RESPONSE = "voice_response"
    VOICE_STREAM_CHUNK = "voice_stream_chunk"

    # Conversation management
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"
    CONVERSATION_STATE = "conversation_state"

    # Status and monitoring
    STATUS = "status"
    ERROR = "error"
    METRICS = "metrics"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""

    type: MessageType
    data: Dict[str, Any]
    message_id: str = None
    timestamp: float = None
    conversation_id: Optional[str] = None

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            data=data["data"],
            message_id=data.get("message_id"),
            timestamp=data.get("timestamp"),
            conversation_id=data.get("conversation_id"),
        )


@dataclass
class ClientConnection:
    """WebSocket client connection information."""

    websocket: ServerConnection
    client_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    connected_at: float = None
    last_activity: float = None
    is_authenticated: bool = False
    processing_mode: VoiceProcessingMode = VoiceProcessingMode.ADAPTIVE

    def __post_init__(self):
        if self.connected_at is None:
            self.connected_at = time.time()
        if self.last_activity is None:
            self.last_activity = time.time()

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()


class VoiceWebSocketHandler:
    """
    WebSocket handler for real-time voice streaming.

    Manages WebSocket connections, handles real-time voice processing,
    and coordinates bidirectional audio streaming with the voice system.

    Features:
    - Real-time bidirectional audio streaming
    - Connection lifecycle management
    - Authentication and authorization
    - Voice processing coordination
    - Performance monitoring and metrics
    - Error handling and recovery

    Example:
        >>> handler = VoiceWebSocketHandler(voice_manager, port=8765)
        >>> await handler.start_server()
    """

    def __init__(
        self,
        voice_manager: VoiceManager,
        host: str = "localhost",
        port: int = 8765,
        max_connections: int = 100,
        auth_required: bool = True,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
    ):
        """
        Initialize WebSocket voice handler.

        Args:
            voice_manager: Voice processing manager
            host: WebSocket server host
            port: WebSocket server port
            max_connections: Maximum concurrent connections
            auth_required: Whether authentication is required
            ping_interval: Ping interval in seconds
            ping_timeout: Ping timeout in seconds
        """
        self.voice_manager = voice_manager
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.auth_required = auth_required
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # Connection management
        self.connections: Dict[str, ClientConnection] = {}
        self.server: Optional[websockets.WebSocketServer] = None
        self.is_running = False

        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.CONNECT: self._handle_connect,
            MessageType.DISCONNECT: self._handle_disconnect,
            MessageType.PING: self._handle_ping,
            MessageType.AUTH: self._handle_auth,
            MessageType.VOICE_START: self._handle_voice_start,
            MessageType.VOICE_CHUNK: self._handle_voice_chunk,
            MessageType.VOICE_END: self._handle_voice_end,
            MessageType.CONVERSATION_START: self._handle_conversation_start,
            MessageType.CONVERSATION_END: self._handle_conversation_end,
        }

        # Performance monitoring
        self.profiler = get_performance_profiler()
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_processed": 0,
            "voice_sessions": 0,
            "errors": 0,
        }

        logger.info(f"VoiceWebSocketHandler initialized on {host}:{port}")

    async def start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                max_size=10 * 1024 * 1024,  # 10MB max message size
                compression=None,  # Disable compression for real-time audio
            )

            self.is_running = True
            logger.info(f"WebSocket server started on {self.host}:{self.port}")

            # Start background tasks
            asyncio.create_task(self._connection_monitor())
            asyncio.create_task(self._metrics_reporter())

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise WebSocketError(f"Server startup failed: {e}")

    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        self.is_running = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

        # Close all connections
        for connection in list(self.connections.values()):
            await self._close_connection(connection.client_id, "Server shutdown")

    async def _handle_client(self, websocket: ServerConnection, path: str) -> None:
        """Handle new WebSocket client connection."""
        client_id = str(uuid.uuid4())

        # Check connection limit
        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Server overloaded")
            return

        # Create connection
        connection = ClientConnection(websocket=websocket, client_id=client_id)

        self.connections[client_id] = connection
        self.metrics["total_connections"] += 1
        self.metrics["active_connections"] = len(self.connections)

        logger.info(f"New WebSocket connection: {client_id}")

        try:
            # Send connection confirmation
            await self._send_message(
                connection,
                MessageType.CONNECT,
                {"client_id": client_id, "server_time": time.time()},
            )

            # Handle messages
            async for message in websocket:
                await self._process_message(connection, message)

        except ConnectionClosed:
            logger.info(f"WebSocket connection closed: {client_id}")
        except WebSocketException as e:
            logger.warning(f"WebSocket error for {client_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {client_id}: {e}")
            await self._send_error(connection, f"Internal server error: {e}")
        finally:
            await self._close_connection(client_id, "Connection ended")

    async def _process_message(self, connection: ClientConnection, raw_message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            with self.profiler.profile_operation("websocket", "process_message"):
                connection.update_activity()

                # Parse message
                message = WebSocketMessage.from_json(raw_message)

                # Check authentication if required
                if (
                    self.auth_required
                    and not connection.is_authenticated
                    and message.type != MessageType.AUTH
                ):
                    await self._send_error(connection, "Authentication required")
                    return

                # Handle message
                handler = self.message_handlers.get(message.type)
                if handler:
                    await handler(connection, message)
                else:
                    await self._send_error(connection, f"Unknown message type: {message.type}")

                self.metrics["messages_processed"] += 1

        except json.JSONDecodeError:
            await self._send_error(connection, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._send_error(connection, f"Message processing error: {e}")
            self.metrics["errors"] += 1

    async def _send_message(
        self,
        connection: ClientConnection,
        message_type: MessageType,
        data: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> None:
        """Send message to WebSocket client."""
        try:
            message = WebSocketMessage(
                type=message_type,
                data=data,
                conversation_id=conversation_id or connection.conversation_id,
            )

            await connection.websocket.send(message.to_json())

        except ConnectionClosed:
            logger.warning(f"Cannot send message to closed connection: {connection.client_id}")
        except Exception as e:
            logger.error(f"Error sending message to {connection.client_id}: {e}")

    async def _send_error(self, connection: ClientConnection, error_message: str) -> None:
        """Send error message to client."""
        await self._send_message(
            connection, MessageType.ERROR, {"error": error_message, "timestamp": time.time()}
        )

    async def _close_connection(self, client_id: str, reason: str = "Unknown") -> None:
        """Close WebSocket connection."""
        if client_id in self.connections:
            connection = self.connections[client_id]

            # End any active conversation
            if connection.conversation_id:
                try:
                    await self.voice_manager.end_conversation(connection.conversation_id)
                except Exception as e:
                    logger.warning(f"Error ending conversation {connection.conversation_id}: {e}")

            # Close WebSocket
            try:
                await connection.websocket.close(reason=reason)
            except Exception:
                pass  # Connection may already be closed

            # Remove from connections
            del self.connections[client_id]
            self.metrics["active_connections"] = len(self.connections)

            logger.info(f"Connection closed: {client_id} - {reason}")

    # Message handlers
    async def _handle_connect(
        self, connection: ClientConnection, message: WebSocketMessage
    ) -> None:
        """Handle connection message."""
        # Connection already handled in _handle_client
        pass

    async def _handle_disconnect(
        self, connection: ClientConnection, message: WebSocketMessage
    ) -> None:
        """Handle disconnect message."""
        await self._close_connection(connection.client_id, "Client requested disconnect")

    async def _handle_ping(self, connection: ClientConnection, message: WebSocketMessage) -> None:
        """Handle ping message."""
        await self._send_message(connection, MessageType.PONG, {"timestamp": time.time()})

    async def _handle_auth(self, connection: ClientConnection, message: WebSocketMessage) -> None:
        """Handle authentication message."""
        try:
            # Extract auth data
            auth_data = message.data
            user_id = auth_data.get("user_id")
            token = auth_data.get("token")

            # Validate authentication (implement your auth logic here)
            if await self._validate_auth(user_id, token):
                connection.is_authenticated = True
                connection.user_id = user_id

                await self._send_message(
                    connection,
                    MessageType.AUTH_SUCCESS,
                    {"user_id": user_id, "authenticated_at": time.time()},
                )
                logger.info(f"Client authenticated: {connection.client_id} as {user_id}")
            else:
                await self._send_message(
                    connection, MessageType.AUTH_FAILED, {"error": "Invalid credentials"}
                )

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self._send_error(connection, "Authentication failed")

    async def _handle_conversation_start(
        self, connection: ClientConnection, message: WebSocketMessage
    ) -> None:
        """Handle conversation start message."""
        try:
            conversation_id = message.data.get("conversation_id")
            if not conversation_id:
                conversation_id = f"ws_{connection.client_id}_{int(time.time())}"

            # Start conversation in voice manager
            conversation_state = await self.voice_manager.start_conversation(conversation_id)

            connection.conversation_id = conversation_id

            await self._send_message(
                connection,
                MessageType.CONVERSATION_START,
                {"conversation_id": conversation_id, "state": asdict(conversation_state)},
                conversation_id,
            )

            logger.info(f"Started conversation: {conversation_id} for {connection.client_id}")

        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            await self._send_error(connection, f"Failed to start conversation: {e}")

    async def _handle_conversation_end(
        self, connection: ClientConnection, message: WebSocketMessage
    ) -> None:
        """Handle conversation end message."""
        try:
            if connection.conversation_id:
                await self.voice_manager.end_conversation(connection.conversation_id)

                await self._send_message(
                    connection,
                    MessageType.CONVERSATION_END,
                    {"conversation_id": connection.conversation_id},
                    connection.conversation_id,
                )

                logger.info(f"Ended conversation: {connection.conversation_id}")
                connection.conversation_id = None

        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
            await self._send_error(connection, f"Failed to end conversation: {e}")

    async def _handle_voice_start(
        self, connection: ClientConnection, message: WebSocketMessage
    ) -> None:
        """Handle voice processing start message."""
        try:
            if not connection.conversation_id:
                await self._send_error(connection, "No active conversation")
                return

            # Extract voice processing parameters
            data = message.data
            processing_mode = VoiceProcessingMode(data.get("mode", "adaptive"))
            connection.processing_mode = processing_mode

            # Send acknowledgment
            await self._send_message(
                connection,
                MessageType.VOICE_START,
                {
                    "conversation_id": connection.conversation_id,
                    "mode": processing_mode.value,
                    "ready": True,
                },
                connection.conversation_id,
            )

            self.metrics["voice_sessions"] += 1
            logger.info(f"Voice processing started for {connection.conversation_id}")

        except Exception as e:
            logger.error(f"Error starting voice processing: {e}")
            await self._send_error(connection, f"Failed to start voice processing: {e}")

    async def _handle_voice_chunk(
        self, connection: ClientConnection, message: WebSocketMessage
    ) -> None:
        """Handle voice audio chunk message."""
        try:
            if not connection.conversation_id:
                await self._send_error(connection, "No active conversation")
                return

            # Extract audio data
            data = message.data
            audio_data = bytes.fromhex(data.get("audio_data", ""))
            text_content = data.get("text_content")
            chunk_id = data.get("chunk_id")
            is_final = data.get("is_final", False)

            if not audio_data and not text_content:
                await self._send_error(connection, "No audio or text data provided")
                return

            # Create voice message
            voice_message = VoiceMessage(
                conversation_id=connection.conversation_id,
                audio_data=audio_data,
                text_content=text_content,
                speaker="user",
                processing_mode=connection.processing_mode,
            )

            # Process voice message
            try:
                if is_final:
                    # Process complete message
                    response = await self.voice_manager.process_voice_input(
                        conversation_id=connection.conversation_id, audio_data=audio_data
                    )

                    # Send response
                    await self._send_voice_response(connection, response)
                else:
                    # Process streaming chunk
                    async for stream_chunk in self.voice_manager.process_voice_stream(
                        connection.conversation_id, self._audio_chunk_generator(audio_data)
                    ):
                        await self._send_stream_chunk(connection, stream_chunk)
            except VoiceProcessingError as e:
                logger.error(f"Voice processing error for {connection.client_id}: {e}")
                await self._send_error(connection, f"Voice processing failed: {e}")
            except VoiceTimeoutError as e:
                logger.error(f"Voice processing timeout for {connection.client_id}: {e}")
                await self._send_error(connection, "Voice processing timeout")
            except Exception as e:
                logger.error(f"Unexpected voice processing error for {connection.client_id}: {e}")
                await self._send_error(connection, "Internal processing error")

        except Exception as e:
            logger.error(f"Error processing voice chunk: {e}")
            await self._send_error(connection, f"Voice processing failed: {e}")

    async def _handle_voice_end(
        self, connection: ClientConnection, message: WebSocketMessage
    ) -> None:
        """Handle voice processing end message."""
        try:
            await self._send_message(
                connection,
                MessageType.VOICE_END,
                {"conversation_id": connection.conversation_id, "ended_at": time.time()},
                connection.conversation_id,
            )

            logger.info(f"Voice processing ended for {connection.conversation_id}")

        except Exception as e:
            logger.error(f"Error ending voice processing: {e}")
            await self._send_error(connection, f"Failed to end voice processing: {e}")

    async def _send_voice_response(
        self, connection: ClientConnection, response: VoiceResponse
    ) -> None:
        """Send voice response to client."""
        response_data = {
            "conversation_id": response.conversation_id,
            "text_content": response.text_content,
            "processing_time": response.processing_time,
            "confidence_score": response.confidence_score,
            "timestamp": response.timestamp,
        }

        # Include audio data if available
        if response.audio_data:
            response_data["audio_data"] = response.audio_data.hex()

        await self._send_message(
            connection, MessageType.VOICE_RESPONSE, response_data, response.conversation_id
        )

    async def _send_stream_chunk(
        self, connection: ClientConnection, chunk: VoiceStreamChunk
    ) -> None:
        """Send voice stream chunk to client."""
        chunk_data = {
            "conversation_id": chunk.conversation_id,
            "text_content": chunk.text_content,
            "is_complete": chunk.is_complete,
            "chunk_index": chunk.chunk_index,
            "timestamp": chunk.timestamp,
        }

        # Include audio data if available
        if chunk.audio_data:
            chunk_data["audio_data"] = chunk.audio_data.hex()

        await self._send_message(
            connection, MessageType.VOICE_STREAM_CHUNK, chunk_data, chunk.conversation_id
        )

    # Utility methods
    async def _validate_auth(self, user_id: str, token: str) -> bool:
        """
        Validate user authentication.

        Args:
            user_id: User identifier
            token: Authentication token

        Returns:
            True if authentication is valid
        """
        # Implement your authentication logic here
        # For now, accept any non-empty user_id and token
        return bool(user_id and token)

    async def _audio_chunk_generator(self, audio_data: bytes):
        """Generate audio chunks for streaming processing."""
        chunk_size = 1024  # 1KB chunks
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i : i + chunk_size]

    async def _connection_monitor(self) -> None:
        """Monitor WebSocket connections and clean up inactive ones."""
        while self.is_running:
            try:
                current_time = time.time()
                inactive_connections = []

                for client_id, connection in self.connections.items():
                    # Check for inactive connections (no activity for 5 minutes)
                    if current_time - connection.last_activity > 300:
                        inactive_connections.append(client_id)

                # Close inactive connections
                for client_id in inactive_connections:
                    await self._close_connection(client_id, "Inactive connection")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Connection monitor error: {e}")
                await asyncio.sleep(60)

    async def _metrics_reporter(self) -> None:
        """Report metrics periodically."""
        while self.is_running:
            try:
                # Update metrics
                self.metrics["active_connections"] = len(self.connections)

                # Log metrics every 5 minutes
                logger.info(f"WebSocket metrics: {self.metrics}")

                # Send metrics to profiler
                self.profiler.record_metric(
                    name="active_websocket_connections",
                    value=self.metrics["active_connections"],
                    unit="count",
                    component="websocket",
                    operation="monitoring",
                )

                await asyncio.sleep(300)  # Report every 5 minutes

            except Exception as e:
                logger.error(f"Metrics reporter error: {e}")
                await asyncio.sleep(300)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics."""
        return {
            "total_connections": self.metrics["total_connections"],
            "active_connections": self.metrics["active_connections"],
            "messages_processed": self.metrics["messages_processed"],
            "voice_sessions": self.metrics["voice_sessions"],
            "errors": self.metrics["errors"],
            "server_uptime": time.time() - (self.metrics.get("start_time", time.time())),
            "connections_by_status": {
                "authenticated": sum(1 for c in self.connections.values() if c.is_authenticated),
                "with_conversation": sum(1 for c in self.connections.values() if c.conversation_id),
            },
        }

    async def broadcast_message(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        conversation_id: Optional[str] = None,
        user_filter: Optional[Callable[[ClientConnection], bool]] = None,
    ) -> int:
        """
        Broadcast message to multiple clients.

        Args:
            message_type: Type of message to broadcast
            data: Message data
            conversation_id: Optional conversation ID filter
            user_filter: Optional function to filter users

        Returns:
            Number of clients message was sent to
        """
        sent_count = 0

        for connection in list(self.connections.values()):
            try:
                # Apply filters
                if conversation_id and connection.conversation_id != conversation_id:
                    continue

                if user_filter and not user_filter(connection):
                    continue

                await self._send_message(connection, message_type, data, conversation_id)
                sent_count += 1

            except Exception as e:
                logger.warning(f"Failed to broadcast to {connection.client_id}: {e}")

        return sent_count

    async def get_conversation_participants(self, conversation_id: str) -> List[str]:
        """Get list of client IDs participating in a conversation."""
        return [
            conn.client_id
            for conn in self.connections.values()
            if conn.conversation_id == conversation_id
        ]
