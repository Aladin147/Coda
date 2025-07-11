"""
WebSocket event broadcasting system for voice processing.

This module provides real-time event broadcasting capabilities,
allowing the voice system to send status updates, progress notifications,
and other events to connected WebSocket clients.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .models import ConversationState, VoiceResponse
from .performance_profiler import get_performance_profiler
from .websocket_handler import ClientConnection, MessageType, VoiceWebSocketHandler

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Voice processing event types."""

    # Processing events
    PROCESSING_STARTED = "processing_started"
    PROCESSING_PROGRESS = "processing_progress"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"

    # Conversation events
    CONVERSATION_CREATED = "conversation_created"
    CONVERSATION_UPDATED = "conversation_updated"
    CONVERSATION_ENDED = "conversation_ended"
    PARTICIPANT_JOINED = "participant_joined"
    PARTICIPANT_LEFT = "participant_left"

    # System events
    SYSTEM_STATUS = "system_status"
    PERFORMANCE_ALERT = "performance_alert"
    ERROR_OCCURRED = "error_occurred"

    # Audio events
    AUDIO_RECEIVED = "audio_received"
    AUDIO_PROCESSED = "audio_processed"
    AUDIO_GENERATED = "audio_generated"

    # Model events
    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"
    MODEL_ERROR = "model_error"


@dataclass
class VoiceEvent:
    """Voice processing event."""

    event_type: EventType
    data: Dict[str, Any]
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: float = None
    event_id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.event_id is None:
            import uuid

            self.event_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)


class VoiceEventBroadcaster:
    """
    Event broadcasting system for voice processing.

    Manages real-time event broadcasting to WebSocket clients,
    with support for filtering, queuing, and delivery guarantees.

    Features:
    - Real-time event broadcasting
    - Event filtering and routing
    - Delivery confirmation
    - Event queuing for offline clients
    - Performance monitoring

    Example:
        >>> broadcaster = VoiceEventBroadcaster(websocket_handler)
        >>> await broadcaster.broadcast_event(
        ...     EventType.PROCESSING_STARTED,
        ...     {"conversation_id": "conv_123"},
        ...     conversation_id="conv_123"
        ... )
    """

    def __init__(
        self,
        websocket_handler: VoiceWebSocketHandler,
        max_queue_size: int = 1000,
        enable_event_history: bool = True,
        history_size: int = 10000,
    ):
        """
        Initialize event broadcaster.

        Args:
            websocket_handler: WebSocket handler for sending events
            max_queue_size: Maximum events to queue per client
            enable_event_history: Whether to maintain event history
            history_size: Maximum events to keep in history
        """
        self.websocket_handler = websocket_handler
        self.max_queue_size = max_queue_size
        self.enable_event_history = enable_event_history
        self.history_size = history_size

        # Event queues for offline clients
        self.client_queues: Dict[str, List[VoiceEvent]] = {}

        # Event history
        self.event_history: List[VoiceEvent] = []

        # Event subscribers
        self.subscribers: Dict[EventType, Set[str]] = {}

        # Performance monitoring
        self.profiler = get_performance_profiler()
        self.metrics = {
            "events_broadcast": 0,
            "events_queued": 0,
            "events_delivered": 0,
            "events_failed": 0,
        }

        logger.info("VoiceEventBroadcaster initialized")

    async def broadcast_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        target_clients: Optional[List[str]] = None,
    ) -> int:
        """
        Broadcast event to clients.

        Args:
            event_type: Type of event to broadcast
            data: Event data
            conversation_id: Optional conversation ID filter
            user_id: Optional user ID filter
            target_clients: Optional list of specific client IDs

        Returns:
            Number of clients event was sent to
        """
        with self.profiler.profile_operation("websocket", "broadcast_event"):
            # Create event
            event = VoiceEvent(
                event_type=event_type, data=data, conversation_id=conversation_id, user_id=user_id
            )

            # Add to history
            if self.enable_event_history:
                self._add_to_history(event)

            # Determine target clients
            if target_clients:
                clients = target_clients
            else:
                clients = self._get_target_clients(event)

            # Broadcast to clients
            sent_count = 0
            for client_id in clients:
                if await self._send_event_to_client(client_id, event):
                    sent_count += 1

            self.metrics["events_broadcast"] += 1
            self.metrics["events_delivered"] += sent_count

            logger.debug(f"Broadcast event {event_type} to {sent_count} clients")
            return sent_count

    async def subscribe_client(self, client_id: str, event_types: List[EventType]) -> None:
        """Subscribe client to specific event types."""
        for event_type in event_types:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = set()
            self.subscribers[event_type].add(client_id)

        logger.info(f"Client {client_id} subscribed to {len(event_types)} event types")

    async def unsubscribe_client(
        self, client_id: str, event_types: Optional[List[EventType]] = None
    ) -> None:
        """Unsubscribe client from event types."""
        if event_types is None:
            # Unsubscribe from all events
            for subscribers in self.subscribers.values():
                subscribers.discard(client_id)
        else:
            for event_type in event_types:
                if event_type in self.subscribers:
                    self.subscribers[event_type].discard(client_id)

        # Clean up empty queues
        if client_id in self.client_queues:
            del self.client_queues[client_id]

        logger.info(f"Client {client_id} unsubscribed from events")

    async def send_queued_events(self, client_id: str) -> int:
        """Send queued events to a client that came online."""
        if client_id not in self.client_queues:
            return 0

        queued_events = self.client_queues[client_id]
        sent_count = 0

        for event in queued_events:
            if await self._send_event_to_client(client_id, event, queue_if_offline=False):
                sent_count += 1

        # Clear queue after sending
        del self.client_queues[client_id]

        logger.info(f"Sent {sent_count} queued events to client {client_id}")
        return sent_count

    def get_event_history(
        self,
        conversation_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        limit: int = 100,
    ) -> List[VoiceEvent]:
        """Get event history with optional filtering."""
        events = self.event_history

        # Apply filters
        if conversation_id:
            events = [e for e in events if e.conversation_id == conversation_id]

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        # Return most recent events
        return events[-limit:] if limit > 0 else events

    async def _send_event_to_client(
        self, client_id: str, event: VoiceEvent, queue_if_offline: bool = True
    ) -> bool:
        """Send event to specific client."""
        try:
            # Check if client is connected
            if client_id not in self.websocket_handler.connections:
                if queue_if_offline:
                    self._queue_event_for_client(client_id, event)
                return False

            connection = self.websocket_handler.connections[client_id]

            # Send event as WebSocket message
            await self.websocket_handler._send_message(
                connection,
                MessageType.STATUS,  # Use STATUS message type for events
                {
                    "event_type": event.event_type.value,
                    "event_id": event.event_id,
                    "data": event.data,
                    "timestamp": event.timestamp,
                },
                event.conversation_id,
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to send event to client {client_id}: {e}")
            self.metrics["events_failed"] += 1
            return False

    def _get_target_clients(self, event: VoiceEvent) -> List[str]:
        """Determine which clients should receive the event."""
        target_clients = set()

        # Get subscribers for this event type
        if event.event_type in self.subscribers:
            target_clients.update(self.subscribers[event.event_type])

        # Add clients in the same conversation
        if event.conversation_id:
            for client_id, connection in self.websocket_handler.connections.items():
                if connection.conversation_id == event.conversation_id:
                    target_clients.add(client_id)

        # Add specific user
        if event.user_id:
            for client_id, connection in self.websocket_handler.connections.items():
                if connection.user_id == event.user_id:
                    target_clients.add(client_id)

        return list(target_clients)

    def _queue_event_for_client(self, client_id: str, event: VoiceEvent) -> None:
        """Queue event for offline client."""
        if client_id not in self.client_queues:
            self.client_queues[client_id] = []

        queue = self.client_queues[client_id]
        queue.append(event)

        # Limit queue size
        if len(queue) > self.max_queue_size:
            queue.pop(0)  # Remove oldest event

        self.metrics["events_queued"] += 1

    def _add_to_history(self, event: VoiceEvent) -> None:
        """Add event to history."""
        self.event_history.append(event)

        # Limit history size
        if len(self.event_history) > self.history_size:
            self.event_history.pop(0)  # Remove oldest event

    def get_metrics(self) -> Dict[str, Any]:
        """Get broadcaster metrics."""
        return {
            **self.metrics,
            "active_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "queued_events": sum(len(queue) for queue in self.client_queues.values()),
            "history_size": len(self.event_history),
        }


# Convenience functions for common events
async def broadcast_processing_started(
    broadcaster: VoiceEventBroadcaster, conversation_id: str, processing_mode: str
) -> None:
    """Broadcast processing started event."""
    await broadcaster.broadcast_event(
        EventType.PROCESSING_STARTED,
        {"processing_mode": processing_mode, "started_at": time.time()},
        conversation_id=conversation_id,
    )


async def broadcast_processing_completed(
    broadcaster: VoiceEventBroadcaster, conversation_id: str, response: VoiceResponse
) -> None:
    """Broadcast processing completed event."""
    await broadcaster.broadcast_event(
        EventType.PROCESSING_COMPLETED,
        {
            "response_id": getattr(response, "response_id", None),
            "processing_time": response.processing_time,
            "confidence_score": response.confidence_score,
            "completed_at": time.time(),
        },
        conversation_id=conversation_id,
    )


async def broadcast_error_occurred(
    broadcaster: VoiceEventBroadcaster,
    conversation_id: Optional[str],
    error_type: str,
    error_message: str,
) -> None:
    """Broadcast error occurred event."""
    await broadcaster.broadcast_event(
        EventType.ERROR_OCCURRED,
        {"error_type": error_type, "error_message": error_message, "occurred_at": time.time()},
        conversation_id=conversation_id,
    )
