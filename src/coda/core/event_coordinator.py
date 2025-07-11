"""
Event Coordination System for Coda.

This module provides centralized event coordination that bridges component events
with GUI clients, WebSocket broadcasting, and real-time updates.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .events import EventType, emit_event, get_event_bus

logger = logging.getLogger("coda.event_coordinator")


class EventCoordinator:
    """
    Centralized event coordinator that manages event flow between components and GUI.

    Features:
    - Event aggregation from all components
    - Real-time broadcasting to GUI clients
    - Event filtering and routing
    - Event history and replay
    - Performance monitoring
    """

    def __init__(self, max_event_history: int = 1000):
        """
        Initialize event coordinator.

        Args:
            max_event_history: Maximum events to keep in history
        """
        self.max_event_history = max_event_history
        self.event_bus = get_event_bus()

        # Event storage and management
        self.event_history: List[Dict[str, Any]] = []
        self.event_subscribers: Dict[str, List[Callable]] = {}
        self.websocket_clients: List[Any] = []

        # Event statistics
        self.event_stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_component": {},
            "last_event_time": None,
        }

        logger.info("EventCoordinator initialized")

    async def initialize(self):
        """Initialize event coordinator and set up event listeners."""
        # Subscribe to all event types for coordination
        await self._setup_event_listeners()
        logger.info("EventCoordinator initialized with event listeners")

    async def _setup_event_listeners(self):
        """Set up listeners for all event types."""
        # Subscribe to system events (subscribe is not async)
        self.event_bus.subscribe_async(EventType.SYSTEM_STARTUP, self._handle_system_event)
        self.event_bus.subscribe_async(EventType.SYSTEM_SHUTDOWN, self._handle_system_event)
        self.event_bus.subscribe_async(EventType.SYSTEM_ERROR, self._handle_system_event)

        # Subscribe to conversation events
        self.event_bus.subscribe_async(
            EventType.CONVERSATION_START, self._handle_conversation_event
        )
        self.event_bus.subscribe_async(EventType.CONVERSATION_END, self._handle_conversation_event)
        self.event_bus.subscribe_async(EventType.CONVERSATION_TURN, self._handle_conversation_event)

        # Subscribe to component events
        self.event_bus.subscribe_async(EventType.LLM_GENERATION_START, self._handle_component_event)
        self.event_bus.subscribe_async(EventType.LLM_GENERATION_TOKEN, self._handle_component_event)
        self.event_bus.subscribe_async(
            EventType.LLM_GENERATION_COMPLETE, self._handle_component_event
        )
        self.event_bus.subscribe_async(EventType.MEMORY_STORE, self._handle_component_event)
        self.event_bus.subscribe_async(EventType.MEMORY_RETRIEVE, self._handle_component_event)
        self.event_bus.subscribe_async(EventType.TOOL_CALL_START, self._handle_component_event)
        self.event_bus.subscribe_async(EventType.TOOL_CALL_COMPLETE, self._handle_component_event)

    async def _handle_system_event(self, event):
        """Handle system-level events."""
        await self._process_event("system", event.type, event.data)

    async def _handle_conversation_event(self, event):
        """Handle conversation-related events."""
        await self._process_event("conversation", event.type, event.data)

    async def _handle_component_event(self, event):
        """Handle component-specific events."""
        component = event.data.get("component", "unknown")
        await self._process_event(component, event.type, event.data)

    async def _process_event(self, category: str, event_type: EventType, data: Dict[str, Any]):
        """Process and coordinate an event."""
        # Create standardized event structure
        event = {
            "id": f"event_{self.event_stats['total_events']}",
            "category": category,
            "type": event_type.value,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "sequence": self.event_stats["total_events"],
        }

        # Update statistics
        self._update_event_stats(event)

        # Store in history
        self._store_event(event)

        # Broadcast to subscribers
        await self._broadcast_event(event)

        # Log important events
        if event_type in [
            EventType.SYSTEM_ERROR,
            EventType.CONVERSATION_START,
            EventType.CONVERSATION_END,
        ]:
            logger.info(f"Event: {event_type.value} - {category}")

    def _update_event_stats(self, event: Dict[str, Any]):
        """Update event statistics."""
        self.event_stats["total_events"] += 1
        self.event_stats["last_event_time"] = event["timestamp"]

        # Count by type
        event_type = event["type"]
        if event_type not in self.event_stats["events_by_type"]:
            self.event_stats["events_by_type"][event_type] = 0
        self.event_stats["events_by_type"][event_type] += 1

        # Count by component
        category = event["category"]
        if category not in self.event_stats["events_by_component"]:
            self.event_stats["events_by_component"][category] = 0
        self.event_stats["events_by_component"][category] += 1

    def _store_event(self, event: Dict[str, Any]):
        """Store event in history with size limit."""
        self.event_history.append(event)

        # Maintain history size limit
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history :]

    async def _broadcast_event(self, event: Dict[str, Any]):
        """Broadcast event to all subscribers."""
        # Broadcast to custom subscribers
        event_type = event["type"]
        if event_type in self.event_subscribers:
            for callback in self.event_subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber callback: {e}")

        # Broadcast to WebSocket clients
        await self._broadcast_to_websocket_clients(event)

    async def _broadcast_to_websocket_clients(self, event: Dict[str, Any]):
        """Broadcast event to WebSocket clients."""
        if not self.websocket_clients:
            return

        # Create WebSocket message
        message = {"type": "event", "event": event}

        # Send to all connected clients
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                if hasattr(client, "send"):
                    await client.send(json.dumps(message))
                elif hasattr(client, "broadcast_event"):
                    await client.broadcast_event(event["type"], event["data"])
            except Exception as e:
                logger.warning(f"Failed to send event to WebSocket client: {e}")
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.remove(client)

    def subscribe_to_events(self, event_type: str, callback: Callable):
        """Subscribe to specific event types."""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        self.event_subscribers[event_type].append(callback)

    def unsubscribe_from_events(self, event_type: str, callback: Callable):
        """Unsubscribe from specific event types."""
        if event_type in self.event_subscribers:
            try:
                self.event_subscribers[event_type].remove(callback)
            except ValueError:
                pass

    def add_websocket_client(self, client):
        """Add a WebSocket client for event broadcasting."""
        if client not in self.websocket_clients:
            self.websocket_clients.append(client)
            logger.info(f"Added WebSocket client for event broadcasting")

    def remove_websocket_client(self, client):
        """Remove a WebSocket client."""
        if client in self.websocket_clients:
            self.websocket_clients.remove(client)
            logger.info(f"Removed WebSocket client from event broadcasting")

    def get_event_history(
        self, limit: Optional[int] = None, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get event history with optional filtering.

        Args:
            limit: Maximum number of events to return
            event_type: Filter by specific event type

        Returns:
            List of events
        """
        events = self.event_history

        # Filter by event type if specified
        if event_type:
            events = [e for e in events if e["type"] == event_type]

        # Apply limit
        if limit:
            events = events[-limit:]

        return events

    def get_event_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        return self.event_stats.copy()

    async def emit_gui_event(self, event_type: str, data: Dict[str, Any]):
        """Emit a GUI-specific event."""
        await emit_event(
            EventType.SYSTEM_STATUS,
            {"gui_event_type": event_type, "gui_data": data, "component": "gui"},
        )
