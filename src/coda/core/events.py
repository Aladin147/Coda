"""
Event system for Coda.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set


class EventType(Enum):
    """Event types for the system."""

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_STATUS = "system.status"

    # Voice events
    VOICE_INPUT_START = "voice.input.start"
    VOICE_INPUT_END = "voice.input.end"
    VOICE_OUTPUT_START = "voice.output.start"
    VOICE_OUTPUT_END = "voice.output.end"

    # LLM events
    LLM_REQUEST_START = "llm.request.start"
    LLM_REQUEST_END = "llm.request.end"
    LLM_RESPONSE_CHUNK = "llm.response.chunk"
    LLM_GENERATION_START = "llm.generation.start"
    LLM_GENERATION_TOKEN = "llm.generation.token"
    LLM_GENERATION_COMPLETE = "llm.generation.complete"

    # Memory events
    MEMORY_STORE = "memory.store"
    MEMORY_RETRIEVE = "memory.retrieve"
    MEMORY_UPDATE = "memory.update"

    # Tool events
    TOOL_EXECUTION_START = "tool.execution.start"
    TOOL_EXECUTION_END = "tool.execution.end"
    TOOL_EXECUTION_ERROR = "tool.execution.error"
    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_COMPLETE = "tool.call.complete"

    # Personality events
    PERSONALITY_UPDATE = "personality.update"
    PERSONALITY_ADAPTATION = "personality.adaptation"

    # Conversation events
    CONVERSATION_START = "conversation.start"
    CONVERSATION_END = "conversation.end"
    CONVERSATION_TURN = "conversation.turn"


@dataclass
class Event:
    """Event data structure."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    correlation_id: Optional[str] = None


class EventBus:
    """Central event bus for component communication."""

    def __init__(self):
        self.logger = logging.getLogger("coda.events")
        self._subscribers: Dict[EventType, Set[Callable]] = {}
        self._async_subscribers: Dict[EventType, Set[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the event bus."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        self.logger.info("Event bus started")

    async def stop(self):
        """Stop the event bus."""
        if not self._running:
            return

        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Event bus stopped")

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to synchronous events."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)
        self.logger.debug(f"Subscribed to {event_type.value}")

    def subscribe_async(self, event_type: EventType, callback: Callable[[Event], Any]):
        """Subscribe to asynchronous events."""
        if event_type not in self._async_subscribers:
            self._async_subscribers[event_type] = set()
        self._async_subscribers[event_type].add(callback)
        self.logger.debug(f"Async subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from events."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
        if event_type in self._async_subscribers:
            self._async_subscribers[event_type].discard(callback)
        self.logger.debug(f"Unsubscribed from {event_type.value}")

    async def emit(self, event: Event):
        """Emit an event."""
        if not self._running:
            self.logger.warning(f"Event bus not running, dropping event: {event.type.value}")
            return

        await self._event_queue.put(event)

    async def emit_sync(self, event: Event):
        """Emit an event synchronously (for immediate processing)."""
        await self._handle_event(event)

    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")

    async def _handle_event(self, event: Event):
        """Handle a single event."""
        self.logger.debug(f"Processing event: {event.type.value}")

        # Handle synchronous subscribers
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in sync event handler: {e}")

        # Handle asynchronous subscribers
        if event.type in self._async_subscribers:
            tasks = []
            for callback in self._async_subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        tasks.append(asyncio.create_task(callback(event)))
                    else:
                        callback(event)
                except Exception as e:
                    self.logger.error(f"Error creating async event handler task: {e}")

            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    self.logger.error(f"Error in async event handlers: {e}")


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


async def emit_event(
    event_type: EventType,
    data: Dict[str, Any] = None,
    source: str = None,
    correlation_id: str = None,
):
    """Convenience function to emit an event."""
    event = Event(type=event_type, data=data or {}, source=source, correlation_id=correlation_id)
    await get_event_bus().emit(event)
