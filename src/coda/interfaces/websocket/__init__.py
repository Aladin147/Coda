"""
WebSocket interface for Coda.

This package provides WebSocket functionality for Coda, allowing clients to
connect and receive real-time events about Coda's operation.

Features:
- Real-time event broadcasting
- Client connection management
- Event replay for new clients
- Performance tracking integration
- Type-safe event system
"""

from .events import (
    BaseEvent,
    ComponentErrorEvent,
    ComponentHealthEvent,
    ComponentStatusEvent,
    ComponentTimingEvent,
    ConversationEndEvent,
    ConversationStartEvent,
    ConversationTurnEvent,
    EventType,
    IntegrationMetricsEvent,
    LatencyTraceEvent,
    LLMErrorEvent,
    LLMResultEvent,
    LLMStartEvent,
    LLMTokenEvent,
    MemoryRetrieveEvent,
    MemoryStoreEvent,
    STTErrorEvent,
    STTInterimEvent,
    STTResultEvent,
    STTStartEvent,
    SystemErrorEvent,
    SystemInfoEvent,
    SystemMetricsEvent,
    SystemStatusEvent,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    TTSErrorEvent,
    TTSProgressEvent,
    TTSResultEvent,
    TTSStartEvent,
    create_event,
    validate_event,
)
from .integration import CodaWebSocketIntegration
from .performance import WebSocketPerfIntegration
from .server import CodaWebSocketServer

__all__ = [
    # Core server
    "CodaWebSocketServer",
    # Event types and classes
    "EventType",
    "BaseEvent",
    "SystemInfoEvent",
    "SystemErrorEvent",
    "SystemMetricsEvent",
    "STTStartEvent",
    "STTInterimEvent",
    "STTResultEvent",
    "STTErrorEvent",
    "LLMStartEvent",
    "LLMTokenEvent",
    "LLMResultEvent",
    "LLMErrorEvent",
    "TTSStartEvent",
    "TTSProgressEvent",
    "TTSResultEvent",
    "TTSErrorEvent",
    "MemoryStoreEvent",
    "MemoryRetrieveEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "ToolErrorEvent",
    "ConversationStartEvent",
    "ConversationTurnEvent",
    "ConversationEndEvent",
    "ComponentStatusEvent",
    "ComponentErrorEvent",
    "ComponentHealthEvent",
    "SystemStatusEvent",
    "IntegrationMetricsEvent",
    "LatencyTraceEvent",
    "ComponentTimingEvent",
    # Utilities
    "create_event",
    "validate_event",
    # Integration
    "CodaWebSocketIntegration",
    "WebSocketPerfIntegration",
]
