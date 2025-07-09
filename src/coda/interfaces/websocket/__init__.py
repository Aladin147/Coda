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

from .server import CodaWebSocketServer
from .events import (
    EventType,
    BaseEvent,
    SystemInfoEvent,
    SystemErrorEvent,
    SystemMetricsEvent,
    STTStartEvent,
    STTInterimEvent,
    STTResultEvent,
    STTErrorEvent,
    LLMStartEvent,
    LLMTokenEvent,
    LLMResultEvent,
    LLMErrorEvent,
    TTSStartEvent,
    TTSProgressEvent,
    TTSResultEvent,
    TTSErrorEvent,
    MemoryStoreEvent,
    MemoryRetrieveEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolErrorEvent,
    ConversationStartEvent,
    ConversationTurnEvent,
    ConversationEndEvent,
    LatencyTraceEvent,
    ComponentTimingEvent,
    create_event,
    validate_event,
)
from .integration import CodaWebSocketIntegration
from .performance import WebSocketPerfIntegration

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
    "LatencyTraceEvent",
    "ComponentTimingEvent",
    
    # Utilities
    "create_event",
    "validate_event",
    
    # Integration
    "CodaWebSocketIntegration",
    "WebSocketPerfIntegration",
]
