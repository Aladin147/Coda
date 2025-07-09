"""
Event system for Coda WebSocket interface.

This module defines all event types and data structures used in the WebSocket
communication between Coda and its clients.
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event types for Coda WebSocket events."""

    # System events
    SYSTEM_INFO = "system_info"
    SYSTEM_ERROR = "system_error"
    SYSTEM_METRICS = "system_metrics"

    # STT events
    STT_START = "stt_start"
    STT_INTERIM = "stt_interim"
    STT_RESULT = "stt_result"
    STT_ERROR = "stt_error"

    # LLM events
    LLM_START = "llm_start"
    LLM_TOKEN = "llm_token"
    LLM_RESULT = "llm_result"
    LLM_ERROR = "llm_error"

    # TTS events
    TTS_START = "tts_start"
    TTS_PROGRESS = "tts_progress"
    TTS_RESULT = "tts_result"
    TTS_ERROR = "tts_error"
    TTS_STATUS = "tts_status"

    # Memory events
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_UPDATE = "memory_update"

    # Tool events
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Conversation events
    CONVERSATION_START = "conversation_start"
    CONVERSATION_TURN = "conversation_turn"
    CONVERSATION_END = "conversation_end"

    # Performance events
    LATENCY_TRACE = "latency_trace"
    COMPONENT_TIMING = "component_timing"
    COMPONENT_STATS = "component_stats"

    # Replay events
    REPLAY = "replay"


class BaseEvent(BaseModel):
    """Base model for all events."""

    version: str = "1.0"
    seq: int
    timestamp: float = Field(default_factory=time.time)
    type: EventType
    session_id: Optional[str] = None


# System Events
class SystemInfoEvent(BaseEvent):
    """System information event."""
    
    type: EventType = EventType.SYSTEM_INFO
    data: Dict[str, Any] = Field(default_factory=dict)


class SystemErrorEvent(BaseEvent):
    """System error event."""
    
    type: EventType = EventType.SYSTEM_ERROR
    level: str  # "warning", "error", "critical"
    message: str
    details: Optional[Dict[str, Any]] = None


class SystemMetricsEvent(BaseEvent):
    """System metrics event."""
    
    type: EventType = EventType.SYSTEM_METRICS
    memory_mb: float
    cpu_percent: float
    gpu_vram_mb: Optional[float] = None
    uptime_seconds: float


# STT Events
class STTStartEvent(BaseEvent):
    """STT start event."""
    
    type: EventType = EventType.STT_START
    mode: str  # "push_to_talk", "continuous", "file"


class STTInterimEvent(BaseEvent):
    """STT interim result event."""
    
    type: EventType = EventType.STT_INTERIM
    text: str
    confidence: float


class STTResultEvent(BaseEvent):
    """STT final result event."""
    
    type: EventType = EventType.STT_RESULT
    text: str
    confidence: float
    duration_ms: float
    language: Optional[str] = None


class STTErrorEvent(BaseEvent):
    """STT error event."""
    
    type: EventType = EventType.STT_ERROR
    error: str
    details: Optional[Dict[str, Any]] = None


# LLM Events
class LLMStartEvent(BaseEvent):
    """LLM processing start event."""
    
    type: EventType = EventType.LLM_START
    prompt: str
    model: str
    temperature: float


class LLMTokenEvent(BaseEvent):
    """LLM token generation event."""
    
    type: EventType = EventType.LLM_TOKEN
    token: str
    cumulative_text: str


class LLMResultEvent(BaseEvent):
    """LLM final result event."""
    
    type: EventType = EventType.LLM_RESULT
    text: str
    duration_ms: float
    token_count: int
    tokens_per_second: float


class LLMErrorEvent(BaseEvent):
    """LLM error event."""
    
    type: EventType = EventType.LLM_ERROR
    error: str
    details: Optional[Dict[str, Any]] = None


# TTS Events
class TTSStartEvent(BaseEvent):
    """TTS start event."""
    
    type: EventType = EventType.TTS_START
    text: str
    voice_id: str
    engine: str


class TTSProgressEvent(BaseEvent):
    """TTS progress event."""
    
    type: EventType = EventType.TTS_PROGRESS
    progress_percent: float
    estimated_duration_ms: Optional[float] = None


class TTSResultEvent(BaseEvent):
    """TTS result event."""
    
    type: EventType = EventType.TTS_RESULT
    duration_ms: float
    audio_duration_ms: float
    success: bool


class TTSErrorEvent(BaseEvent):
    """TTS error event."""
    
    type: EventType = EventType.TTS_ERROR
    error: str
    details: Optional[Dict[str, Any]] = None


class TTSStatusEvent(BaseEvent):
    """TTS status event."""
    
    type: EventType = EventType.TTS_STATUS
    status: str  # "idle", "processing", "playing"


# Memory Events
class MemoryStoreEvent(BaseEvent):
    """Memory store event."""
    
    type: EventType = EventType.MEMORY_STORE
    content_preview: str
    memory_type: str
    importance: float
    memory_id: str


class MemoryRetrieveEvent(BaseEvent):
    """Memory retrieve event."""
    
    type: EventType = EventType.MEMORY_RETRIEVE
    query: str
    results_count: int
    relevance_scores: List[float]


# Tool Events
class ToolCallEvent(BaseEvent):
    """Tool call event."""
    
    type: EventType = EventType.TOOL_CALL
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str


class ToolResultEvent(BaseEvent):
    """Tool result event."""
    
    type: EventType = EventType.TOOL_RESULT
    tool_name: str
    call_id: str
    result: Any
    duration_ms: float


class ToolErrorEvent(BaseEvent):
    """Tool error event."""
    
    type: EventType = EventType.TOOL_ERROR
    tool_name: str
    call_id: str
    error: str
    details: Optional[Dict[str, Any]] = None


# Conversation Events
class ConversationStartEvent(BaseEvent):
    """Conversation start event."""
    
    type: EventType = EventType.CONVERSATION_START
    conversation_id: str


class ConversationTurnEvent(BaseEvent):
    """Conversation turn event."""
    
    type: EventType = EventType.CONVERSATION_TURN
    conversation_id: str
    turn_number: int
    user_input: str
    assistant_response: str


class ConversationEndEvent(BaseEvent):
    """Conversation end event."""
    
    type: EventType = EventType.CONVERSATION_END
    conversation_id: str
    total_turns: int
    duration_seconds: float


# Performance Events
class LatencyTraceEvent(BaseEvent):
    """Latency trace event."""
    
    type: EventType = EventType.LATENCY_TRACE
    component: str
    operation: str
    duration_ms: float
    metadata: Optional[Dict[str, Any]] = None


class ComponentTimingEvent(BaseEvent):
    """Component timing event."""
    
    type: EventType = EventType.COMPONENT_TIMING
    component: str
    timings: Dict[str, float]  # operation -> duration_ms


# Event class mapping for validation
EVENT_CLASS_MAP = {
    EventType.SYSTEM_INFO: SystemInfoEvent,
    EventType.SYSTEM_ERROR: SystemErrorEvent,
    EventType.SYSTEM_METRICS: SystemMetricsEvent,
    EventType.STT_START: STTStartEvent,
    EventType.STT_INTERIM: STTInterimEvent,
    EventType.STT_RESULT: STTResultEvent,
    EventType.STT_ERROR: STTErrorEvent,
    EventType.LLM_START: LLMStartEvent,
    EventType.LLM_TOKEN: LLMTokenEvent,
    EventType.LLM_RESULT: LLMResultEvent,
    EventType.LLM_ERROR: LLMErrorEvent,
    EventType.TTS_START: TTSStartEvent,
    EventType.TTS_PROGRESS: TTSProgressEvent,
    EventType.TTS_RESULT: TTSResultEvent,
    EventType.TTS_ERROR: TTSErrorEvent,
    EventType.TTS_STATUS: TTSStatusEvent,
    EventType.MEMORY_STORE: MemoryStoreEvent,
    EventType.MEMORY_RETRIEVE: MemoryRetrieveEvent,
    EventType.TOOL_CALL: ToolCallEvent,
    EventType.TOOL_RESULT: ToolResultEvent,
    EventType.TOOL_ERROR: ToolErrorEvent,
    EventType.CONVERSATION_START: ConversationStartEvent,
    EventType.CONVERSATION_TURN: ConversationTurnEvent,
    EventType.CONVERSATION_END: ConversationEndEvent,
    EventType.LATENCY_TRACE: LatencyTraceEvent,
    EventType.COMPONENT_TIMING: ComponentTimingEvent,
}


def create_event(event_type: EventType, seq: int, **kwargs) -> BaseEvent:
    """
    Create an event of the specified type.
    
    Args:
        event_type: The type of event to create
        seq: Sequence number for the event
        **kwargs: Additional event data
        
    Returns:
        The created event instance
        
    Raises:
        ValueError: If event_type is not supported
    """
    event_class = EVENT_CLASS_MAP.get(event_type)
    if not event_class:
        raise ValueError(f"Unsupported event type: {event_type}")
    
    return event_class(seq=seq, **kwargs)


def validate_event(event_data: Dict[str, Any]) -> BaseEvent:
    """
    Validate and parse event data.
    
    Args:
        event_data: Raw event data dictionary
        
    Returns:
        Validated event instance
        
    Raises:
        ValueError: If event data is invalid
    """
    event_type = event_data.get("type")
    if not event_type:
        raise ValueError("Event data missing 'type' field")
    
    try:
        event_type_enum = EventType(event_type)
    except ValueError:
        raise ValueError(f"Invalid event type: {event_type}")
    
    event_class = EVENT_CLASS_MAP.get(event_type_enum)
    if not event_class:
        raise ValueError(f"Unsupported event type: {event_type}")
    
    return event_class(**event_data)
