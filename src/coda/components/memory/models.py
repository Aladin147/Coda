"""
Data models for the memory system.

This module defines Pydantic models for type-safe memory operations.
"""

import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryType(str, Enum):
    """Types of memories."""

    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    SYSTEM = "system"
    TOOL_RESULT = "tool_result"


class ConversationRole(str, Enum):
    """Roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ConversationTurn(BaseModel):
    """A single conversation turn."""

    role: ConversationRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    turn_id: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MemoryMetadata(BaseModel):
    """Metadata associated with a memory."""

    source_type: MemoryType
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    topics: List[str] = Field(default_factory=list)
    speakers: List[str] = Field(default_factory=list)
    turn_ids: List[int] = Field(default_factory=list)
    session_id: Optional[str] = None
    additional: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class Memory(BaseModel):
    """A stored memory with content and metadata."""

    id: str
    content: str
    metadata: MemoryMetadata
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_at: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(default=0)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def update_access(self) -> None:
        """Update access tracking."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class MemoryChunk(BaseModel):
    """A chunk of memory content for encoding."""

    content: str
    speakers: List[str]
    turn_ids: List[int]
    timestamp: datetime
    topics: List[str] = Field(default_factory=list)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval."""

    query: str
    limit: int = Field(default=5, ge=1, le=100)
    min_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_types: Optional[List[MemoryType]] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    topics: Optional[List[str]] = None
    include_system: bool = Field(default=False)

    @field_validator("time_range")
    @classmethod
    def validate_time_range(cls, v):
        if v is not None and len(v) == 2:
            start, end = v
            if start > end:
                raise ValueError("Start time must be before end time")
        return v


class MemoryResult(BaseModel):
    """Result of a memory retrieval operation."""

    memory: Memory
    relevance_score: float = Field(ge=0.0, le=1.0)
    time_decay_factor: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MemoryStats(BaseModel):
    """Statistics about the memory system."""

    total_memories: int = 0
    memory_types: Dict[MemoryType, int] = Field(default_factory=dict)
    average_importance: float = 0.0
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None
    total_content_length: int = 0
    average_content_length: float = 0.0
    most_accessed_memory_id: Optional[str] = None
    total_access_count: int = 0

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class ShortTermMemoryConfig(BaseModel):
    """Configuration for short-term memory."""

    max_turns: int = Field(default=20, ge=1, le=1000)
    max_tokens: int = Field(default=800, ge=100, le=10000)
    include_system_in_context: bool = Field(default=True)


class LongTermMemoryConfig(BaseModel):
    """Configuration for long-term memory."""

    storage_path: str = Field(default="data/memory/long_term")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    vector_db_type: str = Field(default="chroma")
    max_memories: int = Field(default=1000, ge=1)
    device: str = Field(default="cpu")
    chunk_size: int = Field(default=200, ge=50, le=1000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    min_chunk_length: int = Field(default=50, ge=10, le=500)
    time_decay_days: float = Field(default=30.0, ge=1.0)
    auto_persist: bool = Field(default=True)
    persist_interval: int = Field(default=5, ge=1)


class MemoryEncoderConfig(BaseModel):
    """Configuration for memory encoder."""

    chunk_size: int = Field(default=200, ge=50, le=1000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    min_chunk_length: int = Field(default=50, ge=10, le=500)
    importance_keywords: List[str] = Field(
        default_factory=lambda: ["important", "remember", "note", "key", "critical", "essential"]
    )
    topic_extraction_enabled: bool = Field(default=True)


class MemoryManagerConfig(BaseModel):
    """Configuration for the memory manager."""

    short_term: ShortTermMemoryConfig = Field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)
    encoder: MemoryEncoderConfig = Field(default_factory=MemoryEncoderConfig)
    auto_persist: bool = Field(default=True)
    persist_interval: int = Field(default=5, ge=1)


# Context models for LLM integration


class ContextMessage(BaseModel):
    """A message in the conversation context."""

    role: ConversationRole
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for LLM."""
        return {"role": self.role.value, "content": self.content}


class ConversationContext(BaseModel):
    """Complete conversation context for LLM."""

    messages: List[ContextMessage]
    total_tokens: int
    short_term_turns: int
    long_term_memories: int

    def to_list(self) -> List[Dict[str, str]]:
        """Convert to list format for LLM."""
        return [msg.to_dict() for msg in self.messages]


# Event models for WebSocket integration


class MemoryEvent(BaseModel):
    """Base class for memory events."""

    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MemoryStoreEvent(MemoryEvent):
    """Event for memory storage."""

    memory_id: str
    content_preview: str
    memory_type: MemoryType
    importance: float


class MemoryRetrieveEvent(MemoryEvent):
    """Event for memory retrieval."""

    query: str
    results_count: int
    relevance_scores: List[float]
    processing_time_ms: float


class MemoryConsolidateEvent(MemoryEvent):
    """Event for memory consolidation."""

    turns_processed: int
    memories_created: int
    processing_time_ms: float
