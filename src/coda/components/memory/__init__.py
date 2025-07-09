"""
Memory management system for Coda.

This package provides comprehensive memory management functionality including:
- Short-term conversation memory
- Long-term persistent memory with vector embeddings
- Memory encoding and chunking
- Semantic search and retrieval
- WebSocket integration for real-time events

Features:
- Dual-layer memory architecture
- Vector-based semantic search with ChromaDB
- Automatic memory consolidation
- Time-based relevance decay
- Importance scoring and filtering
- Type-safe interfaces with Pydantic models
"""

from .interfaces import (
    MemoryInterface,
    ShortTermMemoryInterface,
    LongTermMemoryInterface,
    MemoryEncoderInterface,
)
from .models import (
    ConversationTurn,
    Memory,
    MemoryChunk,
    MemoryQuery,
    MemoryResult,
    MemoryMetadata,
    MemoryStats,
)
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .encoder import MemoryEncoder
from .manager import MemoryManager
from .websocket_integration import WebSocketMemoryManager

__all__ = [
    # Interfaces
    "MemoryInterface",
    "ShortTermMemoryInterface", 
    "LongTermMemoryInterface",
    "MemoryEncoderInterface",
    
    # Models
    "ConversationTurn",
    "Memory",
    "MemoryChunk", 
    "MemoryQuery",
    "MemoryResult",
    "MemoryMetadata",
    "MemoryStats",
    
    # Implementations
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryEncoder",
    "MemoryManager",
    "WebSocketMemoryManager",
]
