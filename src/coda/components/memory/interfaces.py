"""
Interfaces for the memory system.

This module defines abstract base classes for memory components to ensure
consistent APIs and enable easy testing with mocks.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from .models import (
    ConversationContext,
    ConversationTurn,
    Memory,
    MemoryChunk,
    MemoryQuery,
    MemoryResult,
    MemoryStats,
    MemoryType,
)


class ShortTermMemoryInterface(ABC):
    """Interface for short-term memory management."""

    @abstractmethod
    def add_turn(self, role: str, content: str) -> ConversationTurn:
        """Add a conversation turn to short-term memory."""
        pass

    @abstractmethod
    def get_context(self, max_tokens: int = 800) -> ConversationContext:
        """Get conversation context within token budget."""
        pass

    @abstractmethod
    def get_recent_turns(self, count: int = 10) -> List[ConversationTurn]:
        """Get the most recent conversation turns."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all conversation turns."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about short-term memory."""
        pass

    @abstractmethod
    def export_turns(self) -> List[Dict[str, Any]]:
        """Export turns for persistence or debugging."""
        pass


class LongTermMemoryInterface(ABC):
    """Interface for long-term memory management."""

    @abstractmethod
    async def store_memory(
        self, content: str, memory_type: MemoryType, importance: float, metadata: Dict[str, Any]
    ) -> str:
        """Store a memory and return its ID."""
        pass

    @abstractmethod
    async def retrieve_memories(self, query: MemoryQuery) -> List[MemoryResult]:
        """Retrieve memories based on query parameters."""
        pass

    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        pass

    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory."""
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        pass

    @abstractmethod
    async def get_stats(self) -> MemoryStats:
        """Get statistics about long-term memory."""
        pass

    @abstractmethod
    async def cleanup_old_memories(self, max_age_days: int = 365) -> int:
        """Clean up old memories and return count of deleted memories."""
        pass

    @abstractmethod
    async def backup_memories(self, backup_path: str) -> bool:
        """Backup memories to a file."""
        pass

    @abstractmethod
    async def restore_memories(self, backup_path: str) -> int:
        """Restore memories from a backup file."""
        pass


class MemoryEncoderInterface(ABC):
    """Interface for memory encoding and chunking."""

    @abstractmethod
    def encode_conversation(
        self, turns: List[ConversationTurn], include_system: bool = False
    ) -> List[MemoryChunk]:
        """Encode conversation turns into memory chunks."""
        pass

    @abstractmethod
    def encode_fact(
        self, fact: str, source: str = "user", metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryChunk:
        """Encode a fact into a memory chunk."""
        pass

    @abstractmethod
    def calculate_importance(self, content: str) -> float:
        """Calculate importance score for content."""
        pass

    @abstractmethod
    def extract_topics(self, content: str) -> List[str]:
        """Extract topics from content."""
        pass

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass


class MemoryInterface(ABC):
    """Main interface for the complete memory system."""

    @abstractmethod
    def add_turn(self, role: str, content: str) -> ConversationTurn:
        """Add a conversation turn."""
        pass

    @abstractmethod
    def get_context(self, max_tokens: int = 800) -> ConversationContext:
        """Get conversation context."""
        pass

    @abstractmethod
    async def get_enhanced_context(
        self, user_input: str, max_tokens: int = 800, max_memories: int = 3
    ) -> ConversationContext:
        """Get enhanced context with relevant long-term memories."""
        pass

    @abstractmethod
    async def store_fact(self, fact: str, source: str = "user", importance: float = 0.7) -> str:
        """Store a fact in long-term memory."""
        pass

    @abstractmethod
    async def search_memories(
        self, query: str, limit: int = 5, min_relevance: float = 0.0
    ) -> List[MemoryResult]:
        """Search memories by query."""
        pass

    @abstractmethod
    async def consolidate_short_term(self) -> int:
        """Consolidate short-term memory to long-term storage."""
        pass

    @abstractmethod
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        pass

    @abstractmethod
    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        pass

    @abstractmethod
    async def cleanup_old_memories(self, max_age_days: int = 365) -> int:
        """Clean up old memories."""
        pass


class WebSocketMemoryInterface(MemoryInterface):
    """Interface for memory system with WebSocket integration."""

    @abstractmethod
    async def set_websocket_integration(self, integration: Any) -> None:
        """Set WebSocket integration for event broadcasting."""
        pass

    @abstractmethod
    async def broadcast_memory_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a memory event via WebSocket."""
        pass


# Utility interfaces


class MemorySearchInterface(ABC):
    """Interface for memory search functionality."""

    @abstractmethod
    async def semantic_search(self, query: str, limit: int = 10) -> List[MemoryResult]:
        """Perform semantic search on memories."""
        pass

    @abstractmethod
    async def keyword_search(self, keywords: List[str], limit: int = 10) -> List[MemoryResult]:
        """Perform keyword search on memories."""
        pass

    @abstractmethod
    async def filter_by_metadata(
        self, filters: Dict[str, Any], limit: int = 10
    ) -> List[MemoryResult]:
        """Filter memories by metadata."""
        pass

    @abstractmethod
    async def get_similar_memories(self, memory_id: str, limit: int = 5) -> List[MemoryResult]:
        """Get memories similar to a given memory."""
        pass


class MemoryAnalyticsInterface(ABC):
    """Interface for memory analytics and insights."""

    @abstractmethod
    async def get_topic_distribution(self) -> Dict[str, int]:
        """Get distribution of topics in memories."""
        pass

    @abstractmethod
    async def get_memory_timeline(self, days: int = 30) -> Dict[str, int]:
        """Get memory creation timeline."""
        pass

    @abstractmethod
    async def get_importance_distribution(self) -> Dict[str, int]:
        """Get distribution of importance scores."""
        pass

    @abstractmethod
    async def get_access_patterns(self) -> Dict[str, Any]:
        """Get memory access patterns."""
        pass

    @abstractmethod
    async def identify_knowledge_gaps(self, topics: List[str]) -> List[str]:
        """Identify topics with insufficient memory coverage."""
        pass


class MemoryMaintenanceInterface(ABC):
    """Interface for memory maintenance operations."""

    @abstractmethod
    async def optimize_storage(self) -> Dict[str, Any]:
        """Optimize memory storage and return statistics."""
        pass

    @abstractmethod
    async def rebuild_embeddings(self) -> int:
        """Rebuild all memory embeddings."""
        pass

    @abstractmethod
    async def deduplicate_memories(self, similarity_threshold: float = 0.95) -> int:
        """Remove duplicate memories."""
        pass

    @abstractmethod
    async def validate_integrity(self) -> Dict[str, Any]:
        """Validate memory system integrity."""
        pass

    @abstractmethod
    async def repair_corrupted_memories(self) -> int:
        """Repair corrupted memories."""
        pass
