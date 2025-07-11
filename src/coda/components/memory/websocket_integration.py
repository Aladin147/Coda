"""
WebSocket integration for the memory system.

This module provides WebSocketMemoryManager that extends the base MemoryManager
with real-time event broadcasting capabilities.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .interfaces import WebSocketMemoryInterface
from .manager import MemoryManager
from .models import (
    ConversationContext,
    ConversationTurn,
    MemoryConsolidateEvent,
    MemoryManagerConfig,
    MemoryResult,
    MemoryRetrieveEvent,
    MemoryStoreEvent,
)

logger = logging.getLogger("coda.memory.websocket")


class WebSocketMemoryManager(MemoryManager, WebSocketMemoryInterface):
    """
    Memory manager with WebSocket integration for real-time event broadcasting.

    Features:
    - All base MemoryManager functionality
    - Real-time memory events via WebSocket
    - Performance tracking for memory operations
    - Event broadcasting for memory storage, retrieval, and consolidation
    """

    def __init__(self, config: Optional[MemoryManagerConfig] = None):
        """
        Initialize the WebSocket-enabled memory manager.

        Args:
            config: Configuration for memory management
        """
        super().__init__(config)
        self.websocket_integration = None

        logger.info("WebSocketMemoryManager initialized")

    async def set_websocket_integration(self, integration: Any) -> None:
        """
        Set WebSocket integration for event broadcasting.

        Args:
            integration: WebSocket integration instance
        """
        self.websocket_integration = integration
        logger.info("WebSocket integration configured for memory system")

    async def broadcast_memory_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast a memory event via WebSocket.

        Args:
            event_type: Type of memory event
            data: Event data
        """
        if self.websocket_integration:
            try:
                # Map event types to WebSocket integration methods
                if event_type == "memory_store":
                    await self.websocket_integration.memory_store(
                        content=data.get("content_preview", ""),
                        memory_type=data.get("memory_type", "unknown"),
                        importance=data.get("importance", 0.5),
                        memory_id=data.get("memory_id", ""),
                    )
                elif event_type == "memory_retrieve":
                    await self.websocket_integration.memory_retrieve(
                        query=data.get("query", ""),
                        results_count=data.get("results_count", 0),
                        relevance_scores=data.get("relevance_scores", []),
                    )
                elif event_type == "memory_consolidate":
                    # Custom event for consolidation
                    await self.websocket_integration.server.broadcast_event(
                        "memory_consolidate",
                        data,
                        high_priority=True,
                        session_id=self.websocket_integration.session_id,
                    )
                else:
                    logger.warning(f"Unknown memory event type: {event_type}")

            except Exception as e:
                logger.error(f"Failed to broadcast memory event {event_type}: {e}")

    # Override methods to add WebSocket events

    async def store_fact(self, fact: str, source: str = "user", importance: float = 0.7) -> str:
        """Store a fact with WebSocket event broadcasting."""
        start_time = time.time()

        memory_id = await super().store_fact(fact, source, importance)

        # Broadcast memory store event
        await self.broadcast_memory_event(
            "memory_store",
            {
                "memory_id": memory_id,
                "content_preview": fact[:100] + "..." if len(fact) > 100 else fact,
                "memory_type": "fact",
                "importance": importance,
                "source": source,
                "processing_time_ms": (time.time() - start_time) * 1000,
            },
        )

        return memory_id

    async def search_memories(
        self, query: str, limit: int = 5, min_relevance: float = 0.0
    ) -> List[MemoryResult]:
        """Search memories with WebSocket event broadcasting."""
        start_time = time.time()

        results = await super().search_memories(query, limit, min_relevance)

        # Broadcast memory retrieve event
        await self.broadcast_memory_event(
            "memory_retrieve",
            {
                "query": query,
                "results_count": len(results),
                "relevance_scores": [r.relevance_score for r in results],
                "processing_time_ms": (time.time() - start_time) * 1000,
            },
        )

        return results

    async def get_enhanced_context(
        self, user_input: str, max_tokens: int = 800, max_memories: int = 3
    ) -> ConversationContext:
        """Get enhanced context with WebSocket event broadcasting."""
        start_time = time.time()

        context = await super().get_enhanced_context(user_input, max_tokens, max_memories)

        # Broadcast memory retrieve event if long-term memories were used
        if context.long_term_memories > 0:
            await self.broadcast_memory_event(
                "memory_retrieve",
                {
                    "query": user_input,
                    "results_count": context.long_term_memories,
                    "relevance_scores": [],  # Not available in context
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "context_type": "enhanced",
                },
            )

        return context

    async def consolidate_short_term(self) -> int:
        """Consolidate short-term memory with WebSocket event broadcasting."""
        start_time = time.time()

        # Get turn count before consolidation
        turns_before = len(self.short_term.get_recent_turns(self.persist_interval * 2))

        memories_created = await super().consolidate_short_term()

        # Broadcast consolidation event
        await self.broadcast_memory_event(
            "memory_consolidate",
            {
                "turns_processed": turns_before,
                "memories_created": memories_created,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "persist_interval": self.persist_interval,
            },
        )

        return memories_created

    async def cleanup_old_memories(self, max_age_days: int = 365) -> int:
        """Clean up old memories with WebSocket event broadcasting."""
        start_time = time.time()

        deleted_count = await super().cleanup_old_memories(max_age_days)

        # Broadcast cleanup event
        await self.broadcast_memory_event(
            "memory_cleanup",
            {
                "deleted_count": deleted_count,
                "max_age_days": max_age_days,
                "processing_time_ms": (time.time() - start_time) * 1000,
            },
        )

        return deleted_count

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory with WebSocket event broadcasting."""
        # Get memory info before deletion
        memory_info = await self.get_memory_by_id(memory_id)

        success = await super().delete_memory(memory_id)

        if success and memory_info:
            # Broadcast memory deletion event
            await self.broadcast_memory_event(
                "memory_delete",
                {
                    "memory_id": memory_id,
                    "content_preview": (
                        memory_info["content"][:100] + "..."
                        if len(memory_info["content"]) > 100
                        else memory_info["content"]
                    ),
                    "memory_type": memory_info["metadata"]["source_type"],
                    "success": success,
                },
            )

        return success

    async def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """Update memory importance with WebSocket event broadcasting."""
        success = await super().update_memory_importance(memory_id, importance)

        if success:
            # Broadcast memory update event
            await self.broadcast_memory_event(
                "memory_update",
                {
                    "memory_id": memory_id,
                    "field": "importance",
                    "new_value": importance,
                    "success": success,
                },
            )

        return success

    # Additional WebSocket-specific methods

    async def broadcast_memory_stats(self) -> None:
        """Broadcast current memory statistics."""
        stats = await self.get_memory_stats()

        await self.broadcast_memory_event(
            "memory_stats",
            {
                "short_term_turns": stats["short_term"]["total_turns"],
                "long_term_memories": stats["long_term"]["total_memories"],
                "memory_types": stats["long_term"]["memory_types"],
                "average_importance": stats["long_term"]["average_importance"],
                "turns_since_consolidation": stats["consolidation"]["turns_since_last_persist"],
                "auto_persist_enabled": stats["consolidation"]["auto_persist"],
            },
        )

    async def broadcast_conversation_summary(self) -> None:
        """Broadcast conversation summary."""
        summary = await self.get_conversation_summary()

        await self.broadcast_memory_event("conversation_summary", summary)

    def add_turn(self, role: str, content: str) -> ConversationTurn:
        """Add a turn with potential WebSocket events for consolidation."""
        turn = super().add_turn(role, content)

        # If consolidation was triggered, it will broadcast its own event
        # No additional event needed here to avoid spam

        return turn

    async def get_memory_timeline(self, days: int = 7) -> Dict[str, Any]:
        """Get memory creation timeline for dashboard visualization."""
        # This would require additional implementation in long_term.py
        # For now, return basic stats
        stats = await self.get_memory_stats()

        timeline_data = {
            "days": days,
            "total_memories": stats["long_term"]["total_memories"],
            "memory_types": stats["long_term"]["memory_types"],
            "recent_activity": {
                "short_term_turns": stats["short_term"]["total_turns"],
                "session_duration": stats["short_term"]["session_duration_seconds"],
            },
        }

        # Broadcast timeline data
        await self.broadcast_memory_event("memory_timeline", timeline_data)

        return timeline_data

    async def search_and_broadcast(self, query: str, limit: int = 5) -> List[MemoryResult]:
        """Search memories and broadcast detailed results for dashboard."""
        results = await self.search_memories(query, limit)

        # Broadcast detailed search results
        detailed_results = []
        for result in results:
            detailed_results.append(
                {
                    "memory_id": result.memory.id,
                    "content_preview": (
                        result.memory.content[:200] + "..."
                        if len(result.memory.content) > 200
                        else result.memory.content
                    ),
                    "relevance_score": result.relevance_score,
                    "final_score": result.final_score,
                    "importance": result.memory.metadata.importance,
                    "memory_type": result.memory.metadata.source_type.value,
                    "created_at": result.memory.created_at.isoformat(),
                    "topics": result.memory.metadata.topics,
                }
            )

        await self.broadcast_memory_event(
            "memory_search_results",
            {"query": query, "results": detailed_results, "total_results": len(results)},
        )

        return results
