"""
Main memory manager for Coda.

This module provides the MemoryManager class that integrates short-term and
long-term memory with automatic consolidation and enhanced context generation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .encoder import MemoryEncoder
from .interfaces import MemoryInterface
from .long_term import LongTermMemory
from .models import (
    ContextMessage,
    ConversationContext,
    ConversationRole,
    ConversationTurn,
    MemoryManagerConfig,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)
from .short_term import ShortTermMemory

logger = logging.getLogger("coda.memory.manager")


class MemoryManager(MemoryInterface):
    """
    Main memory manager that integrates short-term and long-term memory.

    Features:
    - Dual-layer memory architecture
    - Automatic memory consolidation
    - Enhanced context with relevant memories
    - Configurable persistence intervals
    - Comprehensive memory statistics
    """

    def __init__(self, config: Optional[MemoryManagerConfig] = None):
        """
        Initialize the memory manager.

        Args:
            config: Configuration for memory management
        """
        self.config = config or MemoryManagerConfig()

        # Initialize components
        self.short_term = ShortTermMemory(self.config.short_term)
        self.long_term = LongTermMemory(self.config.long_term)
        self.encoder = MemoryEncoder(self.config.encoder)

        # Consolidation tracking
        self.turn_count_at_last_persist = 0
        self.auto_persist = self.config.auto_persist
        self.persist_interval = self.config.persist_interval

        logger.info("MemoryManager initialized with dual-layer architecture")

    async def initialize(self) -> None:
        """
        Initialize the memory manager and its components.

        This method performs any async initialization needed for:
        - Long-term memory storage setup
        - Vector database initialization
        - Memory encoder preparation
        """
        try:
            # Initialize long-term memory (may need async setup for vector DB)
            if hasattr(self.long_term, "initialize"):
                await self.long_term.initialize()

            # Initialize encoder if needed
            if hasattr(self.encoder, "initialize"):
                await self.encoder.initialize()

            logger.info("MemoryManager async initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")
            raise

    def add_turn(self, role: str, content: str) -> ConversationTurn:
        """
        Add a conversation turn to short-term memory.

        Args:
            role: The speaker role
            content: The message content

        Returns:
            The created conversation turn
        """
        turn = self.short_term.add_turn(role, content)

        # Check if we should consolidate to long-term memory
        if self.auto_persist and role == "assistant":
            turns_since_persist = self.short_term.turn_count - self.turn_count_at_last_persist
            if turns_since_persist >= self.persist_interval:
                # Schedule consolidation (don't block)
                asyncio.create_task(self.consolidate_short_term())

        return turn

    def get_context(self, max_tokens: int = 800) -> ConversationContext:
        """
        Get conversation context from short-term memory.

        Args:
            max_tokens: Maximum number of tokens to include

        Returns:
            Conversation context
        """
        return self.short_term.get_context(max_tokens)

    async def get_enhanced_context(
        self, user_input: str, max_tokens: int = 800, max_memories: int = 3
    ) -> ConversationContext:
        """
        Get enhanced conversation context with relevant long-term memories.

        Args:
            user_input: Current user input for memory retrieval
            max_tokens: Maximum tokens for short-term context
            max_memories: Maximum number of long-term memories to include

        Returns:
            Enhanced conversation context
        """
        # Get short-term context (reserve some tokens for long-term memories)
        short_term_tokens = max_tokens - (max_memories * 100)  # Reserve ~100 tokens per memory
        short_term_context = self.short_term.get_context(max(short_term_tokens, 200))

        # Retrieve relevant long-term memories
        if max_memories > 0:
            query = MemoryQuery(
                query=user_input, limit=max_memories, min_relevance=0.3, include_system=False
            )

            memory_results = await self.long_term.retrieve_memories(query)

            # Convert memories to context messages
            memory_messages = []
            for result in memory_results:
                # Add memory as a system message with context
                memory_content = f"[Relevant memory: {result.memory.content}]"
                memory_messages.append(
                    ContextMessage(role=ConversationRole.SYSTEM, content=memory_content)
                )

            # Insert memory messages after system message but before conversation
            final_messages = []
            system_messages = [
                msg for msg in short_term_context.messages if msg.role == ConversationRole.SYSTEM
            ]
            conversation_messages = [
                msg for msg in short_term_context.messages if msg.role != ConversationRole.SYSTEM
            ]

            final_messages.extend(system_messages)
            final_messages.extend(memory_messages)
            final_messages.extend(conversation_messages)

            return ConversationContext(
                messages=final_messages,
                total_tokens=short_term_context.total_tokens
                + len(memory_messages) * 100,  # Estimate
                short_term_turns=short_term_context.short_term_turns,
                long_term_memories=len(memory_results),
            )

        return short_term_context

    async def store_fact(self, fact: str, source: str = "user", importance: float = 0.7) -> str:
        """
        Store a fact in long-term memory.

        Args:
            fact: The fact to store
            source: Source of the fact
            importance: Importance score

        Returns:
            Memory ID of the stored fact
        """
        memory_id = await self.long_term.store_memory(
            content=fact,
            memory_type=MemoryType.FACT,
            importance=importance,
            metadata={"source": source},
        )

        logger.info(f"Stored fact: {fact[:50]}... (ID: {memory_id})")
        return memory_id

    async def search_memories(
        self, query: str, limit: int = 5, min_relevance: float = 0.0
    ) -> List[MemoryResult]:
        """
        Search memories by query.

        Args:
            query: Search query
            limit: Maximum number of results
            min_relevance: Minimum relevance score

        Returns:
            List of memory results
        """
        memory_query = MemoryQuery(query=query, limit=limit, min_relevance=min_relevance)

        return await self.long_term.retrieve_memories(memory_query)

    async def consolidate_short_term(self) -> int:
        """
        Consolidate short-term memory to long-term storage.

        Returns:
            Number of memories created
        """
        # Get recent turns for consolidation
        recent_turns = self.short_term.get_recent_turns(self.persist_interval * 2)

        if len(recent_turns) < 2:  # Need at least 2 turns for meaningful consolidation
            return 0

        # Encode turns into memory chunks
        memory_chunks = self.encoder.encode_conversation(recent_turns, include_system=False)

        memories_created = 0
        for chunk in memory_chunks:
            # Calculate importance for the chunk
            importance = self.encoder.calculate_importance(chunk.content)

            # Create metadata
            metadata = {
                "speakers": chunk.speakers,
                "turn_ids": chunk.turn_ids,
                "topics": chunk.topics,
                "session_timestamp": chunk.timestamp.isoformat(),
            }

            # Store in long-term memory
            try:
                await self.long_term.store_memory(
                    content=chunk.content,
                    memory_type=MemoryType.CONVERSATION,
                    importance=importance,
                    metadata=metadata,
                )
                memories_created += 1
            except Exception as e:
                logger.error(f"Failed to store memory chunk: {e}")

        # Update consolidation tracking
        self.turn_count_at_last_persist = self.short_term.turn_count

        logger.info(f"Consolidated {memories_created} memories from {len(recent_turns)} turns")
        return memories_created

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        short_term_stats = self.short_term.get_stats()
        long_term_stats = await self.long_term.get_stats()

        return {
            "short_term": short_term_stats,
            "long_term": long_term_stats.model_dump(),
            "consolidation": {
                "auto_persist": self.auto_persist,
                "persist_interval": self.persist_interval,
                "turns_since_last_persist": self.short_term.turn_count
                - self.turn_count_at_last_persist,
            },
        }

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term.clear()
        self.turn_count_at_last_persist = 0
        logger.info("Cleared short-term memory")

    async def cleanup_old_memories(self, max_age_days: int = 365) -> int:
        """
        Clean up old memories from long-term storage.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of memories deleted
        """
        return await self.long_term.cleanup_old_memories(max_age_days)

    async def shutdown(self) -> None:
        """
        Shutdown the memory manager and cleanup resources.

        This method performs:
        - Final consolidation of short-term memory
        - Cleanup of long-term memory resources
        - Encoder cleanup
        """
        try:
            logger.info("Shutting down MemoryManager...")

            # Final consolidation if auto-persist is enabled
            if self.auto_persist:
                await self.consolidate_short_term()

            # Cleanup components
            if hasattr(self.long_term, "shutdown"):
                await self.long_term.shutdown()
            elif hasattr(self.long_term, "cleanup"):
                await self.long_term.cleanup()

            if hasattr(self.encoder, "shutdown"):
                await self.encoder.shutdown()
            elif hasattr(self.encoder, "cleanup"):
                await self.encoder.cleanup()

            # Clear short-term memory
            self.short_term.clear()

            logger.info("MemoryManager shutdown completed")

        except Exception as e:
            logger.error(f"Error during MemoryManager shutdown: {e}")
            raise

    # Additional utility methods

    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        short_term_summary = self.short_term.get_conversation_summary()
        memory_stats = await self.get_memory_stats()

        return {
            "current_conversation": short_term_summary,
            "memory_stats": {
                "short_term_turns": memory_stats["short_term"]["total_turns"],
                "long_term_memories": memory_stats["long_term"]["total_memories"],
                "turns_since_consolidation": memory_stats["consolidation"][
                    "turns_since_last_persist"
                ],
            },
        }

    async def backup_all_memories(self, backup_path: str) -> bool:
        """Backup all memories to a file."""
        return await self.long_term.backup_memories(backup_path)

    async def restore_memories(self, backup_path: str) -> int:
        """Restore memories from a backup file."""
        return await self.long_term.restore_memories(backup_path)

    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        memory = await self.long_term.get_memory(memory_id)
        if memory:
            return {
                "id": memory.id,
                "content": memory.content,
                "metadata": memory.metadata.model_dump(),
                "created_at": memory.created_at.isoformat(),
                "accessed_at": memory.accessed_at.isoformat(),
                "access_count": memory.access_count,
            }
        return None

    async def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """Update the importance score of a memory."""
        return await self.long_term.update_memory(memory_id, importance=importance)

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        return await self.long_term.delete_memory(memory_id)

    def set_auto_persist(self, enabled: bool, interval: Optional[int] = None) -> None:
        """Configure automatic persistence settings."""
        self.auto_persist = enabled
        if interval is not None:
            self.persist_interval = interval

        logger.info(
            f"Auto-persist {'enabled' if enabled else 'disabled'} with interval {self.persist_interval}"
        )

    async def force_consolidation(self) -> int:
        """Force immediate consolidation of short-term memory."""
        return await self.consolidate_short_term()

    # Methods for compatibility with assistant interface
    async def create_session(self, session_id: str) -> None:
        """Create a new session (compatibility method)."""
        # For now, this is a no-op as sessions are handled by the session manager
        logger.debug(f"Session created: {session_id}")

    async def store_interaction(self, session_id: str, user_input: str, assistant_response: str, metadata: dict = None) -> None:
        """Store an interaction in memory."""
        # Add user input
        self.add_turn("user", user_input)
        # Add assistant response
        self.add_turn("assistant", assistant_response)

        # Store important facts if metadata indicates high importance
        if metadata and metadata.get("importance", 0.5) > 0.7:
            await self.store_fact(f"User said: {user_input}", source="interaction")
            await self.store_fact(f"Assistant responded: {assistant_response}", source="interaction")

    async def get_session_history(self, session_id: str, limit: int = 50) -> list:
        """Get session history (compatibility method)."""
        context = self.get_context(max_tokens=2000)
        return [{"role": msg.role, "content": msg.content} for msg in context.messages[-limit:]]

    async def list_conversations(self, limit: int = 10) -> list:
        """List recent conversations (compatibility method)."""
        # Return a simplified conversation list
        return [{"id": "current", "title": "Current Session", "turn_count": self.short_term.turn_count}]

    def is_healthy(self) -> bool:
        """Check if the memory manager is healthy."""
        try:
            # Basic health checks
            return (
                self.short_term is not None and
                self.long_term is not None and
                hasattr(self, 'config') and
                self.config is not None
            )
        except Exception:
            return False
