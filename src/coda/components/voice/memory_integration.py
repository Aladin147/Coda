"""
Voice - Memory Integration

This module provides comprehensive integration between the voice processing system
and the memory manager for context - aware conversations and automatic memory storage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..memory.interfaces import MemoryInterface
from ..memory.models import (
    ConversationContext,
    ConversationRole,
    ConversationTurn,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)
from .models import ConversationState, VoiceMessage, VoiceResponse

logger = logging.getLogger("coda.voice.memory_integration")


@dataclass
class VoiceMemoryConfig:
    """Configuration for voice - memory integration."""

    # Memory context injection
    enable_memory_injection: bool = True
    max_memory_context_tokens: int = 800
    memory_relevance_threshold: float = 0.6
    max_relevant_memories: int = 5

    # Automatic memory storage
    enable_auto_storage: bool = True
    store_user_messages: bool = True
    store_assistant_responses: bool = True
    min_message_length: int = 10  # Minimum characters to store

    # Memory consolidation
    enable_auto_consolidation: bool = True
    consolidation_interval: int = 10  # Messages between consolidations

    # Context enhancement
    enable_context_enhancement: bool = True
    context_window_hours: int = 24  # Hours to look back for context

    # Performance optimization
    memory_cache_ttl_minutes: int = 5
    parallel_memory_operations: bool = True


class VoiceMemoryIntegration:
    """
    Comprehensive integration between voice processing and memory systems.

    Features:
    - Automatic conversation turn storage
    - Relevant memory injection into voice context
    - Enhanced context generation with memory
    - Memory consolidation management
    - Performance optimization with caching
    """

    def __init__(self, memory_manager: MemoryInterface, config: VoiceMemoryConfig):
        """Initialize voice - memory integration."""
        self.memory_manager = memory_manager
        self.config = config

        # Caching for performance
        self.memory_cache: Dict[str, List[MemoryResult]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # Statistics tracking
        self.stats = {
            "turns_stored": 0,
            "memories_retrieved": 0,
            "cache_hits": 0,
            "consolidations_performed": 0,
            "context_enhancements": 0,
        }

        # Message counter for consolidation
        self.message_count = 0

        logger.info("VoiceMemoryIntegration initialized")

    async def process_voice_message(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]:
        """
        Process voice message with memory integration.

        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state

        Returns:
            Enhanced context with memory information
        """

        try:
            # Store user message in memory
            if self.config.enable_auto_storage and self.config.store_user_messages:
                await self._store_user_message(voice_message)

            # Get enhanced context with memory
            memory_context = {}
            if self.config.enable_memory_injection:
                memory_context = await self._get_memory_context(voice_message)

            # Get conversation context
            conversation_context = {}
            if self.config.enable_context_enhancement:
                conversation_context = await self._get_enhanced_conversation_context(voice_message)

            # Combine contexts
            enhanced_context = {
                "memory": memory_context,
                "conversation": conversation_context,
                "message_info": {
                    "message_id": voice_message.message_id,
                    "conversation_id": voice_message.conversation_id,
                    "text_content": voice_message.text_content or "",
                    "timestamp": voice_message.timestamp.isoformat(),
                    "processing_mode": voice_message.processing_mode.value,
                },
            }

            self.stats["context_enhancements"] += 1

            return enhanced_context

        except Exception as e:
            logger.error(f"Voice message processing failed: {e}")
            return {}

    async def process_voice_response(
        self, voice_response: VoiceResponse, conversation_state: Optional[ConversationState] = None
    ) -> None:
        """
        Process voice response for memory storage and learning.

        Args:
            voice_response: The voice response to process
            conversation_state: Current conversation state
        """

        try:
            # Store assistant response in memory
            if self.config.enable_auto_storage and self.config.store_assistant_responses:
                await self._store_assistant_response(voice_response)

            # Check if consolidation is needed
            if self.config.enable_auto_consolidation:
                await self._check_consolidation()

        except Exception as e:
            logger.error(f"Voice response processing failed: {e}")

    async def _store_user_message(self, voice_message: VoiceMessage) -> None:
        """Store user voice message in memory."""

        text_content = voice_message.text_content or ""

        # Check minimum length requirement
        if len(text_content) < self.config.min_message_length:
            return

        try:
            # Add conversation turn
            turn = self.memory_manager.add_turn(
                role=ConversationRole.USER.value, content=text_content
            )

            self.stats["turns_stored"] += 1
            self.message_count += 1

            logger.debug(f"Stored user message: {text_content[:50]}...")

        except Exception as e:
            logger.error(f"Failed to store user message: {e}")

    async def _store_assistant_response(self, voice_response: VoiceResponse) -> None:
        """Store assistant voice response in memory."""

        text_content = voice_response.text_content or ""

        # Check minimum length requirement
        if len(text_content) < self.config.min_message_length:
            return

        try:
            # Add conversation turn
            turn = self.memory_manager.add_turn(
                role=ConversationRole.ASSISTANT.value, content=text_content
            )

            self.stats["turns_stored"] += 1
            self.message_count += 1

            logger.debug(f"Stored assistant response: {text_content[:50]}...")

        except Exception as e:
            logger.error(f"Failed to store assistant response: {e}")

    async def _get_memory_context(self, voice_message: VoiceMessage) -> Dict[str, Any]:
        """Get relevant memory context for voice message."""

        text_content = voice_message.text_content or ""
        if not text_content:
            return {}

        # Check cache first
        cache_key = f"memory_{hash(text_content)}"
        if self._is_memory_cached(cache_key):
            self.stats["cache_hits"] += 1
            return self._get_cached_memory_context(cache_key)

        try:
            # Search for relevant memories
            memory_results = await self.memory_manager.search_memories(
                query=text_content,
                limit=self.config.max_relevant_memories,
                min_relevance=self.config.memory_relevance_threshold,
            )

            # Process memory results
            relevant_memories = []
            total_relevance = 0.0

            for result in memory_results:
                if result.final_score >= self.config.memory_relevance_threshold:
                    memory_info = {
                        "content": result.memory.content,
                        "type": result.memory.type.value if result.memory.type else "unknown",
                        "relevance_score": result.final_score,
                        "timestamp": (
                            result.memory.created_at.isoformat()
                            if result.memory.created_at
                            else None
                        ),
                        "importance": result.memory.importance,
                    }
                    relevant_memories.append(memory_info)
                    total_relevance += result.final_score

            # Create memory context
            memory_context = {
                "relevant_memories": relevant_memories,
                "memory_count": len(relevant_memories),
                "average_relevance": (
                    total_relevance / len(relevant_memories) if relevant_memories else 0.0
                ),
                "context_summary": self._generate_memory_summary(relevant_memories),
                "query": text_content,
            }

            # Cache the result
            self._cache_memory_context(cache_key, memory_context)

            self.stats["memories_retrieved"] += len(relevant_memories)

            logger.debug(f"Retrieved {len(relevant_memories)} relevant memories")

            return memory_context

        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return {}

    async def _get_enhanced_conversation_context(
        self, voice_message: VoiceMessage
    ) -> Dict[str, Any]:
        """Get enhanced conversation context with memory integration."""

        try:
            # Get standard conversation context
            context = self.memory_manager.get_context(
                max_tokens=self.config.max_memory_context_tokens
            )

            # Enhance with additional information
            enhanced_context = {
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    }
                    for msg in context.messages
                ],
                "total_messages": len(context.messages),
                "context_tokens": context.total_tokens,
                "conversation_summary": context.summary or "",
                "conversation_id": voice_message.conversation_id,
            }

            # Add conversation statistics
            memory_stats = await self.memory_manager.get_memory_stats()
            enhanced_context["conversation_stats"] = {
                "short_term_turns": memory_stats.get("short_term", {}).get("turn_count", 0),
                "long_term_memories": memory_stats.get("long_term", {}).get("memory_count", 0),
                "total_tokens": memory_stats.get("short_term", {}).get("total_tokens", 0),
            }

            return enhanced_context

        except Exception as e:
            logger.error(f"Failed to get enhanced conversation context: {e}")
            return {}

    def _generate_memory_summary(self, memories: List[Dict[str, Any]]) -> str:
        """Generate a summary of relevant memories."""

        if not memories:
            return "No relevant memories found."

        # Group memories by type
        memory_types = {}
        for memory in memories:
            mem_type = memory.get("type", "unknown")
            if mem_type not in memory_types:
                memory_types[mem_type] = []
            memory_types[mem_type].append(memory)

        # Generate summary
        summary_parts = []

        for mem_type, type_memories in memory_types.items():
            count = len(type_memories)
            avg_relevance = sum(m["relevance_score"] for m in type_memories) / count

            summary_parts.append(
                f"{count} {mem_type} memories (avg relevance: {avg_relevance:.2f})"
            )

        return f"Found {len(memories)} relevant memories: " + ", ".join(summary_parts)

    async def _check_consolidation(self) -> None:
        """Check if memory consolidation is needed."""

        if (
            self.config.enable_auto_consolidation
            and self.message_count >= self.config.consolidation_interval
        ):

            try:
                consolidated_count = await self.memory_manager.consolidate_short_term()

                if consolidated_count > 0:
                    self.stats["consolidations_performed"] += 1
                    self.message_count = 0  # Reset counter

                    logger.info(f"Consolidated {consolidated_count} memories")

            except Exception as e:
                logger.error(f"Memory consolidation failed: {e}")

    def _is_memory_cached(self, cache_key: str) -> bool:
        """Check if memory context is cached and valid."""

        if cache_key not in self.memory_cache:
            return False

        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return False

        ttl = timedelta(minutes=self.config.memory_cache_ttl_minutes)
        return datetime.now() - cache_time < ttl

    def _get_cached_memory_context(self, cache_key: str) -> Dict[str, Any]:
        """Get cached memory context."""
        return self.memory_cache.get(cache_key, {})

    def _cache_memory_context(self, cache_key: str, context: Dict[str, Any]) -> None:
        """Cache memory context."""
        self.memory_cache[cache_key] = context
        self.cache_timestamps[cache_key] = datetime.now()

        # Clean old cache entries
        self._cleanup_memory_cache()

    def _cleanup_memory_cache(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.now()
        ttl = timedelta(minutes=self.config.memory_cache_ttl_minutes)

        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items() if now - timestamp > ttl
        ]

        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

    async def store_conversation_fact(
        self, fact: str, source: str = "voice_conversation", importance: float = 0.7
    ) -> str:
        """
        Store a fact extracted from voice conversation.

        Args:
            fact: The fact to store
            source: Source of the fact
            importance: Importance score (0.0 to 1.0)

        Returns:
            Memory ID of stored fact
        """

        try:
            memory_id = await self.memory_manager.store_fact(
                fact=fact, source=source, importance=importance
            )

            logger.info(f"Stored conversation fact: {fact[:50]}...")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store conversation fact: {e}")
            raise

    async def search_conversation_memories(
        self, query: str, limit: int = 5, min_relevance: float = 0.5
    ) -> List[MemoryResult]:
        """
        Search conversation memories.

        Args:
            query: Search query
            limit: Maximum number of results
            min_relevance: Minimum relevance score

        Returns:
            List of memory results
        """

        try:
            results = await self.memory_manager.search_memories(
                query=query, limit=limit, min_relevance=min_relevance
            )

            logger.debug(f"Found {len(results)} memories for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get voice - memory integration statistics."""

        return {
            "voice_memory_stats": self.stats.copy(),
            "cache_size": len(self.memory_cache),
            "message_count_since_consolidation": self.message_count,
            "config": {
                "memory_injection_enabled": self.config.enable_memory_injection,
                "auto_storage_enabled": self.config.enable_auto_storage,
                "auto_consolidation_enabled": self.config.enable_auto_consolidation,
                "max_relevant_memories": self.config.max_relevant_memories,
                "relevance_threshold": self.config.memory_relevance_threshold,
            },
        }

    async def cleanup(self) -> None:
        """Clean up integration resources."""

        try:
            # Clear caches
            self.memory_cache.clear()
            self.cache_timestamps.clear()

            # Reset counters
            self.message_count = 0

            logger.info("VoiceMemoryIntegration cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class VoiceMemoryManager:
    """
    High - level manager for voice - memory integration.

    Provides a simplified interface for voice components to interact with memory.
    """

    def __init__(self, memory_manager: MemoryInterface, config: Optional[VoiceMemoryConfig] = None):
        """Initialize voice memory manager."""
        self.memory_manager = memory_manager
        self.config = config or VoiceMemoryConfig()
        self.integration = VoiceMemoryIntegration(memory_manager, self.config)

        logger.info("VoiceMemoryManager initialized")

    async def enhance_voice_context(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]:
        """
        Enhance voice processing context with memory.

        Args:
            voice_message: Voice message to process
            conversation_state: Current conversation state

        Returns:
            Enhanced context dictionary
        """
        return await self.integration.process_voice_message(voice_message, conversation_state)

    async def store_voice_interaction(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None,
    ) -> None:
        """
        Store complete voice interaction in memory.

        Args:
            voice_message: User voice message
            voice_response: Assistant voice response
            conversation_state: Current conversation state
        """

        # Process user message
        await self.integration.process_voice_message(voice_message, conversation_state)

        # Process assistant response
        await self.integration.process_voice_response(voice_response, conversation_state)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return self.integration.get_integration_stats()

    async def cleanup(self) -> None:
        """Clean up manager resources."""
        await self.integration.cleanup()
