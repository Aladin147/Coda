"""
Conversation manager for Coda LLM system.

This module provides conversation management functionality including
message history, context management, and conversation persistence.
"""

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .interfaces import ConversationManagerInterface
from .models import (
    ConversationConfig,
    LLMConversation,
    LLMMessage,
    MessageRole,
)

logger = logging.getLogger("coda.llm.conversation")


class ConversationManager(ConversationManagerInterface):
    """
    Manages LLM conversations with memory and context management.

    Features:
    - Conversation creation and management
    - Message history tracking
    - Context window management
    - Automatic conversation summarization
    - Conversation persistence
    - Token counting and optimization
    """

    def __init__(self, config: Optional[ConversationConfig] = None):
        """
        Initialize the conversation manager.

        Args:
            config: Configuration for conversation management
        """
        self.config = config or ConversationConfig()

        # In-memory conversation storage
        self._conversations: Dict[str, LLMConversation] = {}
        self._conversation_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Context for integration
        self._memory_manager: Optional[Any] = None
        self._llm_provider: Optional[Any] = None

        logger.info("ConversationManager initialized")

    async def create_conversation(self, conversation_id: Optional[str] = None) -> LLMConversation:
        """
        Create a new conversation.

        Args:
            conversation_id: Optional conversation ID, generates one if not provided

        Returns:
            New conversation instance
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # Check if conversation already exists
        if conversation_id in self._conversations:
            logger.warning(f"Conversation {conversation_id} already exists")
            return self._conversations[conversation_id]

        # Create new conversation
        conversation = LLMConversation(
            conversation_id=conversation_id,
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Store conversation
        self._conversations[conversation_id] = conversation
        self._conversation_stats[conversation_id] = {
            "created_at": datetime.now(),
            "message_count": 0,
            "total_tokens": 0,
            "last_activity": datetime.now(),
        }

        logger.info(f"Created conversation: {conversation_id}")
        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation instance or None if not found
        """
        return self._conversations.get(conversation_id)

    async def add_message(self, conversation_id: str, message: LLMMessage) -> None:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation ID
            message: Message to add
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            # Create conversation if it doesn't exist
            conversation = await self.create_conversation(conversation_id)

        # Add message to conversation
        conversation.add_message(message)

        # Update statistics
        stats = self._conversation_stats[conversation_id]
        stats["message_count"] += 1
        stats["last_activity"] = datetime.now()

        # Check if conversation needs summarization
        if (
            len(conversation.messages) >= self.config.summarize_threshold
            and self.config.auto_summarize
        ):
            await self._maybe_summarize_conversation(conversation_id)

        # Store message in memory if enabled
        if self.config.memory_integration_enabled and self._memory_manager:
            await self._store_message_in_memory(conversation_id, message)

        logger.debug(f"Added message to conversation {conversation_id}")

    async def get_conversation_context(
        self, conversation_id: str, max_tokens: Optional[int] = None
    ) -> List[LLMMessage]:
        """
        Get conversation context for LLM with token management.

        Args:
            conversation_id: Conversation ID
            max_tokens: Maximum tokens to include in context

        Returns:
            List of messages within token limit
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []

        max_tokens = max_tokens or self.config.context_window_size

        # Start with all messages
        messages = conversation.messages.copy()

        # If no token limit, return recent messages
        if max_tokens <= 0:
            return messages[-self.config.max_messages :]

        # Estimate tokens and trim if necessary
        context_messages = []
        current_tokens = 0

        # Add messages from most recent backwards
        for message in reversed(messages):
            message_tokens = self._estimate_message_tokens(message)

            if current_tokens + message_tokens > max_tokens:
                break

            context_messages.insert(0, message)
            current_tokens += message_tokens

        # Ensure we have at least the system message if present
        if messages and messages[0].role == MessageRole.SYSTEM:
            if not context_messages or context_messages[0].role != MessageRole.SYSTEM:
                # Remove oldest message to make room for system message
                if context_messages:
                    context_messages.pop()
                context_messages.insert(0, messages[0])

        logger.debug(
            f"Retrieved context for {conversation_id}: {len(context_messages)} messages, ~{current_tokens} tokens"
        )
        return context_messages

    async def summarize_conversation(self, conversation_id: str) -> str:
        """
        Summarize a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation summary
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation or len(conversation.messages) < 2:
            return "No conversation to summarize"

        try:
            # Create summary prompt
            messages_text = []
            for msg in conversation.messages:
                role = msg.role.value.title()
                messages_text.append(f"{role}: {msg.content}")

            conversation_text = "\n".join(messages_text)

            # Use LLM to generate summary if available
            if self._llm_provider:
                summary_prompt = f"""Please provide a concise summary of the following conversation:

{conversation_text}

Summary:"""

                # This would use the LLM provider to generate a summary
                # For now, return a simple summary
                summary = f"Conversation with {len(conversation.messages)} messages covering various topics."
            else:
                # Fallback to simple summary
                summary = f"Conversation with {len(conversation.messages)} messages from {conversation.created_at.strftime('%Y-%m-%d %H:%M')}"

            logger.info(f"Generated summary for conversation {conversation_id}")
            return summary

        except Exception as e:
            logger.error(f"Failed to summarize conversation {conversation_id}: {e}")
            return f"Summary generation failed: {e}"

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if deleted, False if not found
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            del self._conversation_stats[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
            return True

        return False

    async def list_conversations(self, limit: int = 50) -> List[LLMConversation]:
        """
        List recent conversations.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversations sorted by last update
        """
        conversations = list(self._conversations.values())

        # Sort by last update time
        conversations.sort(key=lambda c: c.updated_at, reverse=True)

        return conversations[:limit]

    async def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get statistics for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation statistics
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return {}

        stats = self._conversation_stats.get(conversation_id, {})

        return {
            "conversation_id": conversation_id,
            "message_count": len(conversation.messages),
            "total_tokens": conversation.total_tokens,
            "total_cost": conversation.total_cost,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "duration_minutes": (conversation.updated_at - conversation.created_at).total_seconds()
            / 60,
            **stats,
        }

    async def cleanup_old_conversations(self, days: int = 30) -> int:
        """
        Cleanup old conversations.

        Args:
            days: Delete conversations older than this many days

        Returns:
            Number of conversations deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0

        conversations_to_delete = []
        for conv_id, conversation in self._conversations.items():
            if conversation.updated_at < cutoff_date:
                conversations_to_delete.append(conv_id)

        for conv_id in conversations_to_delete:
            await self.delete_conversation(conv_id)
            deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old conversations")
        return deleted_count

    def set_memory_manager(self, memory_manager: Any) -> None:
        """Set memory manager for integration."""
        self._memory_manager = memory_manager
        logger.info("Memory manager set for conversation manager")

    def set_llm_provider(self, llm_provider: Any) -> None:
        """Set LLM provider for summarization."""
        self._llm_provider = llm_provider
        logger.info("LLM provider set for conversation manager")

    async def _maybe_summarize_conversation(self, conversation_id: str) -> None:
        """Maybe summarize conversation if it's getting too long."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return

        # Check if summarization is needed
        if len(conversation.messages) < self.config.summarize_threshold:
            return

        try:
            # Generate summary
            summary = await self.summarize_conversation(conversation_id)

            # Replace old messages with summary
            system_messages = [
                msg for msg in conversation.messages if msg.role == MessageRole.SYSTEM
            ]
            recent_messages = conversation.messages[-10:]  # Keep last 10 messages

            # Create summary message
            summary_message = LLMMessage(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {summary}",
                timestamp=datetime.now(),
            )

            # Replace messages
            conversation.messages = system_messages + [summary_message] + recent_messages
            conversation.updated_at = datetime.now()

            logger.info(f"Summarized conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to auto-summarize conversation {conversation_id}: {e}")

    async def _store_message_in_memory(self, conversation_id: str, message: LLMMessage) -> None:
        """Store message in memory system."""
        if not self._memory_manager:
            return

        try:
            # Store important messages in memory
            if message.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                await self._memory_manager.store_memory(
                    content=f"{message.role.value}: {message.content}",
                    category="conversation",
                    tags=["llm", "conversation", conversation_id],
                    importance=0.6,
                    source="conversation_manager",
                )
        except Exception as e:
            logger.warning(f"Failed to store message in memory: {e}")

    def _estimate_message_tokens(self, message: LLMMessage) -> int:
        """Estimate token count for a message."""
        # Simple estimation: role + content
        role_tokens = 4  # Approximate tokens for role formatting
        content_tokens = len(message.content) // 4  # Rough approximation
        return role_tokens + content_tokens

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get conversation manager statistics."""
        total_conversations = len(self._conversations)
        total_messages = sum(len(conv.messages) for conv in self._conversations.values())
        total_tokens = sum(conv.total_tokens for conv in self._conversations.values())
        total_cost = sum(conv.total_cost for conv in self._conversations.values())

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "memory_integration_enabled": self.config.memory_integration_enabled,
            "auto_summarize_enabled": self.config.auto_summarize,
            "max_messages_per_conversation": self.config.max_messages,
            "context_window_size": self.config.context_window_size,
        }
