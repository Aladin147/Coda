"""
Short-term memory implementation for Coda.

This module provides a ShortTermMemory class for handling conversation context
within token limits and managing recent conversation history.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from .interfaces import ShortTermMemoryInterface
from .models import (
    ContextMessage,
    ConversationContext,
    ConversationRole,
    ConversationTurn,
    ShortTermMemoryConfig,
)

logger = logging.getLogger("coda.memory.short_term")


class ShortTermMemory(ShortTermMemoryInterface):
    """
    Manages short-term conversation memory for Coda.

    Features:
    - Store conversation turns with automatic pruning
    - Provide context within token limits
    - Support system message handling
    - Export/import for debugging and persistence
    - Thread-safe operations
    """

    def __init__(self, config: Optional[ShortTermMemoryConfig] = None):
        """
        Initialize the short-term memory.

        Args:
            config: Configuration for short-term memory
        """
        self.config = config or ShortTermMemoryConfig()
        self.turns: deque[ConversationTurn] = deque(maxlen=self.config.max_turns)
        self.session_start = datetime.now()
        self.turn_count = 0

        logger.info(f"ShortTermMemory initialized with max_turns={self.config.max_turns}")

    def add_turn(self, role: str, content: str) -> ConversationTurn:
        """
        Add a new conversation turn.

        Args:
            role: The speaker role ("system", "user", or "assistant")
            content: The message content

        Returns:
            The created turn object

        Raises:
            ValueError: If role is invalid
        """
        try:
            conversation_role = ConversationRole(role)
        except ValueError:
            raise ValueError(f"Invalid role '{role}', expected one of: {list(ConversationRole)}")

        turn = ConversationTurn(
            role=conversation_role,
            content=content,
            turn_id=self.turn_count,
            timestamp=datetime.now(),
        )

        self.turns.append(turn)
        self.turn_count += 1

        logger.debug(f"Added turn {self.turn_count} with role '{role}' ({len(content)} chars)")
        return turn

    def get_context(self, max_tokens: int = 800) -> ConversationContext:
        """
        Get conversation context within token budget.

        Args:
            max_tokens: Maximum number of tokens to include

        Returns:
            ConversationContext with messages and metadata
        """
        if max_tokens <= 0:
            return ConversationContext(
                messages=[], total_tokens=0, short_term_turns=0, long_term_memories=0
            )

        messages: List[ContextMessage] = []
        token_count = 0

        # Always include system message if present and configured
        if self.config.include_system_in_context:
            system_turns = [t for t in self.turns if t.role == ConversationRole.SYSTEM]
            if system_turns:
                system_turn = system_turns[0]  # Use the first system message
                system_tokens = self._estimate_tokens(system_turn.content)

                if system_tokens <= max_tokens:
                    messages.append(
                        ContextMessage(role=system_turn.role, content=system_turn.content)
                    )
                    token_count += system_tokens
                    logger.debug(f"Added system message to context ({system_tokens} tokens)")

        # Process non-system turns from newest to oldest
        non_system_turns = [t for t in self.turns if t.role != ConversationRole.SYSTEM]
        selected_turns = []

        for turn in reversed(non_system_turns):
            turn_tokens = self._estimate_tokens(turn.content)

            # Check if we have room for this turn
            if token_count + turn_tokens > max_tokens:
                logger.debug(f"Token limit reached, stopping at turn {turn.turn_id}")
                break

            # Add turn to selected list (will be reversed later to maintain chronological order)
            selected_turns.append(turn)
            token_count += turn_tokens

            logger.debug(f"Added turn {turn.turn_id} to context ({turn_tokens} tokens)")

        # Add selected turns in chronological order (oldest first)
        for turn in reversed(selected_turns):
            messages.append(ContextMessage(role=turn.role, content=turn.content))

        return ConversationContext(
            messages=messages,
            total_tokens=token_count,
            short_term_turns=len([m for m in messages if m.role != ConversationRole.SYSTEM]),
            long_term_memories=0,  # No long-term memories in short-term context
        )

    def get_recent_turns(self, count: int = 10) -> List[ConversationTurn]:
        """
        Get the most recent conversation turns.

        Args:
            count: Number of recent turns to return

        Returns:
            List of recent conversation turns
        """
        if count <= 0:
            return []

        recent_turns = list(self.turns)[-count:]
        logger.debug(f"Retrieved {len(recent_turns)} recent turns")
        return recent_turns

    def clear(self) -> None:
        """Clear all conversation turns."""
        turn_count_before = len(self.turns)
        self.turns.clear()
        self.turn_count = 0
        self.session_start = datetime.now()

        logger.info(f"Cleared {turn_count_before} turns from short-term memory")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about short-term memory.

        Returns:
            Dictionary with memory statistics
        """
        if not self.turns:
            return {
                "total_turns": 0,
                "roles": {},
                "session_duration_seconds": 0,
                "average_content_length": 0,
                "total_content_length": 0,
                "oldest_turn": None,
                "newest_turn": None,
            }

        # Calculate role distribution
        role_counts = {}
        total_content_length = 0

        for turn in self.turns:
            role_str = turn.role.value
            role_counts[role_str] = role_counts.get(role_str, 0) + 1
            total_content_length += len(turn.content)

        # Calculate session duration
        session_duration = (datetime.now() - self.session_start).total_seconds()

        # Get oldest and newest turns
        oldest_turn = min(self.turns, key=lambda t: t.timestamp)
        newest_turn = max(self.turns, key=lambda t: t.timestamp)

        return {
            "total_turns": len(self.turns),
            "roles": role_counts,
            "session_duration_seconds": session_duration,
            "average_content_length": total_content_length / len(self.turns),
            "total_content_length": total_content_length,
            "oldest_turn": oldest_turn.timestamp,
            "newest_turn": newest_turn.timestamp,
            "max_turns": self.config.max_turns,
            "max_tokens": self.config.max_tokens,
        }

    def export_turns(self) -> List[Dict[str, Any]]:
        """
        Export turns for persistence or debugging.

        Returns:
            List of turn dictionaries
        """
        exported = []
        for turn in self.turns:
            exported.append(
                {
                    "role": turn.role.value,
                    "content": turn.content,
                    "timestamp": turn.timestamp.isoformat(),
                    "turn_id": turn.turn_id,
                    "metadata": turn.metadata,
                }
            )

        logger.debug(f"Exported {len(exported)} turns")
        return exported

    def import_turns(self, turns_data: List[Dict[str, Any]]) -> int:
        """
        Import turns from exported data.

        Args:
            turns_data: List of turn dictionaries

        Returns:
            Number of turns imported
        """
        imported_count = 0

        for turn_data in turns_data:
            try:
                turn = ConversationTurn(
                    role=ConversationRole(turn_data["role"]),
                    content=turn_data["content"],
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    turn_id=turn_data["turn_id"],
                    metadata=turn_data.get("metadata", {}),
                )

                self.turns.append(turn)
                imported_count += 1

                # Update turn counter
                if turn.turn_id >= self.turn_count:
                    self.turn_count = turn.turn_id + 1

            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to import turn: {e}")
                continue

        logger.info(f"Imported {imported_count} turns")
        return imported_count

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.

        This is a simple approximation. For more accurate token counting,
        integrate with the actual tokenizer used by the LLM.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token for English text
        return max(1, len(text) // 4)

    def get_turn_by_id(self, turn_id: int) -> Optional[ConversationTurn]:
        """
        Get a specific turn by its ID.

        Args:
            turn_id: The turn ID to search for

        Returns:
            The turn if found, None otherwise
        """
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def get_turns_by_role(self, role: ConversationRole) -> List[ConversationTurn]:
        """
        Get all turns by a specific role.

        Args:
            role: The role to filter by

        Returns:
            List of turns with the specified role
        """
        return [turn for turn in self.turns if turn.role == role]

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.

        Returns:
            Dictionary with conversation summary
        """
        if not self.turns:
            return {"summary": "No conversation yet"}

        stats = self.get_stats()
        recent_topics = self._extract_recent_topics()

        return {
            "total_turns": stats["total_turns"],
            "duration_minutes": stats["session_duration_seconds"] / 60,
            "participants": list(stats["roles"].keys()),
            "recent_topics": recent_topics,
            "last_interaction": stats["newest_turn"],
        }

    def _extract_recent_topics(self, count: int = 5) -> List[str]:
        """
        Extract topics from recent conversation turns.

        This is a simple keyword-based approach. For better topic extraction,
        integrate with NLP libraries or the memory encoder.

        Args:
            count: Number of recent turns to analyze

        Returns:
            List of extracted topics
        """
        recent_turns = self.get_recent_turns(count)

        # Simple keyword extraction (this could be improved)
        keywords = set()
        for turn in recent_turns:
            if turn.role == ConversationRole.USER:
                # Extract potential topics from user messages
                words = turn.content.lower().split()
                # Filter out common words and keep potential topics
                topic_words = [w for w in words if len(w) > 3 and w.isalpha()]
                keywords.update(topic_words[:3])  # Take first 3 words as potential topics

        return list(keywords)[:5]  # Return top 5 topics
