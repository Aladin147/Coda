"""
Memory encoder for Coda.

This module provides a MemoryEncoder class for converting conversation turns
and other content into memory chunks suitable for long-term storage.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .interfaces import MemoryEncoderInterface
from .models import (
    ConversationRole,
    ConversationTurn,
    MemoryChunk,
    MemoryEncoderConfig,
    MemoryType,
)

logger = logging.getLogger("coda.memory.encoder")


class MemoryEncoder(MemoryEncoderInterface):
    """
    Encodes conversations and content into memory chunks for long-term storage.

    Features:
    - Intelligent conversation chunking with overlap
    - Importance scoring based on keywords and patterns
    - Topic extraction from content
    - Speaker grouping and turn consolidation
    - Configurable chunk sizes and overlap
    """

    def __init__(self, config: Optional[MemoryEncoderConfig] = None):
        """
        Initialize the memory encoder.

        Args:
            config: Configuration for memory encoding
        """
        self.config = config or MemoryEncoderConfig()

        # Compile regex patterns for efficiency
        self._importance_pattern = re.compile(
            r"\b(" + "|".join(self.config.importance_keywords) + r")\b", re.IGNORECASE
        )
        self._topic_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

        logger.info(f"MemoryEncoder initialized with chunk_size={self.config.chunk_size}")

    def encode_conversation(
        self, turns: List[ConversationTurn], include_system: bool = False
    ) -> List[MemoryChunk]:
        """
        Encode conversation turns into memory chunks.

        Args:
            turns: List of conversation turns
            include_system: Whether to include system messages

        Returns:
            List of memory chunks
        """
        if not turns:
            return []

        # Filter turns if needed
        filtered_turns = turns
        if not include_system:
            filtered_turns = [t for t in turns if t.role != ConversationRole.SYSTEM]

        if not filtered_turns:
            return []

        # Group consecutive turns by the same speaker
        grouped_turns = self._group_consecutive_turns(filtered_turns)

        # Create chunks from grouped turns
        chunks = []
        for group in grouped_turns:
            group_chunks = self._create_chunks_from_group(group)
            chunks.extend(group_chunks)

        logger.info(f"Encoded {len(chunks)} memory chunks from {len(turns)} turns")
        return chunks

    def encode_fact(
        self, fact: str, source: str = "user", metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryChunk:
        """
        Encode a fact into a memory chunk.

        Args:
            fact: The fact text
            source: Source of the fact
            metadata: Additional metadata

        Returns:
            Memory chunk for the fact
        """
        if metadata is None:
            metadata = {}

        # Extract topics from the fact
        topics = self.extract_topics(fact) if self.config.topic_extraction_enabled else []

        return MemoryChunk(
            content=fact, speakers=[source], turn_ids=[], timestamp=datetime.now(), topics=topics
        )

    def calculate_importance(self, content: str) -> float:
        """
        Calculate importance score for content.

        Args:
            content: Content to analyze

        Returns:
            Importance score between 0.0 and 1.0
        """
        if not content.strip():
            return 0.0

        base_score = 0.5

        # Check for importance keywords
        importance_matches = self._importance_pattern.findall(content)
        keyword_boost = min(len(importance_matches) * 0.1, 0.3)

        # Length factor (longer content might be more important)
        length_factor = min(len(content) / 1000, 0.2)

        # Question factor (questions might be more important)
        question_count = content.count("?")
        question_boost = min(question_count * 0.05, 0.1)

        # Exclamation factor (emphasis might indicate importance)
        exclamation_count = content.count("!")
        exclamation_boost = min(exclamation_count * 0.02, 0.05)

        # Calculate final score
        final_score = (
            base_score + keyword_boost + length_factor + question_boost + exclamation_boost
        )

        return min(final_score, 1.0)

    def extract_topics(self, content: str) -> List[str]:
        """
        Extract topics from content.

        Args:
            content: Content to analyze

        Returns:
            List of extracted topics
        """
        if not self.config.topic_extraction_enabled:
            return []

        topics = set()

        # Extract capitalized phrases (potential proper nouns/topics)
        topic_matches = self._topic_pattern.findall(content)
        for match in topic_matches:
            # Filter out common words and short phrases
            if len(match) > 3 and not self._is_common_word(match.lower()):
                topics.add(match.lower())

        # Extract hashtag-like patterns
        hashtag_pattern = re.compile(r"#(\w+)")
        hashtag_matches = hashtag_pattern.findall(content)
        topics.update(tag.lower() for tag in hashtag_matches)

        # Extract quoted phrases as potential topics
        quote_pattern = re.compile(r'"([^"]+)"')
        quote_matches = quote_pattern.findall(content)
        for quote in quote_matches:
            if 3 < len(quote) < 50:  # Reasonable length for topics
                topics.add(quote.lower())

        return list(topics)[:10]  # Limit to top 10 topics

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the overlap region
                search_start = max(start, end - self.config.chunk_overlap)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > start:
                    end = sentence_end

            chunk = text[start:end].strip()
            if len(chunk) >= self.config.min_chunk_length:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.config.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _group_consecutive_turns(
        self, turns: List[ConversationTurn]
    ) -> List[List[ConversationTurn]]:
        """Group consecutive turns by the same speaker."""
        if not turns:
            return []

        groups = []
        current_group = [turns[0]]

        for turn in turns[1:]:
            if turn.role == current_group[-1].role:
                current_group.append(turn)
            else:
                groups.append(current_group)
                current_group = [turn]

        if current_group:
            groups.append(current_group)

        return groups

    def _create_chunks_from_group(self, group: List[ConversationTurn]) -> List[MemoryChunk]:
        """Create memory chunks from a group of turns by the same speaker."""
        if not group:
            return []

        # Combine content from all turns in the group
        combined_content = " ".join(turn.content for turn in group)
        speakers = [group[0].role.value]
        turn_ids = [turn.turn_id for turn in group]
        timestamp = group[0].timestamp

        # Extract topics from combined content
        topics = (
            self.extract_topics(combined_content) if self.config.topic_extraction_enabled else []
        )

        # Split into chunks if content is too long
        text_chunks = self.chunk_text(combined_content)

        memory_chunks = []
        for chunk_text in text_chunks:
            memory_chunks.append(
                MemoryChunk(
                    content=chunk_text,
                    speakers=speakers,
                    turn_ids=turn_ids,
                    timestamp=timestamp,
                    topics=topics,
                )
            )

        return memory_chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find a good sentence boundary within the given range."""
        # Look for sentence endings (., !, ?)
        sentence_endings = [".", "!", "?"]

        # Search backwards from end to start
        for i in range(end - 1, start - 1, -1):
            if text[i] in sentence_endings:
                # Make sure it's not an abbreviation or decimal
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1

        # If no sentence boundary found, look for other boundaries
        other_boundaries = ["\n", ";", ","]
        for i in range(end - 1, start - 1, -1):
            if text[i] in other_boundaries:
                return i + 1

        # If no good boundary found, return the original end
        return end

    def _is_common_word(self, word: str) -> bool:
        """Check if a word is a common word that shouldn't be a topic."""
        common_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "myself",
            "yourself",
            "himself",
            "herself",
            "itself",
            "ourselves",
            "yourselves",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "where",
            "when",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "can",
            "will",
            "just",
            "should",
            "now",
            "hello",
            "hi",
            "hey",
            "okay",
            "ok",
            "yes",
            "no",
            "please",
            "thank",
            "thanks",
            "sorry",
            "excuse",
            "well",
            "actually",
            "really",
            "maybe",
            "perhaps",
            "probably",
            "definitely",
            "certainly",
        }
        return word.lower() in common_words

    def get_encoding_stats(self, chunks: List[MemoryChunk]) -> Dict[str, Any]:
        """Get statistics about encoded chunks."""
        if not chunks:
            return {
                "total_chunks": 0,
                "average_length": 0,
                "total_length": 0,
                "unique_speakers": 0,
                "unique_topics": 0,
                "average_topics_per_chunk": 0,
            }

        total_length = sum(len(chunk.content) for chunk in chunks)
        all_speakers = set()
        all_topics = set()

        for chunk in chunks:
            all_speakers.update(chunk.speakers)
            all_topics.update(chunk.topics)

        return {
            "total_chunks": len(chunks),
            "average_length": total_length / len(chunks),
            "total_length": total_length,
            "unique_speakers": len(all_speakers),
            "unique_topics": len(all_topics),
            "average_topics_per_chunk": sum(len(chunk.topics) for chunk in chunks) / len(chunks),
        }
