"""
Session management for Coda.

This module provides the SessionManager class for tracking conversation
sessions and managing session closure behavior.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import SessionManagerInterface
from .models import (
    PersonalityTraitType,
    SessionManagerConfig,
    SessionState,
)

logger = logging.getLogger("coda.personality.session_manager")


class SessionManager(SessionManagerInterface):
    """
    Manages conversation session state and closure behavior.

    Features:
    - Session timing and interaction tracking
    - Intelligent closure mode detection
    - Session summary generation
    - Personality adjustments for session closure
    - Session analytics and patterns
    """

    def __init__(
        self, config: Optional[SessionManagerConfig] = None, session_id: Optional[str] = None
    ):
        """
        Initialize the session manager.

        Args:
            config: Configuration for session management
            session_id: Optional session ID (generates one if not provided)
        """
        self.config = config or SessionManagerConfig()

        # Initialize session state
        self.session_state = SessionState(session_id=session_id or str(uuid.uuid4()))

        # Session closure thresholds
        self.long_session_threshold = timedelta(minutes=self.config.long_session_threshold_minutes)
        self.idle_threshold = timedelta(minutes=self.config.idle_threshold_minutes)

        # Session analytics
        self.interaction_history: List[Dict[str, Any]] = []
        self.session_topics: List[str] = []
        self.user_engagement_scores: List[float] = []

        logger.info(f"SessionManager initialized for session {self.session_state.session_id}")

    def get_session_state(self) -> SessionState:
        """Get the current session state."""
        self.session_state.update()
        return self.session_state

    def update_session(self) -> SessionState:
        """Update session timing and state."""
        self.session_state.update()

        # Check if we should enter closure mode
        should_close, reason = self.should_enter_closure_mode()
        if should_close and not self.session_state.in_closure_mode:
            self.session_state.enter_closure_mode(reason)
            logger.info(f"Session entered closure mode: {reason}")

        return self.session_state

    def process_interaction(self, role: str, content: str) -> None:
        """
        Process a new interaction in the session.

        Args:
            role: Role of the speaker ("user" or "assistant")
            content: Content of the interaction
        """
        # Update session state
        self.session_state.add_interaction()

        # Record interaction
        interaction = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "turn_number": self.session_state.turn_count,
            "session_duration": self.session_state.duration_seconds,
        }

        self.interaction_history.append(interaction)

        # Analyze user engagement if it's a user interaction
        if role == "user":
            engagement_score = self._analyze_user_engagement(content)
            self.user_engagement_scores.append(engagement_score)

            # Keep only recent engagement scores
            if len(self.user_engagement_scores) > 20:
                self.user_engagement_scores = self.user_engagement_scores[-20:]

        # Trim interaction history if needed
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

        logger.debug(f"Processed {role} interaction (turn {self.session_state.turn_count})")

    def should_enter_closure_mode(self) -> Tuple[bool, str]:
        """
        Check if session should enter closure mode.

        Returns:
            Tuple of (should_close, reason)
        """
        if self.session_state.in_closure_mode:
            return False, ""

        # Check session duration
        if self.session_state.duration_seconds > self.long_session_threshold.total_seconds():
            return True, f"Long session ({self.session_state.duration_seconds/60:.1f} minutes)"

        # Check idle time
        if self.session_state.idle_seconds > self.idle_threshold.total_seconds():
            return True, f"Idle timeout ({self.session_state.idle_seconds/60:.1f} minutes)"

        # Check for natural conversation endings
        if self._detect_natural_ending():
            return True, "Natural conversation ending detected"

        # Check for low engagement patterns
        if self._detect_low_engagement():
            return True, "Low user engagement detected"

        return False, ""

    def enter_closure_mode(self, reason: str) -> Dict[PersonalityTraitType, float]:
        """
        Enter closure mode and get personality adjustments.

        Args:
            reason: Reason for entering closure mode

        Returns:
            Dictionary of personality trait adjustments
        """
        self.session_state.enter_closure_mode(reason)

        # Default closure adjustments
        closure_adjustments = {
            PersonalityTraitType.PROACTIVITY: 0.3,  # More proactive in wrapping up
            PersonalityTraitType.EMPATHY: 0.2,  # More empathetic in closure
            PersonalityTraitType.VERBOSITY: -0.2,  # More concise
            PersonalityTraitType.FORMALITY: 0.1,  # Slightly more formal
        }

        # Merge with config adjustments if provided
        if self.config.closure_mode_adjustments:
            closure_adjustments.update(self.config.closure_mode_adjustments)

        logger.info(f"Applied closure mode adjustments: {closure_adjustments}")
        return closure_adjustments

    def generate_session_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the current session.

        Returns:
            Dictionary with session summary
        """
        if not self.interaction_history:
            return {"summary": "No interactions in this session"}

        # Basic session metrics
        user_interactions = [i for i in self.interaction_history if i["role"] == "user"]
        assistant_interactions = [i for i in self.interaction_history if i["role"] == "assistant"]

        # Calculate engagement metrics
        avg_engagement = (
            sum(self.user_engagement_scores) / len(self.user_engagement_scores)
            if self.user_engagement_scores
            else 0.0
        )

        # Analyze interaction patterns
        interaction_intervals = []
        for i in range(1, len(self.interaction_history)):
            prev_time = self.interaction_history[i - 1]["timestamp"]
            curr_time = self.interaction_history[i]["timestamp"]
            interval = (curr_time - prev_time).total_seconds()
            interaction_intervals.append(interval)

        avg_response_time = (
            sum(interaction_intervals) / len(interaction_intervals)
            if interaction_intervals
            else 0.0
        )

        # Extract topics (simple keyword extraction)
        all_content = " ".join([i["content"] for i in user_interactions])
        topics = self._extract_session_topics(all_content)

        # Generate summary
        summary = {
            "session_id": self.session_state.session_id,
            "duration_minutes": self.session_state.duration_seconds / 60,
            "total_turns": self.session_state.turn_count,
            "user_messages": len(user_interactions),
            "assistant_messages": len(assistant_interactions),
            "average_engagement": avg_engagement,
            "average_response_time_seconds": avg_response_time,
            "topics_discussed": topics,
            "session_quality": self._assess_session_quality(),
            "closure_reason": self.session_state.closure_reason,
            "started_at": self.session_state.start_time.isoformat(),
            "last_interaction": self.session_state.last_interaction.isoformat(),
        }

        return summary

    def reset_session(self, new_session_id: str = None) -> None:
        """
        Reset session state.

        Args:
            new_session_id: Optional new session ID
        """
        old_session_id = self.session_state.session_id

        # Create new session state
        self.session_state = SessionState(session_id=new_session_id or str(uuid.uuid4()))

        # Clear history
        self.interaction_history.clear()
        self.session_topics.clear()
        self.user_engagement_scores.clear()

        logger.info(f"Session reset: {old_session_id} -> {self.session_state.session_id}")

    def get_session_analytics(self) -> Dict[str, Any]:
        """Get detailed session analytics."""
        if not self.interaction_history:
            return {"analytics": "No data available"}

        # Interaction patterns
        user_message_lengths = [
            len(i["content"]) for i in self.interaction_history if i["role"] == "user"
        ]
        assistant_message_lengths = [
            len(i["content"]) for i in self.interaction_history if i["role"] == "assistant"
        ]

        # Timing patterns
        interaction_times = [i["timestamp"] for i in self.interaction_history]
        if len(interaction_times) > 1:
            intervals = [
                (interaction_times[i] - interaction_times[i - 1]).total_seconds()
                for i in range(1, len(interaction_times))
            ]
        else:
            intervals = []

        # Engagement trends
        engagement_trend = "stable"
        if len(self.user_engagement_scores) >= 3:
            recent_avg = sum(self.user_engagement_scores[-3:]) / 3
            early_avg = sum(self.user_engagement_scores[:3]) / 3
            if recent_avg > early_avg + 0.2:
                engagement_trend = "increasing"
            elif recent_avg < early_avg - 0.2:
                engagement_trend = "decreasing"

        return {
            "interaction_patterns": {
                "avg_user_message_length": (
                    sum(user_message_lengths) / len(user_message_lengths)
                    if user_message_lengths
                    else 0
                ),
                "avg_assistant_message_length": (
                    sum(assistant_message_lengths) / len(assistant_message_lengths)
                    if assistant_message_lengths
                    else 0
                ),
                "avg_interaction_interval": sum(intervals) / len(intervals) if intervals else 0,
                "total_interactions": len(self.interaction_history),
            },
            "engagement_analysis": {
                "average_engagement": (
                    sum(self.user_engagement_scores) / len(self.user_engagement_scores)
                    if self.user_engagement_scores
                    else 0
                ),
                "engagement_trend": engagement_trend,
                "engagement_scores": self.user_engagement_scores[-10:],  # Last 10 scores
            },
            "session_flow": {
                "session_phase": self._determine_session_phase(),
                "conversation_momentum": self._calculate_conversation_momentum(),
                "topic_coherence": self._assess_topic_coherence(),
            },
        }

    def _analyze_user_engagement(self, content: str) -> float:
        """Analyze user engagement from message content."""
        engagement_score = 0.5  # Base score

        # Length factor (longer messages often indicate higher engagement)
        length_factor = min(len(content) / 100, 0.3)
        engagement_score += length_factor

        # Question factor (questions indicate engagement)
        question_count = content.count("?")
        engagement_score += min(question_count * 0.1, 0.2)

        # Exclamation factor (enthusiasm)
        exclamation_count = content.count("!")
        engagement_score += min(exclamation_count * 0.05, 0.1)

        # Positive words
        positive_words = [
            "great",
            "awesome",
            "excellent",
            "love",
            "amazing",
            "fantastic",
            "wonderful",
        ]
        positive_count = sum(1 for word in positive_words if word.lower() in content.lower())
        engagement_score += min(positive_count * 0.1, 0.2)

        # Negative indicators
        short_responses = ["ok", "sure", "fine", "whatever", "yes", "no"]
        if content.lower().strip() in short_responses:
            engagement_score -= 0.3

        return max(0.0, min(1.0, engagement_score))

    def _detect_natural_ending(self) -> bool:
        """Detect if the conversation is naturally ending."""
        if len(self.interaction_history) < 4:
            return False

        # Check recent user messages for ending indicators
        recent_user_messages = [
            i["content"] for i in self.interaction_history[-4:] if i["role"] == "user"
        ]

        ending_phrases = [
            "thanks",
            "thank you",
            "goodbye",
            "bye",
            "see you",
            "that's all",
            "that helps",
            "perfect",
            "got it",
            "understood",
            "that's enough",
        ]

        for message in recent_user_messages:
            if any(phrase in message.lower() for phrase in ending_phrases):
                return True

        return False

    def _detect_low_engagement(self) -> bool:
        """Detect patterns of low user engagement."""
        if len(self.user_engagement_scores) < 5:
            return False

        # Check if recent engagement is consistently low
        recent_scores = self.user_engagement_scores[-5:]
        avg_recent_engagement = sum(recent_scores) / len(recent_scores)

        return avg_recent_engagement < 0.3

    def _extract_session_topics(self, content: str) -> List[str]:
        """Extract main topics from session content."""
        # Simple keyword-based topic extraction
        words = content.lower().split()

        # Common topic keywords
        topic_keywords = {
            "programming": ["code", "programming", "python", "javascript", "software"],
            "learning": ["learn", "study", "understand", "explain", "teach"],
            "creative": ["create", "design", "art", "music", "write"],
            "personal": ["feel", "think", "personal", "life", "experience"],
            "work": ["work", "job", "business", "project", "career"],
            "technology": ["technology", "computer", "ai", "machine", "data"],
        }

        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in words for keyword in keywords):
                detected_topics.append(topic)

        return detected_topics[:5]  # Return top 5 topics

    def _assess_session_quality(self) -> str:
        """Assess the overall quality of the session."""
        if not self.user_engagement_scores:
            return "unknown"

        avg_engagement = sum(self.user_engagement_scores) / len(self.user_engagement_scores)
        turn_count = self.session_state.turn_count

        if avg_engagement > 0.7 and turn_count > 5:
            return "excellent"
        elif avg_engagement > 0.5 and turn_count > 3:
            return "good"
        elif avg_engagement > 0.3:
            return "fair"
        else:
            return "poor"

    def _determine_session_phase(self) -> str:
        """Determine the current phase of the session."""
        if self.session_state.turn_count <= 2:
            return "opening"
        elif self.session_state.in_closure_mode:
            return "closing"
        elif (
            self.session_state.duration_seconds > self.long_session_threshold.total_seconds() * 0.7
        ):
            return "late"
        else:
            return "active"

    def _calculate_conversation_momentum(self) -> float:
        """Calculate the momentum of the conversation."""
        if len(self.interaction_history) < 3:
            return 0.5

        # Recent interaction frequency
        recent_interactions = self.interaction_history[-5:]
        if len(recent_interactions) < 2:
            return 0.5

        intervals = []
        for i in range(1, len(recent_interactions)):
            interval = (
                recent_interactions[i]["timestamp"] - recent_interactions[i - 1]["timestamp"]
            ).total_seconds()
            intervals.append(interval)

        avg_interval = sum(intervals) / len(intervals)

        # Convert to momentum score (faster = higher momentum)
        if avg_interval < 30:
            return 0.9
        elif avg_interval < 60:
            return 0.7
        elif avg_interval < 120:
            return 0.5
        else:
            return 0.3

    def _assess_topic_coherence(self) -> float:
        """Assess how coherent the topic flow has been."""
        if len(self.session_topics) < 2:
            return 1.0

        # Simple coherence based on topic consistency
        # This is a placeholder - could be enhanced with more sophisticated analysis
        unique_topics = len(set(self.session_topics))
        total_topics = len(self.session_topics)

        coherence = 1.0 - (unique_topics / total_topics) if total_topics > 0 else 1.0
        return max(0.0, min(1.0, coherence))
