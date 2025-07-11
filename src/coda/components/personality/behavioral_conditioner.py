"""
Behavioral conditioning system for Coda.

This module provides the BehavioralConditioner class for learning user behavior
patterns and adapting personality accordingly.
"""

import asyncio
import logging
import re
from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .interfaces import BehavioralConditionerInterface
from .models import (
    BehavioralConditioningConfig,
    BehaviorPattern,
    BehaviorProfile,
    BehaviorSignal,
    PersonalityTraitType,
)

logger = logging.getLogger("coda.personality.behavioral_conditioner")


class BehavioralConditioner(BehavioralConditionerInterface):
    """
    Learns from user behavior patterns and suggests personality adjustments.

    Features:
    - Explicit feedback detection and processing
    - Implicit behavior pattern recognition
    - User preference learning with confidence scoring
    - Personality adjustment suggestions based on learned patterns
    - Interaction history analysis
    """

    def __init__(self, config: Optional[BehavioralConditioningConfig] = None):
        """
        Initialize the behavioral conditioner.

        Args:
            config: Configuration for behavioral conditioning
        """
        self.config = config or BehavioralConditioningConfig()
        self.behavior_profile = BehaviorProfile()
        self.recent_interactions = deque(maxlen=self.config.max_recent_interactions)

        # Compile regex patterns for efficiency
        self._compile_patterns()

        logger.info("BehavioralConditioner initialized")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for behavior detection."""
        # Explicit preference patterns
        self.preference_patterns = {
            PersonalityTraitType.VERBOSITY: {
                "increase": re.compile(
                    r"\b(more detail|elaborate|explain more|be more verbose|longer response)\b",
                    re.IGNORECASE,
                ),
                "decrease": re.compile(
                    r"\b(be brief|shorter|concise|less detail|summarize)\b", re.IGNORECASE
                ),
            },
            PersonalityTraitType.HUMOR: {
                "increase": re.compile(
                    r"\b(be funny|more humor|make jokes|be playful|lighten up)\b", re.IGNORECASE
                ),
                "decrease": re.compile(
                    r"\b(be serious|no jokes|formal|professional tone)\b", re.IGNORECASE
                ),
            },
            PersonalityTraitType.FORMALITY: {
                "increase": re.compile(
                    r"\b(be formal|professional|proper|official)\b", re.IGNORECASE
                ),
                "decrease": re.compile(r"\b(be casual|informal|relaxed|friendly)\b", re.IGNORECASE),
            },
            PersonalityTraitType.ASSERTIVENESS: {
                "increase": re.compile(
                    r"\b(be confident|assertive|direct|decisive)\b", re.IGNORECASE
                ),
                "decrease": re.compile(
                    r"\b(be gentle|tentative|careful|uncertain)\b", re.IGNORECASE
                ),
            },
            PersonalityTraitType.EMPATHY: {
                "increase": re.compile(
                    r"\b(be understanding|empathetic|caring|supportive)\b", re.IGNORECASE
                ),
                "decrease": re.compile(
                    r"\b(be objective|factual|neutral|detached)\b", re.IGNORECASE
                ),
            },
        }

        # Feedback patterns
        self.feedback_patterns = {
            "positive": re.compile(
                r"\b(good|great|excellent|perfect|helpful|thanks|appreciate)\b", re.IGNORECASE
            ),
            "negative": re.compile(
                r"\b(bad|wrong|unhelpful|confusing|too much|too little)\b", re.IGNORECASE
            ),
            "style_feedback": re.compile(r"\b(tone|style|way you|how you)\b", re.IGNORECASE),
        }

        # Engagement patterns
        self.engagement_patterns = {
            "high": re.compile(
                r"\b(interesting|tell me more|continue|what else|amazing)\b", re.IGNORECASE
            ),
            "low": re.compile(r"\b(okay|fine|whatever|sure|next)\b", re.IGNORECASE),
        }

    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input for behavioral patterns.

        Args:
            user_input: User input text

        Returns:
            Dictionary with detected patterns and suggestions
        """
        # Add to recent interactions
        interaction = {
            "text": user_input,
            "timestamp": datetime.now(),
            "type": "user_input",
            "length": len(user_input),
            "word_count": len(user_input.split()),
        }
        self.recent_interactions.append(interaction)

        # Detect explicit preferences
        explicit_patterns = self._detect_explicit_preferences(user_input)

        # Detect engagement level
        engagement = self._detect_engagement_level(user_input)

        # Analyze interaction style
        style_preferences = self._analyze_interaction_style(user_input)

        results = {
            "explicit_patterns": explicit_patterns,
            "engagement": engagement,
            "style_preferences": style_preferences,
            "suggestions": {},
        }

        # Generate suggestions based on patterns
        if explicit_patterns:
            suggestions = await self._generate_adjustment_suggestions(explicit_patterns)
            results["suggestions"] = suggestions

        return results

    async def process_user_feedback(
        self, feedback: str, feedback_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Process explicit user feedback.

        Args:
            feedback: User feedback text
            feedback_type: Type of feedback (general, style, content, etc.)

        Returns:
            Dictionary with feedback analysis and suggestions
        """
        # Add to recent interactions
        interaction = {
            "text": feedback,
            "timestamp": datetime.now(),
            "type": "user_feedback",
            "feedback_type": feedback_type,
            "length": len(feedback),
        }
        self.recent_interactions.append(interaction)

        # Analyze feedback sentiment
        sentiment = self._analyze_feedback_sentiment(feedback)

        # Detect specific style feedback
        style_feedback = self._detect_style_feedback(feedback)

        # Create behavior pattern
        pattern = BehaviorPattern(
            pattern_type=(
                BehaviorSignal.POSITIVE_FEEDBACK
                if sentiment > 0
                else BehaviorSignal.NEGATIVE_FEEDBACK
            ),
            confidence=abs(sentiment),
            evidence=[feedback],
            suggested_adjustments=style_feedback,
        )

        # Update behavior profile
        await self._update_profile_from_pattern(pattern)

        return {
            "sentiment": sentiment,
            "style_feedback": style_feedback,
            "pattern": pattern.model_dump(),
            "profile_updated": True,
        }

    async def analyze_interaction_patterns(self) -> List[BehaviorPattern]:
        """
        Analyze recent interactions for behavior patterns.

        Returns:
            List of detected behavior patterns
        """
        if len(self.recent_interactions) < self.config.pattern_detection_window:
            return []

        patterns = []

        # Analyze response length preferences
        length_pattern = self._analyze_length_preferences()
        if length_pattern:
            patterns.append(length_pattern)

        # Analyze engagement patterns
        engagement_pattern = self._analyze_engagement_patterns()
        if engagement_pattern:
            patterns.append(engagement_pattern)

        # Analyze interaction frequency
        frequency_pattern = self._analyze_interaction_frequency()
        if frequency_pattern:
            patterns.append(frequency_pattern)

        # Update behavior profile with detected patterns
        for pattern in patterns:
            await self._update_profile_from_pattern(pattern)

        return patterns

    def get_behavior_profile(self) -> BehaviorProfile:
        """Get the current user behavior profile."""
        return self.behavior_profile

    async def update_behavior_profile(self, patterns: List[BehaviorPattern]) -> bool:
        """
        Update behavior profile based on detected patterns.

        Args:
            patterns: List of behavior patterns to incorporate

        Returns:
            True if profile was updated
        """
        try:
            for pattern in patterns:
                await self._update_profile_from_pattern(pattern)
            return True
        except Exception as e:
            logger.error(f"Failed to update behavior profile: {e}")
            return False

    async def suggest_personality_adjustments(self) -> Dict[PersonalityTraitType, float]:
        """
        Suggest personality adjustments based on learned behavior.

        Returns:
            Dictionary of suggested trait adjustments
        """
        suggestions = {}

        # Only suggest adjustments if we have sufficient confidence
        for trait_type, preference in self.behavior_profile.user_preferences.items():
            confidence = self.behavior_profile.confidence_scores.get(trait_type, 0.0)

            if confidence >= self.config.confidence_threshold:
                # Calculate adjustment based on preference and confidence
                # Preference is the target value, so we suggest moving towards it
                current_value = 0.5  # This would come from personality parameters
                target_value = preference

                # Suggest a gradual adjustment weighted by learning rate and confidence
                adjustment = (target_value - current_value) * self.config.learning_rate * confidence

                if abs(adjustment) > 0.05:  # Only suggest significant adjustments
                    suggestions[trait_type] = adjustment

        return suggestions

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about behavioral learning."""
        return {
            "total_interactions": len(self.recent_interactions),
            "observation_count": self.behavior_profile.observation_count,
            "learned_preferences": len(self.behavior_profile.user_preferences),
            "average_confidence": (
                sum(self.behavior_profile.confidence_scores.values())
                / len(self.behavior_profile.confidence_scores)
                if self.behavior_profile.confidence_scores
                else 0.0
            ),
            "detected_patterns": len(self.behavior_profile.detected_patterns),
            "explicit_feedback_count": len(self.behavior_profile.explicit_feedback),
            "last_updated": self.behavior_profile.last_updated.isoformat(),
        }

    def _detect_explicit_preferences(self, text: str) -> Dict[PersonalityTraitType, float]:
        """Detect explicit personality preferences in text."""
        preferences = {}

        for trait_type, patterns in self.preference_patterns.items():
            if patterns["increase"].search(text):
                preferences[trait_type] = 0.8  # High preference for increase
            elif patterns["decrease"].search(text):
                preferences[trait_type] = 0.2  # High preference for decrease

        return preferences

    def _detect_engagement_level(self, text: str) -> Dict[str, Any]:
        """Detect user engagement level from text."""
        high_engagement = len(self.engagement_patterns["high"].findall(text))
        low_engagement = len(self.engagement_patterns["low"].findall(text))

        # Calculate engagement score
        total_signals = high_engagement + low_engagement
        if total_signals > 0:
            engagement_score = high_engagement / total_signals
        else:
            # Use text length and punctuation as indicators
            engagement_score = min(1.0, len(text) / 100)  # Longer text = higher engagement
            if "!" in text or "?" in text:
                engagement_score += 0.2

        return {
            "score": engagement_score,
            "high_signals": high_engagement,
            "low_signals": low_engagement,
            "level": (
                "high" if engagement_score > 0.6 else "low" if engagement_score < 0.4 else "medium"
            ),
        }

    def _analyze_interaction_style(self, text: str) -> Dict[str, Any]:
        """Analyze user's interaction style preferences."""
        style = {"formality": 0.5, "directness": 0.5, "detail_preference": 0.5}

        # Analyze formality
        formal_indicators = len(
            re.findall(r"\b(please|thank you|could you|would you)\b", text, re.IGNORECASE)
        )
        casual_indicators = len(re.findall(r"\b(hey|yeah|ok|cool|awesome)\b", text, re.IGNORECASE))

        if formal_indicators + casual_indicators > 0:
            style["formality"] = formal_indicators / (formal_indicators + casual_indicators)

        # Analyze directness
        direct_indicators = len(
            re.findall(r"\b(tell me|show me|give me|I want|I need)\b", text, re.IGNORECASE)
        )
        indirect_indicators = len(
            re.findall(r"\b(maybe|perhaps|could|might|possibly)\b", text, re.IGNORECASE)
        )

        if direct_indicators + indirect_indicators > 0:
            style["directness"] = direct_indicators / (direct_indicators + indirect_indicators)

        # Analyze detail preference based on question complexity
        question_words = len(
            re.findall(r"\b(how|why|what|when|where|which)\b", text, re.IGNORECASE)
        )
        style["detail_preference"] = min(1.0, question_words / 3)

        return style

    def _analyze_feedback_sentiment(self, feedback: str) -> float:
        """Analyze sentiment of feedback (-1 to 1)."""
        positive_matches = len(self.feedback_patterns["positive"].findall(feedback))
        negative_matches = len(self.feedback_patterns["negative"].findall(feedback))

        total_matches = positive_matches + negative_matches
        if total_matches == 0:
            return 0.0

        sentiment = (positive_matches - negative_matches) / total_matches
        return max(-1.0, min(1.0, sentiment))

    def _detect_style_feedback(self, feedback: str) -> Dict[PersonalityTraitType, float]:
        """Detect specific style feedback and convert to trait adjustments."""
        adjustments = {}

        # Check if feedback mentions style
        if self.feedback_patterns["style_feedback"].search(feedback):
            # Analyze what aspects of style are mentioned
            for trait_type, patterns in self.preference_patterns.items():
                if patterns["increase"].search(feedback):
                    adjustments[trait_type] = 0.2  # Positive adjustment
                elif patterns["decrease"].search(feedback):
                    adjustments[trait_type] = -0.2  # Negative adjustment

        return adjustments

    async def _update_profile_from_pattern(self, pattern: BehaviorPattern) -> None:
        """Update behavior profile from a detected pattern."""
        # Add pattern to profile
        self.behavior_profile.detected_patterns.append(pattern)

        # Trim patterns if needed
        if len(self.behavior_profile.detected_patterns) > 50:
            self.behavior_profile.detected_patterns = self.behavior_profile.detected_patterns[-50:]

        # Update preferences based on pattern
        for trait_type, adjustment in pattern.suggested_adjustments.items():
            # Convert adjustment to preference value
            current_preference = self.behavior_profile.user_preferences.get(trait_type, 0.5)
            new_preference = max(0.0, min(1.0, current_preference + adjustment))

            # Update with confidence weighting
            self.behavior_profile.update_preference(trait_type, new_preference, pattern.confidence)

        self.behavior_profile.last_updated = datetime.now()

    def _analyze_length_preferences(self) -> Optional[BehaviorPattern]:
        """Analyze user preferences for response length."""
        recent_inputs = [i for i in self.recent_interactions if i["type"] == "user_input"]
        if len(recent_inputs) < 5:
            return None

        # Calculate average input length
        avg_length = sum(i["length"] for i in recent_inputs) / len(recent_inputs)

        # Determine preference based on input length patterns
        if avg_length > 100:  # Long inputs suggest preference for detailed responses
            return BehaviorPattern(
                pattern_type=BehaviorSignal.PREFERENCE_IMPLICIT,
                confidence=0.6,
                evidence=[f"Average input length: {avg_length:.1f} characters"],
                suggested_adjustments={PersonalityTraitType.VERBOSITY: 0.1},
            )
        elif avg_length < 30:  # Short inputs suggest preference for brief responses
            return BehaviorPattern(
                pattern_type=BehaviorSignal.PREFERENCE_IMPLICIT,
                confidence=0.6,
                evidence=[f"Average input length: {avg_length:.1f} characters"],
                suggested_adjustments={PersonalityTraitType.VERBOSITY: -0.1},
            )

        return None

    def _analyze_engagement_patterns(self) -> Optional[BehaviorPattern]:
        """Analyze user engagement patterns."""
        recent_interactions = list(self.recent_interactions)[-10:]
        if len(recent_interactions) < 5:
            return None

        # Count engagement indicators
        high_engagement_count = 0
        for interaction in recent_interactions:
            engagement = self._detect_engagement_level(interaction["text"])
            if engagement["level"] == "high":
                high_engagement_count += 1

        engagement_ratio = high_engagement_count / len(recent_interactions)

        if engagement_ratio > 0.7:
            return BehaviorPattern(
                pattern_type=BehaviorSignal.ENGAGEMENT_HIGH,
                confidence=0.7,
                evidence=[
                    f"High engagement in {high_engagement_count}/{len(recent_interactions)} interactions"
                ],
                suggested_adjustments={
                    PersonalityTraitType.ENTHUSIASM: 0.1,
                    PersonalityTraitType.PROACTIVITY: 0.1,
                },
            )
        elif engagement_ratio < 0.3:
            return BehaviorPattern(
                pattern_type=BehaviorSignal.ENGAGEMENT_LOW,
                confidence=0.7,
                evidence=[
                    f"Low engagement in {len(recent_interactions) - high_engagement_count}/{len(recent_interactions)} interactions"
                ],
                suggested_adjustments={
                    PersonalityTraitType.ENTHUSIASM: -0.1,
                    PersonalityTraitType.VERBOSITY: -0.1,
                },
            )

        return None

    def _analyze_interaction_frequency(self) -> Optional[BehaviorPattern]:
        """Analyze interaction frequency patterns."""
        if len(self.recent_interactions) < 3:
            return None

        # Calculate time between interactions
        times = [i["timestamp"] for i in self.recent_interactions]
        intervals = [(times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))]

        avg_interval = sum(intervals) / len(intervals)

        # Quick interactions suggest high engagement
        if avg_interval < 30:  # Less than 30 seconds between interactions
            return BehaviorPattern(
                pattern_type=BehaviorSignal.ENGAGEMENT_HIGH,
                confidence=0.6,
                evidence=[f"Average interaction interval: {avg_interval:.1f} seconds"],
                suggested_adjustments={PersonalityTraitType.PROACTIVITY: 0.1},
            )

        return None

    async def _generate_adjustment_suggestions(
        self, patterns: Dict[PersonalityTraitType, float]
    ) -> Dict[PersonalityTraitType, float]:
        """Generate personality adjustment suggestions from detected patterns."""
        suggestions = {}

        for trait_type, preference_value in patterns.items():
            # Convert preference to adjustment suggestion
            # If preference is high (0.8), suggest positive adjustment
            # If preference is low (0.2), suggest negative adjustment
            if preference_value > 0.6:
                suggestions[trait_type] = 0.15  # Moderate positive adjustment
            elif preference_value < 0.4:
                suggestions[trait_type] = -0.15  # Moderate negative adjustment

        return suggestions
