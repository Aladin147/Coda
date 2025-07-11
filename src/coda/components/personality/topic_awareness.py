"""
Topic awareness system for Coda.

This module provides the TopicAwareness class for detecting conversation topics
and adjusting personality based on context.
"""

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from .interfaces import TopicAwarenessInterface
from .models import (
    PersonalityTraitType,
    TopicAwarenessConfig,
    TopicCategory,
    TopicContext,
)

logger = logging.getLogger("coda.personality.topic_awareness")


class TopicAwareness(TopicAwarenessInterface):
    """
    Detects and tracks conversation topics for personality adaptation.

    Features:
    - Topic detection using keyword matching and patterns
    - Topic categorization into predefined categories
    - Topic history tracking
    - Personality adjustments based on topic context
    - Confidence scoring for topic detection
    """

    def __init__(self, config: Optional[TopicAwarenessConfig] = None):
        """
        Initialize the topic awareness system.

        Args:
            config: Configuration for topic awareness
        """
        self.config = config or TopicAwarenessConfig()
        self.current_topic = TopicContext()

        # Initialize topic categories and keywords
        self._initialize_topic_categories()
        self._compile_patterns()

        logger.info("TopicAwareness initialized")

    def _initialize_topic_categories(self) -> None:
        """Initialize topic categories with default keywords."""
        default_categories = {
            TopicCategory.TECHNICAL: [
                "programming",
                "code",
                "software",
                "algorithm",
                "database",
                "api",
                "framework",
                "python",
                "javascript",
                "java",
                "c++",
                "html",
                "css",
                "sql",
                "git",
                "linux",
                "machine learning",
                "ai",
                "artificial intelligence",
                "neural network",
                "data science",
                "computer",
                "technology",
                "tech",
                "development",
                "debugging",
                "server",
                "cloud",
            ],
            TopicCategory.CREATIVE: [
                "art",
                "music",
                "writing",
                "poetry",
                "story",
                "novel",
                "painting",
                "drawing",
                "design",
                "creative",
                "imagination",
                "inspiration",
                "artistic",
                "compose",
                "create",
                "craft",
                "photography",
                "film",
                "movie",
                "theater",
                "dance",
            ],
            TopicCategory.EDUCATIONAL: [
                "learn",
                "study",
                "education",
                "school",
                "university",
                "course",
                "lesson",
                "teach",
                "explain",
                "understand",
                "knowledge",
                "research",
                "academic",
                "science",
                "math",
                "physics",
                "chemistry",
                "biology",
                "history",
                "geography",
            ],
            TopicCategory.ENTERTAINMENT: [
                "game",
                "fun",
                "play",
                "entertainment",
                "movie",
                "tv",
                "show",
                "series",
                "book",
                "novel",
                "joke",
                "funny",
                "humor",
                "comedy",
                "sport",
                "music",
                "video",
                "youtube",
                "netflix",
                "gaming",
                "puzzle",
                "quiz",
            ],
            TopicCategory.PERSONAL: [
                "feel",
                "emotion",
                "personal",
                "family",
                "friend",
                "relationship",
                "love",
                "sad",
                "happy",
                "angry",
                "worried",
                "excited",
                "nervous",
                "confident",
                "life",
                "experience",
                "memory",
                "childhood",
                "dream",
                "goal",
                "hope",
            ],
            TopicCategory.PROFESSIONAL: [
                "work",
                "job",
                "career",
                "business",
                "company",
                "office",
                "meeting",
                "project",
                "deadline",
                "client",
                "customer",
                "sales",
                "marketing",
                "management",
                "leadership",
                "team",
                "colleague",
                "professional",
                "industry",
            ],
            TopicCategory.CASUAL: [
                "weather",
                "food",
                "cooking",
                "recipe",
                "restaurant",
                "travel",
                "vacation",
                "hobby",
                "weekend",
                "daily",
                "routine",
                "shopping",
                "fashion",
                "health",
                "exercise",
                "fitness",
                "news",
                "current events",
                "chat",
                "conversation",
            ],
            TopicCategory.EMERGENCY: [
                "emergency",
                "urgent",
                "help",
                "crisis",
                "problem",
                "issue",
                "error",
                "broken",
                "fix",
                "repair",
                "immediate",
                "asap",
                "quickly",
                "fast",
                "important",
                "critical",
                "serious",
                "danger",
                "risk",
                "safety",
            ],
        }

        # Merge with config if provided
        if self.config.topic_categories:
            for category, keywords in self.config.topic_categories.items():
                if category in default_categories:
                    default_categories[category].extend(keywords)
                else:
                    default_categories[category] = keywords

        self.topic_categories = default_categories

        # Build keyword to category mapping
        self.keyword_to_categories = {}
        for category, keywords in self.topic_categories.items():
            for keyword in keywords:
                if keyword not in self.keyword_to_categories:
                    self.keyword_to_categories[keyword] = []
                self.keyword_to_categories[keyword].append(category)

    def _compile_patterns(self) -> None:
        """Compile regex patterns for topic detection."""
        # Question patterns that might indicate specific topics
        self.question_patterns = {
            TopicCategory.TECHNICAL: re.compile(
                r"\b(how to|how do|what is|explain|implement|debug|code|program)\b", re.IGNORECASE
            ),
            TopicCategory.EDUCATIONAL: re.compile(
                r"\b(learn|study|understand|teach|explain|what is|how does)\b", re.IGNORECASE
            ),
            TopicCategory.CREATIVE: re.compile(
                r"\b(create|make|design|write|compose|draw|paint)\b", re.IGNORECASE
            ),
            TopicCategory.PERSONAL: re.compile(
                r"\b(feel|think|believe|personal|my|me|i am|i feel)\b", re.IGNORECASE
            ),
            TopicCategory.EMERGENCY: re.compile(
                r"\b(help|urgent|emergency|problem|broken|fix|error)\b", re.IGNORECASE
            ),
        }

        # Context patterns
        self.context_patterns = {
            "question": re.compile(r"\?"),
            "request": re.compile(
                r"\b(please|can you|could you|would you|help me)\b", re.IGNORECASE
            ),
            "statement": re.compile(r"\."),
            "exclamation": re.compile(r"!"),
        }

    async def detect_topic(self, text: str) -> TopicContext:
        """
        Detect the topic and category of given text.

        Args:
            text: Text to analyze for topic detection

        Returns:
            TopicContext with detected topic information
        """
        # Normalize text
        normalized_text = text.lower()
        words = re.findall(r"\b\w+\b", normalized_text)

        # Count category matches
        category_scores = Counter()
        matched_keywords = []

        # Check keyword matches
        for word in words:
            if word in self.keyword_to_categories:
                for category in self.keyword_to_categories[word]:
                    weight = self.config.keyword_weights.get(word, 1.0)
                    category_scores[category] += weight
                    matched_keywords.append(word)

        # Check pattern matches
        for category, pattern in self.question_patterns.items():
            if pattern.search(text):
                category_scores[category] += 2.0  # Higher weight for patterns

        # Determine primary category
        if not category_scores:
            return TopicContext(
                category=TopicCategory.UNKNOWN,
                confidence=0.0,
                keywords=[],
                detected_at=datetime.now(),
            )

        primary_category = category_scores.most_common(1)[0][0]
        total_score = sum(category_scores.values())
        confidence = category_scores[primary_category] / total_score

        # Extract most relevant topic keyword
        topic_keyword = None
        if matched_keywords:
            # Find the most specific keyword for this category
            category_keywords = [
                kw
                for kw in matched_keywords
                if primary_category in self.keyword_to_categories.get(kw, [])
            ]
            if category_keywords:
                # Use the longest/most specific keyword
                topic_keyword = max(category_keywords, key=len)

        return TopicContext(
            current_topic=topic_keyword,
            category=primary_category,
            confidence=confidence,
            keywords=matched_keywords[:10],  # Limit keywords
            detected_at=datetime.now(),
        )

    async def process_user_input(self, user_input: str) -> TopicContext:
        """
        Process user input for topic detection and tracking.

        Args:
            user_input: User input text

        Returns:
            Updated topic context
        """
        # Detect topic in input
        detected_topic = await self.detect_topic(user_input)

        # Update current topic if confidence is sufficient
        if detected_topic.confidence >= self.config.confidence_threshold:
            # Add current topic to history before updating
            if self.current_topic.current_topic:
                self.current_topic.topic_history.append(
                    {
                        "topic": self.current_topic.current_topic,
                        "category": self.current_topic.category.value,
                        "confidence": self.current_topic.confidence,
                        "duration": (
                            datetime.now() - self.current_topic.detected_at
                        ).total_seconds(),
                        "ended_at": datetime.now().isoformat(),
                    }
                )

            # Update current topic
            self.current_topic.update_topic(
                detected_topic.current_topic or "general",
                detected_topic.category,
                detected_topic.confidence,
                detected_topic.keywords,
            )

            # Trim history
            if len(self.current_topic.topic_history) > self.config.max_topic_history:
                self.current_topic.topic_history = self.current_topic.topic_history[
                    -self.config.max_topic_history :
                ]

            logger.info(
                f"Topic updated: {self.current_topic.current_topic} ({self.current_topic.category.value}, confidence: {self.current_topic.confidence:.2f})"
            )

        return self.current_topic

    def get_current_topic(self) -> TopicContext:
        """Get the current topic context."""
        return self.current_topic

    def get_topic_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent topic history.

        Args:
            limit: Maximum number of topics to return

        Returns:
            List of recent topics
        """
        return self.current_topic.topic_history[-limit:] if self.current_topic.topic_history else []

    async def get_topic_personality_adjustments(
        self, topic_context: TopicContext
    ) -> Dict[PersonalityTraitType, float]:
        """
        Get personality adjustments for the current topic.

        Args:
            topic_context: Current topic context

        Returns:
            Dictionary of personality trait adjustments
        """
        adjustments = {}
        category = topic_context.category
        confidence = topic_context.confidence

        # Define personality adjustments for each topic category
        category_adjustments = {
            TopicCategory.TECHNICAL: {
                PersonalityTraitType.VERBOSITY: 0.2,
                PersonalityTraitType.ANALYTICAL: 0.3,
                PersonalityTraitType.ASSERTIVENESS: 0.2,
                PersonalityTraitType.HUMOR: -0.2,
                PersonalityTraitType.FORMALITY: 0.1,
            },
            TopicCategory.CREATIVE: {
                PersonalityTraitType.CREATIVITY: 0.3,
                PersonalityTraitType.ENTHUSIASM: 0.2,
                PersonalityTraitType.HUMOR: 0.1,
                PersonalityTraitType.FORMALITY: -0.2,
                PersonalityTraitType.ANALYTICAL: -0.1,
            },
            TopicCategory.EDUCATIONAL: {
                PersonalityTraitType.VERBOSITY: 0.2,
                PersonalityTraitType.ANALYTICAL: 0.2,
                PersonalityTraitType.PROACTIVITY: 0.2,
                PersonalityTraitType.CONFIDENCE: 0.1,
                PersonalityTraitType.FORMALITY: 0.1,
            },
            TopicCategory.ENTERTAINMENT: {
                PersonalityTraitType.HUMOR: 0.3,
                PersonalityTraitType.ENTHUSIASM: 0.3,
                PersonalityTraitType.CREATIVITY: 0.2,
                PersonalityTraitType.FORMALITY: -0.3,
                PersonalityTraitType.ANALYTICAL: -0.2,
            },
            TopicCategory.PERSONAL: {
                PersonalityTraitType.EMPATHY: 0.3,
                PersonalityTraitType.FORMALITY: -0.2,
                PersonalityTraitType.ASSERTIVENESS: -0.1,
                PersonalityTraitType.HUMOR: -0.1,
                PersonalityTraitType.CONFIDENCE: -0.1,
            },
            TopicCategory.PROFESSIONAL: {
                PersonalityTraitType.FORMALITY: 0.3,
                PersonalityTraitType.ASSERTIVENESS: 0.2,
                PersonalityTraitType.CONFIDENCE: 0.2,
                PersonalityTraitType.ANALYTICAL: 0.1,
                PersonalityTraitType.HUMOR: -0.2,
            },
            TopicCategory.CASUAL: {
                PersonalityTraitType.FORMALITY: -0.3,
                PersonalityTraitType.HUMOR: 0.1,
                PersonalityTraitType.ENTHUSIASM: 0.1,
                PersonalityTraitType.EMPATHY: 0.1,
                PersonalityTraitType.ANALYTICAL: -0.1,
            },
            TopicCategory.EMERGENCY: {
                PersonalityTraitType.ASSERTIVENESS: 0.4,
                PersonalityTraitType.CONFIDENCE: 0.3,
                PersonalityTraitType.PROACTIVITY: 0.4,
                PersonalityTraitType.VERBOSITY: -0.2,
                PersonalityTraitType.HUMOR: -0.4,
            },
        }

        # Apply adjustments for the current category
        if category in category_adjustments:
            base_adjustments = category_adjustments[category]

            # Scale adjustments by confidence
            for trait_type, adjustment in base_adjustments.items():
                scaled_adjustment = adjustment * confidence
                adjustments[trait_type] = scaled_adjustment

        return adjustments

    def clear_topic_history(self) -> None:
        """Clear topic history."""
        self.current_topic.topic_history.clear()
        logger.info("Topic history cleared")

    def get_topic_stats(self) -> Dict[str, Any]:
        """Get statistics about topic detection and history."""
        if not self.current_topic.topic_history:
            return {
                "total_topics": 0,
                "current_topic": self.current_topic.current_topic,
                "current_category": self.current_topic.category.value,
                "current_confidence": self.current_topic.confidence,
            }

        # Analyze topic history
        categories = [topic["category"] for topic in self.current_topic.topic_history]
        category_counts = Counter(categories)

        durations = [topic.get("duration", 0) for topic in self.current_topic.topic_history]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_topics": len(self.current_topic.topic_history),
            "current_topic": self.current_topic.current_topic,
            "current_category": self.current_topic.category.value,
            "current_confidence": self.current_topic.confidence,
            "category_distribution": dict(category_counts),
            "most_common_category": (
                category_counts.most_common(1)[0][0] if category_counts else None
            ),
            "average_topic_duration": avg_duration,
            "topic_switches": len(self.current_topic.topic_history),
        }

    def get_category_keywords(self, category: TopicCategory) -> List[str]:
        """Get keywords for a specific category."""
        return self.topic_categories.get(category, [])

    def add_category_keywords(self, category: TopicCategory, keywords: List[str]) -> None:
        """Add keywords to a category."""
        if category not in self.topic_categories:
            self.topic_categories[category] = []

        self.topic_categories[category].extend(keywords)

        # Update keyword to category mapping
        for keyword in keywords:
            if keyword not in self.keyword_to_categories:
                self.keyword_to_categories[keyword] = []
            if category not in self.keyword_to_categories[keyword]:
                self.keyword_to_categories[keyword].append(category)

        logger.info(f"Added {len(keywords)} keywords to category {category.value}")

    def predict_topic_transition(self, new_input: str) -> Dict[str, Any]:
        """Predict if the topic is likely to change based on new input."""
        # Detect topic in new input
        detected_topic = asyncio.run(self.detect_topic(new_input))

        current_category = self.current_topic.category
        detected_category = detected_topic.category

        # Calculate transition probability
        if detected_category == TopicCategory.UNKNOWN:
            transition_probability = 0.0
        elif detected_category == current_category:
            transition_probability = 0.1  # Low probability of staying in same category
        else:
            # Higher confidence in detection = higher transition probability
            transition_probability = detected_topic.confidence

        return {
            "current_category": current_category.value,
            "predicted_category": detected_category.value,
            "transition_probability": transition_probability,
            "confidence": detected_topic.confidence,
            "keywords": detected_topic.keywords,
        }
