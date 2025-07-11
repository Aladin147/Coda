"""
Data models for the personality system.

This module defines Pydantic models for type-safe personality operations.
"""

import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


class PersonalityTraitType(str, Enum):
    """Types of personality traits."""

    VERBOSITY = "verbosity"
    ASSERTIVENESS = "assertiveness"
    HUMOR = "humor"
    FORMALITY = "formality"
    PROACTIVITY = "proactivity"
    CONFIDENCE = "confidence"
    EMPATHY = "empathy"
    CREATIVITY = "creativity"
    ANALYTICAL = "analytical"
    ENTHUSIASM = "enthusiasm"


class TopicCategory(str, Enum):
    """Categories of conversation topics."""

    TECHNICAL = "technical"
    CASUAL = "casual"
    CREATIVE = "creative"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class BehaviorSignal(str, Enum):
    """Types of behavior signals from users."""

    POSITIVE_FEEDBACK = "positive_feedback"
    NEGATIVE_FEEDBACK = "negative_feedback"
    PREFERENCE_EXPLICIT = "preference_explicit"
    PREFERENCE_IMPLICIT = "preference_implicit"
    ENGAGEMENT_HIGH = "engagement_high"
    ENGAGEMENT_LOW = "engagement_low"
    STYLE_REQUEST = "style_request"


class PersonalityTrait(BaseModel):
    """A single personality trait with value and constraints."""

    name: PersonalityTraitType
    value: float = Field(ge=0.0, le=1.0)
    default_value: float = Field(ge=0.0, le=1.0)
    min_value: float = Field(default=0.0, ge=0.0, le=1.0)
    max_value: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str
    context_adjustments: Dict[str, float] = Field(default_factory=dict)
    last_adjusted: datetime = Field(default_factory=datetime.now)
    adjustment_count: int = Field(default=0)

    @field_validator("value")
    @classmethod
    def validate_value_range(cls, v, info):
        # In Pydantic V2, we need to access other field values differently
        # For now, we'll validate against the default range and let model_validator handle complex validation
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value {v} must be between 0.0 and 1.0")
        return v

    def adjust(self, delta: float, reason: str = "") -> float:
        """Adjust the trait value by delta amount."""
        old_value = self.value
        self.value = max(self.min_value, min(self.max_value, self.value + delta))
        self.last_adjusted = datetime.now()
        self.adjustment_count += 1
        return self.value - old_value

    def reset_to_default(self) -> None:
        """Reset trait to its default value."""
        self.value = self.default_value
        self.last_adjusted = datetime.now()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PersonalityParameters(BaseModel):
    """Collection of personality traits with management methods."""

    traits: Dict[PersonalityTraitType, PersonalityTrait] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="2.0.0")

    def get_trait(self, trait_type: PersonalityTraitType) -> Optional[PersonalityTrait]:
        """Get a specific trait."""
        return self.traits.get(trait_type)

    def get_trait_value(self, trait_type: PersonalityTraitType) -> float:
        """Get the current value of a trait."""
        trait = self.traits.get(trait_type)
        return trait.value if trait else 0.5

    def adjust_trait(
        self, trait_type: PersonalityTraitType, delta: float, reason: str = ""
    ) -> bool:
        """Adjust a trait value."""
        trait = self.traits.get(trait_type)
        if trait:
            trait.adjust(delta, reason)
            self.last_updated = datetime.now()
            return True
        return False

    def apply_context_adjustments(self, context: str) -> Dict[PersonalityTraitType, float]:
        """Apply context-based adjustments to all traits."""
        adjustments = {}
        for trait_type, trait in self.traits.items():
            if context in trait.context_adjustments:
                adjustment = trait.context_adjustments[context]
                actual_adjustment = trait.adjust(adjustment, f"Context: {context}")
                adjustments[trait_type] = actual_adjustment

        if adjustments:
            self.last_updated = datetime.now()

        return adjustments

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BehaviorPattern(BaseModel):
    """A detected user behavior pattern."""

    pattern_type: BehaviorSignal
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    suggested_adjustments: Dict[PersonalityTraitType, float] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BehaviorProfile(BaseModel):
    """User behavior profile learned over time."""

    user_preferences: Dict[PersonalityTraitType, float] = Field(default_factory=dict)
    confidence_scores: Dict[PersonalityTraitType, float] = Field(default_factory=dict)
    observation_count: int = Field(default=0)
    last_updated: datetime = Field(default_factory=datetime.now)
    detected_patterns: List[BehaviorPattern] = Field(default_factory=list)
    explicit_feedback: List[Dict[str, Any]] = Field(default_factory=list)

    def update_preference(self, trait_type: PersonalityTraitType, value: float, confidence: float):
        """Update a user preference with confidence weighting."""
        current_value = self.user_preferences.get(trait_type, 0.5)
        current_confidence = self.confidence_scores.get(trait_type, 0.0)

        # Weighted average based on confidence
        total_confidence = current_confidence + confidence
        if total_confidence > 0:
            new_value = (current_value * current_confidence + value * confidence) / total_confidence
            self.user_preferences[trait_type] = new_value
            self.confidence_scores[trait_type] = min(1.0, total_confidence)

        self.observation_count += 1
        self.last_updated = datetime.now()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TopicContext(BaseModel):
    """Current topic context information."""

    current_topic: Optional[str] = None
    category: TopicCategory = TopicCategory.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.now)
    topic_history: List[Dict[str, Any]] = Field(default_factory=list)

    def update_topic(
        self, topic: str, category: TopicCategory, confidence: float, keywords: List[str]
    ):
        """Update the current topic context."""
        # Add current topic to history
        if self.current_topic:
            self.topic_history.append(
                {
                    "topic": self.current_topic,
                    "category": self.category.value,
                    "confidence": self.confidence,
                    "duration": (datetime.now() - self.detected_at).total_seconds(),
                }
            )

        # Update current topic
        self.current_topic = topic
        self.category = category
        self.confidence = confidence
        self.keywords = keywords
        self.detected_at = datetime.now()

        # Trim history
        if len(self.topic_history) > 10:
            self.topic_history = self.topic_history[-10:]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PersonalLoreElement(BaseModel):
    """A single element of personal lore."""

    type: str  # "backstory", "trait", "quirk", "memory", "preference"
    content: str
    context_triggers: List[str] = Field(default_factory=list)
    usage_count: int = Field(default=0)
    last_used: Optional[datetime] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    def use(self) -> None:
        """Mark this lore element as used."""
        self.usage_count += 1
        self.last_used = datetime.now()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PersonalLore(BaseModel):
    """Complete personal lore collection."""

    backstory: Dict[str, str] = Field(default_factory=dict)
    traits: List[str] = Field(default_factory=list)
    quirks: List[PersonalLoreElement] = Field(default_factory=list)
    memories: List[PersonalLoreElement] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    anchors: Dict[str, List[str]] = Field(default_factory=dict)  # Context -> phrases

    def get_relevant_lore(self, context: str, keywords: List[str]) -> List[PersonalLoreElement]:
        """Get lore elements relevant to the current context."""
        relevant = []

        # Check quirks
        for quirk in self.quirks:
            if any(trigger.lower() in context.lower() for trigger in quirk.context_triggers):
                relevant.append(quirk)
            elif any(keyword.lower() in quirk.content.lower() for keyword in keywords):
                relevant.append(quirk)

        # Check memories
        for memory in self.memories:
            if any(trigger.lower() in context.lower() for trigger in memory.context_triggers):
                relevant.append(memory)
            elif any(keyword.lower() in memory.content.lower() for keyword in keywords):
                relevant.append(memory)

        # Sort by importance and usage (prefer important but less used)
        relevant.sort(key=lambda x: (x.importance, -x.usage_count), reverse=True)
        return relevant[:3]  # Return top 3

    def get_anchor_phrases(self, context: str) -> List[str]:
        """Get anchor phrases for the given context."""
        return self.anchors.get(context, [])


class SessionState(BaseModel):
    """Current session state information."""

    session_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    last_interaction: datetime = Field(default_factory=datetime.now)
    turn_count: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
    idle_seconds: float = Field(default=0.0)
    in_closure_mode: bool = Field(default=False)
    closure_reason: Optional[str] = None

    def update(self) -> None:
        """Update session timing information."""
        now = datetime.now()
        self.duration_seconds = (now - self.start_time).total_seconds()
        self.idle_seconds = (now - self.last_interaction).total_seconds()

    def add_interaction(self) -> None:
        """Record a new interaction."""
        self.last_interaction = datetime.now()
        self.turn_count += 1
        self.idle_seconds = 0.0

    def enter_closure_mode(self, reason: str) -> None:
        """Enter session closure mode."""
        self.in_closure_mode = True
        self.closure_reason = reason

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PersonalityAdjustment(BaseModel):
    """Record of a personality adjustment."""

    trait_type: PersonalityTraitType
    old_value: float
    new_value: float
    delta: float
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PersonalityEvent(BaseModel):
    """Event for WebSocket broadcasting."""

    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# Configuration models


class PersonalityParameterConfig(BaseModel):
    """Configuration for personality parameters."""

    default_traits: Dict[PersonalityTraitType, Dict[str, Any]] = Field(default_factory=dict)
    context_categories: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    adjustment_limits: Dict[str, float] = Field(default_factory=dict)


class BehavioralConditioningConfig(BaseModel):
    """Configuration for behavioral conditioning."""

    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_recent_interactions: int = Field(default=20, ge=1)
    pattern_detection_window: int = Field(default=10, ge=1)
    explicit_feedback_weight: float = Field(default=0.8, ge=0.0, le=1.0)


class TopicAwarenessConfig(BaseModel):
    """Configuration for topic awareness."""

    topic_categories: Dict[TopicCategory, List[str]] = Field(default_factory=dict)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_topic_history: int = Field(default=10, ge=1)
    keyword_weights: Dict[str, float] = Field(default_factory=dict)


class PersonalLoreConfig(BaseModel):
    """Configuration for personal lore."""

    lore_injection_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    max_lore_elements_per_response: int = Field(default=2, ge=0)
    quirk_trigger_sensitivity: float = Field(default=0.7, ge=0.0, le=1.0)
    backstory_elements: Dict[str, str] = Field(default_factory=dict)


class SessionManagerConfig(BaseModel):
    """Configuration for session management."""

    long_session_threshold_minutes: int = Field(default=30, ge=1)
    idle_threshold_minutes: int = Field(default=10, ge=1)
    closure_mode_adjustments: Dict[PersonalityTraitType, float] = Field(default_factory=dict)


class PersonalityConfig(BaseModel):
    """Complete personality system configuration."""

    parameters: PersonalityParameterConfig = Field(default_factory=PersonalityParameterConfig)
    behavioral_conditioning: BehavioralConditioningConfig = Field(
        default_factory=BehavioralConditioningConfig
    )
    topic_awareness: TopicAwarenessConfig = Field(default_factory=TopicAwarenessConfig)
    personal_lore: PersonalLoreConfig = Field(default_factory=PersonalLoreConfig)
    session_manager: SessionManagerConfig = Field(default_factory=SessionManagerConfig)
    websocket_events_enabled: bool = Field(default=True)
    analytics_enabled: bool = Field(default=True)
