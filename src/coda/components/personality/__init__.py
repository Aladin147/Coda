"""
Personality system for Coda.

This package provides sophisticated personality management functionality including:
- Dynamic personality parameter adjustment
- Behavioral conditioning and user preference learning
- Topic-aware personality adaptation
- Personal lore and backstory management
- Session state tracking and closure management
- Real-time personality event broadcasting

Features:
- Adaptive personality traits (verbosity, assertiveness, humor, etc.)
- User behavior pattern recognition and learning
- Context-aware personality adjustments
- Personal backstory, quirks, and memory injection
- Session management with intelligent closure detection
- WebSocket integration for real-time monitoring
"""

from .behavioral_conditioner import BehavioralConditioner
from .interfaces import (
    BehavioralConditionerInterface,
    PersonalityManagerInterface,
    PersonalityParameterInterface,
    PersonalLoreInterface,
    SessionManagerInterface,
    TopicAwarenessInterface,
    WebSocketPersonalityInterface,
)
from .manager import PersonalityManager
from .models import (
    BehaviorProfile,
    PersonalityAdjustment,
    PersonalityConfig,
    PersonalityEvent,
    PersonalityParameters,
    PersonalityTrait,
    PersonalLore,
    SessionState,
    TopicContext,
)
from .parameters import PersonalityParameterManager
from .personal_lore import PersonalLoreManager
from .prompt_enhancer import PersonalityPromptEnhancer
from .session_manager import SessionManager
from .topic_awareness import TopicAwareness
from .websocket_integration import WebSocketPersonalityManager

__all__ = [
    # Interfaces
    "PersonalityParameterInterface",
    "BehavioralConditionerInterface",
    "TopicAwarenessInterface",
    "PersonalLoreInterface",
    "SessionManagerInterface",
    "PersonalityManagerInterface",
    "WebSocketPersonalityInterface",
    # Models
    "PersonalityTrait",
    "PersonalityParameters",
    "BehaviorProfile",
    "TopicContext",
    "PersonalLore",
    "SessionState",
    "PersonalityAdjustment",
    "PersonalityEvent",
    "PersonalityConfig",
    # Implementations
    "PersonalityParameterManager",
    "BehavioralConditioner",
    "TopicAwareness",
    "PersonalLoreManager",
    "SessionManager",
    "PersonalityManager",
    "WebSocketPersonalityManager",
    "PersonalityPromptEnhancer",
]
