"""
Interfaces for the personality system.

This module defines abstract base classes for personality components to ensure
consistent APIs and enable easy testing with mocks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from .models import (
    PersonalityTrait,
    PersonalityParameters,
    PersonalityTraitType,
    BehaviorProfile,
    BehaviorPattern,
    TopicContext,
    TopicCategory,
    PersonalLore,
    PersonalLoreElement,
    SessionState,
    PersonalityAdjustment,
    PersonalityEvent,
)


class PersonalityParameterInterface(ABC):
    """Interface for personality parameter management."""
    
    @abstractmethod
    def get_parameters(self) -> PersonalityParameters:
        """Get current personality parameters."""
        pass
    
    @abstractmethod
    def get_trait_value(self, trait_type: PersonalityTraitType) -> float:
        """Get the current value of a specific trait."""
        pass
    
    @abstractmethod
    def adjust_trait(self, trait_type: PersonalityTraitType, delta: float, 
                    reason: str = "", confidence: float = 1.0) -> PersonalityAdjustment:
        """Adjust a personality trait value."""
        pass
    
    @abstractmethod
    def apply_context_adjustments(self, context: str) -> Dict[PersonalityTraitType, float]:
        """Apply context-based adjustments to personality traits."""
        pass
    
    @abstractmethod
    def reset_trait(self, trait_type: PersonalityTraitType) -> bool:
        """Reset a trait to its default value."""
        pass
    
    @abstractmethod
    def reset_all_traits(self) -> None:
        """Reset all traits to their default values."""
        pass
    
    @abstractmethod
    def get_adjustment_history(self, limit: int = 10) -> List[PersonalityAdjustment]:
        """Get recent personality adjustments."""
        pass


class BehavioralConditionerInterface(ABC):
    """Interface for behavioral conditioning and learning."""
    
    @abstractmethod
    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input for behavioral patterns."""
        pass
    
    @abstractmethod
    async def process_user_feedback(self, feedback: str, feedback_type: str = "general") -> Dict[str, Any]:
        """Process explicit user feedback."""
        pass
    
    @abstractmethod
    async def analyze_interaction_patterns(self) -> List[BehaviorPattern]:
        """Analyze recent interactions for behavior patterns."""
        pass
    
    @abstractmethod
    def get_behavior_profile(self) -> BehaviorProfile:
        """Get the current user behavior profile."""
        pass
    
    @abstractmethod
    async def update_behavior_profile(self, patterns: List[BehaviorPattern]) -> bool:
        """Update behavior profile based on detected patterns."""
        pass
    
    @abstractmethod
    async def suggest_personality_adjustments(self) -> Dict[PersonalityTraitType, float]:
        """Suggest personality adjustments based on learned behavior."""
        pass
    
    @abstractmethod
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about behavioral learning."""
        pass


class TopicAwarenessInterface(ABC):
    """Interface for topic detection and awareness."""
    
    @abstractmethod
    async def detect_topic(self, text: str) -> TopicContext:
        """Detect the topic and category of given text."""
        pass
    
    @abstractmethod
    async def process_user_input(self, user_input: str) -> TopicContext:
        """Process user input for topic detection and tracking."""
        pass
    
    @abstractmethod
    def get_current_topic(self) -> TopicContext:
        """Get the current topic context."""
        pass
    
    @abstractmethod
    def get_topic_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent topic history."""
        pass
    
    @abstractmethod
    async def get_topic_personality_adjustments(self, topic_context: TopicContext) -> Dict[PersonalityTraitType, float]:
        """Get personality adjustments for the current topic."""
        pass
    
    @abstractmethod
    def clear_topic_history(self) -> None:
        """Clear topic history."""
        pass


class PersonalLoreInterface(ABC):
    """Interface for personal lore management."""
    
    @abstractmethod
    def get_lore(self) -> PersonalLore:
        """Get the complete personal lore."""
        pass
    
    @abstractmethod
    def get_relevant_lore(self, context: str, keywords: List[str]) -> List[PersonalLoreElement]:
        """Get lore elements relevant to the current context."""
        pass
    
    @abstractmethod
    def get_backstory_element(self, key: str) -> Optional[str]:
        """Get a specific backstory element."""
        pass
    
    @abstractmethod
    def get_anchor_phrases(self, context: str) -> List[str]:
        """Get anchor phrases for the given context."""
        pass
    
    @abstractmethod
    def add_lore_element(self, element_type: str, content: str, 
                        triggers: List[str] = None, importance: float = 0.5) -> bool:
        """Add a new lore element."""
        pass
    
    @abstractmethod
    def enhance_prompt(self, prompt: str, context: str, keywords: List[str]) -> str:
        """Enhance a prompt with relevant personal lore."""
        pass
    
    @abstractmethod
    def format_response_with_lore(self, response: str, context: str, 
                                 trigger_words: List[str] = None) -> str:
        """Format a response with personal lore elements."""
        pass
    
    @abstractmethod
    def get_lore_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about lore usage."""
        pass


class SessionManagerInterface(ABC):
    """Interface for session state management."""
    
    @abstractmethod
    def get_session_state(self) -> SessionState:
        """Get the current session state."""
        pass
    
    @abstractmethod
    def update_session(self) -> SessionState:
        """Update session timing and state."""
        pass
    
    @abstractmethod
    def process_interaction(self, role: str, content: str) -> None:
        """Process a new interaction in the session."""
        pass
    
    @abstractmethod
    def should_enter_closure_mode(self) -> Tuple[bool, str]:
        """Check if session should enter closure mode."""
        pass
    
    @abstractmethod
    def enter_closure_mode(self, reason: str) -> Dict[PersonalityTraitType, float]:
        """Enter closure mode and get personality adjustments."""
        pass
    
    @abstractmethod
    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate a summary of the current session."""
        pass
    
    @abstractmethod
    def reset_session(self, new_session_id: str = None) -> None:
        """Reset session state."""
        pass


class PersonalityManagerInterface(ABC):
    """Interface for the main personality manager."""
    
    @abstractmethod
    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input through all personality components."""
        pass
    
    @abstractmethod
    async def process_assistant_response(self, response: str) -> Dict[str, Any]:
        """Process assistant response for learning and analysis."""
        pass
    
    @abstractmethod
    async def enhance_prompt(self, prompt: str, context: str = "") -> str:
        """Enhance a prompt with personality and lore."""
        pass
    
    @abstractmethod
    async def format_response(self, response: str, context: str = "") -> str:
        """Format a response with personality elements."""
        pass
    
    @abstractmethod
    async def process_feedback(self, feedback: str, feedback_type: str = "general") -> Dict[str, Any]:
        """Process user feedback for learning."""
        pass
    
    @abstractmethod
    def get_personality_state(self) -> Dict[str, Any]:
        """Get complete personality state."""
        pass
    
    @abstractmethod
    async def apply_learned_adjustments(self) -> Dict[str, Any]:
        """Apply personality adjustments based on learning."""
        pass
    
    @abstractmethod
    def get_analytics(self) -> Dict[str, Any]:
        """Get personality analytics and insights."""
        pass


class WebSocketPersonalityInterface(PersonalityManagerInterface):
    """Interface for personality manager with WebSocket integration."""
    
    @abstractmethod
    async def set_websocket_integration(self, integration: Any) -> None:
        """Set WebSocket integration for event broadcasting."""
        pass
    
    @abstractmethod
    async def broadcast_personality_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a personality event via WebSocket."""
        pass
    
    @abstractmethod
    async def broadcast_personality_state(self) -> None:
        """Broadcast current personality state."""
        pass
    
    @abstractmethod
    async def broadcast_adjustment(self, adjustment: PersonalityAdjustment) -> None:
        """Broadcast a personality adjustment event."""
        pass


# Utility interfaces

class PersonalityAnalyticsInterface(ABC):
    """Interface for personality analytics and insights."""
    
    @abstractmethod
    def get_trait_evolution(self, trait_type: PersonalityTraitType, days: int = 7) -> List[Dict[str, Any]]:
        """Get trait value evolution over time."""
        pass
    
    @abstractmethod
    def get_behavior_insights(self) -> Dict[str, Any]:
        """Get insights about user behavior patterns."""
        pass
    
    @abstractmethod
    def get_topic_distribution(self) -> Dict[TopicCategory, int]:
        """Get distribution of conversation topics."""
        pass
    
    @abstractmethod
    def get_lore_effectiveness(self) -> Dict[str, Any]:
        """Get effectiveness metrics for personal lore."""
        pass
    
    @abstractmethod
    def get_session_patterns(self) -> Dict[str, Any]:
        """Get patterns in session behavior."""
        pass


class PersonalityExplainerInterface(ABC):
    """Interface for explaining personality decisions."""
    
    @abstractmethod
    def explain_adjustment(self, adjustment: PersonalityAdjustment) -> str:
        """Explain why a personality adjustment was made."""
        pass
    
    @abstractmethod
    def explain_response_style(self, response: str, parameters: PersonalityParameters) -> str:
        """Explain how personality influenced response style."""
        pass
    
    @abstractmethod
    def explain_lore_injection(self, lore_elements: List[PersonalLoreElement]) -> str:
        """Explain why specific lore elements were used."""
        pass
    
    @abstractmethod
    def get_decision_trace(self, interaction_id: str) -> List[Dict[str, Any]]:
        """Get detailed trace of personality decisions for an interaction."""
        pass


class PersonalityConfigInterface(ABC):
    """Interface for personality configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> bool:
        """Load personality configuration from file."""
        pass
    
    @abstractmethod
    def save_config(self, config_path: str) -> bool:
        """Save current personality configuration to file."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate personality configuration."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default personality configuration."""
        pass
    
    @abstractmethod
    def merge_configs(self, base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two personality configurations."""
        pass
