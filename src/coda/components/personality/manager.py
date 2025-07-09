"""
Main personality manager for Coda.

This module provides the PersonalityManager class that orchestrates all
personality components for intelligent, adaptive behavior.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .interfaces import PersonalityManagerInterface
from .models import (
    PersonalityConfig,
    PersonalityTraitType,
    PersonalityAdjustment,
    TopicCategory,
)
from .parameters import PersonalityParameterManager
from .behavioral_conditioner import BehavioralConditioner
from .topic_awareness import TopicAwareness
from .personal_lore import PersonalLoreManager
from .session_manager import SessionManager

logger = logging.getLogger("coda.personality.manager")


class PersonalityManager(PersonalityManagerInterface):
    """
    Main personality manager that orchestrates all personality components.
    
    Features:
    - Integrated personality parameter management
    - Behavioral learning and adaptation
    - Topic-aware personality adjustments
    - Personal lore injection and management
    - Session state tracking and closure management
    - Comprehensive personality analytics
    """
    
    def __init__(self, config: Optional[PersonalityConfig] = None, 
                 session_id: Optional[str] = None):
        """
        Initialize the personality manager.
        
        Args:
            config: Configuration for personality system
            session_id: Optional session ID
        """
        self.config = config or PersonalityConfig()
        
        # Initialize components
        self.parameters = PersonalityParameterManager(self.config.parameters)
        self.behavioral_conditioner = BehavioralConditioner(self.config.behavioral_conditioning)
        self.topic_awareness = TopicAwareness(self.config.topic_awareness)
        self.personal_lore = PersonalLoreManager(self.config.personal_lore)
        self.session_manager = SessionManager(self.config.session_manager, session_id)
        
        # Analytics tracking
        self.interaction_count = 0
        self.adjustment_history: List[PersonalityAdjustment] = []
        
        logger.info(f"PersonalityManager initialized for session {self.session_manager.session_state.session_id}")
    
    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input through all personality components.
        
        Args:
            user_input: User input text
            
        Returns:
            Dictionary with processing results and adjustments
        """
        self.interaction_count += 1
        
        # Update session
        self.session_manager.process_interaction("user", user_input)
        session_state = self.session_manager.update_session()
        
        # Process through behavioral conditioner
        behavior_results = await self.behavioral_conditioner.process_user_input(user_input)
        
        # Process through topic awareness
        topic_context = await self.topic_awareness.process_user_input(user_input)
        
        # Get personality adjustments based on topic
        topic_adjustments = await self.topic_awareness.get_topic_personality_adjustments(topic_context)
        
        # Apply topic adjustments
        applied_adjustments = {}
        for trait_type, adjustment in topic_adjustments.items():
            if abs(adjustment) > 0.05:  # Only apply significant adjustments
                personality_adjustment = self.parameters.adjust_trait(
                    trait_type, 
                    adjustment, 
                    f"Topic context: {topic_context.category.value}",
                    confidence=topic_context.confidence
                )
                applied_adjustments[trait_type] = personality_adjustment
        
        # Apply behavioral learning adjustments
        if behavior_results.get("suggestions"):
            for trait_type, adjustment in behavior_results["suggestions"].items():
                personality_adjustment = self.parameters.adjust_trait(
                    trait_type,
                    adjustment,
                    "Behavioral learning",
                    confidence=0.7
                )
                applied_adjustments[trait_type] = personality_adjustment
        
        # Apply session closure adjustments if needed
        if session_state.in_closure_mode:
            closure_adjustments = self.session_manager.enter_closure_mode(session_state.closure_reason)
            for trait_type, adjustment in closure_adjustments.items():
                personality_adjustment = self.parameters.adjust_trait(
                    trait_type,
                    adjustment,
                    f"Session closure: {session_state.closure_reason}",
                    confidence=0.8
                )
                applied_adjustments[trait_type] = personality_adjustment
        
        # Store adjustments in history
        for adjustment in applied_adjustments.values():
            self.adjustment_history.append(adjustment)
        
        # Trim adjustment history
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]
        
        results = {
            "session_state": session_state.model_dump(),
            "topic_context": topic_context.model_dump(),
            "behavior_analysis": behavior_results,
            "personality_adjustments": {k: v.model_dump() for k, v in applied_adjustments.items()},
            "current_parameters": {
                trait.name.value: trait.value 
                for trait in self.parameters.get_parameters().traits.values()
            }
        }
        
        logger.info(f"Processed user input: {len(applied_adjustments)} personality adjustments applied")
        return results
    
    async def process_assistant_response(self, response: str) -> Dict[str, Any]:
        """
        Process assistant response for learning and analysis.
        
        Args:
            response: Assistant response text
            
        Returns:
            Dictionary with analysis results
        """
        # Update session
        self.session_manager.process_interaction("assistant", response)
        
        # Update personal lore based on interaction
        current_topic = self.topic_awareness.get_current_topic()
        lore_updates = self.personal_lore.update_lore_from_interaction(
            "", response, current_topic.category.value
        )
        
        # Analyze response for behavioral patterns
        behavior_patterns = await self.behavioral_conditioner.analyze_interaction_patterns()
        
        return {
            "lore_updates": lore_updates,
            "behavior_patterns": [p.model_dump() for p in behavior_patterns],
            "response_length": len(response),
            "session_turn": self.session_manager.session_state.turn_count
        }
    
    async def enhance_prompt(self, prompt: str, context: str = "") -> str:
        """
        Enhance a prompt with personality and lore.
        
        Args:
            prompt: Original prompt
            context: Additional context information
            
        Returns:
            Enhanced prompt with personality elements
        """
        # Get current personality parameters
        parameters = self.parameters.get_parameters()
        
        # Get current topic context
        topic_context = self.topic_awareness.get_current_topic()
        
        # Extract keywords from context and topic
        keywords = topic_context.keywords + context.split()
        
        # Enhance with personal lore
        enhanced_prompt = self.personal_lore.enhance_prompt(
            prompt, 
            topic_context.category.value, 
            keywords
        )
        
        # Add personality parameter guidance
        personality_guidance = self._generate_personality_guidance(parameters)
        if personality_guidance:
            enhanced_prompt += f"\n\nPersonality Guidance:\n{personality_guidance}"
        
        # Add session context if in closure mode
        session_state = self.session_manager.get_session_state()
        if session_state.in_closure_mode:
            enhanced_prompt += f"\n\nSession Context: This conversation is winding down ({session_state.closure_reason}). Consider providing a thoughtful conclusion."
        
        return enhanced_prompt
    
    async def format_response(self, response: str, context: str = "") -> str:
        """
        Format a response with personality elements.
        
        Args:
            response: Original response
            context: Current context
            
        Returns:
            Response formatted with personality elements
        """
        # Get current topic for context
        topic_context = self.topic_awareness.get_current_topic()
        
        # Extract trigger words from response
        trigger_words = response.split()
        
        # Format with personal lore
        formatted_response = self.personal_lore.format_response_with_lore(
            response, 
            topic_context.category.value, 
            trigger_words
        )
        
        return formatted_response
    
    async def process_feedback(self, feedback: str, feedback_type: str = "general") -> Dict[str, Any]:
        """
        Process user feedback for learning.
        
        Args:
            feedback: User feedback text
            feedback_type: Type of feedback
            
        Returns:
            Dictionary with feedback processing results
        """
        # Process through behavioral conditioner
        feedback_results = await self.behavioral_conditioner.process_user_feedback(feedback, feedback_type)
        
        # Apply any suggested adjustments
        applied_adjustments = {}
        if "style_feedback" in feedback_results:
            for trait_type, adjustment in feedback_results["style_feedback"].items():
                personality_adjustment = self.parameters.adjust_trait(
                    trait_type,
                    adjustment,
                    f"User feedback: {feedback_type}",
                    confidence=0.9  # High confidence for explicit feedback
                )
                applied_adjustments[trait_type] = personality_adjustment
        
        return {
            "feedback_analysis": feedback_results,
            "personality_adjustments": {k: v.model_dump() for k, v in applied_adjustments.items()},
            "learning_updated": True
        }
    
    def get_personality_state(self) -> Dict[str, Any]:
        """Get complete personality state."""
        parameters = self.parameters.get_parameters()
        topic_context = self.topic_awareness.get_current_topic()
        behavior_profile = self.behavioral_conditioner.get_behavior_profile()
        session_state = self.session_manager.get_session_state()
        
        return {
            "parameters": {
                trait.name.value: {
                    "value": trait.value,
                    "default": trait.default_value,
                    "deviation": trait.value - trait.default_value,
                    "adjustments": trait.adjustment_count
                }
                for trait in parameters.traits.values()
            },
            "topic_context": {
                "current_topic": topic_context.current_topic,
                "category": topic_context.category.value,
                "confidence": topic_context.confidence,
                "keywords": topic_context.keywords
            },
            "behavior_profile": {
                "preferences": {k.value: v for k, v in behavior_profile.user_preferences.items()},
                "confidence_scores": {k.value: v for k, v in behavior_profile.confidence_scores.items()},
                "observation_count": behavior_profile.observation_count
            },
            "session": {
                "id": session_state.session_id,
                "duration_minutes": session_state.duration_seconds / 60,
                "turn_count": session_state.turn_count,
                "in_closure_mode": session_state.in_closure_mode,
                "closure_reason": session_state.closure_reason
            },
            "lore_stats": self.personal_lore.get_lore_usage_stats()
        }
    
    async def apply_learned_adjustments(self) -> Dict[str, Any]:
        """Apply personality adjustments based on learning."""
        # Get suggestions from behavioral conditioner
        suggestions = await self.behavioral_conditioner.suggest_personality_adjustments()
        
        applied_adjustments = {}
        for trait_type, adjustment in suggestions.items():
            personality_adjustment = self.parameters.adjust_trait(
                trait_type,
                adjustment,
                "Learned behavior pattern",
                confidence=0.6
            )
            applied_adjustments[trait_type] = personality_adjustment
        
        return {
            "adjustments_applied": len(applied_adjustments),
            "adjustments": {k: v.model_dump() for k, v in applied_adjustments.items()}
        }
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get personality analytics and insights."""
        # Get component analytics
        parameter_stats = self.parameters.get_trait_summary()
        behavior_stats = self.behavioral_conditioner.get_learning_stats()
        topic_stats = self.topic_awareness.get_topic_stats()
        lore_stats = self.personal_lore.get_lore_usage_stats()
        session_analytics = self.session_manager.get_session_analytics()
        
        # Calculate overall metrics
        total_adjustments = len(self.adjustment_history)
        recent_adjustments = len([adj for adj in self.adjustment_history 
                                if (datetime.now() - adj.timestamp).total_seconds() < 3600])  # Last hour
        
        # Most adjusted traits
        trait_adjustment_counts = {}
        for adj in self.adjustment_history:
            trait_name = adj.trait_type.value
            trait_adjustment_counts[trait_name] = trait_adjustment_counts.get(trait_name, 0) + 1
        
        most_adjusted_trait = max(trait_adjustment_counts.items(), key=lambda x: x[1])[0] if trait_adjustment_counts else None
        
        return {
            "overview": {
                "total_interactions": self.interaction_count,
                "total_adjustments": total_adjustments,
                "recent_adjustments": recent_adjustments,
                "most_adjusted_trait": most_adjusted_trait,
                "session_quality": session_analytics.get("session_flow", {}).get("conversation_momentum", 0.5)
            },
            "parameters": parameter_stats,
            "behavior_learning": behavior_stats,
            "topic_awareness": topic_stats,
            "personal_lore": lore_stats,
            "session_analytics": session_analytics,
            "adjustment_history": [adj.model_dump() for adj in self.adjustment_history[-10:]]  # Last 10 adjustments
        }
    
    def _generate_personality_guidance(self, parameters) -> str:
        """Generate personality guidance for prompt enhancement."""
        guidance_parts = []
        
        # High-impact traits
        verbosity = parameters.get_trait_value(PersonalityTraitType.VERBOSITY)
        if verbosity > 0.7:
            guidance_parts.append("Provide detailed, comprehensive responses")
        elif verbosity < 0.3:
            guidance_parts.append("Keep responses concise and to the point")
        
        assertiveness = parameters.get_trait_value(PersonalityTraitType.ASSERTIVENESS)
        if assertiveness > 0.7:
            guidance_parts.append("Be confident and direct in your responses")
        elif assertiveness < 0.3:
            guidance_parts.append("Use tentative language and acknowledge uncertainty")
        
        humor = parameters.get_trait_value(PersonalityTraitType.HUMOR)
        if humor > 0.6:
            guidance_parts.append("Include appropriate humor when suitable")
        elif humor < 0.2:
            guidance_parts.append("Maintain a serious, professional tone")
        
        empathy = parameters.get_trait_value(PersonalityTraitType.EMPATHY)
        if empathy > 0.7:
            guidance_parts.append("Show understanding and emotional awareness")
        
        creativity = parameters.get_trait_value(PersonalityTraitType.CREATIVITY)
        if creativity > 0.7:
            guidance_parts.append("Feel free to be creative and imaginative")
        
        return ". ".join(guidance_parts) if guidance_parts else ""
    
    def reset_session(self, new_session_id: str = None) -> None:
        """Reset session state while preserving learned personality."""
        self.session_manager.reset_session(new_session_id)
        self.topic_awareness.clear_topic_history()
        
        # Keep behavioral learning and personality parameters
        logger.info(f"Session reset to {self.session_manager.session_state.session_id}")
    
    def export_personality_state(self) -> Dict[str, Any]:
        """Export complete personality state for persistence."""
        return {
            "parameters": self.parameters.export_parameters(),
            "behavior_profile": self.behavioral_conditioner.get_behavior_profile().model_dump(),
            "lore": self.personal_lore.get_lore().model_dump(),
            "adjustment_history": [adj.model_dump() for adj in self.adjustment_history],
            "exported_at": datetime.now().isoformat()
        }
    
    def import_personality_state(self, data: Dict[str, Any]) -> bool:
        """Import personality state from exported data."""
        try:
            if "parameters" in data:
                self.parameters.import_parameters(data["parameters"])
            
            if "adjustment_history" in data:
                self.adjustment_history = [
                    PersonalityAdjustment(**adj) for adj in data["adjustment_history"]
                ]
            
            logger.info("Successfully imported personality state")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import personality state: {e}")
            return False
