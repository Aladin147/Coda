"""
Voice - Personality Integration

This module provides comprehensive integration between the voice processing system
and the personality manager for adaptive, context - aware responses that reflect
the assistant's personality and evolve based on conversation feedback.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..personality.interfaces import PersonalityManagerInterface
from ..personality.models import (
    BehaviorPattern,
    PersonalityAdjustment,
    PersonalityParameters,
    PersonalityTraitType,
)
from .models import ConversationState, VoiceMessage, VoiceResponse

logger = logging.getLogger("coda.voice.personality_integration")


@dataclass
class VoicePersonalityConfig:
    """Configuration for voice - personality integration."""

    # Personality injection
    enable_personality_injection: bool = True
    personality_context_length: int = 300  # Max characters from personality
    include_trait_values: bool = True
    include_behavior_patterns: bool = True

    # Response adaptation
    enable_response_adaptation: bool = True
    adapt_speaking_style: bool = True
    adapt_formality_level: bool = True
    adapt_enthusiasm_level: bool = True

    # Learning and evolution
    enable_personality_learning: bool = True
    learn_from_user_feedback: bool = True
    learn_from_conversation_flow: bool = True
    adjustment_sensitivity: float = 0.1  # How quickly personality adapts

    # Voice - specific traits
    enable_voice_traits: bool = True
    voice_confidence_factor: float = 1.2  # Voice interactions boost confidence
    voice_engagement_factor: float = 1.1  # Voice interactions boost engagement

    # Performance optimization
    personality_cache_ttl_minutes: int = 10
    parallel_personality_operations: bool = True


class VoicePersonalityIntegration:
    """
    Comprehensive integration between voice processing and personality systems.

    Features:
    - Personality - aware response generation
    - Dynamic personality adaptation based on voice interactions
    - Speaking style and tone adjustment
    - Conversation flow analysis for personality learning
    - Voice - specific personality trait evolution
    """

    def __init__(
        self, personality_manager: PersonalityManagerInterface, config: VoicePersonalityConfig
    ):
        """Initialize voice - personality integration."""
        self.personality_manager = personality_manager
        self.config = config

        # Caching for performance
        self.personality_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # Statistics tracking
        self.stats = {
            "personality_injections": 0,
            "response_adaptations": 0,
            "personality_adjustments": 0,
            "cache_hits": 0,
            "voice_interactions": 0,
        }

        # Voice interaction tracking
        self.conversation_flows: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("VoicePersonalityIntegration initialized")

    async def enhance_voice_context(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]:
        """
        Enhance voice processing context with personality information.

        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state

        Returns:
            Enhanced context with personality information
        """

        try:
            if not self.config.enable_personality_injection:
                return {}

            # Check cache first
            cache_key = f"personality_{voice_message.conversation_id}"
            if self._is_personality_cached(cache_key):
                self.stats["cache_hits"] += 1
                return self._get_cached_personality_context(cache_key)

            # Get current personality state
            personality_context = await self._get_personality_context(voice_message)

            # Cache the result
            self._cache_personality_context(cache_key, personality_context)

            self.stats["personality_injections"] += 1

            return personality_context

        except Exception as e:
            logger.error(f"Failed to enhance voice context with personality: {e}")
            return {}

    async def adapt_voice_response(
        self,
        voice_response: VoiceResponse,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None,
    ) -> VoiceResponse:
        """
        Adapt voice response based on personality and conversation context.

        Args:
            voice_response: The voice response to adapt
            voice_message: Original voice message
            conversation_state: Current conversation state

        Returns:
            Personality - adapted voice response
        """

        try:
            if not self.config.enable_response_adaptation:
                return voice_response

            # Get current personality parameters
            personality_params = await self._get_personality_parameters()

            # Adapt response text
            adapted_text = await self._adapt_response_text(
                voice_response.text_content or "", personality_params, voice_message
            )

            # Create adapted response
            adapted_response = VoiceResponse(
                response_id=voice_response.response_id,
                conversation_id=voice_response.conversation_id,
                message_id=voice_response.message_id,
                text_content=adapted_text,
                audio_data=voice_response.audio_data,
                processing_mode=voice_response.processing_mode,
                total_latency_ms=voice_response.total_latency_ms,
                moshi_latency_ms=voice_response.moshi_latency_ms,
                llm_latency_ms=voice_response.llm_latency_ms,
                response_relevance=voice_response.response_relevance,
            )

            self.stats["response_adaptations"] += 1

            return adapted_response

        except Exception as e:
            logger.error(f"Failed to adapt voice response: {e}")
            return voice_response

    async def learn_from_voice_interaction(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None,
        user_feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Learn and adapt personality based on voice interaction.

        Args:
            voice_message: User voice message
            voice_response: Assistant voice response
            conversation_state: Current conversation state
            user_feedback: Optional user feedback
        """

        try:
            if not self.config.enable_personality_learning:
                return

            # Track voice interaction
            self.stats["voice_interactions"] += 1

            # Record conversation flow
            await self._record_conversation_flow(voice_message, voice_response)

            # Learn from user feedback if provided
            if user_feedback and self.config.learn_from_user_feedback:
                await self._learn_from_user_feedback(voice_message, voice_response, user_feedback)

            # Learn from conversation flow patterns
            if self.config.learn_from_conversation_flow:
                await self._learn_from_conversation_flow(voice_message, voice_response)

            # Apply voice - specific personality adjustments
            if self.config.enable_voice_traits:
                await self._apply_voice_trait_adjustments(voice_message, voice_response)

        except Exception as e:
            logger.error(f"Failed to learn from voice interaction: {e}")

    async def _get_personality_context(self, voice_message: VoiceMessage) -> Dict[str, Any]:
        """Get comprehensive personality context."""

        try:
            # Get current personality parameters
            personality_params = await self._get_personality_parameters()

            # Build personality context
            personality_context = {
                "traits": {},
                "speaking_style": {},
                "current_state": {},
                "behavior_patterns": [],
                "voice_adaptations": {},
            }

            # Extract trait values
            if self.config.include_trait_values:
                personality_context["traits"] = {
                    trait.name: trait.current_value for trait in personality_params.traits
                }

            # Extract speaking style parameters
            if self.config.adapt_speaking_style:
                personality_context["speaking_style"] = {
                    "formality": personality_context["traits"].get("formality", 0.5),
                    "enthusiasm": personality_context["traits"].get("enthusiasm", 0.7),
                    "verbosity": personality_context["traits"].get("verbosity", 0.6),
                    "technical_depth": personality_context["traits"].get(
                        "technical_expertise", 0.8
                    ),
                }

            # Get current personality state
            personality_context["current_state"] = {
                "mood": "engaged",  # This would come from personality manager
                "energy_level": 0.8,
                "confidence": personality_context["traits"].get("confidence", 0.7),
                "adaptability": personality_context["traits"].get("adaptability", 0.6),
            }

            # Voice - specific adaptations
            if self.config.enable_voice_traits:
                personality_context["voice_adaptations"] = {
                    "voice_confidence_boost": self.config.voice_confidence_factor,
                    "voice_engagement_boost": self.config.voice_engagement_factor,
                    "prefers_voice_interaction": True,
                }

            # Generate personality description
            personality_context["description"] = self._generate_personality_description(
                personality_context
            )

            return personality_context

        except Exception as e:
            logger.error(f"Failed to get personality context: {e}")
            return {}

    async def _get_personality_parameters(self) -> PersonalityParameters:
        """Get current personality parameters from personality manager."""

        try:
            # This would call the actual personality manager
            # For now, return a mock implementation

            class MockPersonalityParameters:
                def __init__(self):
                    self.traits = [
                        type(
                            "Trait",
                            (),
                            {
                                "name": "empathy",
                                "current_value": 0.9,
                                "trait_type": PersonalityTraitType.EMPATHY,
                            },
                        )(),
                        type(
                            "Trait",
                            (),
                            {
                                "name": "enthusiasm",
                                "current_value": 0.8,
                                "trait_type": PersonalityTraitType.ENTHUSIASM,
                            },
                        )(),
                        type(
                            "Trait",
                            (),
                            {
                                "name": "formality",
                                "current_value": 0.3,
                                "trait_type": PersonalityTraitType.FORMALITY,
                            },
                        )(),
                        type(
                            "Trait",
                            (),
                            {
                                "name": "analytical",
                                "current_value": 0.85,
                                "trait_type": PersonalityTraitType.ANALYTICAL,
                            },
                        )(),
                        type(
                            "Trait",
                            (),
                            {
                                "name": "confidence",
                                "current_value": 0.75,
                                "trait_type": PersonalityTraitType.CONFIDENCE,
                            },
                        )(),
                    ]

            return MockPersonalityParameters()

        except Exception as e:
            logger.error(f"Failed to get personality parameters: {e}")
            return None

    async def _adapt_response_text(
        self, response_text: str, personality_params: Any, voice_message: VoiceMessage
    ) -> str:
        """Adapt response text based on personality parameters."""

        try:
            if not response_text or not personality_params:
                return response_text

            # Get trait values
            traits = {trait.name: trait.current_value for trait in personality_params.traits}

            adapted_text = response_text

            # Adapt formality level
            if self.config.adapt_formality_level:
                formality = traits.get("formality", 0.5)
                if formality < 0.3:  # Very casual
                    adapted_text = self._make_more_casual(adapted_text)
                elif formality > 0.7:  # Very formal
                    adapted_text = self._make_more_formal(adapted_text)

            # Adapt enthusiasm level
            if self.config.adapt_enthusiasm_level:
                enthusiasm = traits.get("enthusiasm", 0.7)
                if enthusiasm > 0.8:  # High enthusiasm
                    adapted_text = self._add_enthusiasm(adapted_text)
                elif enthusiasm < 0.4:  # Low enthusiasm
                    adapted_text = self._reduce_enthusiasm(adapted_text)

            # Apply voice - specific adaptations
            if self.config.enable_voice_traits:
                adapted_text = self._apply_voice_adaptations(adapted_text, traits)

            return adapted_text

        except Exception as e:
            logger.error(f"Failed to adapt response text: {e}")
            return response_text

    def _make_more_casual(self, text: str) -> str:
        """Make text more casual."""
        # Simple casual adaptations
        casual_replacements = {
            "I would": "I'd",
            "you would": "you'd",
            "cannot": "can't",
            "do not": "don't",
            "will not": "won't",
            "However,": "But",
            "Therefore,": "So",
            "Additionally,": "Also",
        }

        adapted_text = text
        for formal, casual in casual_replacements.items():
            adapted_text = adapted_text.replace(formal, casual)

        return adapted_text

    def _make_more_formal(self, text: str) -> str:
        """Make text more formal."""
        # Simple formal adaptations
        formal_replacements = {
            "I'd": "I would",
            "you'd": "you would",
            "can't": "cannot",
            "don't": "do not",
            "won't": "will not",
            "But": "However,",
            "So": "Therefore,",
            "Also": "Additionally",
        }

        adapted_text = text
        for casual, formal in formal_replacements.items():
            adapted_text = adapted_text.replace(casual, formal)

        return adapted_text

    def _add_enthusiasm(self, text: str) -> str:
        """Add enthusiasm to text."""
        # Add enthusiasm markers
        if not text.endswith(("!", "?", ".")):
            text += "!"
        elif text.endswith("."):
            text = text[:-1] + "!"

        # Add enthusiastic phrases
        enthusiastic_starters = ["Great question! ", "Absolutely! ", "Fantastic! ", "Excellent! "]

        # Randomly add enthusiasm (simplified)
        if len(text) > 50 and not any(
            text.startswith(starter) for starter in enthusiastic_starters
        ):
            text = "Great! " + text

        return text

    def _reduce_enthusiasm(self, text: str) -> str:
        """Reduce enthusiasm in text."""
        # Remove excessive exclamation marks
        text = text.replace("!!", ".")
        text = text.replace("!", ".")

        # Remove enthusiastic starters
        enthusiastic_starters = [
            "Great question! ",
            "Absolutely! ",
            "Fantastic! ",
            "Excellent! ",
            "Great! ",
        ]

        for starter in enthusiastic_starters:
            if text.startswith(starter):
                text = text[len(starter) :]
                break

        return text

    def _apply_voice_adaptations(self, text: str, traits: Dict[str, float]) -> str:
        """Apply voice - specific adaptations."""

        # Voice interactions tend to be more conversational
        confidence = traits.get("confidence", 0.7) * self.config.voice_confidence_factor

        if confidence > 0.9:
            # High confidence - more direct statements
            text = text.replace("I think", "I know")
            text = text.replace("maybe", "definitely")
            text = text.replace("might", "will")

        # Voice interactions are more personal
        if not any(word in text.lower() for word in ["you", "your"]):
            # Add personal touch if missing
            if text.startswith("The "):
                text = "Your " + text[4:]

        return text

    def _generate_personality_description(self, personality_context: Dict[str, Any]) -> str:
        """Generate a description of current personality state."""

        traits = personality_context.get("traits", {})
        speaking_style = personality_context.get("speaking_style", {})

        # Build personality description
        description_parts = []

        # Empathy level
        empathy = traits.get("empathy", 0.7)
        if empathy > 0.8:
            description_parts.append("highly empathetic and supportive")
        elif empathy > 0.6:
            description_parts.append("empathetic and accommodating")
        else:
            description_parts.append("moderately empathetic")

        # Enthusiasm level
        enthusiasm = traits.get("enthusiasm", 0.7)
        if enthusiasm > 0.8:
            description_parts.append("enthusiastic and energetic")
        elif enthusiasm > 0.6:
            description_parts.append("engaged and positive")
        else:
            description_parts.append("calm and measured")

        # Formality level
        formality = speaking_style.get("formality", 0.5)
        if formality > 0.7:
            description_parts.append("formal and professional")
        elif formality < 0.3:
            description_parts.append("casual and friendly")
        else:
            description_parts.append("conversational")

        # Analytical thinking
        analytical = traits.get("analytical", 0.8)
        if analytical > 0.8:
            description_parts.append("analytically minded")

        return f"Assistant personality: {', '.join(description_parts)}."

    async def _record_conversation_flow(
        self, voice_message: VoiceMessage, voice_response: VoiceResponse
    ) -> None:
        """Record conversation flow for learning."""

        conversation_id = voice_message.conversation_id

        if conversation_id not in self.conversation_flows:
            self.conversation_flows[conversation_id] = []

        flow_entry = {
            "timestamp": datetime.now(),
            "user_message_length": len(voice_message.text_content or ""),
            "response_length": len(voice_response.text_content or ""),
            "response_latency": voice_response.total_latency_ms,
            "response_relevance": voice_response.response_relevance or 0.5,
            "processing_mode": voice_response.processing_mode.value,
        }

        self.conversation_flows[conversation_id].append(flow_entry)

        # Keep only recent flow entries
        if len(self.conversation_flows[conversation_id]) > 50:
            self.conversation_flows[conversation_id] = self.conversation_flows[conversation_id][
                -50:
            ]

    async def _learn_from_user_feedback(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        user_feedback: Dict[str, Any],
    ) -> None:
        """Learn from explicit user feedback."""

        try:
            feedback_type = user_feedback.get("type", "general")
            feedback_score = user_feedback.get("score", 0.5)  # 0.0 to 1.0
            feedback_text = user_feedback.get("text", "")

            # Determine personality adjustments based on feedback
            adjustments = []

            if feedback_type == "too_formal" or "too formal" in feedback_text.lower():
                adjustments.append(("formality", -self.config.adjustment_sensitivity))
            elif feedback_type == "too_casual" or "too casual" in feedback_text.lower():
                adjustments.append(("formality", self.config.adjustment_sensitivity))

            if feedback_type == "too_enthusiastic" or "too excited" in feedback_text.lower():
                adjustments.append(("enthusiasm", -self.config.adjustment_sensitivity))
            elif feedback_type == "not_enthusiastic" or "more energy" in feedback_text.lower():
                adjustments.append(("enthusiasm", self.config.adjustment_sensitivity))

            # Apply adjustments
            for trait_name, adjustment in adjustments:
                await self._apply_personality_adjustment(trait_name, adjustment, "user_feedback")

            logger.info(f"Applied {len(adjustments)} personality adjustments from user feedback")

        except Exception as e:
            logger.error(f"Failed to learn from user feedback: {e}")

    async def _learn_from_conversation_flow(
        self, voice_message: VoiceMessage, voice_response: VoiceResponse
    ) -> None:
        """Learn from conversation flow patterns."""

        try:
            conversation_id = voice_message.conversation_id
            flow_history = self.conversation_flows.get(conversation_id, [])

            if len(flow_history) < 3:  # Need some history to learn
                return

            # Analyze recent conversation patterns
            recent_flows = flow_history[-5:]  # Last 5 interactions

            # Check response relevance trends
            relevance_scores = [flow.get("response_relevance", 0.5) for flow in recent_flows]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)

            # Check response latency trends
            latencies = [flow.get("response_latency", 200) for flow in recent_flows]
            avg_latency = sum(latencies) / len(latencies)

            # Adjust personality based on performance
            adjustments = []

            if avg_relevance < 0.6:  # Low relevance - be more empathetic
                adjustments.append(("empathy", self.config.adjustment_sensitivity * 0.5))

            if avg_latency > 1000:  # High latency - be more concise
                adjustments.append(("verbosity", -self.config.adjustment_sensitivity * 0.3))

            # Apply adjustments
            for trait_name, adjustment in adjustments:
                await self._apply_personality_adjustment(
                    trait_name, adjustment, "conversation_flow"
                )

            if adjustments:
                logger.debug(
                    f"Applied {len(adjustments)} personality adjustments from conversation flow"
                )

        except Exception as e:
            logger.error(f"Failed to learn from conversation flow: {e}")

    async def _apply_voice_trait_adjustments(
        self, voice_message: VoiceMessage, voice_response: VoiceResponse
    ) -> None:
        """Apply voice - specific personality trait adjustments."""

        try:
            # Voice interactions boost confidence and engagement
            confidence_boost = self.config.adjustment_sensitivity * 0.2
            engagement_boost = self.config.adjustment_sensitivity * 0.1

            await self._apply_personality_adjustment(
                "confidence", confidence_boost, "voice_interaction"
            )
            await self._apply_personality_adjustment(
                "engagement", engagement_boost, "voice_interaction"
            )

            # Successful voice interactions (high relevance) boost enthusiasm
            if voice_response.response_relevance and voice_response.response_relevance > 0.8:
                enthusiasm_boost = self.config.adjustment_sensitivity * 0.15
                await self._apply_personality_adjustment(
                    "enthusiasm", enthusiasm_boost, "successful_voice_interaction"
                )

        except Exception as e:
            logger.error(f"Failed to apply voice trait adjustments: {e}")

    async def _apply_personality_adjustment(
        self, trait_name: str, adjustment: float, source: str
    ) -> None:
        """Apply a personality trait adjustment."""

        try:
            # This would call the actual personality manager to adjust traits
            # For now, just log the adjustment

            self.stats["personality_adjustments"] += 1

            logger.debug(
                f"Personality adjustment: {trait_name} {adjustment:+.3f} (source: {source})"
            )

            # In a real implementation, this would call:
            # await self.personality_manager.adjust_trait(trait_name, adjustment, source)

        except Exception as e:
            logger.error(f"Failed to apply personality adjustment: {e}")

    def _is_personality_cached(self, cache_key: str) -> bool:
        """Check if personality context is cached and valid."""

        if cache_key not in self.personality_cache:
            return False

        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return False

        ttl = timedelta(minutes=self.config.personality_cache_ttl_minutes)
        return datetime.now() - cache_time < ttl

    def _get_cached_personality_context(self, cache_key: str) -> Dict[str, Any]:
        """Get cached personality context."""
        return self.personality_cache.get(cache_key, {})

    def _cache_personality_context(self, cache_key: str, context: Dict[str, Any]) -> None:
        """Cache personality context."""
        self.personality_cache[cache_key] = context
        self.cache_timestamps[cache_key] = datetime.now()

        # Clean old cache entries
        self._cleanup_personality_cache()

    def _cleanup_personality_cache(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.now()
        ttl = timedelta(minutes=self.config.personality_cache_ttl_minutes)

        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items() if now - timestamp > ttl
        ]

        for key in expired_keys:
            self.personality_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get voice - personality integration statistics."""

        return {
            "voice_personality_stats": self.stats.copy(),
            "cache_size": len(self.personality_cache),
            "active_conversations": len(self.conversation_flows),
            "config": {
                "personality_injection_enabled": self.config.enable_personality_injection,
                "response_adaptation_enabled": self.config.enable_response_adaptation,
                "personality_learning_enabled": self.config.enable_personality_learning,
                "voice_traits_enabled": self.config.enable_voice_traits,
                "adjustment_sensitivity": self.config.adjustment_sensitivity,
            },
        }

    async def cleanup(self) -> None:
        """Clean up integration resources."""

        try:
            # Clear caches
            self.personality_cache.clear()
            self.cache_timestamps.clear()

            # Clear conversation flows
            self.conversation_flows.clear()

            logger.info("VoicePersonalityIntegration cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class VoicePersonalityManager:
    """
    High - level manager for voice - personality integration.

    Provides a simplified interface for voice components to interact with personality.
    """

    def __init__(
        self,
        personality_manager: PersonalityManagerInterface,
        config: Optional[VoicePersonalityConfig] = None,
    ):
        """Initialize voice personality manager."""
        self.personality_manager = personality_manager
        self.config = config or VoicePersonalityConfig()
        self.integration = VoicePersonalityIntegration(personality_manager, self.config)

        logger.info("VoicePersonalityManager initialized")

    async def enhance_voice_context(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]:
        """
        Enhance voice processing context with personality.

        Args:
            voice_message: Voice message to process
            conversation_state: Current conversation state

        Returns:
            Enhanced context dictionary
        """
        return await self.integration.enhance_voice_context(voice_message, conversation_state)

    async def adapt_voice_response(
        self,
        voice_response: VoiceResponse,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None,
    ) -> VoiceResponse:
        """
        Adapt voice response based on personality.

        Args:
            voice_response: Voice response to adapt
            voice_message: Original voice message
            conversation_state: Current conversation state

        Returns:
            Personality - adapted voice response
        """
        return await self.integration.adapt_voice_response(
            voice_response, voice_message, conversation_state
        )

    async def learn_from_interaction(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None,
        user_feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Learn from voice interaction for personality evolution.

        Args:
            voice_message: User voice message
            voice_response: Assistant voice response
            conversation_state: Current conversation state
            user_feedback: Optional user feedback
        """
        await self.integration.learn_from_voice_interaction(
            voice_message, voice_response, conversation_state, user_feedback
        )

    def get_personality_stats(self) -> Dict[str, Any]:
        """Get comprehensive personality statistics."""
        return self.integration.get_integration_stats()

    async def cleanup(self) -> None:
        """Clean up manager resources."""
        await self.integration.cleanup()
