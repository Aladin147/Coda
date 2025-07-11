"""
Voice - LLM Context Integration

This module provides integration between voice processing, LLMs, and context systems
(memory, personality, tools) for enhanced conversation capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import Memory, MemoryMetadata, MemoryType
from ..personality.models import PersonalityParameters, PersonalityTrait
from .llm_integration import VoiceLLMConfig, VoiceLLMProcessor
from .model_manager import VoiceModelManager
from .models import ConversationState, VoiceMessage, VoiceResponse

logger = logging.getLogger("coda.voice.context_integration")


@dataclass
class ContextConfig:
    """Configuration for context integration."""

    # Memory integration
    enable_memory_injection: bool = True
    memory_context_length: int = 500  # Max characters from memory
    memory_relevance_threshold: float = 0.7

    # Personality integration
    enable_personality_injection: bool = True
    personality_context_length: int = 200  # Max characters from personality

    # Conversation context
    conversation_history_length: int = 10  # Number of previous messages
    context_window_minutes: int = 30  # Time window for relevant context

    # Dynamic model selection
    enable_dynamic_models: bool = True
    model_switch_threshold: float = 0.8  # Confidence threshold for switching

    # Performance optimization
    context_cache_ttl_minutes: int = 5  # Cache TTL for context data
    parallel_context_loading: bool = True


class ContextInjector:
    """
    Handles injection of context from various sources into LLM prompts.

    Features:
    - Memory system integration
    - Personality system integration
    - Conversation history management
    - Dynamic context optimization
    """

    def __init__(self, config: ContextConfig):
        """Initialize the context injector."""
        self.config = config
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        logger.info("ContextInjector initialized")

    async def build_context(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive context for LLM processing.

        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state

        Returns:
            Context dictionary with all relevant information
        """
        conversation_id = voice_message.conversation_id

        # Check cache first
        if self._is_context_cached(conversation_id):
            cached_context = self.context_cache[conversation_id].copy()
            # Update with current message
            cached_context["current_message"] = self._extract_message_context(voice_message)
            return cached_context

        # Build fresh context
        context = {}

        if self.config.parallel_context_loading:
            # Load context components in parallel
            tasks = []

            if self.config.enable_memory_injection:
                tasks.append(self._load_memory_context(voice_message))

            if self.config.enable_personality_injection:
                tasks.append(self._load_personality_context(conversation_id))

            tasks.append(self._load_conversation_context(conversation_id, conversation_state))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            result_index = 0
            if self.config.enable_memory_injection:
                context["memory"] = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else {}
                )
                result_index += 1

            if self.config.enable_personality_injection:
                context["personality"] = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else {}
                )
                result_index += 1

            context["conversation"] = (
                results[result_index] if not isinstance(results[result_index], Exception) else {}
            )

            # Extract message context separately (not async)
            context["current_message"] = self._extract_message_context(voice_message)

        else:
            # Load context components sequentially
            if self.config.enable_memory_injection:
                context["memory"] = await self._load_memory_context(voice_message)

            if self.config.enable_personality_injection:
                context["personality"] = await self._load_personality_context(conversation_id)

            context["conversation"] = await self._load_conversation_context(
                conversation_id, conversation_state
            )
            context["current_message"] = self._extract_message_context(voice_message)

        # Cache the context
        self._cache_context(conversation_id, context)

        return context

    async def _load_memory_context(self, voice_message: VoiceMessage) -> Dict[str, Any]:
        """Load relevant memory context."""
        try:
            # This would integrate with the actual memory system
            # For now, return a mock implementation

            memory_context = {
                "recent_topics": [
                    "User asked about machine learning yesterday",
                    "User is interested in voice AI development",
                    "User prefers technical explanations",
                ],
                "user_preferences": {
                    "communication_style": "technical",
                    "detail_level": "high",
                    "topics_of_interest": ["AI", "programming", "technology"],
                },
                "conversation_summary": "User is developing a voice AI system and has been asking technical questions about implementation.",
                "relevance_score": 0.8,
            }

            logger.debug(
                f"Loaded memory context with {len(memory_context['recent_topics'])} topics"
            )
            return memory_context

        except Exception as e:
            logger.error(f"Failed to load memory context: {e}")
            return {}

    async def _load_personality_context(self, conversation_id: str) -> Dict[str, Any]:
        """Load personality context."""
        try:
            # This would integrate with the actual personality system
            # For now, return a mock implementation

            personality_context = {
                "traits": {
                    "helpfulness": 0.9,
                    "technical_expertise": 0.8,
                    "friendliness": 0.7,
                    "curiosity": 0.8,
                },
                "communication_style": {
                    "formality": 0.3,  # Casual
                    "verbosity": 0.6,  # Moderately detailed
                    "enthusiasm": 0.7,  # Enthusiastic
                },
                "description": "I am Coda, a helpful AI assistant specializing in voice technology and AI development. I'm enthusiastic about helping with technical projects and enjoy explaining complex concepts clearly.",
                "current_mood": "engaged",
                "adaptation_level": 0.6,
            }

            logger.debug("Loaded personality context")
            return personality_context

        except Exception as e:
            logger.error(f"Failed to load personality context: {e}")
            return {}

    async def _load_conversation_context(
        self, conversation_id: str, conversation_state: Optional[ConversationState]
    ) -> Dict[str, Any]:
        """Load conversation context."""
        try:
            conversation_context = {
                "conversation_id": conversation_id,
                "message_count": 0,
                "duration_minutes": 0,
                "topics": [],
                "sentiment": "neutral",
                "engagement_level": 0.5,
            }

            if conversation_state:
                conversation_context.update(
                    {
                        "message_count": (
                            len(conversation_state.messages)
                            if hasattr(conversation_state, "messages")
                            else 0
                        ),
                        "is_active": conversation_state.is_active,
                        "start_time": (
                            conversation_state.start_time.isoformat()
                            if hasattr(conversation_state, "start_time")
                            else None
                        ),
                    }
                )

            logger.debug(f"Loaded conversation context for {conversation_id}")
            return conversation_context

        except Exception as e:
            logger.error(f"Failed to load conversation context: {e}")
            return {}

    def _extract_message_context(self, voice_message: VoiceMessage) -> Dict[str, Any]:
        """Extract context from the current voice message."""
        try:
            message_context = {
                "message_id": voice_message.message_id,
                "text_content": voice_message.text_content or "",
                "processing_mode": voice_message.processing_mode.value,
                "timestamp": voice_message.timestamp.isoformat(),
                "audio_length_seconds": (
                    len(voice_message.audio_data) / (24000 * 2) if voice_message.audio_data else 0
                ),
                "complexity_estimate": self._estimate_complexity(voice_message.text_content or ""),
                "intent_category": self._categorize_intent(voice_message.text_content or ""),
            }

            return message_context

        except Exception as e:
            logger.error(f"Failed to extract message context: {e}")
            return {}

    def _estimate_complexity(self, text: str) -> str:
        """Estimate the complexity of the user's request."""
        if not text:
            return "low"

        # Simple heuristics for complexity estimation
        word_count = len(text.split())

        # Check for complex keywords
        complex_keywords = [
            "implement",
            "algorithm",
            "architecture",
            "optimization",
            "performance",
            "integration",
            "system",
            "design",
            "analysis",
            "research",
            "compare",
            "explain",
            "detailed",
            "comprehensive",
            "advanced",
        ]

        complex_count = sum(1 for keyword in complex_keywords if keyword.lower() in text.lower())

        if word_count > 50 or complex_count > 2:
            return "high"
        elif word_count > 20 or complex_count > 0:
            return "medium"
        else:
            return "low"

    def _categorize_intent(self, text: str) -> str:
        """Categorize the user's intent."""
        if not text:
            return "unknown"

        text_lower = text.lower()

        # Intent categories with keywords
        intent_keywords = {
            "question": ["what", "how", "why", "when", "where", "which", "?"],
            "request": ["please", "can you", "could you", "would you", "help"],
            "coding": ["code", "program", "function", "class", "implement", "debug"],
            "explanation": ["explain", "describe", "tell me about", "what is"],
            "comparison": ["compare", "difference", "better", "vs", "versus"],
            "troubleshooting": ["error", "problem", "issue", "fix", "debug", "not working"],
            "general": ["hello", "hi", "thanks", "thank you", "goodbye"],
        }

        for intent, keywords in intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent

        return "general"

    def _is_context_cached(self, conversation_id: str) -> bool:
        """Check if context is cached and still valid."""
        if conversation_id not in self.context_cache:
            return False

        cache_time = self.cache_timestamps.get(conversation_id)
        if not cache_time:
            return False

        ttl = timedelta(minutes=self.config.context_cache_ttl_minutes)
        return datetime.now() - cache_time < ttl

    def _cache_context(self, conversation_id: str, context: Dict[str, Any]) -> None:
        """Cache context data."""
        self.context_cache[conversation_id] = context.copy()
        self.cache_timestamps[conversation_id] = datetime.now()

        # Clean old cache entries
        self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.now()
        ttl = timedelta(minutes=self.config.context_cache_ttl_minutes)

        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items() if now - timestamp > ttl
        ]

        for key in expired_keys:
            self.context_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)


class VoiceContextManager:
    """
    Main context manager that orchestrates voice processing with full context integration.

    Features:
    - Dynamic model selection based on context
    - Memory and personality integration
    - Performance optimization
    - Context - aware response generation
    """

    def __init__(
        self,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        moshi_vram_usage: float = 15.0,
    ):
        """Initialize the voice context manager."""
        self.context_config = context_config
        self.llm_config = llm_config

        # Initialize components
        self.context_injector = ContextInjector(context_config)
        self.model_manager = (
            VoiceModelManager(moshi_vram_usage) if context_config.enable_dynamic_models else None
        )
        self.current_processor: Optional[VoiceLLMProcessor] = None

        # Performance tracking
        self.processing_stats = {
            "total_requests": 0,
            "context_cache_hits": 0,
            "model_switches": 0,
            "average_latency_ms": 0.0,
        }

        logger.info("VoiceContextManager initialized")

    async def initialize(self) -> None:
        """Initialize the context manager."""
        try:
            # Initialize default processor
            self.current_processor = VoiceLLMProcessor(self.llm_config)
            await self.current_processor.initialize()

            logger.info("VoiceContextManager initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize VoiceContextManager: {e}")
            raise

    async def process_with_context(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """
        Process voice message with full context integration.

        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state

        Returns:
            Enhanced voice response with context
        """
        start_time = time.time()

        try:
            # Build comprehensive context
            context = await self.context_injector.build_context(voice_message, conversation_state)

            # Select optimal model if dynamic selection is enabled
            if self.context_config.enable_dynamic_models and self.model_manager:
                optimal_model = self.model_manager.get_optimal_model(context["current_message"])
                processor = await self._ensure_model_loaded(optimal_model)
            else:
                processor = self.current_processor

            # Process with context
            response = await processor.process_voice_message(
                voice_message, conversation_state, context
            )

            # Update statistics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            self._update_stats(latency_ms)

            return response

        except Exception as e:
            logger.error(f"Context processing failed: {e}")
            # Fallback to basic processing
            if self.current_processor:
                return await self.current_processor.process_voice_message(
                    voice_message, conversation_state
                )
            raise

    async def _ensure_model_loaded(self, model_name: str) -> VoiceLLMProcessor:
        """Ensure the specified model is loaded and return its processor."""
        if not self.model_manager:
            return self.current_processor

        current_model = self.model_manager.current_model
        if current_model != model_name:
            logger.info(f"Switching from {current_model} to {model_name}")
            processor = await self.model_manager.switch_model(model_name)
            self.processing_stats["model_switches"] += 1
            return processor

        return self.model_manager.loaded_models.get(model_name, self.current_processor)

    def _update_stats(self, latency_ms: float) -> None:
        """Update processing statistics."""
        self.processing_stats["total_requests"] += 1

        # Update average latency
        current_avg = self.processing_stats["average_latency_ms"]
        total_requests = self.processing_stats["total_requests"]

        new_avg = ((current_avg * (total_requests - 1)) + latency_ms) / total_requests
        self.processing_stats["average_latency_ms"] = new_avg

    def get_context_status(self) -> Dict[str, Any]:
        """Get current context manager status."""
        status = {
            "processing_stats": self.processing_stats.copy(),
            "cache_size": len(self.context_injector.context_cache),
            "current_model": None,
            "available_models": [],
        }

        if self.model_manager:
            vram_status = self.model_manager.get_vram_status()
            status.update(
                {
                    "current_model": vram_status["current_model"],
                    "loaded_models": vram_status["loaded_models"],
                    "vram_usage": vram_status,
                }
            )

            status["available_models"] = list(self.model_manager.available_models.keys())

        return status

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.current_processor:
                await self.current_processor.cleanup()

            if self.model_manager:
                await self.model_manager.cleanup()

            # Clear caches
            self.context_injector.context_cache.clear()
            self.context_injector.cache_timestamps.clear()

            logger.info("VoiceContextManager cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
