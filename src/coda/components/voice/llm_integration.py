"""
Voice - LLM Integration Module

This module provides integration between the voice processing system and external LLMs
for enhanced reasoning and conversation capabilities.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ..llm.manager import LLMManager
from ..llm.models import (
    LLMConfig,
    LLMConversation,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    MessageRole,
    ProviderConfig,
)
from .exceptions import (
    ComponentFailureError,
    ErrorCodes,
    LLMIntegrationError,
    NetworkError,
    VoiceTimeoutError,
    create_error,
    wrap_exception,
)
from .models import ConversationState, VoiceMessage, VoiceProcessingMode, VoiceResponse
from .resource_management import CircuitBreaker, with_retry, with_timeout
from .utils import LatencyTracker, track_latency
from .validation import validate_timeout

logger = logging.getLogger("coda.voice.llm_integration")


@dataclass
class VoiceLLMConfig:
    """Configuration for voice - LLM integration."""

    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_model: str = "llama3:8b - instruct - q4_0"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512

    # Integration Settings
    enable_streaming: bool = True
    enable_context_injection: bool = True
    max_context_length: int = 4000

    # Performance Settings
    llm_timeout_seconds: float = 5.0
    parallel_processing: bool = True
    fallback_to_moshi: bool = True

    # Voice - specific Settings
    voice_context_weight: float = 0.8
    personality_injection: bool = True
    memory_injection: bool = True


class VoiceLLMProcessor:
    """
    Processes voice messages using external LLMs for enhanced reasoning.

    This class integrates the existing LLM system with voice processing,
        providing enhanced conversation capabilities beyond Moshi's built - in responses.
    """

    def __init__(self, config: VoiceLLMConfig, moshi_client=None):
        """Initialize the voice - LLM processor."""
        self.config = config
        self.llm_manager: Optional[LLMManager] = None
        self.moshi_client = moshi_client  # For TTS synthesis
        self.latency_tracker = LatencyTracker("voice_llm")
        self.conversation_contexts: Dict[str, List[LLMMessage]] = {}
        self.active_streams: Dict[str, AsyncGenerator] = {}

        logger.info("VoiceLLMProcessor initialized")

    async def initialize(self) -> None:
        """Initialize the LLM manager and models."""
        try:
            # Configure LLM system
            llm_config = LLMConfig(
                default_provider=self.config.llm_provider,
                providers={
                    LLMProvider.OLLAMA: ProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        api_base="http://localhost:11434",
                        model=self.config.llm_model,
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_tokens,
                    )
                },
                enable_streaming=self.config.enable_streaming,
                timeout_seconds=self.config.llm_timeout_seconds,
            )

            # Initialize LLM manager (no separate initialize method needed)
            self.llm_manager = LLMManager(llm_config)

            logger.info(f"LLM manager initialized with model: {self.config.llm_model}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM manager: {e}")
            raise wrap_exception(
                e,
                LLMIntegrationError,
                "Failed to initialize LLM integration",
                ErrorCodes.LLM_INTEGRATION_FAILED,
            )

    @with_timeout(30.0, "LLM processing timed out")
    @with_retry(max_retries=2, delay=1.0, exceptions=(NetworkError, VoiceTimeoutError))
    async def process_voice_message(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VoiceResponse:
        """
        Process a voice message using LLM for enhanced reasoning.

        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state
            context: Additional context (memory, personality, etc.)

        Returns:
            Enhanced voice response with LLM reasoning
        """
        with track_latency(self.latency_tracker) as timing:
            try:
                # Extract text from voice message (from Moshi inner monologue)
                input_text = voice_message.text_content or ""
                if not input_text.strip():
                    logger.warning("No text content in voice message for LLM processing")
                    return self._create_fallback_response(voice_message)

                # Build conversation context
                conversation_id = voice_message.conversation_id
                llm_messages = await self._build_conversation_context(
                    conversation_id, input_text, context
                )

                # Process with LLM
                if self.config.enable_streaming:
                    return await self._process_streaming(voice_message, llm_messages)
                else:
                    return await self._process_non_streaming(voice_message, llm_messages)

            except Exception as e:
                logger.error(f"LLM processing failed: {e}")
                if self.config.fallback_to_moshi:
                    return self._create_fallback_response(voice_message)
                raise

    async def _build_conversation_context(
        self, conversation_id: str, input_text: str, context: Optional[Dict[str, Any]] = None
    ) -> List[LLMMessage]:
        """Build conversation context for LLM processing."""
        messages = []

        # Add system message with personality and context
        system_content = self._build_system_prompt(context)
        messages.append(LLMMessage(role=MessageRole.SYSTEM, content=system_content))

        # Add conversation history (limited by max_context_length)
        if conversation_id in self.conversation_contexts:
            history = self.conversation_contexts[conversation_id]
            # Keep recent messages within context limit
            recent_history = self._trim_context(history)
            messages.extend(recent_history)

        # Add current user message
        messages.append(LLMMessage(role=MessageRole.USER, content=input_text))

        return messages

    def _build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt with personality and context injection."""
        base_prompt = (
            "You are Coda, an advanced AI assistant with voice capabilities. "
            "Respond naturally and conversationally. Keep responses concise but helpful. "
        )

        if context and self.config.personality_injection:
            personality = context.get("personality", {})
            if personality:
                base_prompt += f"Your personality: {personality.get('description', '')} "

        if context and self.config.memory_injection:
            memory = context.get("memory", {})
            if memory:
                base_prompt += f"Relevant context: {memory.get('summary', '')} "

        return base_prompt

    def _trim_context(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Trim conversation context to fit within limits."""
        # Simple implementation: keep last N messages
        # TODO: Implement smarter context trimming based on token count
        max_messages = 10
        return messages[-max_messages:] if len(messages) > max_messages else messages

    async def _process_streaming(
        self, voice_message: VoiceMessage, llm_messages: List[LLMMessage]
    ) -> VoiceResponse:
        """Process with streaming LLM response."""
        try:
            # Get user message text
            user_text = llm_messages[-1].content if llm_messages else ""

            # Start streaming response (await to get AsyncGenerator)
            stream_generator = await self.llm_manager.generate_response(
                prompt=user_text, conversation_id=voice_message.conversation_id, stream=True
            )

            # Collect streaming response
            response_text = ""
            async for chunk in stream_generator:
                if hasattr(chunk, "delta"):
                    response_text += chunk.delta
                elif hasattr(chunk, "content"):
                    response_text += chunk.content

            # Update conversation context
            self._update_conversation_context(
                voice_message.conversation_id, llm_messages, response_text
            )

            return VoiceResponse(
                response_id=f"llm_{voice_message.message_id}",
                conversation_id=voice_message.conversation_id,
                message_id=voice_message.message_id,
                text_content=response_text,
                audio_data=await self._synthesize_speech(response_text),
                processing_mode=VoiceProcessingMode.HYBRID,
                total_latency_ms=self.latency_tracker.get_latest_latency(),
                llm_latency_ms=self.latency_tracker.get_latest_latency(),
                response_relevance=0.9,  # High relevance for LLM responses
            )

        except Exception as e:
            logger.error(f"Streaming LLM processing failed: {e}")
            raise

    async def _process_non_streaming(
        self, voice_message: VoiceMessage, llm_messages: List[LLMMessage]
    ) -> VoiceResponse:
        """Process with non - streaming LLM response."""
        try:
            # Get user message text
            user_text = llm_messages[-1].content if llm_messages else ""

            # Get LLM response
            response = await self.llm_manager.generate_response(
                prompt=user_text, conversation_id=voice_message.conversation_id, stream=False
            )

            response_text = response.content if hasattr(response, "content") else str(response)

            # Update conversation context
            self._update_conversation_context(
                voice_message.conversation_id, llm_messages, response_text
            )

            return VoiceResponse(
                response_id=f"llm_{voice_message.message_id}",
                conversation_id=voice_message.conversation_id,
                message_id=voice_message.message_id,
                text_content=response_text,
                audio_data=await self._synthesize_speech(response_text),
                processing_mode=VoiceProcessingMode.HYBRID,
                total_latency_ms=self.latency_tracker.get_latest_latency(),
                llm_latency_ms=self.latency_tracker.get_latest_latency(),
                response_relevance=0.9,
            )

        except Exception as e:
            logger.error(f"Non - streaming LLM processing failed: {e}")
            raise

    def _update_conversation_context(
        self, conversation_id: str, messages: List[LLMMessage], response_text: str
    ) -> None:
        """Update conversation context with new messages."""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = []

        # Add user message and assistant response
        context = self.conversation_contexts[conversation_id]

        # Add user message (last message in the list)
        if messages:
            user_message = messages[-1]
            if user_message.role == MessageRole.USER:
                context.append(user_message)

        # Add assistant response
        context.append(LLMMessage(role=MessageRole.ASSISTANT, content=response_text))

        # Trim context if too long
        self.conversation_contexts[conversation_id] = self._trim_context(context)

    def _create_fallback_response(self, voice_message: VoiceMessage) -> VoiceResponse:
        """Create a fallback response when LLM processing fails."""
        return VoiceResponse(
            response_id=f"fallback_{voice_message.message_id}",
            conversation_id=voice_message.conversation_id,
            message_id=voice_message.message_id,
            text_content="I'm having trouble processing that right now. Could you try again?",
            audio_data=b"",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=0.0,
            response_relevance=0.5,
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.llm_manager:
                await self.llm_manager.cleanup()

            # Clear conversation contexts
            self.conversation_contexts.clear()
            self.active_streams.clear()

            logger.info("VoiceLLMProcessor cleaned up")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def _synthesize_speech(self, text: str) -> bytes:
        """Convert text to speech using available TTS engines."""
        try:
            if not text or not text.strip():
                return b""

            # Use Moshi client for TTS if available
            if self.moshi_client and hasattr(self.moshi_client, 'inject_text'):
                try:
                    audio_data = await self.moshi_client.inject_text(text)
                    if audio_data and len(audio_data) > 0:
                        logger.debug(f"Synthesized speech via Moshi: {len(audio_data)} bytes")
                        return audio_data
                except Exception as e:
                    logger.warning(f"Moshi TTS failed: {e}")

            # Fallback: Generate synthetic audio data
            # This creates a simple audio representation for testing
            audio_data = self._generate_synthetic_audio(text)
            logger.debug(f"Generated synthetic audio: {len(audio_data)} bytes")
            return audio_data

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return b""

    def _generate_synthetic_audio(self, text: str) -> bytes:
        """Generate synthetic audio data as fallback."""
        try:
            # Create a simple audio representation based on text length
            # This is a placeholder that generates valid audio data
            import struct
            import math

            # Audio parameters
            sample_rate = 16000
            duration = max(0.5, len(text) * 0.1)  # Roughly 100ms per character
            num_samples = int(sample_rate * duration)

            # Generate simple sine wave audio
            audio_samples = []
            frequency = 440  # A4 note

            for i in range(num_samples):
                # Simple sine wave with envelope
                t = i / sample_rate
                envelope = math.exp(-t * 2)  # Decay envelope
                sample = envelope * math.sin(2 * math.pi * frequency * t)
                # Convert to 16-bit PCM
                sample_int = int(sample * 32767)
                audio_samples.append(struct.pack('<h', sample_int))

            return b''.join(audio_samples)

        except Exception as e:
            logger.error(f"Synthetic audio generation failed: {e}")
            return b""

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the voice - LLM integration."""
        return {
            "latency_stats": self.latency_tracker.get_stats(),
            "active_conversations": len(self.conversation_contexts),
            "active_streams": len(self.active_streams),
            "llm_model": self.config.llm_model,
            "llm_provider": self.config.llm_provider.value,
        }
