"""
Voice processing interfaces for Coda 2.0.

This module defines the abstract interfaces for voice processing components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from .models import (
    VoiceConfig,
    VoiceMessage,
    VoiceResponse,
    VoiceStreamChunk,
    ConversationState,
    VoiceEvent,
    VoiceAnalytics,
    AudioConfig,
)


class VoiceProcessorInterface(ABC):
    """Abstract interface for voice processors."""
    
    @abstractmethod
    async def initialize(self, config: VoiceConfig) -> None:
        """Initialize the voice processor."""
        pass
    
    @abstractmethod
    async def start_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> ConversationState:
        """Start a new voice conversation."""
        pass
    
    @abstractmethod
    async def process_audio(self, message: VoiceMessage) -> VoiceResponse:
        """Process audio input and generate response."""
        pass
    
    @abstractmethod
    async def process_audio_stream(
        self, 
        conversation_id: str, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[VoiceStreamChunk, None]:
        """Process streaming audio input."""
        pass
    
    @abstractmethod
    async def end_conversation(self, conversation_id: str) -> None:
        """End a voice conversation."""
        pass
    
    @abstractmethod
    async def get_conversation_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Get current conversation state."""
        pass
    
    @abstractmethod
    async def get_analytics(self) -> VoiceAnalytics:
        """Get voice processing analytics."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup voice processor resources."""
        pass


class AudioProcessorInterface(ABC):
    """Abstract interface for audio processing."""
    
    @abstractmethod
    async def initialize(self, config: AudioConfig) -> None:
        """Initialize the audio processor."""
        pass
    
    @abstractmethod
    async def process_input_audio(self, audio_data: bytes) -> bytes:
        """Process input audio (noise reduction, enhancement, etc.)."""
        pass
    
    @abstractmethod
    async def process_output_audio(self, audio_data: bytes) -> bytes:
        """Process output audio (enhancement, normalization, etc.)."""
        pass
    
    @abstractmethod
    async def detect_voice_activity(self, audio_data: bytes) -> bool:
        """Detect voice activity in audio."""
        pass
    
    @abstractmethod
    async def extract_features(self, audio_data: bytes) -> Dict[str, Any]:
        """Extract audio features for analysis."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        pass
    
    @abstractmethod
    async def convert_format(self, audio_data: bytes, source_format: str, target_format: str) -> bytes:
        """Convert audio between formats."""
        pass


class SpeechToTextInterface(ABC):
    """Abstract interface for speech-to-text processing."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the STT processor."""
        pass
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio to text."""
        pass
    
    @abstractmethod
    async def transcribe_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        """Transcribe streaming audio to text."""
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        pass
    
    @abstractmethod
    async def detect_language(self, audio_data: bytes) -> str:
        """Detect language in audio."""
        pass
    
    @abstractmethod
    def get_confidence_score(self) -> float:
        """Get confidence score of last transcription."""
        pass


class TextToSpeechInterface(ABC):
    """Abstract interface for text-to-speech processing."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the TTS processor."""
        pass
    
    @abstractmethod
    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        """Synthesize text to audio."""
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self, 
        text: str, 
        voice_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text to streaming audio."""
        pass
    
    @abstractmethod
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voices."""
        pass
    
    @abstractmethod
    async def clone_voice(self, audio_sample: bytes, voice_name: str) -> str:
        """Clone voice from audio sample."""
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        pass


class VoiceActivityDetectorInterface(ABC):
    """Abstract interface for voice activity detection."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the VAD processor."""
        pass
    
    @abstractmethod
    async def detect_activity(self, audio_data: bytes) -> bool:
        """Detect voice activity in audio chunk."""
        pass
    
    @abstractmethod
    async def detect_activity_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bool, None]:
        """Detect voice activity in streaming audio."""
        pass
    
    @abstractmethod
    def set_sensitivity(self, sensitivity: float) -> None:
        """Set VAD sensitivity (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def get_confidence_score(self) -> float:
        """Get confidence score of last detection."""
        pass
    
    @abstractmethod
    async def calibrate(self, background_audio: bytes) -> None:
        """Calibrate VAD with background audio."""
        pass


class MoshiInterface(ABC):
    """Abstract interface for Kyutai Moshi integration."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize Moshi processor."""
        pass
    
    @abstractmethod
    async def start_conversation(self, conversation_id: str) -> None:
        """Start a Moshi conversation."""
        pass
    
    @abstractmethod
    async def process_audio(self, audio_data: bytes) -> bytes:
        """Process audio through Moshi."""
        pass
    
    @abstractmethod
    async def process_audio_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """Process streaming audio through Moshi."""
        pass
    
    @abstractmethod
    async def extract_text(self, audio_data: bytes) -> str:
        """Extract text from Moshi's inner monologue."""
        pass
    
    @abstractmethod
    async def inject_text(self, text: str) -> bytes:
        """Inject text into Moshi for speech synthesis."""
        pass
    
    @abstractmethod
    async def get_conversation_state(self) -> Dict[str, Any]:
        """Get Moshi conversation state."""
        pass
    
    @abstractmethod
    async def end_conversation(self, conversation_id: str) -> None:
        """End Moshi conversation."""
        pass


class ExternalLLMInterface(ABC):
    """Abstract interface for external LLM integration."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize external LLM."""
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response from external LLM."""
        pass
    
    @abstractmethod
    async def generate_response_stream(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from external LLM."""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get external LLM model information."""
        pass
    
    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if external LLM is available."""
        pass
    
    @abstractmethod
    def get_vram_usage(self) -> str:
        """Get current VRAM usage."""
        pass


class WebSocketVoiceInterface(ABC):
    """Abstract interface for WebSocket voice integration."""
    
    @abstractmethod
    async def set_websocket_integration(self, integration: Any) -> None:
        """Set WebSocket integration."""
        pass
    
    @abstractmethod
    async def broadcast_voice_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast voice event via WebSocket."""
        pass
    
    @abstractmethod
    async def broadcast_conversation_start(self, conversation_id: str) -> None:
        """Broadcast conversation start event."""
        pass
    
    @abstractmethod
    async def broadcast_conversation_end(self, conversation_id: str) -> None:
        """Broadcast conversation end event."""
        pass
    
    @abstractmethod
    async def broadcast_audio_chunk(self, conversation_id: str, chunk: VoiceStreamChunk) -> None:
        """Broadcast audio chunk event."""
        pass
    
    @abstractmethod
    async def broadcast_response_generated(self, conversation_id: str, response: VoiceResponse) -> None:
        """Broadcast response generated event."""
        pass
    
    @abstractmethod
    async def broadcast_analytics_update(self) -> None:
        """Broadcast analytics update."""
        pass


class VoiceManagerInterface(ABC):
    """Abstract interface for the main voice manager."""
    
    @abstractmethod
    async def initialize(self, config: VoiceConfig) -> None:
        """Initialize the voice manager."""
        pass
    
    @abstractmethod
    async def start_conversation(
        self, 
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """Start a new voice conversation."""
        pass
    
    @abstractmethod
    async def process_voice_input(
        self, 
        conversation_id: str, 
        audio_data: bytes
    ) -> VoiceResponse:
        """Process voice input and generate response."""
        pass
    
    @abstractmethod
    async def process_voice_stream(
        self, 
        conversation_id: str, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[VoiceStreamChunk, None]:
        """Process streaming voice input."""
        pass
    
    @abstractmethod
    async def end_conversation(self, conversation_id: str) -> None:
        """End a voice conversation."""
        pass
    
    @abstractmethod
    async def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs."""
        pass
    
    @abstractmethod
    async def get_conversation_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation state."""
        pass
    
    @abstractmethod
    async def set_memory_manager(self, memory_manager: Any) -> None:
        """Set memory manager for integration."""
        pass
    
    @abstractmethod
    async def set_personality_manager(self, personality_manager: Any) -> None:
        """Set personality manager for integration."""
        pass
    
    @abstractmethod
    async def set_tool_manager(self, tool_manager: Any) -> None:
        """Set tool manager for integration."""
        pass
    
    @abstractmethod
    async def get_analytics(self) -> VoiceAnalytics:
        """Get voice processing analytics."""
        pass
    
    @abstractmethod
    async def get_system_status(self) -> Dict[str, Any]:
        """Get voice system status."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup voice manager resources."""
        pass
