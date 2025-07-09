"""
Voice processing system for Coda 2.0.

This package provides comprehensive voice processing capabilities including:
- Kyutai Moshi integration for real-time speech conversation
- External LLM integration for enhanced reasoning
- Hybrid processing modes for optimal performance
- Audio streaming and WebSocket integration
- Voice activity detection and audio processing
"""

from .models import (
    VoiceConfig,
    VoiceMessage,
    VoiceResponse,
    VoiceStreamChunk,
    AudioConfig,
    ConversationState,
    VoiceEvent,
)

from .interfaces import (
    VoiceProcessorInterface,
    AudioProcessorInterface,
    SpeechToTextInterface,
    TextToSpeechInterface,
    VoiceActivityDetectorInterface,
)

from .manager import VoiceManager
from .moshi_integration import MoshiVoiceProcessor
from .hybrid_processor import HybridVoiceProcessor
from .audio_processor import AudioProcessor
from .websocket_integration import WebSocketVoiceManager

__all__ = [
    # Models
    "VoiceConfig",
    "VoiceMessage", 
    "VoiceResponse",
    "VoiceStreamChunk",
    "AudioConfig",
    "ConversationState",
    "VoiceEvent",
    
    # Interfaces
    "VoiceProcessorInterface",
    "AudioProcessorInterface", 
    "SpeechToTextInterface",
    "TextToSpeechInterface",
    "VoiceActivityDetectorInterface",
    
    # Core components
    "VoiceManager",
    "MoshiVoiceProcessor",
    "HybridVoiceProcessor", 
    "AudioProcessor",
    "WebSocketVoiceManager",
]
