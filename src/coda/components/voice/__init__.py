"""
Voice processing system for Coda 2.0.

This package provides comprehensive voice processing capabilities including:
- Kyutai Moshi integration for real - time speech conversation
- External LLM integration for enhanced reasoning
- Hybrid processing modes for optimal performance
- Audio streaming and WebSocket integration
- Voice activity detection and audio processing
"""

# Phase 1 components only
from .audio_processor import AudioProcessor
from .interfaces import (
    AudioProcessorInterface,
    SpeechToTextInterface,
    TextToSpeechInterface,
    VoiceActivityDetectorInterface,
    VoiceProcessorInterface,
)
from .manager import VoiceManager
from .models import (
    AudioConfig,
    ConversationState,
    VoiceConfig,
    VoiceEvent,
    VoiceMessage,
    VoiceResponse,
    VoiceStreamChunk,
)

# Placeholder imports for Phase 2+ components
try:
    from .moshi_integration import MoshiVoiceProcessor
except ImportError:
    from .manager import MoshiVoiceProcessor

try:
    from .hybrid_processor import HybridVoiceProcessor
except ImportError:
    from .manager import HybridVoiceProcessor

try:
    from .websocket_integration import WebSocketVoiceManager
except ImportError:
    from .manager import WebSocketVoiceManager

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
