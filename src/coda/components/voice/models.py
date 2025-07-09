"""
Voice processing models for Coda 2.0.

This module defines the data models and configurations for the voice processing system.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class VoiceProcessingMode(str, Enum):
    """Voice processing modes."""
    MOSHI_ONLY = "moshi_only"
    HYBRID = "hybrid"
    TRADITIONAL = "traditional"
    ADAPTIVE = "adaptive"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    WEBM = "webm"


class VoiceProvider(str, Enum):
    """Voice processing providers."""
    MOSHI = "moshi"
    WHISPER = "whisper"
    ELEVENLABS = "elevenlabs"
    OPENAI_TTS = "openai_tts"
    COQUI_TTS = "coqui_tts"
    OLLAMA = "ollama"


class ConversationMode(str, Enum):
    """Conversation modes."""
    FULL_DUPLEX = "full_duplex"
    TURN_BASED = "turn_based"
    PUSH_TO_TALK = "push_to_talk"


class AudioConfig(BaseModel):
    """Audio processing configuration."""
    model_config = ConfigDict(extra="forbid")
    
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    bit_depth: int = Field(default=16, description="Audio bit depth")
    chunk_size: int = Field(default=1024, description="Audio chunk size for streaming")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")
    
    # Voice Activity Detection
    vad_enabled: bool = Field(default=True, description="Enable voice activity detection")
    vad_threshold: float = Field(default=0.5, description="VAD sensitivity threshold")
    silence_duration_ms: int = Field(default=1000, description="Silence duration to end speech")
    
    # Audio enhancement
    noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    echo_cancellation: bool = Field(default=True, description="Enable echo cancellation")
    auto_gain_control: bool = Field(default=True, description="Enable automatic gain control")


class MoshiConfig(BaseModel):
    """Kyutai Moshi configuration."""
    model_config = ConfigDict(extra="forbid")
    
    model_path: str = Field(default="kyutai/moshika-pytorch-bf16", description="Moshi model path")
    device: str = Field(default="cuda", description="Device for inference")
    optimization: str = Field(default="bf16", description="Model optimization level")
    max_conversation_length: int = Field(default=300, description="Max conversation length in seconds")
    
    # Performance settings
    target_latency_ms: int = Field(default=200, description="Target latency in milliseconds")
    vram_allocation: str = Field(default="8GB", description="VRAM allocation for Moshi")
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    
    # Integration settings
    external_llm_enabled: bool = Field(default=True, description="Enable external LLM integration")
    inner_monologue_enabled: bool = Field(default=True, description="Enable inner monologue text extraction")


class ExternalLLMConfig(BaseModel):
    """External LLM configuration for hybrid processing."""
    model_config = ConfigDict(extra="forbid")
    
    provider: str = Field(default="ollama", description="LLM provider")
    model: str = Field(default="llama3.1:70b-instruct-q4_K_M", description="LLM model")
    vram_allocation: str = Field(default="20GB", description="VRAM allocation for LLM")
    
    # Processing settings
    reasoning_mode: str = Field(default="enhanced", description="Reasoning mode")
    context_window: int = Field(default=8192, description="Context window size")
    temperature: float = Field(default=0.7, description="Generation temperature")
    
    # Integration settings
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    fallback_enabled: bool = Field(default=True, description="Enable fallback to Moshi-only")


class VoiceConfig(BaseModel):
    """Main voice processing configuration."""
    model_config = ConfigDict(extra="forbid")
    
    # Core settings
    mode: VoiceProcessingMode = Field(default=VoiceProcessingMode.HYBRID, description="Voice processing mode")
    conversation_mode: ConversationMode = Field(default=ConversationMode.FULL_DUPLEX, description="Conversation mode")
    
    # Component configurations
    audio: AudioConfig = Field(default_factory=AudioConfig, description="Audio configuration")
    moshi: MoshiConfig = Field(default_factory=MoshiConfig, description="Moshi configuration")
    external_llm: ExternalLLMConfig = Field(default_factory=ExternalLLMConfig, description="External LLM configuration")
    
    # Integration settings
    memory_integration_enabled: bool = Field(default=True, description="Enable memory integration")
    personality_integration_enabled: bool = Field(default=True, description="Enable personality integration")
    tools_integration_enabled: bool = Field(default=True, description="Enable tools integration")
    websocket_events_enabled: bool = Field(default=True, description="Enable WebSocket events")
    
    # Performance settings
    total_vram: str = Field(default="32GB", description="Total available VRAM")
    reserved_system: str = Field(default="4GB", description="Reserved VRAM for system")
    dynamic_allocation: bool = Field(default=True, description="Enable dynamic VRAM allocation")
    
    # Fallback settings
    enable_traditional_pipeline: bool = Field(default=False, description="Enable traditional STT/TTS fallback")
    fallback_whisper_model: str = Field(default="large-v3", description="Fallback Whisper model")
    fallback_tts_model: str = Field(default="xtts_v2", description="Fallback TTS model")


class VoiceMessage(BaseModel):
    """Voice message model."""
    model_config = ConfigDict(extra="forbid")
    
    message_id: str = Field(description="Unique message identifier")
    conversation_id: str = Field(description="Conversation identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    
    # Content
    audio_data: Optional[bytes] = Field(default=None, description="Raw audio data")
    text_content: Optional[str] = Field(default=None, description="Transcribed text content")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    duration_ms: Optional[int] = Field(default=None, description="Audio duration in milliseconds")
    audio_format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")
    
    # Processing info
    processing_mode: VoiceProcessingMode = Field(description="Processing mode used")
    latency_ms: Optional[float] = Field(default=None, description="Processing latency")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score")
    
    # Integration data
    memory_context: Optional[Dict[str, Any]] = Field(default=None, description="Memory context used")
    personality_state: Optional[Dict[str, Any]] = Field(default=None, description="Personality state")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool calls made")


class VoiceResponse(BaseModel):
    """Voice response model."""
    model_config = ConfigDict(extra="forbid")
    
    response_id: str = Field(description="Unique response identifier")
    conversation_id: str = Field(description="Conversation identifier")
    message_id: str = Field(description="Original message identifier")
    
    # Content
    audio_data: Optional[bytes] = Field(default=None, description="Generated audio data")
    text_content: str = Field(description="Response text content")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    duration_ms: Optional[int] = Field(default=None, description="Audio duration in milliseconds")
    audio_format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")
    
    # Processing info
    processing_mode: VoiceProcessingMode = Field(description="Processing mode used")
    total_latency_ms: float = Field(description="Total processing latency")
    moshi_latency_ms: Optional[float] = Field(default=None, description="Moshi processing latency")
    llm_latency_ms: Optional[float] = Field(default=None, description="LLM processing latency")
    
    # Quality metrics
    audio_quality_score: Optional[float] = Field(default=None, description="Audio quality score")
    response_relevance: Optional[float] = Field(default=None, description="Response relevance score")
    
    # Integration results
    memory_updates: Optional[List[Dict[str, Any]]] = Field(default=None, description="Memory updates made")
    personality_changes: Optional[Dict[str, Any]] = Field(default=None, description="Personality changes")
    tool_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool execution results")


class VoiceStreamChunk(BaseModel):
    """Voice stream chunk for real-time processing."""
    model_config = ConfigDict(extra="forbid")
    
    chunk_id: str = Field(description="Unique chunk identifier")
    conversation_id: str = Field(description="Conversation identifier")
    sequence_number: int = Field(description="Chunk sequence number")
    
    # Content
    audio_data: Optional[bytes] = Field(default=None, description="Audio chunk data")
    text_delta: Optional[str] = Field(default=None, description="Text delta for this chunk")
    accumulated_text: Optional[str] = Field(default=None, description="Accumulated text so far")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Chunk timestamp")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    chunk_type: str = Field(default="audio", description="Type of chunk (audio, text, control)")
    
    # Processing info
    processing_latency_ms: Optional[float] = Field(default=None, description="Chunk processing latency")
    confidence_score: Optional[float] = Field(default=None, description="Chunk confidence score")


class ConversationState(BaseModel):
    """Voice conversation state."""
    model_config = ConfigDict(extra="forbid")
    
    conversation_id: str = Field(description="Conversation identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    
    # State
    is_active: bool = Field(default=False, description="Whether conversation is active")
    is_speaking: bool = Field(default=False, description="Whether user is speaking")
    is_listening: bool = Field(default=True, description="Whether system is listening")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now, description="Conversation start time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity time")
    total_duration_ms: int = Field(default=0, description="Total conversation duration")
    
    # Statistics
    message_count: int = Field(default=0, description="Number of messages")
    total_audio_duration_ms: int = Field(default=0, description="Total audio duration")
    average_latency_ms: float = Field(default=0.0, description="Average response latency")
    
    # Processing info
    current_mode: VoiceProcessingMode = Field(description="Current processing mode")
    moshi_active: bool = Field(default=False, description="Whether Moshi is active")
    llm_active: bool = Field(default=False, description="Whether external LLM is active")
    
    # Context
    memory_context: Optional[Dict[str, Any]] = Field(default=None, description="Current memory context")
    personality_state: Optional[Dict[str, Any]] = Field(default=None, description="Current personality state")


class VoiceEvent(BaseModel):
    """Voice processing event for WebSocket broadcasting."""
    model_config = ConfigDict(extra="forbid")
    
    event_id: str = Field(description="Unique event identifier")
    event_type: str = Field(description="Type of event")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier")
    
    # Event data
    data: Dict[str, Any] = Field(description="Event data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    
    # Metadata
    source: str = Field(description="Event source (moshi, llm, audio, etc.)")
    priority: str = Field(default="normal", description="Event priority")
    requires_response: bool = Field(default=False, description="Whether event requires response")


class VoiceAnalytics(BaseModel):
    """Voice processing analytics."""
    model_config = ConfigDict(extra="forbid")
    
    # Usage statistics
    total_conversations: int = Field(default=0, description="Total conversations")
    total_messages: int = Field(default=0, description="Total messages processed")
    total_audio_duration_ms: int = Field(default=0, description="Total audio duration")
    
    # Performance metrics
    average_latency_ms: float = Field(default=0.0, description="Average processing latency")
    moshi_latency_ms: float = Field(default=0.0, description="Average Moshi latency")
    llm_latency_ms: float = Field(default=0.0, description="Average LLM latency")
    
    # Quality metrics
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    audio_quality: float = Field(default=0.0, description="Average audio quality")
    
    # Resource usage
    vram_usage: Dict[str, str] = Field(default_factory=dict, description="VRAM usage by component")
    cpu_usage: float = Field(default=0.0, description="CPU usage percentage")
    
    # Error statistics
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    fallback_rate: float = Field(default=0.0, description="Fallback usage rate")
    
    # Integration statistics
    memory_integration_usage: float = Field(default=0.0, description="Memory integration usage rate")
    personality_integration_usage: float = Field(default=0.0, description="Personality integration usage rate")
    tools_integration_usage: float = Field(default=0.0, description="Tools integration usage rate")
