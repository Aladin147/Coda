# Voice Processing API Reference

## Overview

The Coda Voice Processing System provides a comprehensive API for real-time voice conversation processing, combining Kyutai Moshi for fast speech-to-speech processing with external LLMs for enhanced reasoning capabilities.

## Core Classes

### VoiceManager

Main voice manager that orchestrates all voice processing components and integrations.

```python
class VoiceManager(VoiceManagerInterface):
    """
    Main voice manager for orchestrating voice processing.

    The VoiceManager coordinates between different processing modes (Moshi-only,
    LLM-enhanced, hybrid) and manages integrations with memory, personality,
    and tools systems.

    Args:
        config: Voice configuration object. If None, loads development config.

    Example:
        >>> from src.coda.components.voice import VoiceManager
        >>> manager = VoiceManager()
        >>> await manager.initialize()
        >>> response = await manager.process_voice_message(voice_message)
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        """Initialize voice manager with configuration."""

    async def initialize(self) -> None:
        """
        Initialize all voice processing components.

        This method must be called before processing any voice messages.
        It initializes VRAM management, audio processing, and integration components.

        Raises:
            VoiceInitializationError: If initialization fails
            VRAMAllocationError: If VRAM allocation fails
        """

    async def cleanup(self) -> None:
        """
        Clean up all resources and stop processing.

        This method should be called when shutting down to ensure
        proper cleanup of VRAM, audio resources, and background threads.
        """

    async def process_voice_message(
        self,
        voice_message: VoiceMessage,
        mode: Optional[VoiceProcessingMode] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> VoiceResponse:
        """
        Process a voice message and return a response.

        Args:
            voice_message: The voice message to process
            mode: Processing mode (adaptive, moshi_only, llm_enhanced, hybrid)
            context: Additional context for processing

        Returns:
            VoiceResponse containing the processed response

        Raises:
            VoiceProcessingError: If processing fails
            ValidationError: If voice message is invalid
            TimeoutError: If processing exceeds timeout

        Example:
            >>> voice_msg = VoiceMessage(
            ...     conversation_id="conv_123",
            ...     audio_data=audio_bytes,
            ...     text_content="Hello, how are you?"
            ... )
            >>> response = await manager.process_voice_message(voice_msg)
            >>> print(response.text_content)
        """

    async def process_streaming(
        self,
        voice_message: VoiceMessage,
        mode: Optional[VoiceProcessingMode] = None
    ) -> AsyncGenerator[VoiceStreamChunk, None]:
        """
        Process voice message with streaming response.

        Args:
            voice_message: The voice message to process
            mode: Processing mode for streaming

        Yields:
            VoiceStreamChunk: Incremental response chunks

        Example:
            >>> async for chunk in manager.process_streaming(voice_msg):
            ...     print(chunk.text_content)
            ...     if chunk.audio_data:
            ...         play_audio(chunk.audio_data)
        """

    async def start_conversation(self, conversation_id: str) -> ConversationState:
        """
        Start a new voice conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            ConversationState: Initial conversation state

        Raises:
            ValidationError: If conversation_id is invalid
            ConversationError: If conversation already exists
        """

    async def end_conversation(self, conversation_id: str) -> None:
        """
        End an active voice conversation.

        Args:
            conversation_id: Conversation to end

        Raises:
            ConversationError: If conversation not found
        """

### VoiceMessage

Represents a user voice input message with audio data and metadata.

```python
@dataclass
class VoiceMessage:
    """
    Voice message containing audio data and processing metadata.

    This class encapsulates all information needed to process a voice input,
    including the audio data, conversation context, and processing preferences.

    Attributes:
        conversation_id: Unique identifier for the conversation
        audio_data: Raw audio data in bytes (WAV format preferred)
        text_content: Optional transcribed text content
        timestamp: When the message was created
        speaker: Speaker identifier (default: "user")
        processing_mode: Preferred processing mode
        metadata: Additional metadata for processing

    Example:
        >>> voice_msg = VoiceMessage(
        ...     conversation_id="conv_123",
        ...     audio_data=wav_audio_bytes,
        ...     text_content="What's the weather like?",
        ...     speaker="user"
        ... )
    """

    conversation_id: str
    audio_data: bytes
    text_content: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    speaker: str = "user"
    processing_mode: VoiceProcessingMode = VoiceProcessingMode.ADAPTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Audio metadata (auto-populated during validation)
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_seconds: Optional[float] = None
    format: Optional[str] = None

    def validate(self) -> None:
        """
        Validate the voice message data.

        Raises:
            ValidationError: If message data is invalid
        """

    def get_audio_metadata(self) -> Dict[str, Any]:
        """
        Get audio metadata from the audio data.

        Returns:
            Dictionary containing audio format information
        """
```

### VoiceResponse

Represents a processed voice response with audio and text content.

```python
@dataclass
class VoiceResponse:
    """
    Voice response containing processed audio and text.

    This class encapsulates the result of voice processing, including
    generated audio, text content, and processing metadata.

    Attributes:
        conversation_id: Conversation this response belongs to
        text_content: Generated text response
        audio_data: Generated audio response (optional)
        timestamp: When the response was generated
        processing_time: Time taken to process (milliseconds)
        confidence_score: Confidence in the response quality (0.0-1.0)
        processing_mode: Mode used for processing
        metadata: Additional response metadata

    Example:
        >>> response = VoiceResponse(
        ...     conversation_id="conv_123",
        ...     text_content="The weather is sunny today.",
        ...     audio_data=generated_audio_bytes,
        ...     confidence_score=0.95
        ... )
    """

    conversation_id: str
    text_content: str
    audio_data: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None
    processing_mode: Optional[VoiceProcessingMode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Response quality metrics
    relevance_score: Optional[float] = None
    coherence_score: Optional[float] = None
    audio_quality_score: Optional[float] = None

    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """
        Check if response meets quality threshold.

        Args:
            threshold: Quality threshold (0.0-1.0)

        Returns:
            True if response quality is above threshold
        """

    def get_quality_metrics(self) -> Dict[str, float]:
        """
        Get all quality metrics for the response.

        Returns:
            Dictionary of quality metrics
        """

## Processing Modes

### VoiceProcessingMode

Enumeration of available voice processing modes.

```python
class VoiceProcessingMode(str, Enum):
    """
    Voice processing modes for different use cases.

    Each mode provides different trade-offs between speed, quality, and capabilities:

    - MOSHI_ONLY: Fastest, uses only Moshi for speech-to-speech
    - LLM_ENHANCED: Highest quality, uses external LLM for reasoning
    - HYBRID: Balanced, uses both Moshi and LLM in parallel
    - ADAPTIVE: Intelligent, automatically selects best mode based on context
    """

    MOSHI_ONLY = "moshi_only"
    """
    Use only Moshi for processing.

    Characteristics:
    - Fastest response time (100-200ms)
    - Natural speech patterns
    - Limited reasoning capabilities
    - Best for: Simple conversations, quick responses
    """

    LLM_ENHANCED = "llm_enhanced"
    """
    Use external LLM for enhanced reasoning.

    Characteristics:
    - Slower response time (500-1500ms)
    - Advanced reasoning and knowledge
    - Text-to-speech conversion required
    - Best for: Complex questions, analysis, creative tasks
    """

    HYBRID = "hybrid"
    """
    Use both Moshi and LLM in parallel, select best response.

    Characteristics:
    - Moderate response time (200-800ms)
    - Quality-based response selection
    - Fallback capabilities
    - Best for: Balanced conversations requiring both speed and quality
    """

    ADAPTIVE = "adaptive"
    """
    Automatically select optimal mode based on context.

    Characteristics:
    - Variable response time based on selected mode
    - Context-aware mode selection
    - Learning from conversation patterns
    - Best for: General conversations, production use
    """
```

## Configuration Classes

### VoiceConfig

Main configuration class for the voice processing system.

```python
@dataclass
class VoiceConfig:
    """
    Comprehensive configuration for voice processing system.

    This class contains all configuration options for audio processing,
    model settings, performance tuning, and integration settings.

    Example:
        >>> config = VoiceConfig(
        ...     audio=AudioConfig(sample_rate=24000),
        ...     moshi=MoshiConfig(device="cuda"),
        ...     llm=LLMConfig(provider="ollama")
        ... )
    """

    # Core configuration
    audio: AudioConfig
    moshi: MoshiConfig
    llm: Optional[LLMConfig] = None

    # Processing settings
    default_mode: VoiceProcessingMode = VoiceProcessingMode.ADAPTIVE
    enable_streaming: bool = True
    max_conversation_length: int = 100

    # Performance settings
    enable_caching: bool = True
    enable_optimization: bool = True
    max_concurrent_conversations: int = 10

    # Integration settings
    enable_memory_integration: bool = True
    enable_personality_integration: bool = True
    enable_tools_integration: bool = True

    # Monitoring and logging
    enable_analytics: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = True
```

### AudioConfig

Configuration for audio processing settings.

```python
@dataclass
class AudioConfig:
    """
    Audio processing configuration.

    Defines audio format, quality settings, and processing options.

    Attributes:
        sample_rate: Audio sample rate in Hz (16000, 24000, 44100, 48000)
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Audio format ("wav", "mp3", "flac")
        bit_depth: Audio bit depth (16, 24, 32)
        enable_vad: Enable Voice Activity Detection
        enable_noise_reduction: Enable noise reduction
        enable_echo_cancellation: Enable echo cancellation

    Example:
        >>> audio_config = AudioConfig(
        ...     sample_rate=24000,
        ...     channels=1,
        ...     format="wav",
        ...     enable_vad=True
        ... )
    """

    sample_rate: int = 24000
    channels: int = 1
    format: str = "wav"
    bit_depth: int = 16

    # Audio processing features
    enable_vad: bool = True
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    enable_auto_gain_control: bool = True

    # Quality settings
    vad_threshold: float = 0.5
    noise_reduction_level: float = 0.7
    max_audio_length_seconds: float = 30.0

    def validate(self) -> None:
        """
        Validate audio configuration.

        Raises:
            ConfigurationError: If configuration is invalid
        """

## Integration Classes

### VoiceMemoryIntegration

Integrates voice processing with memory system.

```python
class VoiceMemoryIntegration:
    async def enhance_voice_context(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]
    
    async def learn_from_voice_interaction(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None
    ) -> None
    
    def get_integration_stats() -> Dict[str, Any]
```

### VoicePersonalityIntegration

Integrates voice processing with personality system.

```python
class VoicePersonalityIntegration:
    async def enhance_voice_context(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]
    
    async def adapt_voice_response(
        self,
        voice_response: VoiceResponse,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse
    
    async def learn_from_voice_interaction(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None,
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> None
```

### VoiceToolsIntegration

Integrates voice processing with tools system.

```python
class VoiceToolsIntegration:
    async def enhance_voice_context(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None
    ) -> Dict[str, Any]
    
    async def process_function_calls(
        self,
        voice_message: VoiceMessage,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None
    ) -> Tuple[VoiceResponse, List[ToolResult]]
    
    async def get_tool_suggestions(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None
    ) -> List[Dict[str, Any]]
```

### VoiceLLMManagerIntegration

Integrates voice processing with LLM manager.

```python
class VoiceLLMManagerIntegration:
    async def generate_voice_response(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None,
        enhanced_context: Optional[Dict[str, Any]] = None
    ) -> VoiceResponse
    
    async def generate_streaming_voice_response(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None,
        enhanced_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[VoiceResponse, None]
    
    async def get_conversation_history(
        self,
        voice_conversation_id: str,
        limit: int = 10
    ) -> List[LLMMessage]
```

### VoiceConversationSync

Synchronizes voice conversations with conversation manager.

```python
class VoiceConversationSync:
    async def sync_voice_message(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None
    ) -> bool
    
    async def sync_voice_response(
        self,
        voice_response: VoiceResponse,
        conversation_state: Optional[ConversationState] = None
    ) -> bool
    
    async def sync_conversation_state(
        self,
        conversation_state: ConversationState
    ) -> bool
    
    async def handle_conversation_update(
        self,
        conversation_id: str,
        update_data: Dict[str, Any]
    ) -> bool
```

## Configuration Classes

### VoiceProcessingConfig

Main configuration for voice processing system.

```python
@dataclass
class VoiceProcessingConfig:
    # Processing modes
    default_mode: VoiceProcessingMode = VoiceProcessingMode.ADAPTIVE
    enable_moshi: bool = True
    enable_external_llm: bool = True
    enable_hybrid_processing: bool = True
    
    # Performance settings
    max_latency_ms: float = 1000.0
    enable_parallel_processing: bool = True
    vram_limit_gb: float = 32.0
    
    # Audio settings
    sample_rate: int = 24000
    channels: int = 1
    chunk_size: int = 1024
    enable_streaming: bool = True
    
    # Integration settings
    enable_memory_integration: bool = True
    enable_personality_integration: bool = True
    enable_tools_integration: bool = True
    enable_conversation_sync: bool = True
```

### Integration Configs

```python
@dataclass
class VoiceMemoryConfig:
    enable_context_injection: bool = True
    max_relevant_memories: int = 10
    memory_relevance_threshold: float = 0.7
    enable_learning: bool = True
    learning_rate: float = 0.1

@dataclass
class VoicePersonalityConfig:
    enable_personality_injection: bool = True
    enable_response_adaptation: bool = True
    enable_personality_learning: bool = True
    voice_confidence_factor: float = 1.2
    voice_engagement_factor: float = 1.1
    adjustment_sensitivity: float = 0.1

@dataclass
class VoiceToolsConfig:
    enable_tool_injection: bool = True
    enable_function_calling: bool = True
    auto_execute_tools: bool = True
    max_tools_in_context: int = 10
    tool_relevance_threshold: float = 0.6

@dataclass
class VoiceLLMManagerConfig:
    llm_provider: str = "ollama"
    llm_model: str = "gemma3:1b"
    voice_response_max_tokens: int = 300
    voice_temperature: float = 0.8
    enable_streaming_for_voice: bool = True
    voice_timeout_seconds: float = 8.0

@dataclass
class VoiceConversationSyncConfig:
    sync_enabled: bool = True
    real_time_sync: bool = True
    bidirectional_sync: bool = True
    conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.VOICE_PRIORITY
```

## Enums

### VoiceProcessingMode

```python
class VoiceProcessingMode(str, Enum):
    MOSHI_ONLY = "moshi_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
```

### ConflictResolutionStrategy

```python
class ConflictResolutionStrategy(str, Enum):
    VOICE_PRIORITY = "voice_priority"
    CONVERSATION_PRIORITY = "conversation_priority"
    TIMESTAMP_PRIORITY = "timestamp_priority"
    MERGE = "merge"
```

### SyncEventType

```python
class SyncEventType(str, Enum):
    VOICE_MESSAGE_ADDED = "voice_message_added"
    VOICE_RESPONSE_ADDED = "voice_response_added"
    CONVERSATION_STATE_UPDATED = "conversation_state_updated"
    SYNC_CONFLICT_DETECTED = "sync_conflict_detected"
    SYNC_COMPLETED = "sync_completed"
```

## Usage Examples

### Basic Voice Processing

```python
# Initialize voice processor
config = VoiceProcessingConfig(
    default_mode=VoiceProcessingMode.HYBRID,
    enable_parallel_processing=True
)
voice_processor = VoiceProcessor(config)
await voice_processor.start()

# Process voice message
voice_message = VoiceMessage(
    message_id="msg_001",
    conversation_id="conv_001",
    audio_data=audio_bytes,
    text_content="What's the weather like?"
)

response = await voice_processor.process_voice_message(voice_message)
print(f"Response: {response.text_content}")
```

### Streaming Processing

```python
# Stream voice responses
async for chunk in voice_processor.process_streaming(voice_message):
    # Handle streaming response
    if chunk.audio_data:
        await play_audio(chunk.audio_data)
    if chunk.text_content:
        print(f"Partial response: {chunk.text_content}")
```

### Integration Usage

```python
# Enhanced processing with all integrations
enhanced_response = await voice_processor.process_with_integrations(
    voice_message,
    include_memory=True,
    include_personality=True,
    include_tools=True
)

# Manual integration usage
memory_integration = VoiceMemoryIntegration(memory_manager, memory_config)
personality_integration = VoicePersonalityIntegration(personality_manager, personality_config)

# Enhance context
memory_context = await memory_integration.enhance_voice_context(voice_message)
personality_context = await personality_integration.enhance_voice_context(voice_message)

# Adapt response
adapted_response = await personality_integration.adapt_voice_response(
    response, voice_message
)
```

### Event Handling

```python
# Add sync event handlers
conversation_sync = VoiceConversationSync(conversation_manager, sync_config)

def handle_voice_message_added(event_type, data):
    print(f"Voice message added: {data['voice_message_id']}")

conversation_sync.add_event_handler(
    SyncEventType.VOICE_MESSAGE_ADDED,
    handle_voice_message_added
)
```

## Error Handling

### Common Exceptions

```python
class VoiceProcessingError(Exception):
    """Base exception for voice processing errors."""

class AudioProcessingError(VoiceProcessingError):
    """Audio processing related errors."""

class MoshiConnectionError(VoiceProcessingError):
    """Moshi connection related errors."""

class LLMIntegrationError(VoiceProcessingError):
    """LLM integration related errors."""

class SyncError(VoiceProcessingError):
    """Conversation synchronization errors."""
```

### Error Handling Example

```python
try:
    response = await voice_processor.process_voice_message(voice_message)
except AudioProcessingError as e:
    logger.error(f"Audio processing failed: {e}")
    # Handle audio error
except MoshiConnectionError as e:
    logger.error(f"Moshi connection failed: {e}")
    # Fallback to LLM-only mode
except LLMIntegrationError as e:
    logger.error(f"LLM integration failed: {e}")
    # Fallback to Moshi-only mode
except VoiceProcessingError as e:
    logger.error(f"Voice processing failed: {e}")
    # General error handling
```
