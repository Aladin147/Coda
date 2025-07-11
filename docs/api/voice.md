# Voice Processing API

The Voice Processing API provides real-time voice conversation capabilities using Kyutai Moshi for fast speech-to-speech processing combined with external LLMs for enhanced reasoning.

## Quick Start

```python
from coda.components.voice import VoiceManager, VoiceConfig, VoiceMessage

# Initialize voice manager
config = VoiceConfig()
voice_manager = VoiceManager(config)
await voice_manager.initialize()

# Process voice input
voice_message = VoiceMessage(
    conversation_id="conv_123",
    audio_data=audio_bytes,
    text_content="Hello, how are you?"
)

response = await voice_manager.process_voice_input(
    conversation_id="conv_123",
    audio_data=audio_bytes
)

print(f"Response: {response.text_content}")
```

## Core Classes

### VoiceManager

Main orchestrator for voice processing operations.

```python
class VoiceManager:
    def __init__(self, config: Optional[VoiceConfig] = None) -> None
    
    async def initialize(self) -> None
    async def cleanup(self) -> None
    
    async def start_conversation(
        self, 
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str
    
    async def end_conversation(self, conversation_id: str) -> None
    
    async def process_voice_input(
        self,
        conversation_id: str,
        audio_data: bytes
    ) -> VoiceResponse
    
    async def process_voice_stream(
        self,
        conversation_id: str,
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[VoiceStreamChunk]
    
    async def get_conversation_state(
        self, 
        conversation_id: str
    ) -> Optional[ConversationState]
```

#### Methods

**`initialize()`**
- Initializes all voice processing components
- Must be called before processing any voice messages
- Raises: `VoiceInitializationError`, `VRAMAllocationError`

**`start_conversation(user_id, conversation_id)`**
- Starts a new voice conversation
- Returns: Conversation ID (generated if not provided)
- Raises: `ComponentNotInitializedError`, `ValidationError`

**`process_voice_input(conversation_id, audio_data)`**
- Processes voice input and returns response
- Supports multiple processing modes (adaptive, moshi-only, hybrid)
- Returns: `VoiceResponse` with text and audio content
- Raises: `VoiceProcessingError`, `ValidationError`

### VoiceMessage

Represents user voice input with audio data and metadata.

```python
@dataclass
class VoiceMessage:
    message_id: str
    conversation_id: str
    audio_data: Optional[bytes] = None
    text_content: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    speaker: str = "user"
    processing_mode: VoiceProcessingMode = VoiceProcessingMode.ADAPTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Audio metadata (auto-populated)
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_seconds: Optional[float] = None
    format: Optional[str] = None
```

### VoiceResponse

Represents processed voice response with audio and text content.

```python
@dataclass
class VoiceResponse:
    response_id: str
    conversation_id: str
    message_id: str
    text_content: str
    audio_data: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)
    processing_mode: VoiceProcessingMode
    total_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    confidence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    audio_quality_score: Optional[float] = None
```

## Processing Modes

### VoiceProcessingMode

```python
class VoiceProcessingMode(str, Enum):
    MOSHI_ONLY = "moshi_only"      # Fastest, Moshi speech-to-speech only
    LLM_ENHANCED = "llm_enhanced"  # Highest quality, external LLM reasoning
    HYBRID = "hybrid"              # Balanced, both Moshi and LLM in parallel
    ADAPTIVE = "adaptive"          # Intelligent mode selection based on context
```

**Mode Characteristics:**

- **MOSHI_ONLY**: 100-200ms latency, natural speech, limited reasoning
- **LLM_ENHANCED**: 500-1500ms latency, advanced reasoning, TTS conversion
- **HYBRID**: 200-800ms latency, quality-based selection, fallback support
- **ADAPTIVE**: Variable latency, context-aware mode selection, learning

## Configuration

### VoiceConfig

```python
@dataclass
class VoiceConfig:
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
    
    # Monitoring
    enable_analytics: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = True
```

### AudioConfig

```python
@dataclass
class AudioConfig:
    sample_rate: int = 24000
    channels: int = 1
    format: str = "wav"
    bit_depth: int = 16
    
    # Processing features
    enable_vad: bool = True
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    enable_auto_gain_control: bool = True
    
    # Quality settings
    vad_threshold: float = 0.5
    noise_reduction_level: float = 0.7
    max_audio_length_seconds: float = 30.0
```

## Streaming Processing

For real-time applications, use streaming processing:

```python
async def process_voice_stream(voice_manager, conversation_id, audio_stream):
    async for chunk in voice_manager.process_voice_stream(
        conversation_id, audio_stream
    ):
        # Handle streaming response
        if chunk.audio_data:
            await play_audio(chunk.audio_data)
        
        if chunk.text_content:
            print(f"Partial: {chunk.text_content}")
        
        if chunk.is_final:
            print(f"Final: {chunk.text_content}")
            break
```

## Error Handling

### Exception Hierarchy

```python
class VoiceProcessingError(Exception):
    """Base exception for voice processing errors."""

class VoiceInitializationError(VoiceProcessingError):
    """Voice system initialization errors."""

class AudioProcessingError(VoiceProcessingError):
    """Audio processing related errors."""

class MoshiConnectionError(VoiceProcessingError):
    """Moshi connection related errors."""

class VRAMAllocationError(VoiceProcessingError):
    """VRAM allocation and management errors."""

class ValidationError(VoiceProcessingError):
    """Input validation errors."""

class ComponentNotInitializedError(VoiceProcessingError):
    """Component not properly initialized."""
```

### Error Handling Example

```python
try:
    response = await voice_manager.process_voice_input(
        conversation_id, audio_data
    )
except ValidationError as e:
    logger.error(f"Invalid input: {e}")
    # Handle validation error
except AudioProcessingError as e:
    logger.error(f"Audio processing failed: {e}")
    # Handle audio error
except MoshiConnectionError as e:
    logger.error(f"Moshi connection failed: {e}")
    # Fallback to LLM-only mode
except VRAMAllocationError as e:
    logger.error(f"VRAM allocation failed: {e}")
    # Reduce model size or cleanup
except VoiceProcessingError as e:
    logger.error(f"Voice processing failed: {e}")
    # General error handling
```

## Performance Monitoring

### VRAM Management

```python
# Check VRAM usage
vram_stats = await voice_manager.get_vram_stats()
print(f"VRAM: {vram_stats.used_gb:.1f}GB / {vram_stats.total_gb:.1f}GB")

# Monitor performance
performance_stats = await voice_manager.get_performance_stats()
print(f"Average latency: {performance_stats.avg_latency_ms:.1f}ms")
print(f"Success rate: {performance_stats.success_rate:.2%}")
```

### Analytics

```python
# Get conversation analytics
analytics = await voice_manager.get_analytics()
print(f"Total conversations: {analytics.total_conversations}")
print(f"Average conversation length: {analytics.avg_conversation_length}")
print(f"Mode distribution: {analytics.mode_distribution}")
```

## Integration with Other Systems

The Voice Processing API integrates seamlessly with other Coda components:

- **Memory Integration**: Automatic context enhancement and learning
- **Personality Integration**: Adaptive response styling and personality learning
- **Tools Integration**: Function calling and tool suggestions
- **LLM Integration**: Enhanced reasoning and conversation management
- **WebSocket Integration**: Real-time streaming and event broadcasting

See [Integration APIs](integrations/) for detailed integration documentation.
