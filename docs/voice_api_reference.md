# Voice Processing API Reference

## Core Classes

### VoiceProcessor

Main voice processing orchestrator that coordinates all voice operations.

```python
class VoiceProcessor:
    def __init__(self, config: VoiceProcessingConfig)
    
    async def start() -> None
    async def stop() -> None
    
    async def process_voice_message(
        self, 
        voice_message: VoiceMessage,
        mode: Optional[VoiceProcessingMode] = None
    ) -> VoiceResponse
    
    async def process_streaming(
        self,
        voice_message: VoiceMessage
    ) -> AsyncGenerator[VoiceResponse, None]
    
    async def process_with_integrations(
        self,
        voice_message: VoiceMessage,
        include_memory: bool = True,
        include_personality: bool = True,
        include_tools: bool = True
    ) -> VoiceResponse
```

### VoiceMessage

Represents a user voice input message.

```python
@dataclass
class VoiceMessage:
    message_id: str
    conversation_id: str
    audio_data: bytes
    text_content: Optional[str] = None
    processing_mode: VoiceProcessingMode = VoiceProcessingMode.ADAPTIVE
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Audio metadata
    sample_rate: int = 24000
    channels: int = 1
    audio_duration_ms: Optional[float] = None
    
    # Processing metadata
    confidence_score: Optional[float] = None
    language: Optional[str] = None
```

### VoiceResponse

Represents an assistant voice response.

```python
@dataclass
class VoiceResponse:
    response_id: str
    conversation_id: str
    message_id: str
    text_content: Optional[str] = None
    audio_data: bytes = b""
    processing_mode: VoiceProcessingMode
    
    # Performance metrics
    total_latency_ms: float = 0.0
    moshi_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    
    # Quality metrics
    response_relevance: Optional[float] = None
```

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
