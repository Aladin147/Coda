# Coda API Documentation

Welcome to the comprehensive API documentation for Coda - Core Operations & Digital Assistant.

## Overview

Coda is a next-generation, local-first voice assistant built for real-time interaction and extensibility. This documentation covers all public APIs across the system's core components.

## Quick Navigation

### Core Systems
- [Voice Processing API](voice.md) - Real-time voice conversation processing
- [Memory System API](memory.md) - Intelligent memory management and retrieval
- [Personality Engine API](personality.md) - Adaptive personality system
- [Tools System API](tools.md) - Extensible function calling framework
- [LLM Manager API](llm.md) - Multi-provider language model integration
- [WebSocket API](websocket.md) - Real-time communication interface

### Integration APIs
- [Voice-Memory Integration](integrations/voice-memory.md)
- [Voice-Personality Integration](integrations/voice-personality.md)
- [Voice-Tools Integration](integrations/voice-tools.md)
- [Voice-LLM Integration](integrations/voice-llm.md)
- [Conversation Synchronization](integrations/conversation-sync.md)

### Configuration & Setup
- [Configuration Reference](configuration.md)
- [Installation Guide](installation.md)
- [Development Setup](development.md)

## Getting Started

### Basic Usage

```python
from coda import Coda

# Initialize Coda with default configuration
coda = Coda()
await coda.initialize()

# Start a voice conversation
conversation_id = await coda.start_voice_conversation()

# Process voice input
response = await coda.process_voice_input(
    conversation_id=conversation_id,
    audio_data=audio_bytes
)

print(f"Response: {response.text_content}")
```

### Advanced Usage

```python
from coda.components.voice import VoiceManager, VoiceConfig
from coda.components.memory import MemoryManager
from coda.components.personality import PersonalityManager

# Custom configuration
config = VoiceConfig(
    audio=AudioConfig(sample_rate=24000),
    moshi=MoshiConfig(device="cuda"),
    enable_memory_integration=True,
    enable_personality_integration=True
)

# Initialize components
voice_manager = VoiceManager(config)
await voice_manager.initialize()

# Process with full integration
response = await voice_manager.process_voice_message(
    voice_message,
    mode=VoiceProcessingMode.HYBRID
)
```

## System Architecture

Coda follows a modular, event-driven architecture with the following key principles:

- **Component Isolation**: Each system (voice, memory, personality, tools) is independently testable
- **Integration Layers**: Dedicated integration classes handle cross-component communication
- **Configuration-Driven**: All behavior is configurable through structured configuration objects
- **Async-First**: Built for high-performance async/await patterns
- **Type Safety**: Comprehensive type hints and validation throughout

## API Conventions

### Async/Await Pattern
All I/O operations use async/await:

```python
# Correct
response = await voice_manager.process_voice_message(message)

# Incorrect
response = voice_manager.process_voice_message(message)  # Missing await
```

### Error Handling
APIs use specific exception types for different error conditions:

```python
try:
    response = await voice_manager.process_voice_message(message)
except VoiceProcessingError as e:
    # Handle voice processing errors
    logger.error(f"Voice processing failed: {e}")
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Invalid input: {e}")
```

### Configuration Objects
All components use structured configuration objects:

```python
# Use configuration objects
config = VoiceConfig(
    audio=AudioConfig(sample_rate=24000),
    moshi=MoshiConfig(device="cuda")
)

# Not raw dictionaries
config = {"audio": {"sample_rate": 24000}}  # Avoid this
```

### Response Objects
APIs return structured response objects with metadata:

```python
response = await voice_manager.process_voice_message(message)

# Access response data
print(response.text_content)
print(response.audio_data)
print(response.processing_time)
print(response.confidence_score)
```

## Performance Considerations

### VRAM Management
Voice processing requires careful VRAM management:

```python
# Monitor VRAM usage
vram_stats = await voice_manager.get_vram_stats()
print(f"VRAM usage: {vram_stats.used_gb:.1f}GB / {vram_stats.total_gb:.1f}GB")

# Cleanup when done
await voice_manager.cleanup()
```

### Streaming Responses
Use streaming for real-time applications:

```python
async for chunk in voice_manager.process_streaming(message):
    if chunk.audio_data:
        await play_audio(chunk.audio_data)
    if chunk.text_content:
        print(chunk.text_content)
```

### Conversation Management
Properly manage conversation lifecycle:

```python
# Start conversation
conversation_id = await voice_manager.start_conversation()

try:
    # Process messages
    response = await voice_manager.process_voice_input(
        conversation_id, audio_data
    )
finally:
    # Always cleanup
    await voice_manager.end_conversation(conversation_id)
```

## Production APIs

### System Management API
- [System Launcher API](system_launcher.md) - Unified system startup and management
- [Error Handling API](error_handling.md) - Comprehensive error management and recovery
- [Performance Optimization API](performance.md) - System performance monitoring and optimization

### Integration APIs
- [Component Integration API](component_integration.md) - Cross-component communication
- [Event Coordination API](event_coordination.md) - System-wide event management
- [Session Management API](session_management.md) - Conversation session handling

## REST API Endpoints

### System Status
```http
GET /api/status
GET /api/health
GET /api/metrics
GET /api/components
```

### Session Management
```http
POST /api/sessions
GET /api/sessions/{session_id}
DELETE /api/sessions/{session_id}
PUT /api/sessions/{session_id}
```

### Voice Processing
```http
POST /api/voice/process
GET /api/voice/status
POST /api/voice/settings
```

### Memory Operations
```http
POST /api/memory/store
GET /api/memory/search
GET /api/memory/stats
DELETE /api/memory/{memory_id}
```

### Tool Execution
```http
POST /api/tools/execute
GET /api/tools/list
GET /api/tools/{tool_name}
```

## WebSocket Events

### System Events
- `system_startup` - System initialization complete
- `system_shutdown` - System shutdown initiated
- `system_error` - System-level errors
- `component_status_change` - Component status updates
- `performance_metrics` - Real-time performance data

### Conversation Events
- `session_created` - New conversation session
- `session_updated` - Session state changes
- `message_received` - New message in session
- `message_processed` - Message processing complete

### Voice Events
- `voice_input_start` - Voice input detected
- `voice_input_end` - Voice input complete
- `voice_processing` - Voice processing status
- `voice_output_ready` - Voice response ready

### Error Events
- `error_occurred` - Error with recovery suggestions
- `error_resolved` - Error successfully resolved
- `recovery_initiated` - Automatic recovery started

## Examples

See the [examples directory](../examples/) for complete working examples:

- [Basic Voice Chat](../examples/basic_voice_chat.py)
- [Memory Integration](../examples/memory_integration.py)
- [Personality Adaptation](../examples/personality_adaptation.py)
- [Tool Integration](../examples/tool_integration.py)
- [WebSocket Server](../examples/websocket_server.py)
- [System Launcher](../examples/system_launcher.py)
- [Performance Monitoring](../examples/performance_monitoring.py)
- [Error Handling](../examples/error_handling.py)

## Authentication

### API Key Authentication
```python
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}
```

### WebSocket Authentication
```javascript
const ws = new WebSocket('ws://localhost:8765', [], {
    headers: {
        'Authorization': 'Bearer YOUR_API_KEY'
    }
});
```

## Rate Limiting

- **REST API**: 100 requests per minute per IP
- **WebSocket**: 1000 messages per minute per connection
- **Voice Processing**: 10 concurrent streams per session

## Error Handling

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Response Format
```json
{
    "error": {
        "code": "COMPONENT_ERROR",
        "message": "Component initialization failed",
        "details": {
            "component": "voice",
            "reason": "GPU not available"
        },
        "suggestions": [
            "Check GPU drivers",
            "Use CPU mode instead"
        ],
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Support

- [GitHub Issues](https://github.com/your-repo/coda/issues)
- [Discussions](https://github.com/your-repo/coda/discussions)
- [Contributing Guide](../development/contributing.md)
- [Installation Guide](../INSTALLATION_GUIDE.md)
- [Troubleshooting](../TROUBLESHOOTING.md)

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.
