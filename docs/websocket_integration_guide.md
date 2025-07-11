# WebSocket Integration Guide

## Overview

The Coda Voice WebSocket Integration provides real-time, bidirectional voice communication capabilities through WebSocket connections. This system enables low-latency voice processing, streaming audio, and real-time event broadcasting for interactive voice applications.

## Architecture

### Core Components

1. **VoiceWebSocketHandler** - Main WebSocket connection handler
2. **VoiceEventBroadcaster** - Real-time event broadcasting system
3. **AudioStreamProcessor** - Bidirectional audio streaming
4. **WebSocketMonitor** - Performance monitoring and analytics
5. **VoiceWebSocketServer** - Complete server integration

### System Flow

```
Client (Browser/App) ←→ WebSocket ←→ Voice Processing ←→ Response
                                ↓
                         Event Broadcasting
                                ↓
                         Real-time Monitoring
```

## Quick Start

### Server Setup

```python
from src.coda.components.voice.websocket_server import VoiceWebSocketServer
from src.coda.components.voice.models import VoiceConfig, AudioConfig

# Create server configuration
voice_config = VoiceConfig(
    audio=AudioConfig(
        sample_rate=16000,
        channels=1,
        format="wav"
    )
)

# Initialize and start server
server = VoiceWebSocketServer(
    voice_config=voice_config,
    host="localhost",
    port=8765,
    max_connections=100,
    auth_required=False
)

await server.start()
```

### Client Setup (JavaScript)

```javascript
// Initialize WebSocket client
const client = new VoiceWebSocketClient({
    serverUrl: 'ws://localhost:8765',
    sampleRate: 16000,
    enableVAD: true
});

// Connect to server
await client.connect();

// Start conversation
const conversationId = await client.startConversation();

// Set up event handlers
client.on('voiceResponse', (response) => {
    console.log('Received response:', response.text_content);
    // Handle audio playback if audio_data is present
});

// Start voice recording
await client.startVoiceRecording('adaptive');
```

## WebSocket Protocol

### Message Format

All WebSocket messages follow this JSON structure:

```json
{
    "type": "message_type",
    "data": { /* message-specific data */ },
    "message_id": "unique_id",
    "timestamp": 1234567890.123,
    "conversation_id": "optional_conversation_id"
}
```

### Message Types

#### Connection Management
- `connect` - Connection establishment
- `disconnect` - Connection termination
- `ping` / `pong` - Connection health check
- `auth` / `auth_success` / `auth_failed` - Authentication

#### Voice Processing
- `voice_start` - Begin voice processing session
- `voice_chunk` - Audio data chunk
- `voice_end` - End voice processing session
- `voice_response` - Complete voice response
- `voice_stream_chunk` - Streaming response chunk

#### Conversation Management
- `conversation_start` - Start new conversation
- `conversation_end` - End conversation
- `conversation_state` - Conversation status update

#### Status and Events
- `status` - General status updates
- `error` - Error notifications
- `metrics` - Performance metrics

## Audio Streaming

### Supported Formats

- **WAV** - Recommended for best quality
- **PCM** - Raw audio data
- **OPUS** - Compressed audio (future)
- **MP3** - Compressed audio (future)

### Streaming Configuration

```python
from src.coda.components.voice.websocket_audio_streaming import AudioStreamConfig

config = AudioStreamConfig(
    sample_rate=16000,      # Audio sample rate
    channels=1,             # Mono audio
    format=AudioFormat.WAV, # Audio format
    chunk_size_ms=100,      # 100ms chunks
    enable_vad=True,        # Voice Activity Detection
    vad_threshold=0.5,      # VAD sensitivity
    streaming_mode=StreamingMode.VOICE_ACTIVITY
)
```

### Streaming Modes

1. **PUSH_TO_TALK** - Manual recording control
2. **CONTINUOUS** - Continuous recording
3. **VOICE_ACTIVITY** - Automatic based on voice detection

## Event Broadcasting

### Event Types

```python
from src.coda.components.voice.websocket_events import EventType

# Processing events
EventType.PROCESSING_STARTED
EventType.PROCESSING_PROGRESS
EventType.PROCESSING_COMPLETED
EventType.PROCESSING_FAILED

# System events
EventType.SYSTEM_STATUS
EventType.PERFORMANCE_ALERT
EventType.ERROR_OCCURRED
```

### Broadcasting Events

```python
# Broadcast to all clients in conversation
await event_broadcaster.broadcast_event(
    EventType.PROCESSING_STARTED,
    {"mode": "hybrid", "estimated_time": 2.5},
    conversation_id="conv_123"
)

# Broadcast to specific user
await event_broadcaster.broadcast_event(
    EventType.SYSTEM_STATUS,
    {"message": "System maintenance in 5 minutes"},
    user_id="user_456"
)
```

### Client Event Handling

```javascript
// Subscribe to specific events
client.on('status', (data) => {
    if (data.event_type === 'processing_started') {
        showProcessingIndicator();
    }
});

// Handle errors
client.on('error', (error) => {
    console.error('Voice processing error:', error);
    showErrorMessage(error.error_message);
});
```

## Real-time Monitoring

### Performance Metrics

The monitoring system tracks:

- **Connection Metrics** - Active connections, message rates
- **Latency Metrics** - Processing times, network latency
- **Error Metrics** - Error rates, failure types
- **Resource Metrics** - CPU, memory, VRAM usage
- **Audio Metrics** - Stream quality, processing efficiency

### Monitoring Dashboard

```python
# Get real-time metrics
metrics = monitor.get_real_time_metrics()

# Get historical data
history = monitor.get_historical_metrics(minutes=60)

# Get connection details
connection_info = monitor.get_connection_details("client_123")
```

### Alert Configuration

```python
# Set custom alert thresholds
monitor.set_alert_threshold("max_latency_ms", 1000.0)
monitor.set_alert_threshold("max_error_rate", 0.05)
monitor.set_alert_threshold("max_cpu_usage", 80.0)
```

## Authentication and Security

### Basic Authentication

```python
# Server with authentication required
server = VoiceWebSocketServer(auth_required=True)

# Client authentication
client = new VoiceWebSocketClient({
    authToken: 'your_auth_token',
    userId: 'user_123'
});
```

### Custom Authentication

```python
class CustomWebSocketHandler(VoiceWebSocketHandler):
    async def _validate_auth(self, user_id: str, token: str) -> bool:
        # Implement your authentication logic
        return await your_auth_service.validate(user_id, token)
```

## Error Handling

### Server-side Error Handling

```python
try:
    response = await voice_manager.process_voice_input(
        conversation_id, audio_data
    )
except VoiceProcessingError as e:
    await websocket_handler._send_error(
        connection, f"Processing failed: {e}"
    )
except TimeoutError as e:
    await websocket_handler._send_error(
        connection, "Processing timeout"
    )
```

### Client-side Error Handling

```javascript
client.on('error', (error) => {
    switch (error.type) {
        case 'websocket':
            handleConnectionError(error);
            break;
        case 'audio':
            handleAudioError(error);
            break;
        case 'server':
            handleServerError(error);
            break;
    }
});

// Automatic reconnection
client.autoReconnect = true;
client.maxReconnectAttempts = 5;
```

## Performance Optimization

### Connection Management

```python
# Optimize for high-concurrency
server = VoiceWebSocketServer(
    max_connections=1000,
    ping_interval=30.0,
    ping_timeout=10.0
)
```

### Audio Optimization

```javascript
// Optimize audio settings
const client = new VoiceWebSocketClient({
    chunkSizeMs: 50,        // Smaller chunks for lower latency
    enableVAD: true,        // Reduce unnecessary processing
    vadThreshold: 0.02      // Adjust sensitivity
});
```

### Memory Management

```python
# Configure buffer management
from src.coda.components.voice.audio_buffer_pool import AudioBufferPool

buffer_pool = AudioBufferPool(
    max_buffers=100,
    cleanup_interval=30.0
)
```

## Deployment

### Production Configuration

```python
# Production server setup
server = VoiceWebSocketServer(
    host="0.0.0.0",
    port=8765,
    max_connections=500,
    auth_required=True
)

# Enable monitoring
await server.monitor.start_monitoring()

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/
WORKDIR /app

EXPOSE 8765

CMD ["python", "-m", "src.coda.components.voice.websocket_server", "--host", "0.0.0.0"]
```

### Load Balancing

For high-traffic deployments, use multiple server instances behind a load balancer:

```nginx
upstream voice_websocket {
    server voice-server-1:8765;
    server voice-server-2:8765;
    server voice-server-3:8765;
}

server {
    location /voice {
        proxy_pass http://voice_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Testing

### Unit Tests

```bash
# Run WebSocket integration tests
python -m pytest tests/voice/test_websocket_integration.py -v

# Run with coverage
python -m pytest tests/voice/test_websocket_integration.py --cov=src.coda.components.voice
```

### Load Testing

```python
import asyncio
import websockets

async def load_test():
    tasks = []
    for i in range(100):  # 100 concurrent connections
        task = asyncio.create_task(test_client_connection(i))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

async def test_client_connection(client_id):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Simulate voice interaction
        await websocket.send(json.dumps({
            "type": "conversation_start",
            "data": {"conversation_id": f"load_test_{client_id}"}
        }))
        
        response = await websocket.recv()
        # Process response...
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check server is running
   - Verify port is not blocked
   - Check firewall settings

2. **Audio Not Processing**
   - Verify microphone permissions
   - Check audio format compatibility
   - Ensure voice manager is initialized

3. **High Latency**
   - Reduce chunk size
   - Check network conditions
   - Monitor server resources

4. **Memory Leaks**
   - Enable buffer pool cleanup
   - Monitor connection cleanup
   - Check for unclosed streams

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('src.coda.components.voice').setLevel(logging.DEBUG)

# Monitor performance
profiler = get_performance_profiler()
profiler.start_monitoring(interval=1.0)
```

## API Reference

For complete API documentation, see:
- [WebSocket Handler API](./voice_api_reference.md#websocket-handler)
- [Event Broadcasting API](./voice_api_reference.md#event-broadcasting)
- [Audio Streaming API](./voice_api_reference.md#audio-streaming)
- [Monitoring API](./voice_api_reference.md#monitoring)

## Examples

Complete examples are available in:
- `examples/websocket_voice_chat.py` - Basic voice chat
- `examples/websocket_voice_assistant.py` - Voice assistant
- `examples/websocket_monitoring_dashboard.py` - Monitoring dashboard
- `client/examples/` - JavaScript client examples
