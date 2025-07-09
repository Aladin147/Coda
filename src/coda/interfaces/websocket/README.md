# Coda WebSocket Interface

> **Real-time event broadcasting system for Coda**

The WebSocket interface provides real-time communication between Coda's core components and external clients (dashboards, monitoring tools, etc.).

## Features

- 🔄 **Real-time event broadcasting** to all connected clients
- 📝 **Type-safe event system** with Pydantic models
- 🔌 **Client connection management** with automatic cleanup
- 📚 **Event replay buffer** for new clients
- 📊 **Performance monitoring** integration
- 🛡️ **Error handling** and recovery
- ⚡ **Async/await** based architecture

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Coda Core     │───▶│  WebSocket       │───▶│   Dashboard     │
│   Components    │    │  Server          │    │   Clients       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Integration    │    │  Event System    │    │  Monitoring     │
│  Layer          │    │  (Type-safe)     │    │  Tools          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Basic Server Usage

```python
import asyncio
from coda.interfaces.websocket import CodaWebSocketServer

async def main():
    server = CodaWebSocketServer(host="localhost", port=8765)
    
    async with server.run_context():
        # Broadcast system info
        await server.broadcast_system_info({"version": "2.0.0"})
        
        # Keep server running
        await asyncio.Future()  # Run forever

asyncio.run(main())
```

### Integration with Coda Components

```python
from coda.interfaces.websocket import CodaWebSocketServer, CodaWebSocketIntegration

async def main():
    server = CodaWebSocketServer()
    integration = CodaWebSocketIntegration(server)
    
    await server.start()
    
    # STT events
    await integration.stt_start("continuous")
    await integration.stt_final_result("Hello world", 0.95, 1500.0)
    
    # LLM events
    await integration.llm_start("Hello world", "llama3")
    await integration.llm_result("Hi there!", 800.0, 15, 18.75)
    
    # TTS events
    await integration.tts_start("Hi there!", "default", "elevenlabs")
    await integration.tts_result(600.0, 1200.0, True)
    
    await server.stop()

asyncio.run(main())
```

### Performance Monitoring

```python
from coda.interfaces.websocket import WebSocketPerfIntegration

async def main():
    server = CodaWebSocketServer()
    perf = WebSocketPerfIntegration(server, metrics_interval=5.0)
    
    await server.start()
    await perf.start()
    
    # Track operations
    async with perf.time_operation("stt", "transcribe"):
        # Your STT code here
        await asyncio.sleep(1.5)  # Simulated processing
    
    # Manual tracking
    await perf.track_operation("llm", "generate", 1200.0, {"tokens": 45})
    
    await perf.stop()
    await server.stop()

asyncio.run(main())
```

## Event Types

### System Events
- `system_info` - System information and status
- `system_error` - System errors and warnings  
- `system_metrics` - Performance metrics (CPU, memory, GPU)

### Voice Processing Events
- `stt_start`, `stt_interim`, `stt_result`, `stt_error` - Speech-to-text
- `llm_start`, `llm_token`, `llm_result`, `llm_error` - Language model
- `tts_start`, `tts_progress`, `tts_result`, `tts_error` - Text-to-speech

### Memory Events
- `memory_store` - Memory storage operations
- `memory_retrieve` - Memory retrieval operations

### Tool Events
- `tool_call` - Tool execution requests
- `tool_result` - Tool execution results
- `tool_error` - Tool execution errors

### Conversation Events
- `conversation_start` - Conversation initiation
- `conversation_turn` - Individual conversation turns
- `conversation_end` - Conversation completion

### Performance Events
- `latency_trace` - Individual operation timing
- `component_timing` - Component performance summary

## Event Structure

All events follow a consistent structure:

```json
{
  "version": "1.0",
  "seq": 123,
  "timestamp": 1704067200.123,
  "type": "stt_result",
  "session_id": "uuid-here",
  "text": "Hello world",
  "confidence": 0.95,
  "duration_ms": 1500.0,
  "language": "en"
}
```

## Client Connection

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event:', data.type, data);
};

ws.onopen = () => {
    console.log('Connected to Coda WebSocket');
};
```

### Event Replay
New clients automatically receive a replay buffer of recent high-priority events to catch up on the current state.

## Configuration

### Server Configuration
```python
server = CodaWebSocketServer(
    host="localhost",           # Bind host
    port=8765,                 # Bind port
    max_replay_events=50,      # Replay buffer size
    ping_interval=20,          # Ping interval (seconds)
    ping_timeout=10,           # Ping timeout (seconds)
)
```

### Performance Integration
```python
perf = WebSocketPerfIntegration(
    server=server,
    metrics_interval=5.0,      # Metrics broadcast interval
)
```

## Testing

### Run Unit Tests
```bash
pytest tests/unit/test_websocket_server.py -v
```

### Run Demo
```bash
python scripts/websocket_demo.py
```

Then connect with a WebSocket client:
```bash
# Using wscat (npm install -g wscat)
wscat -c ws://localhost:8765

# Using websocat (cargo install websocat)
websocat ws://localhost:8765
```

## Error Handling

The WebSocket system includes comprehensive error handling:

- **Connection errors** - Automatic client cleanup
- **Message parsing errors** - Invalid JSON handling
- **Event validation errors** - Type checking with Pydantic
- **Broadcasting errors** - Graceful degradation
- **Server shutdown** - Graceful client disconnection

## Performance Considerations

- **Async/await** throughout for non-blocking operations
- **Batch broadcasting** to multiple clients
- **Automatic buffer management** for replay events
- **Connection pooling** and cleanup
- **Metrics collection** with minimal overhead

## Integration Points

The WebSocket system integrates with:

- **Core Assistant** - Main conversation loop events
- **Memory System** - Storage and retrieval events
- **Personality Engine** - Behavioral adaptation events
- **Tool System** - Function call events
- **Voice Processing** - STT/TTS events
- **Performance Monitor** - System metrics

## Migration from Coda Lite

Key improvements over the original implementation:

✅ **Fixed duplicate `__init__` methods**  
✅ **Type-safe event system** with Pydantic  
✅ **Modern async/await** architecture  
✅ **Comprehensive error handling**  
✅ **Performance monitoring** integration  
✅ **Better client lifecycle** management  
✅ **Graceful shutdown** handling  
✅ **Unit test coverage**  

## Next Steps

- [ ] Dashboard client implementation
- [ ] Authentication and authorization
- [ ] Event filtering and subscriptions
- [ ] Horizontal scaling support
- [ ] Metrics persistence
- [ ] Real-time debugging tools
