# Coda Examples

This directory contains comprehensive examples demonstrating Coda's capabilities across all integrated systems.

## Quick Start Examples

### Basic Usage

```bash
# Run the comprehensive demo
python scripts/comprehensive_demo.py

# Try basic voice chat
python examples/basic_voice_chat.py

# Explore memory integration
python examples/memory_integration.py --interactive
```

## Available Examples

### 1. Basic Voice Chat (`basic_voice_chat.py`)

Demonstrates fundamental voice processing capabilities:

- Real-time voice input processing
- Multiple processing modes (moshi-only, hybrid, adaptive)
- Basic conversation management
- Audio format handling

**Usage:**
```bash
# Default adaptive mode
python examples/basic_voice_chat.py

# Fast moshi-only mode
python examples/basic_voice_chat.py --mode moshi-only

# High-quality hybrid mode
python examples/basic_voice_chat.py --mode hybrid
```

**Key Features:**
- Voice-to-voice conversation
- Automatic mode selection
- Real-time audio processing
- Simple conversation flow

### 2. Memory Integration (`memory_integration.py`)

Showcases intelligent memory management:

- Semantic memory storage and retrieval
- Different memory types (episodic, semantic, preference, etc.)
- Conversation context enhancement
- Memory-based learning

**Usage:**
```bash
# Run full demo
python examples/memory_integration.py

# Interactive exploration
python examples/memory_integration.py --interactive

# Verbose output
python examples/memory_integration.py --verbose
```

**Interactive Commands:**
- `search <query>` - Search memories semantically
- `store <content>` - Store new memory
- `stats` - Show memory statistics
- `recent` - Show recent memories
- `quit` - Exit interactive mode

### 3. Comprehensive Demo (`../scripts/comprehensive_demo.py`)

Full system integration demonstration:

- All systems working together
- Memory, personality, tools, voice integration
- Performance statistics
- Real conversation simulation

**Usage:**
```bash
# Full demo
python scripts/comprehensive_demo.py

# Specific components
python scripts/comprehensive_demo.py --mode memory
python scripts/comprehensive_demo.py --mode personality
python scripts/comprehensive_demo.py --mode tools
python scripts/comprehensive_demo.py --mode conversation
```

## Component-Specific Examples

### Memory System

```python
from coda.components.memory import MemoryManager, MemoryConfig, MemoryType

# Initialize memory system
config = MemoryConfig(enable_auto_learning=True)
memory_manager = MemoryManager(config)
await memory_manager.initialize()

# Store different types of memories
await memory_manager.store_memory(
    content="User prefers morning meetings",
    memory_type=MemoryType.PREFERENCE,
    importance_score=0.8
)

# Semantic search
memories = await memory_manager.retrieve_memories(
    query="schedule a meeting",
    limit=5
)
```

### Personality System

```python
from coda.components.personality import PersonalityManager, PersonalityConfig

# Initialize personality system
config = PersonalityConfig(enable_adaptation=True)
personality_manager = PersonalityManager(config)
await personality_manager.initialize()

# Adapt personality based on feedback
await personality_manager.adapt_personality(
    user_feedback={"engagement": 0.9, "helpfulness": 0.8},
    context="technical discussion"
)

# Get personality-aware response context
context = await personality_manager.get_response_context(
    user_input="Explain machine learning",
    conversation_context=["User is a software engineer"]
)
```

### Voice Processing

```python
from coda.components.voice import VoiceManager, VoiceConfig, VoiceProcessingMode

# Initialize voice system
config = VoiceConfig(default_mode=VoiceProcessingMode.ADAPTIVE)
voice_manager = VoiceManager(config)
await voice_manager.initialize()

# Process voice input
response = await voice_manager.process_voice_input(
    conversation_id="conv_123",
    audio_data=audio_bytes
)

print(f"Response: {response.text_content}")
# Play audio: response.audio_data
```

### Tools System

```python
from coda.components.tools import ToolsManager, ToolsConfig

# Initialize tools system
config = ToolsConfig(enable_auto_discovery=True)
tools_manager = ToolsManager(config)
await tools_manager.initialize()

# Get tool suggestions
suggestions = await tools_manager.get_tool_suggestions(
    query="What time is it?"
)

# Execute tool
if suggestions:
    result = await tools_manager.execute_tool(
        tool_name=suggestions[0].name,
        parameters={}
    )
    print(f"Result: {result}")
```

## Integration Examples

### Voice + Memory Integration

```python
# Enhanced voice processing with memory context
async def process_with_memory(user_input, audio_data):
    # Get relevant memories
    memories = await memory_manager.retrieve_memories(
        query=user_input,
        limit=3
    )
    
    # Process voice with context
    response = await voice_manager.process_voice_input(
        conversation_id="conv_123",
        audio_data=audio_data
    )
    
    # Store interaction
    await memory_manager.store_memory(
        content=f"User said: {user_input}",
        memory_type=MemoryType.EPISODIC
    )
    
    return response
```

### Personality + Memory Learning

```python
# Learn personality preferences from memory
async def adapt_from_memory():
    # Get preference memories
    preferences = await memory_manager.retrieve_memories(
        query="user preferences",
        memory_types=[MemoryType.PREFERENCE]
    )
    
    # Extract personality insights
    for pref in preferences:
        if "technical" in pref.content.lower():
            await personality_manager.adjust_trait(
                trait="technical_depth",
                adjustment=0.1
            )
        elif "casual" in pref.content.lower():
            await personality_manager.adjust_trait(
                trait="formality",
                adjustment=-0.1
            )
```

## WebSocket Examples

### Real-time Voice Streaming

```javascript
// Client-side WebSocket for voice streaming
const ws = new WebSocket('ws://localhost:8765');

// Send voice data
function sendVoiceData(audioBlob) {
    const reader = new FileReader();
    reader.onload = function() {
        ws.send(JSON.stringify({
            type: 'voice_input',
            conversation_id: 'conv_123',
            audio_data: Array.from(new Uint8Array(reader.result))
        }));
    };
    reader.readAsArrayBuffer(audioBlob);
}

// Receive voice response
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'voice_response') {
        // Play audio response
        playAudio(data.audio_data);
        // Display text
        console.log(data.text_content);
    }
};
```

### Memory Updates via WebSocket

```javascript
// Subscribe to memory updates
ws.send(JSON.stringify({
    type: 'subscribe',
    events: ['memory_stored', 'memory_retrieved']
}));

// Handle memory events
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'memory_stored') {
        console.log('New memory stored:', data.memory.content);
    }
};
```

## Performance Examples

### Benchmarking Voice Processing

```python
import time
import statistics

async def benchmark_voice_processing():
    latencies = []
    
    for i in range(100):
        start_time = time.time()
        
        response = await voice_manager.process_voice_input(
            conversation_id="benchmark",
            audio_data=test_audio_data
        )
        
        latency = (time.time() - start_time) * 1000  # ms
        latencies.append(latency)
    
    print(f"Mean latency: {statistics.mean(latencies):.1f}ms")
    print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.1f}ms")
```

### Memory Performance Testing

```python
async def benchmark_memory_operations():
    # Test storage performance
    start_time = time.time()
    for i in range(1000):
        await memory_manager.store_memory(
            content=f"Test memory {i}",
            memory_type=MemoryType.EPISODIC
        )
    storage_time = time.time() - start_time
    
    # Test retrieval performance
    start_time = time.time()
    for i in range(100):
        memories = await memory_manager.retrieve_memories(
            query=f"test query {i}",
            limit=10
        )
    retrieval_time = time.time() - start_time
    
    print(f"Storage: {1000/storage_time:.1f} ops/sec")
    print(f"Retrieval: {100/retrieval_time:.1f} ops/sec")
```

## Error Handling Examples

### Robust Voice Processing

```python
async def robust_voice_processing(audio_data):
    try:
        response = await voice_manager.process_voice_input(
            conversation_id="conv_123",
            audio_data=audio_data
        )
        return response
    except VoiceProcessingError as e:
        logger.error(f"Voice processing failed: {e}")
        # Fallback to text processing
        return await fallback_text_processing(audio_data)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return create_error_response("I'm sorry, I couldn't process that.")
```

### Memory Error Recovery

```python
async def safe_memory_operation(content):
    try:
        return await memory_manager.store_memory(content)
    except MemoryStorageError as e:
        logger.warning(f"Memory storage failed: {e}")
        # Try with reduced importance
        return await memory_manager.store_memory(
            content=content,
            importance_score=0.1
        )
    except Exception as e:
        logger.error(f"Memory operation failed: {e}")
        return None
```

## Running the Examples

### Prerequisites

```bash
# Install Coda with all features
pip install -e ".[dev]"

# For voice examples, ensure GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment Setup

```bash
# Set up environment variables
export CODA_CONFIG_PATH="configs/development.yaml"
export CODA_LOG_LEVEL="INFO"
export CODA_VOICE_ENABLED="true"
```

### Running Examples

```bash
# Make examples executable
chmod +x examples/*.py

# Run with Python
python examples/basic_voice_chat.py

# Run with verbose output
python examples/memory_integration.py --verbose

# Run comprehensive demo
python scripts/comprehensive_demo.py --mode full
```

## Troubleshooting

### Common Issues

1. **Voice not working**: Check GPU availability and CUDA installation
2. **Memory errors**: Ensure sufficient RAM and proper configuration
3. **Import errors**: Verify all dependencies are installed
4. **Performance issues**: Check system resources and configuration

### Debug Mode

```bash
# Enable debug logging
export CODA_LOG_LEVEL="DEBUG"
python examples/basic_voice_chat.py --verbose
```

## Contributing Examples

We welcome contributions of new examples! Please:

1. Follow the existing code style
2. Include comprehensive documentation
3. Add error handling
4. Test on multiple configurations
5. Submit a pull request

---

**Explore, learn, and build amazing things with Coda!** 🚀
