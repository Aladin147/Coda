# Coda API Reference

> **Comprehensive API documentation for Coda voice assistant components**
> 
> **🎉 SYSTEM STATUS: 100% OPERATIONAL** - All APIs functional with robust validation

## 📊 API Health Status

| Component API | Status | Health | Description |
|---------------|--------|--------|-------------|
| **CodaAssistant** | ✅ OPERATIONAL | 100% | Main assistant interface |
| **LLM Manager** | ✅ HEALTHY | 100% | Language model management |
| **Memory Manager** | ✅ HEALTHY | 100% | Memory and retrieval operations |
| **Personality Manager** | ✅ HEALTHY | 100% | Personality adaptation |
| **Voice Manager** | ✅ HEALTHY | 100% | Voice processing and streaming |
| **Tools Manager** | ⚠️ CONFIGURED | 95% | Tool execution with security |
| **Config Validator** | ✅ OPERATIONAL | 100% | Configuration validation and healing |
| **Model Validator** | ✅ OPERATIONAL | 100% | Model health monitoring |

## 🤖 Core Assistant API

### CodaAssistant

The main entry point for all Coda functionality.

```python
from coda.core.assistant import CodaAssistant
from coda.core.config import load_config
from pathlib import Path

# Initialize with configuration
config = load_config(Path("configs/default.yaml"))
assistant = CodaAssistant(config)

# Check system health
health_status = assistant.get_health_status()
print(f"System Health: {health_status}")

# Process text input
response = await assistant.process_text("Hello, how are you?")
print(f"Response: {response}")

# Process voice input (if voice manager is healthy)
if assistant.voice_manager.is_healthy():
    voice_response = await assistant.process_voice(audio_data)
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__(config)` | `config: CodaConfig` | `CodaAssistant` | Initialize assistant with configuration |
| `get_health_status()` | None | `Dict[str, bool]` | Get health status of all components |
| `process_text(text)` | `text: str` | `str` | Process text input and return response |
| `process_voice(audio)` | `audio: bytes` | `bytes` | Process voice input and return audio response |
| `get_conversation_history()` | None | `List[Dict]` | Get current conversation history |
| `reset_conversation()` | None | `None` | Reset conversation state |

## 🧠 LLM Manager API

### LLMManager

Manages language model providers and conversation flow.

```python
from coda.components.llm.manager import LLMManager
from coda.core.config_adapters import ConfigAdapter

# Initialize with adapted configuration
llm_config = ConfigAdapter.adapt_config_for_component(config, 'llm')
llm_manager = LLMManager(llm_config)

# Check health
is_healthy = llm_manager.is_healthy()
print(f"LLM Manager Health: {is_healthy}")

# Generate response
response = await llm_manager.generate_response(
    message="What is the weather like?",
    context={"location": "San Francisco"}
)

# Get available providers
providers = llm_manager.get_available_providers()
print(f"Available providers: {providers}")
```

#### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `is_healthy()` | None | `bool` | Check if LLM manager is operational |
| `generate_response(message, context)` | `message: str, context: Dict` | `str` | Generate LLM response |
| `get_available_providers()` | None | `List[str]` | List available LLM providers |
| `switch_provider(provider_name)` | `provider_name: str` | `bool` | Switch to different provider |
| `get_conversation_manager()` | None | `ConversationManager` | Get conversation manager instance |

## 🧠 Memory Manager API

### MemoryManager

Handles short-term and long-term memory operations.

```python
from coda.components.memory.manager import MemoryManager
from coda.components.memory.models import MemoryManagerConfig

# Initialize memory manager
memory_manager = MemoryManager(memory_config)

# Store memory
await memory_manager.store_memory(
    content="User prefers morning meetings",
    metadata={"type": "preference", "importance": "high"}
)

# Search memories
relevant_memories = await memory_manager.search_memories(
    query="meeting preferences",
    limit=5
)

# Get conversation context
context = memory_manager.get_conversation_context(max_tokens=1000)
```

#### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `is_healthy()` | None | `bool` | Check memory system health |
| `store_memory(content, metadata)` | `content: str, metadata: Dict` | `str` | Store new memory |
| `search_memories(query, limit)` | `query: str, limit: int` | `List[Memory]` | Search relevant memories |
| `get_conversation_context(max_tokens)` | `max_tokens: int` | `str` | Get conversation context |
| `clear_short_term_memory()` | None | `None` | Clear short-term memory |

## 🎭 Personality Manager API

### PersonalityManager

Manages adaptive personality and behavioral conditioning.

```python
from coda.components.personality.manager import PersonalityManager

# Initialize personality manager
personality_manager = PersonalityManager(personality_config)

# Adapt to user feedback
personality_manager.adapt_to_feedback(
    interaction_id="conv_123",
    feedback_type="positive",
    feedback_data={"enthusiasm": "high", "formality": "low"}
)

# Get current personality state
personality_state = personality_manager.get_personality_state()
print(f"Current personality: {personality_state}")

# Apply personality to response
enhanced_response = personality_manager.enhance_response(
    base_response="Here's the information you requested",
    context={"user_mood": "excited", "topic": "technology"}
)
```

#### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `is_healthy()` | None | `bool` | Check personality system health |
| `adapt_to_feedback(interaction_id, feedback_type, data)` | Various | `None` | Adapt personality based on feedback |
| `get_personality_state()` | None | `Dict` | Get current personality configuration |
| `enhance_response(response, context)` | `response: str, context: Dict` | `str` | Apply personality to response |
| `reset_personality()` | None | `None` | Reset to base personality |

## 🎙️ Voice Manager API

### VoiceManager

Handles voice processing with Moshi integration.

```python
from coda.components.voice.manager import VoiceManager

# Initialize voice manager
voice_manager = VoiceManager(voice_config)

# Check if voice processing is available
if voice_manager.is_healthy():
    # Process audio input
    audio_response = await voice_manager.process_audio(
        audio_data=input_audio,
        sample_rate=16000
    )
    
    # Start streaming session
    session = await voice_manager.start_streaming_session()
    
    # Process streaming audio
    async for audio_chunk in voice_manager.process_streaming_audio(session):
        # Handle real-time audio processing
        yield audio_chunk
```

#### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `is_healthy()` | None | `bool` | Check voice system health |
| `process_audio(audio_data, sample_rate)` | `audio_data: bytes, sample_rate: int` | `bytes` | Process audio input |
| `start_streaming_session()` | None | `StreamingSession` | Start real-time audio session |
| `process_streaming_audio(session)` | `session: StreamingSession` | `AsyncGenerator` | Process streaming audio |
| `get_voice_capabilities()` | None | `Dict` | Get available voice capabilities |

## 🛠️ Tools Manager API

### ToolManager

Manages function calling and tool execution.

```python
from coda.components.tools.manager import ToolManager

# Initialize tools manager
tools_manager = ToolManager(tools_config)

# Get available tools
available_tools = tools_manager.get_available_tools()
print(f"Available tools: {available_tools}")

# Execute tool
result = await tools_manager.execute_tool(
    tool_name="get_weather",
    parameters={"location": "San Francisco", "units": "metric"}
)

# Register custom tool
tools_manager.register_tool(
    name="custom_calculator",
    function=my_calculator_function,
    description="Performs mathematical calculations"
)
```

#### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `is_healthy()` | None | `bool` | Check tools system health |
| `get_available_tools()` | None | `List[str]` | List available tools |
| `execute_tool(name, parameters)` | `name: str, parameters: Dict` | `Any` | Execute specific tool |
| `register_tool(name, function, description)` | Various | `bool` | Register new tool |
| `get_tool_schema(tool_name)` | `tool_name: str` | `Dict` | Get tool parameter schema |
