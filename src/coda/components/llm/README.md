# Coda LLM Integration System

> **Advanced multi-provider LLM system with conversation management, function calling, and real-time streaming**

The LLM integration system provides comprehensive language model capabilities for Coda, featuring multiple provider support, intelligent conversation management, function calling orchestration, and seamless integration with memory and personality systems.

## Features

- 🤖 **Multi-Provider Support** - OpenAI, Anthropic, Ollama, and local models
- 💬 **Conversation Management** - Persistent conversations with context optimization
- ⚡ **Function Calling** - Seamless tool integration with LLM responses
- 📡 **Streaming Responses** - Real-time response generation with WebSocket support
- 🧠 **Memory Integration** - Context-aware responses using persistent memory
- 🎭 **Personality Enhancement** - Dynamic personality-driven prompt enhancement
- 🌐 **WebSocket Events** - Real-time LLM operation broadcasting
- 📊 **Analytics & Monitoring** - Comprehensive usage tracking and performance metrics

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  LLM            │───▶│  Conversation    │───▶│  Prompt         │
│  Manager        │    │  Manager         │    │  Enhancer       │
│  (Orchestrator) │    │  (History)       │    │  (Context)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Provider       │    │  Function        │    │  WebSocket      │
│  Management     │    │  Calling         │    │  Integration    │
│  (Multi-LLM)    │    │  (Tools)         │    │  (Events)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Basic Usage

```python
from coda.components.llm import LLMManager, LLMConfig, ProviderConfig, LLMProvider

# Create configuration
config = LLMConfig(
    providers={
        "openai": ProviderConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="your-api-key",
            temperature=0.7
        )
    },
    default_provider="openai"
)

# Create LLM manager
llm_manager = LLMManager(config)

# Generate response
response = await llm_manager.generate_response("Hello, how can you help me?")
print(f"Response: {response.content}")

# Continue conversation
response2 = await llm_manager.continue_conversation(
    response.conversation_id,
    "Tell me about artificial intelligence"
)
```

### Streaming Responses

```python
# Generate streaming response
async for chunk in llm_manager.generate_response(
    "Tell me a story",
    stream=True
):
    print(chunk.delta, end="", flush=True)
```

### Multiple Providers

```python
config = LLMConfig(
    providers={
        "openai": ProviderConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="openai-key"
        ),
        "anthropic": ProviderConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet",
            api_key="anthropic-key"
        ),
        "ollama": ProviderConfig(
            provider=LLMProvider.OLLAMA,
            model="llama2",
            api_base="http://localhost:11434"
        )
    },
    default_provider="openai"
)

# Use specific provider
response = await llm_manager.generate_response(
    "Explain quantum computing",
    provider="anthropic"
)
```

## Supported Providers

### OpenAI
- **Models**: GPT-3.5, GPT-4, GPT-4 Turbo
- **Features**: Function calling, streaming, token counting, cost calculation
- **Configuration**: API key, base URL, model parameters

```python
ProviderConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=2000,
    top_p=1.0
)
```

### Anthropic
- **Models**: Claude 3 (Opus, Sonnet, Haiku), Claude 2.1
- **Features**: Tool calling, streaming, safety features
- **Configuration**: API key, model parameters

```python
ProviderConfig(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-sonnet-20240229",
    api_key="sk-ant-...",
    temperature=0.7,
    max_tokens=4096
)
```

### Ollama
- **Models**: Any Ollama-compatible model (Llama 2, Mistral, etc.)
- **Features**: Local inference, streaming, no API costs
- **Configuration**: Base URL, model name

```python
ProviderConfig(
    provider=LLMProvider.OLLAMA,
    model="llama2",
    api_base="http://localhost:11434",
    temperature=0.7
)
```

### Local Models
- **Models**: Hugging Face transformers models
- **Features**: Complete local inference, GPU acceleration
- **Configuration**: Model path, device settings

```python
ProviderConfig(
    provider=LLMProvider.LOCAL,
    model="microsoft/DialoGPT-medium",
    temperature=0.7,
    extra_params={"device": "cuda"}
)
```

## Conversation Management

### Creating and Managing Conversations

```python
# Create conversation manager
from coda.components.llm import ConversationManager

conv_manager = ConversationManager()

# Create conversation
conversation = await conv_manager.create_conversation("my-conversation")

# Add messages
from coda.components.llm import LLMMessage, MessageRole

message = LLMMessage(
    role=MessageRole.USER,
    content="Hello, world!"
)
await conv_manager.add_message("my-conversation", message)

# Get conversation context
context = await conv_manager.get_conversation_context(
    "my-conversation",
    max_tokens=4000
)
```

### Conversation Features

- **Context Management**: Automatic token limit handling
- **Auto-Summarization**: Long conversations automatically summarized
- **Message History**: Persistent conversation storage
- **Token Optimization**: Smart context window management

## Function Calling Integration

### Setting Up Function Calling

```python
from coda.components.tools import ToolManager

# Create tool manager
tool_manager = ToolManager()

# Integrate with LLM manager
await llm_manager.set_tool_manager(tool_manager)

# Enable function calling
config.function_calling.enabled = True
config.function_calling.auto_execute = True
```

### Function Calling Flow

1. **LLM generates function calls** in response
2. **System parses and validates** function calls
3. **Tools are executed** with provided parameters
4. **Results are formatted** and sent back to LLM
5. **LLM incorporates results** into final response

```python
# Example: LLM can now use tools
response = await llm_manager.generate_response(
    "What time is it and what's 2+2?"
)
# LLM will automatically call get_time and calculator tools
```

## Memory and Personality Integration

### Memory Integration

```python
from coda.components.memory import MemoryManager

# Create and integrate memory manager
memory_manager = MemoryManager()
await llm_manager.set_memory_manager(memory_manager)

# Memory is automatically used for context enhancement
response = await llm_manager.generate_response(
    "Remember that I prefer detailed explanations"
)
# This preference will be stored and used in future conversations
```

### Personality Integration

```python
from coda.components.personality import PersonalityManager

# Create and integrate personality manager
personality_manager = PersonalityManager()
await llm_manager.set_personality_manager(personality_manager)

# Personality influences response style and content
response = await llm_manager.generate_response("How are you feeling?")
# Response will reflect current personality state
```

## WebSocket Integration

### Real-Time Events

```python
from coda.components.llm import WebSocketLLMManager
from coda.interfaces.websocket import CodaWebSocketServer, CodaWebSocketIntegration

# Set up WebSocket server
server = CodaWebSocketServer()
integration = CodaWebSocketIntegration(server)

# Create WebSocket-enabled LLM manager
llm_manager = WebSocketLLMManager(config)
await llm_manager.set_websocket_integration(integration)

# All LLM operations now broadcast real-time events
await llm_manager.generate_response("Hello")  # Broadcasts events
```

### Event Types

- `response_start` - Response generation begins
- `response_chunk` - Streaming response chunk
- `response_complete` - Response generation complete
- `function_calls` - Function calls detected
- `conversation_created` - New conversation created
- `provider_status_change` - Provider availability change
- `analytics_update` - Usage analytics update

## Prompt Enhancement

### Automatic Enhancement

The system automatically enhances prompts with:

- **Personality Context**: Current personality traits and mood
- **Memory Context**: Relevant memories and preferences
- **Conversation History**: Formatted conversation context
- **System Information**: Current time, available tools

### Custom Enhancement

```python
from coda.components.llm import PromptEnhancer

enhancer = PromptEnhancer()

# Enhance system prompt
enhanced_system = await enhancer.enhance_system_prompt(
    "You are a helpful assistant",
    context={"user_preferences": ["detailed", "technical"]}
)

# Enhance user prompt
enhanced_user = await enhancer.enhance_user_prompt(
    "Explain machine learning",
    context={"conversation_id": "conv-123"}
)
```

## Configuration

### Complete Configuration

```python
from coda.components.llm import LLMConfig, ConversationConfig, FunctionCallingConfig

config = LLMConfig(
    # Provider configurations
    providers={
        "openai": ProviderConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="your-key",
            temperature=0.7,
            max_tokens=2000,
            timeout=120.0,
            max_retries=3
        )
    },
    default_provider="openai",
    
    # Conversation management
    conversation=ConversationConfig(
        max_messages=50,
        max_tokens_per_conversation=8000,
        auto_summarize=True,
        summarize_threshold=40,
        memory_integration_enabled=True,
        personality_integration_enabled=True
    ),
    
    # Function calling
    function_calling=FunctionCallingConfig(
        enabled=True,
        auto_execute=True,
        max_function_calls_per_turn=5,
        parallel_execution=True,
        tool_integration_enabled=True
    ),
    
    # WebSocket integration
    websocket_events_enabled=True,
    stream_responses=True,
    
    # System integration
    memory_integration_enabled=True,
    personality_integration_enabled=True,
    tool_integration_enabled=True
)
```

## Analytics and Monitoring

### Usage Analytics

```python
# Get comprehensive analytics
analytics = await llm_manager.get_analytics()

print(f"Total conversations: {analytics['conversations']['total_conversations']}")
print(f"Total tokens: {analytics['conversations']['total_tokens']}")
print(f"Total cost: ${analytics['conversations']['total_cost']:.4f}")

# Provider-specific analytics
for provider, stats in analytics['providers'].items():
    print(f"{provider}: {stats['request_count']} requests, {stats['total_tokens']} tokens")
```

### Provider Status

```python
# Check provider status
providers = await llm_manager.list_providers()

for provider in providers:
    status = await llm_manager.get_provider_status(provider)
    print(f"{provider}: {'✅' if status['available'] else '❌'}")
    print(f"  Model: {status.get('model', 'unknown')}")
    print(f"  Requests: {status.get('request_count', 0)}")
    print(f"  Tokens: {status.get('total_tokens', 0)}")
```

## Testing

### Run Unit Tests
```bash
pytest tests/unit/test_llm_system.py -v
```

### Run Demo
```bash
python scripts/llm_demo.py
```

## Error Handling

### Provider Errors

```python
from coda.components.llm.base_provider import (
    LLMError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthenticationError
)

try:
    response = await llm_manager.generate_response("Hello")
except LLMAuthenticationError:
    print("Invalid API key")
except LLMRateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after} seconds")
except LLMTimeoutError:
    print("Request timed out")
except LLMError as e:
    print(f"LLM error: {e}")
```

### Automatic Retry

Providers automatically retry failed requests with:
- **Exponential backoff** for rate limits
- **Configurable retry attempts** (default: 3)
- **Timeout handling** with configurable limits
- **Error classification** for recovery decisions

## Performance Optimization

### Token Management

- **Context window optimization** - Smart message truncation
- **Token counting** - Accurate token estimation per provider
- **Cost tracking** - Real-time cost calculation
- **Memory efficiency** - Conversation summarization

### Caching

```python
# Response caching (if implemented)
config.enable_caching = True
config.cache_ttl_seconds = 3600

# Cached responses for identical prompts
response1 = await llm_manager.generate_response("What is AI?")
response2 = await llm_manager.generate_response("What is AI?")  # From cache
```

## Migration from Coda Lite

Key improvements over the original implementation:

✅ **Multi-provider architecture** with unified interface  
✅ **Type-safe models** with Pydantic validation  
✅ **Advanced conversation management** with context optimization  
✅ **Function calling integration** with tool orchestration  
✅ **Streaming responses** with WebSocket support  
✅ **Memory and personality integration** for context-aware responses  
✅ **Comprehensive error handling** with automatic retry  
✅ **Real-time monitoring** and analytics  
✅ **Production-grade configuration** and deployment support  

## Advanced Features

### Custom Providers

```python
from coda.components.llm.base_provider import BaseLLMProvider

class CustomProvider(BaseLLMProvider):
    def get_provider_name(self):
        return LLMProvider.CUSTOM
    
    async def _make_request(self, messages, **kwargs):
        # Custom implementation
        pass
    
    def _parse_response(self, response):
        # Custom response parsing
        pass

# Register custom provider
llm_manager._providers["custom"] = CustomProvider(config)
```

### Response Post-Processing

```python
# Custom response processing
class CustomLLMManager(LLMManager):
    async def _process_response(self, response):
        # Custom post-processing
        response.content = self.apply_custom_filters(response.content)
        return response
```

## Next Steps

- [ ] Advanced caching and response optimization
- [ ] Multi-modal support (images, audio)
- [ ] Fine-tuning integration
- [ ] Advanced prompt engineering tools
- [ ] Cost optimization and budget management
- [ ] A/B testing for different providers
- [ ] Advanced analytics and insights

## Dependencies

- `openai` - OpenAI API integration
- `anthropic` - Anthropic API integration
- `aiohttp` - HTTP client for Ollama
- `transformers` - Local model support
- `torch` - PyTorch for local models
- `pydantic` - Data validation and settings
- `asyncio` - Asynchronous operations
