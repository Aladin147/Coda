"""
LLM integration system for Coda.

This package provides comprehensive LLM integration functionality including:
- Multiple LLM provider support (OpenAI, Anthropic, Ollama, local models)
- Conversation management with memory integration
- Response generation with personality enhancement
- Streaming responses with WebSocket integration
- Function calling orchestration
- Advanced prompt engineering

Features:
- Type-safe LLM operations with Pydantic models
- Async LLM interactions with error handling
- WebSocket integration for real-time events
- Memory and personality system integration
- Function calling with tool orchestration
- Comprehensive analytics and monitoring
"""

from .base_provider import BaseLLMProvider, LLMError, LLMTimeoutError
from .conversation_manager import ConversationManager
from .function_calling_orchestrator import FunctionCallingOrchestrator
from .interfaces import (
    ConversationManagerInterface,
    LLMManagerInterface,
    LLMProviderInterface,
    WebSocketLLMInterface,
)
from .manager import LLMManager
from .models import (
    ConversationConfig,
    FunctionCall,
    FunctionCallResult,
    LLMConfig,
    LLMConversation,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMRole,
    LLMStreamChunk,
    MessageRole,
)
from .prompt_enhancer import PromptEnhancer
from .providers import (
    AnthropicProvider,
    LocalModelProvider,
    OllamaProvider,
    OpenAIProvider,
)
from .websocket_integration import WebSocketLLMManager

__all__ = [
    # Interfaces
    "LLMProviderInterface",
    "ConversationManagerInterface",
    "LLMManagerInterface",
    "WebSocketLLMInterface",
    # Models
    "LLMMessage",
    "LLMConversation",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMProvider",
    "LLMConfig",
    "ConversationConfig",
    "FunctionCall",
    "FunctionCallResult",
    "MessageRole",
    "LLMRole",
    # Core Components
    "BaseLLMProvider",
    "LLMError",
    "LLMTimeoutError",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LocalModelProvider",
    "ConversationManager",
    "PromptEnhancer",
    "FunctionCallingOrchestrator",
    "LLMManager",
    "WebSocketLLMManager",
]
