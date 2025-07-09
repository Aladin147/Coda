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

from .interfaces import (
    LLMProviderInterface,
    ConversationManagerInterface,
    LLMManagerInterface,
    WebSocketLLMInterface,
)
from .models import (
    LLMMessage,
    LLMConversation,
    LLMResponse,
    LLMStreamChunk,
    LLMProvider,
    LLMConfig,
    ConversationConfig,
    FunctionCall,
    FunctionCallResult,
)
from .base_provider import BaseLLMProvider, LLMError, LLMTimeoutError
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    LocalModelProvider,
)
from .conversation_manager import ConversationManager
from .prompt_enhancer import PromptEnhancer
from .function_calling_orchestrator import FunctionCallingOrchestrator
from .manager import LLMManager
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
