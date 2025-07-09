"""
Data models for the LLM system.

This module defines Pydantic models for type-safe LLM operations.
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Literal, AsyncGenerator
from enum import Enum
from pydantic import BaseModel, Field, validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


class MessageRole(str, Enum):
    """Message roles in conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class LLMMessage(BaseModel):
    """A message in an LLM conversation."""
    
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        
        if self.name:
            result["name"] = self.name
        
        if self.function_call:
            result["function_call"] = self.function_call
        
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        
        return result


class LLMConversation(BaseModel):
    """A conversation with an LLM."""
    
    conversation_id: str
    messages: List[LLMMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Conversation statistics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_message(self, message: LLMMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages_for_llm(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """Get messages formatted for LLM API calls."""
        messages = []
        
        for message in self.messages:
            if not include_system and message.role == MessageRole.SYSTEM:
                continue
            messages.append(message.to_dict())
        
        return messages
    
    def get_last_n_messages(self, n: int) -> List[LLMMessage]:
        """Get the last n messages from the conversation."""
        return self.messages[-n:] if n > 0 else []
    
    def get_token_count(self) -> int:
        """Get total token count for the conversation."""
        return self.total_tokens
    
    def update_token_usage(self, prompt_tokens: int, completion_tokens: int, cost: float = 0.0) -> None:
        """Update token usage statistics."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.total_cost += cost
        self.updated_at = datetime.now()


class LLMStreamChunk(BaseModel):
    """A chunk of streamed LLM response."""
    
    chunk_id: str
    content: str
    delta: str = ""
    finish_reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LLMResponse(BaseModel):
    """Response from an LLM."""
    
    response_id: str
    content: str
    provider: LLMProvider
    model: str
    conversation_id: Optional[str] = None
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Timing
    response_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Function calling
    function_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def has_function_calls(self) -> bool:
        """Check if response contains function calls."""
        return len(self.function_calls) > 0 or len(self.tool_calls) > 0
    
    def get_cost_estimate(self) -> float:
        """Get estimated cost for this response."""
        # This would be implemented based on provider pricing
        return 0.0


class FunctionCall(BaseModel):
    """A function call from an LLM."""
    
    call_id: str
    name: str
    arguments: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FunctionCallResult(BaseModel):
    """Result of a function call execution."""
    
    call_id: str
    function_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Configuration models

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    
    # Provider-specific settings
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class ConversationConfig(BaseModel):
    """Configuration for conversation management."""
    
    max_messages: int = Field(default=50, gt=0)
    max_tokens_per_conversation: int = Field(default=8000, gt=0)
    auto_summarize: bool = Field(default=True)
    summarize_threshold: int = Field(default=40, gt=0)
    context_window_size: int = Field(default=4000, gt=0)
    
    # Memory integration
    memory_integration_enabled: bool = Field(default=True)
    store_conversations: bool = Field(default=True)
    
    # Personality integration
    personality_integration_enabled: bool = Field(default=True)
    personality_prompt_enhancement: bool = Field(default=True)


class FunctionCallingConfig(BaseModel):
    """Configuration for function calling."""
    
    enabled: bool = Field(default=True)
    auto_execute: bool = Field(default=True)
    max_function_calls_per_turn: int = Field(default=5, gt=0)
    function_call_timeout: float = Field(default=30.0, gt=0)
    parallel_execution: bool = Field(default=True)
    
    # Tool integration
    tool_integration_enabled: bool = Field(default=True)
    allowed_tool_categories: List[str] = Field(default_factory=list)
    dangerous_tools_allowed: bool = Field(default=False)


class LLMConfig(BaseModel):
    """Complete LLM system configuration."""
    
    # Provider configurations
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    default_provider: str = "openai"
    
    # Conversation management
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    
    # Function calling
    function_calling: FunctionCallingConfig = Field(default_factory=FunctionCallingConfig)
    
    # WebSocket integration
    websocket_events_enabled: bool = Field(default=True)
    stream_responses: bool = Field(default=True)
    
    # Analytics and monitoring
    analytics_enabled: bool = Field(default=True)
    performance_monitoring: bool = Field(default=True)
    
    # Integration settings
    memory_integration_enabled: bool = Field(default=True)
    personality_integration_enabled: bool = Field(default=True)
    tool_integration_enabled: bool = Field(default=True)


class LLMEvent(BaseModel):
    """Event for WebSocket broadcasting."""
    
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    conversation_id: Optional[str] = None
    provider: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LLMStats(BaseModel):
    """Statistics about LLM usage."""
    
    total_conversations: int = 0
    total_messages: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    
    # Provider breakdown
    provider_usage: Dict[str, int] = Field(default_factory=dict)
    model_usage: Dict[str, int] = Field(default_factory=dict)
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    success_rate: float = 0.0
    
    # Function calling stats
    total_function_calls: int = 0
    function_call_success_rate: float = 0.0
    
    # Recent activity
    conversations_last_24h: int = 0
    tokens_last_24h: int = 0
    cost_last_24h: float = 0.0


# Utility types

StreamingResponse = AsyncGenerator[LLMStreamChunk, None]
LLMResponseType = Union[LLMResponse, StreamingResponse]
