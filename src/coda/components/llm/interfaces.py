"""
Interfaces for the LLM system.

This module defines abstract base classes for LLM components to ensure
consistent APIs and enable easy testing with mocks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from .models import (
    LLMMessage,
    LLMConversation,
    LLMResponse,
    LLMStreamChunk,
    LLMProvider,
    ProviderConfig,
    FunctionCall,
    FunctionCallResult,
    LLMStats,
    StreamingResponse,
    LLMResponseType,
)


class LLMProviderInterface(ABC):
    """Interface for LLM providers."""
    
    @abstractmethod
    def get_provider_name(self) -> LLMProvider:
        """Get the provider name."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponseType:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[LLMMessage],
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate a streaming response from the LLM."""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection to the LLM provider."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass
    
    @abstractmethod
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage."""
        pass
    
    @abstractmethod
    async def list_available_models(self) -> List[str]:
        """List available models for this provider."""
        pass
    
    @abstractmethod
    def supports_function_calling(self) -> bool:
        """Check if provider supports function calling."""
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        pass


class ConversationManagerInterface(ABC):
    """Interface for conversation management."""
    
    @abstractmethod
    async def create_conversation(self, conversation_id: Optional[str] = None) -> LLMConversation:
        """Create a new conversation."""
        pass
    
    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        """Get a conversation by ID."""
        pass
    
    @abstractmethod
    async def add_message(self, conversation_id: str, message: LLMMessage) -> None:
        """Add a message to a conversation."""
        pass
    
    @abstractmethod
    async def get_conversation_context(
        self,
        conversation_id: str,
        max_tokens: Optional[int] = None
    ) -> List[LLMMessage]:
        """Get conversation context for LLM."""
        pass
    
    @abstractmethod
    async def summarize_conversation(self, conversation_id: str) -> str:
        """Summarize a conversation."""
        pass
    
    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        pass
    
    @abstractmethod
    async def list_conversations(self, limit: int = 50) -> List[LLMConversation]:
        """List recent conversations."""
        pass
    
    @abstractmethod
    async def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation."""
        pass
    
    @abstractmethod
    async def cleanup_old_conversations(self, days: int = 30) -> int:
        """Cleanup old conversations."""
        pass


class PromptEnhancerInterface(ABC):
    """Interface for prompt enhancement."""
    
    @abstractmethod
    async def enhance_system_prompt(
        self,
        base_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhance system prompt with personality and context."""
        pass
    
    @abstractmethod
    async def enhance_user_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhance user prompt with context."""
        pass
    
    @abstractmethod
    async def inject_memory_context(
        self,
        prompt: str,
        conversation_id: str,
        max_memories: int = 5
    ) -> str:
        """Inject relevant memories into prompt."""
        pass
    
    @abstractmethod
    async def inject_personality_context(
        self,
        prompt: str,
        personality_state: Dict[str, Any]
    ) -> str:
        """Inject personality context into prompt."""
        pass
    
    @abstractmethod
    def format_conversation_history(
        self,
        messages: List[LLMMessage],
        max_length: int = 2000
    ) -> str:
        """Format conversation history for context."""
        pass


class FunctionCallingOrchestratorInterface(ABC):
    """Interface for function calling orchestration."""
    
    @abstractmethod
    async def process_function_calls(
        self,
        function_calls: List[FunctionCall],
        context: Optional[Dict[str, Any]] = None
    ) -> List[FunctionCallResult]:
        """Process function calls and return results."""
        pass
    
    @abstractmethod
    async def get_available_functions(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get available functions for LLM."""
        pass
    
    @abstractmethod
    def parse_function_calls_from_response(
        self,
        response: LLMResponse
    ) -> List[FunctionCall]:
        """Parse function calls from LLM response."""
        pass
    
    @abstractmethod
    def format_function_results_for_llm(
        self,
        results: List[FunctionCallResult]
    ) -> List[LLMMessage]:
        """Format function results as messages for LLM."""
        pass
    
    @abstractmethod
    async def validate_function_call(
        self,
        function_call: FunctionCall
    ) -> bool:
        """Validate a function call."""
        pass


class LLMManagerInterface(ABC):
    """Interface for the main LLM manager."""
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        provider: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponseType:
        """Generate a response using the LLM system."""
        pass
    
    @abstractmethod
    async def continue_conversation(
        self,
        conversation_id: str,
        message: str,
        stream: bool = False,
        **kwargs
    ) -> LLMResponseType:
        """Continue an existing conversation."""
        pass
    
    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        """Get a conversation."""
        pass
    
    @abstractmethod
    async def list_providers(self) -> List[str]:
        """List available providers."""
        pass
    
    @abstractmethod
    async def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """Get status of a provider."""
        pass
    
    @abstractmethod
    async def get_analytics(self) -> Dict[str, Any]:
        """Get LLM system analytics."""
        pass
    
    @abstractmethod
    async def set_memory_manager(self, memory_manager: Any) -> None:
        """Set memory manager for integration."""
        pass
    
    @abstractmethod
    async def set_personality_manager(self, personality_manager: Any) -> None:
        """Set personality manager for integration."""
        pass
    
    @abstractmethod
    async def set_tool_manager(self, tool_manager: Any) -> None:
        """Set tool manager for integration."""
        pass


class WebSocketLLMInterface(LLMManagerInterface):
    """Interface for LLM manager with WebSocket integration."""
    
    @abstractmethod
    async def set_websocket_integration(self, integration: Any) -> None:
        """Set WebSocket integration for event broadcasting."""
        pass
    
    @abstractmethod
    async def broadcast_llm_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an LLM event via WebSocket."""
        pass
    
    @abstractmethod
    async def broadcast_response_start(
        self,
        conversation_id: str,
        provider: str,
        model: str
    ) -> None:
        """Broadcast response generation start."""
        pass
    
    @abstractmethod
    async def broadcast_response_chunk(
        self,
        conversation_id: str,
        chunk: LLMStreamChunk
    ) -> None:
        """Broadcast response chunk."""
        pass
    
    @abstractmethod
    async def broadcast_response_complete(
        self,
        conversation_id: str,
        response: LLMResponse
    ) -> None:
        """Broadcast response completion."""
        pass
    
    @abstractmethod
    async def broadcast_function_calls(
        self,
        conversation_id: str,
        function_calls: List[FunctionCall]
    ) -> None:
        """Broadcast function calls."""
        pass


# Utility interfaces

class LLMAnalyticsInterface(ABC):
    """Interface for LLM analytics."""
    
    @abstractmethod
    def record_response(self, response: LLMResponse) -> None:
        """Record an LLM response for analytics."""
        pass
    
    @abstractmethod
    def get_usage_stats(self, time_range_hours: int = 24) -> LLMStats:
        """Get usage statistics for a time range."""
        pass
    
    @abstractmethod
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis and breakdown."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        pass
    
    @abstractmethod
    def get_provider_comparison(self) -> Dict[str, Any]:
        """Get provider performance comparison."""
        pass


class LLMCacheInterface(ABC):
    """Interface for LLM response caching."""
    
    @abstractmethod
    async def get_cached_response(
        self,
        messages: List[LLMMessage],
        provider: str,
        model: str
    ) -> Optional[LLMResponse]:
        """Get cached response for messages."""
        pass
    
    @abstractmethod
    async def cache_response(
        self,
        messages: List[LLMMessage],
        response: LLMResponse
    ) -> None:
        """Cache an LLM response."""
        pass
    
    @abstractmethod
    async def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class LLMSecurityInterface(ABC):
    """Interface for LLM security management."""
    
    @abstractmethod
    def validate_prompt_safety(self, prompt: str) -> bool:
        """Validate that a prompt is safe."""
        pass
    
    @abstractmethod
    def sanitize_response(self, response: str) -> str:
        """Sanitize LLM response for safety."""
        pass
    
    @abstractmethod
    def check_content_policy(self, content: str) -> Dict[str, Any]:
        """Check content against policies."""
        pass
    
    @abstractmethod
    def audit_conversation(self, conversation: LLMConversation) -> None:
        """Audit conversation for security."""
        pass
