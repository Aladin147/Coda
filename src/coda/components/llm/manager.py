"""
LLM manager for Coda.

This module provides the main LLMManager class that orchestrates
all LLM system components.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from .base_provider import LLMError
from .conversation_manager import ConversationManager
from .function_calling_orchestrator import FunctionCallingOrchestrator
from .interfaces import LLMManagerInterface
from .models import (
    LLMConfig,
    LLMConversation,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMResponseType,
    LLMStreamChunk,
    MessageRole,
    ProviderConfig,
    StreamingResponse,
)
from .prompt_enhancer import PromptEnhancer
from .providers import (
    AnthropicProvider,
    LocalModelProvider,
    OllamaProvider,
    OpenAIProvider,
)

logger = logging.getLogger("coda.llm.manager")


class LLMManager(LLMManagerInterface):
    """
    Main LLM manager that orchestrates all LLM system components.

    Features:
    - Multiple provider support and management
    - Conversation management with memory integration
    - Prompt enhancement with personality
    - Function calling orchestration
    - Streaming response support
    - Analytics and monitoring
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM manager.

        Args:
            config: LLM system configuration
        """
        self.config = config or LLMConfig()

        # Initialize components
        conversation_config = getattr(self.config, "conversation", None)
        self.conversation_manager = ConversationManager(conversation_config)
        self.prompt_enhancer = PromptEnhancer()
        self.function_calling = FunctionCallingOrchestrator()

        # Provider management
        self._providers: Dict[str, Any] = {}
        self._default_provider: Optional[str] = None

        # Integration context
        self._memory_manager: Optional[Any] = None
        self._personality_manager: Optional[Any] = None
        self._tool_manager: Optional[Any] = None

        # Initialize providers
        self._initialize_providers()

        logger.info("LLMManager initialized")

    async def initialize(self) -> None:
        """
        Perform async initialization of the LLM manager.

        This method performs any additional async setup needed for:
        - Provider connection testing
        - Integration component setup
        - Performance monitoring initialization
        """
        try:
            # Test provider connections if configured
            for provider_name, provider in self._providers.items():
                if hasattr(provider, "test_connection"):
                    try:
                        await provider.test_connection()
                        logger.debug(f"Provider {provider_name} connection test passed")
                    except Exception as e:
                        logger.warning(f"Provider {provider_name} connection test failed: {e}")

            # Initialize conversation manager if needed
            if hasattr(self.conversation_manager, "initialize"):
                await self.conversation_manager.initialize()

            logger.info("LLMManager async initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {e}")
            raise

    async def generate_response(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        provider: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponseType:
        """
        Generate a response using the LLM system.

        Args:
            prompt: User prompt
            conversation_id: Optional conversation ID
            provider: Optional provider name
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            LLM response or streaming response
        """
        try:
            # Create conversation if needed
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())

            conversation = await self.conversation_manager.get_conversation(conversation_id)
            if not conversation:
                conversation = await self.conversation_manager.create_conversation(conversation_id)

            # Get provider
            provider_name = provider or self._default_provider
            if not provider_name or provider_name not in self._providers:
                raise LLMError(f"Provider not available: {provider_name}")

            llm_provider = self._providers[provider_name]

            # Enhance prompt with context
            context = {"conversation_id": conversation_id, "provider": provider_name, **kwargs}

            enhanced_prompt = await self.prompt_enhancer.enhance_user_prompt(prompt, context)

            # Create user message
            user_message = LLMMessage(
                role=MessageRole.USER, content=enhanced_prompt, timestamp=datetime.now()
            )

            # Add to conversation
            await self.conversation_manager.add_message(conversation_id, user_message)

            # Get conversation context
            context_messages = await self.conversation_manager.get_conversation_context(
                conversation_id, max_tokens=kwargs.get("max_context_tokens", 4000)
            )

            # Enhance system prompt if needed
            if not context_messages or context_messages[0].role != MessageRole.SYSTEM:
                system_prompt = await self.prompt_enhancer.enhance_system_prompt("", context)
                system_message = LLMMessage(
                    role=MessageRole.SYSTEM, content=system_prompt, timestamp=datetime.now()
                )
                context_messages.insert(0, system_message)

            # Get available functions
            functions = None
            function_calling_config = getattr(self.config, "function_calling", None)
            if (
                function_calling_config
                and getattr(function_calling_config, "enabled", False)
                and self._tool_manager
            ):
                functions = await self.function_calling.get_available_functions(context)

            # Generate response
            if stream:
                return await self._generate_streaming_response(
                    llm_provider, context_messages, conversation_id, functions, **kwargs
                )
            else:
                return await self._generate_single_response(
                    llm_provider, context_messages, conversation_id, functions, **kwargs
                )

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise LLMError(f"Response generation failed: {e}")

    async def continue_conversation(
        self, conversation_id: str, message: str, stream: bool = False, **kwargs
    ) -> LLMResponseType:
        """
        Continue an existing conversation.

        Args:
            conversation_id: Conversation ID
            message: User message
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            LLM response or streaming response
        """
        return await self.generate_response(
            prompt=message, conversation_id=conversation_id, stream=stream, **kwargs
        )

    async def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        """Get a conversation."""
        return await self.conversation_manager.get_conversation(conversation_id)

    async def list_providers(self) -> List[str]:
        """List available providers."""
        return list(self._providers.keys())

    async def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """Get status of a provider."""
        if provider not in self._providers:
            return {"available": False, "error": "Provider not found"}

        llm_provider = self._providers[provider]

        try:
            is_available = await llm_provider.validate_connection()

            return {
                "available": is_available,
                "provider": provider,
                "model": llm_provider.get_model_name(),
                "supports_streaming": llm_provider.supports_streaming(),
                "supports_function_calling": llm_provider.supports_function_calling(),
                "request_count": llm_provider.get_request_count(),
                "total_tokens": llm_provider.get_total_tokens(),
                "total_cost": llm_provider.get_total_cost(),
                "last_error": llm_provider.get_last_error(),
            }

        except Exception as e:
            return {"available": False, "provider": provider, "error": str(e)}

    async def get_analytics(self) -> Dict[str, Any]:
        """Get LLM system analytics."""
        # Provider analytics
        provider_stats = {}
        for name, provider in self._providers.items():
            provider_stats[name] = {
                "request_count": provider.get_request_count(),
                "total_tokens": provider.get_total_tokens(),
                "total_cost": provider.get_total_cost(),
                "available": provider.is_available(),
            }

        # Conversation analytics
        conversation_stats = self.conversation_manager.get_manager_stats()

        # Function calling analytics
        function_stats = self.function_calling.get_execution_stats()

        function_calling_config = getattr(self.config, "function_calling", None)
        function_calling_enabled = function_calling_config and getattr(
            function_calling_config, "enabled", False
        )

        return {
            "providers": provider_stats,
            "conversations": conversation_stats,
            "function_calling": function_stats,
            "config": {
                "default_provider": self._default_provider,
                "function_calling_enabled": function_calling_enabled,
                "memory_integration_enabled": getattr(
                    self.config, "memory_integration_enabled", False
                ),
                "personality_integration_enabled": getattr(
                    self.config, "personality_integration_enabled", False
                ),
            },
        }

    async def set_memory_manager(self, memory_manager: Any) -> None:
        """Set memory manager for integration."""
        self._memory_manager = memory_manager
        self.conversation_manager.set_memory_manager(memory_manager)
        self.prompt_enhancer.set_memory_manager(memory_manager)
        logger.info("Memory manager set for LLM system")

    async def set_personality_manager(self, personality_manager: Any) -> None:
        """Set personality manager for integration."""
        self._personality_manager = personality_manager
        self.prompt_enhancer.set_personality_manager(personality_manager)
        logger.info("Personality manager set for LLM system")

    async def set_tool_manager(self, tool_manager: Any) -> None:
        """Set tool manager for integration."""
        self._tool_manager = tool_manager
        self.function_calling.set_tool_manager(tool_manager)
        logger.info("Tool manager set for LLM system")

    def _initialize_providers(self) -> None:
        """Initialize LLM providers from configuration."""
        # Handle different config structures
        providers_config = getattr(self.config, "providers", None)
        if not providers_config:
            # Try single provider config
            provider_config = getattr(self.config, "provider", None)
            if provider_config:
                providers_config = {"default": provider_config}
            else:
                logger.warning("No provider configuration found")
                return

        for provider_name, provider_config in providers_config.items():
            try:
                provider = self._create_provider(provider_config)
                self._providers[provider_name] = provider

                # Set default provider
                default_provider = getattr(self.config, "default_provider", provider_name)
                if provider_name == default_provider:
                    self._default_provider = provider_name

                logger.info(f"Initialized provider: {provider_name}")

            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")

        # Set fallback default provider
        if not self._default_provider and self._providers:
            self._default_provider = list(self._providers.keys())[0]
            logger.info(f"Set fallback default provider: {self._default_provider}")

    def _create_provider(self, config: ProviderConfig) -> Any:
        """Create a provider instance from configuration."""
        if config.provider == LLMProvider.OPENAI:
            return OpenAIProvider(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicProvider(config)
        elif config.provider == LLMProvider.OLLAMA:
            return OllamaProvider(config)
        elif config.provider == LLMProvider.LOCAL:
            return LocalModelProvider(config)
        else:
            raise LLMError(f"Unsupported provider: {config.provider}")

    async def _generate_single_response(
        self,
        provider: Any,
        messages: List[LLMMessage],
        conversation_id: str,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a single (non-streaming) response."""
        # Generate response
        response = await provider.generate_response(
            messages=messages, stream=False, functions=functions, **kwargs
        )

        # Set conversation ID
        response.conversation_id = conversation_id

        # Add assistant message to conversation
        assistant_message = LLMMessage(
            role=MessageRole.ASSISTANT,
            content=response.content,
            timestamp=datetime.now(),
            function_call=response.function_calls[0] if response.function_calls else None,
            tool_calls=response.tool_calls,
        )

        await self.conversation_manager.add_message(conversation_id, assistant_message)

        # Process function calls if present
        function_calling_config = getattr(self.config, "function_calling", None)
        auto_execute = function_calling_config and getattr(
            function_calling_config, "auto_execute", False
        )
        if response.has_function_calls() and auto_execute:
            await self._process_function_calls(response, conversation_id)

        return response

    async def _generate_streaming_response(
        self,
        provider: Any,
        messages: List[LLMMessage],
        conversation_id: str,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate a streaming response."""
        accumulated_content = ""

        async for chunk in provider.generate_streaming_response(
            messages=messages, functions=functions, **kwargs
        ):
            accumulated_content += chunk.delta
            yield chunk

        # Add final assistant message to conversation
        if accumulated_content:
            assistant_message = LLMMessage(
                role=MessageRole.ASSISTANT, content=accumulated_content, timestamp=datetime.now()
            )

            await self.conversation_manager.add_message(conversation_id, assistant_message)

    async def _process_function_calls(self, response: LLMResponse, conversation_id: str) -> None:
        """Process function calls from LLM response."""
        try:
            # Parse function calls
            function_calls = self.function_calling.parse_function_calls_from_response(response)

            if not function_calls:
                return

            # Execute function calls
            results = await self.function_calling.process_function_calls(function_calls)

            # Format results as messages
            result_messages = self.function_calling.format_function_results_for_llm(results)

            # Add result messages to conversation
            for message in result_messages:
                await self.conversation_manager.add_message(conversation_id, message)

            logger.info(
                f"Processed {len(function_calls)} function calls for conversation {conversation_id}"
            )

        except Exception as e:
            logger.error(f"Failed to process function calls: {e}")

    async def cleanup(self) -> None:
        """Cleanup LLM manager resources."""
        logger.info("Cleaning up LLM manager")

        # Cleanup all providers
        for provider in self._providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up provider: {e}")

        logger.info("LLM manager cleanup completed")

    async def shutdown(self) -> None:
        """Shutdown LLM manager (alias for cleanup)."""
        await self.cleanup()

    async def set_memory_manager(self, memory_manager) -> None:
        """Set memory manager for context enhancement."""
        self.memory_manager = memory_manager
        logger.info("Memory manager integrated with LLM manager")

    async def set_personality_manager(self, personality_manager) -> None:
        """Set personality manager for response adaptation."""
        self.personality_manager = personality_manager
        logger.info("Personality manager integrated with LLM manager")

    async def set_tools_manager(self, tools_manager) -> None:
        """Set tools manager for function calling."""
        self.tools_manager = tools_manager
        logger.info("Tools manager integrated with LLM manager")

    def is_healthy(self) -> bool:
        """Check if the LLM manager is healthy."""
        try:
            return (
                self.config is not None and
                self._providers is not None and
                len(self._providers) > 0 and
                self.conversation_manager is not None and
                self.prompt_enhancer is not None
            )
        except Exception:
            return False
