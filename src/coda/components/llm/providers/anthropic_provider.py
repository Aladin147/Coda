"""
Anthropic provider for Coda LLM system.

This module provides integration with Anthropic's Claude models.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..base_provider import (
    BaseLLMProvider,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ..models import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    MessageRole,
    ProviderConfig,
)

logger = logging.getLogger("coda.llm.anthropic")


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic provider for Claude models.

    Features:
    - Support for Claude models
    - Tool calling support (Claude 3+)
    - Streaming responses
    - Token counting and cost calculation
    """

    def __init__(self, config: ProviderConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)

        self._client = None

        # Model pricing (per 1M tokens)
        self._pricing = {
            "claude-3-opus": {"prompt": 15.0, "completion": 75.0},
            "claude-3-sonnet": {"prompt": 3.0, "completion": 15.0},
            "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
            "claude-2.1": {"prompt": 8.0, "completion": 24.0},
            "claude-2.0": {"prompt": 8.0, "completion": 24.0},
        }

        # Initialize client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key, base_url=self.config.api_base
            )

            logger.info("Anthropic client initialized successfully")

        except ImportError:
            raise LLMError(
                "Anthropic library not installed. Install with: pip install anthropic", "anthropic"
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize Anthropic client: {e}", "anthropic")

    def get_provider_name(self) -> LLMProvider:
        """Get the provider name."""
        return LLMProvider.ANTHROPIC

    async def _make_request(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Make request to Anthropic API."""
        try:
            # Format messages for Anthropic
            formatted_messages = self._format_messages_for_anthropic(messages)

            # Extract system message
            system_message = None
            user_messages = []

            for msg in formatted_messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            # Prepare request parameters
            request_params = {
                "model": self.config.model,
                "messages": user_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
                "stream": stream,
            }

            # Add system message if present
            if system_message:
                request_params["system"] = system_message

            # Add optional parameters
            if "temperature" in kwargs:
                request_params["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                request_params["top_p"] = kwargs["top_p"]

            # Add tools if provided (Claude 3+ only)
            if functions and self.supports_function_calling():
                request_params["tools"] = self._convert_functions_to_tools(functions)

            # Make the request
            response = await self._client.messages.create(**request_params)

            return response

        except Exception as e:
            self._handle_anthropic_error(e)

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response to LLMResponse."""
        try:
            # Extract content
            content = ""
            tool_calls = []

            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": content_block.id,
                            "type": "function",
                            "function": {
                                "name": content_block.name,
                                "arguments": json.dumps(content_block.input),
                            },
                        }
                    )

            # Extract usage information
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            return LLMResponse(
                response_id=self._create_response_id(),
                content=content,
                provider=self.get_provider_name(),
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tool_calls=tool_calls,
                finish_reason=getattr(response, "stop_reason", None),
                metadata={
                    "cost": cost,
                    "model": self.config.model,
                    "usage": usage.__dict__ if usage else {},
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse Anthropic response: {e}")
            raise LLMError(f"Failed to parse response: {e}", "anthropic")

    async def _parse_streaming_response(
        self, response: Any
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Parse Anthropic streaming response."""
        try:
            accumulated_content = ""

            async for event in response:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        delta = event.delta.text
                        accumulated_content += delta

                        yield LLMStreamChunk(
                            chunk_id=self._create_chunk_id(),
                            content=accumulated_content,
                            delta=delta,
                            metadata={"model": self.config.model, "provider": "anthropic"},
                        )

                elif event.type == "message_stop":
                    yield LLMStreamChunk(
                        chunk_id=self._create_chunk_id(),
                        content=accumulated_content,
                        delta="",
                        finish_reason="stop",
                        metadata={"model": self.config.model, "provider": "anthropic"},
                    )

        except Exception as e:
            logger.error(f"Failed to parse Anthropic streaming response: {e}")
            raise LLMError(f"Failed to parse streaming response: {e}", "anthropic")

    def _format_messages_for_anthropic(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Format messages for Anthropic API."""
        formatted = []

        for message in messages:
            # Convert role names
            role = message.role.value
            if role == "assistant":
                role = "assistant"
            elif role == "user":
                role = "user"
            elif role == "system":
                role = "system"
            else:
                continue  # Skip unsupported roles

            formatted.append({"role": role, "content": message.content})

        return formatted

    def _convert_functions_to_tools(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI function format to Anthropic tools format."""
        tools = []

        for function in functions:
            tool = {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
            tools.append(tool)

        return tools

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for Anthropic token usage."""
        model_key = self.config.model.lower()

        # Find matching pricing
        pricing = None
        for model_name, model_pricing in self._pricing.items():
            if model_name in model_key:
                pricing = model_pricing
                break

        if not pricing:
            # Default to Claude 3 Haiku pricing
            pricing = self._pricing["claude-3-haiku"]

        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]

        return prompt_cost + completion_cost

    async def list_available_models(self) -> List[str]:
        """List available Anthropic models."""
        # Anthropic doesn't have a models endpoint, return known models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
        ]

    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        model = self.config.model.lower()
        return "claude-3" in model

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return True

    def _handle_anthropic_error(self, error: Exception) -> None:
        """Handle Anthropic-specific errors."""
        error_str = str(error).lower()

        if "rate limit" in error_str or "429" in error_str:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {error}", "anthropic")

        elif "unauthorized" in error_str or "401" in error_str:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {error}", "anthropic")

        elif "timeout" in error_str:
            raise LLMTimeoutError(f"Anthropic request timed out: {error}", "anthropic")

        else:
            raise LLMError(f"Anthropic error: {error}", "anthropic")

    async def validate_connection(self) -> bool:
        """Validate connection to Anthropic."""
        try:
            # Try a simple test request
            test_messages = [LLMMessage(role=MessageRole.USER, content="Hello")]

            await self._make_request(test_messages)
            self.set_available(True)
            return True

        except Exception as e:
            logger.warning(f"Anthropic connection validation failed: {e}")
            self.set_available(False)
            return False
