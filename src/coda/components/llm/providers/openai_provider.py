"""
OpenAI provider for Coda LLM system.

This module provides integration with OpenAI's GPT models.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..base_provider import (
    BaseLLMProvider,
    LLMAuthenticationError,
    LLMError,
    LLMQuotaExceededError,
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

logger = logging.getLogger("coda.llm.openai")


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider for GPT models.

    Features:
    - Support for GPT-3.5 and GPT-4 models
    - Function calling support
    - Streaming responses
    - Token counting and cost calculation
    - Rate limit handling
    """

    def __init__(self, config: ProviderConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)

        self._client = None
        self._async_client = None

        # Model pricing (tokens per dollar)
        self._pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
        }

        # Initialize client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            # Set API key and base URL
            if self.config.api_key:
                openai.api_key = self.config.api_key

            if self.config.api_base:
                openai.api_base = self.config.api_base

            # Create async client for newer versions
            try:
                from openai import AsyncOpenAI

                self._async_client = AsyncOpenAI(
                    api_key=self.config.api_key, base_url=self.config.api_base
                )
            except ImportError:
                # Fallback for older versions
                self._client = openai

            logger.info("OpenAI client initialized successfully")

        except ImportError:
            raise LLMError(
                "OpenAI library not installed. Install with: pip install openai", "openai"
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}", "openai")

    def get_provider_name(self) -> LLMProvider:
        """Get the provider name."""
        return LLMProvider.OPENAI

    async def _make_request(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Make request to OpenAI API."""
        try:
            # Format messages for OpenAI
            formatted_messages = self._format_messages_for_api(messages)

            # Prepare request parameters
            request_params = {
                "model": self.config.model,
                "messages": formatted_messages,
                "stream": stream,
                **kwargs,
            }

            # Add functions if provided
            if functions and self.supports_function_calling():
                request_params["functions"] = functions
                request_params["function_call"] = "auto"

            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}

            # Make the request
            if self._async_client:
                # Use new async client
                response = await self._async_client.chat.completions.create(**request_params)
            else:
                # Use legacy client
                response = await self._client.ChatCompletion.acreate(**request_params)

            return response

        except Exception as e:
            self._handle_openai_error(e)

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response to LLMResponse."""
        try:
            # Handle different response formats
            if hasattr(response, "choices"):
                choice = response.choices[0]
                message = choice.message

                # Extract content
                content = getattr(message, "content", "") or ""

                # Extract function calls
                function_calls = []
                tool_calls = []

                if hasattr(message, "function_call") and message.function_call:
                    function_calls.append(
                        {
                            "name": message.function_call.name,
                            "arguments": message.function_call.arguments,
                        }
                    )

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_calls.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )

                # Extract usage information
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

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
                    function_calls=function_calls,
                    tool_calls=tool_calls,
                    finish_reason=getattr(choice, "finish_reason", None),
                    metadata={
                        "cost": cost,
                        "model": self.config.model,
                        "usage": usage.__dict__ if usage else {},
                    },
                )
            else:
                raise LLMError("Invalid response format from OpenAI", "openai")

        except Exception as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise LLMError(f"Failed to parse response: {e}", "openai")

    async def _parse_streaming_response(
        self, response: Any
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Parse OpenAI streaming response."""
        try:
            accumulated_content = ""

            async for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]

                    # Extract delta content
                    delta = ""
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        delta = choice.delta.content or ""
                        accumulated_content += delta

                    # Get finish reason
                    finish_reason = getattr(choice, "finish_reason", None)

                    yield LLMStreamChunk(
                        chunk_id=self._create_chunk_id(),
                        content=accumulated_content,
                        delta=delta,
                        finish_reason=finish_reason,
                        metadata={"model": self.config.model, "provider": "openai"},
                    )

        except Exception as e:
            logger.error(f"Failed to parse OpenAI streaming response: {e}")
            raise LLMError(f"Failed to parse streaming response: {e}", "openai")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using tiktoken if available."""
        try:
            import tiktoken

            # Get encoding for model
            if "gpt-4" in self.config.model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.config.model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))

        except ImportError:
            # Fallback to rough approximation
            return super().estimate_tokens(text)
        except Exception:
            # Fallback on any error
            return super().estimate_tokens(text)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for OpenAI token usage."""
        model_key = self.config.model.lower()

        # Find matching pricing
        pricing = None
        for model_name, model_pricing in self._pricing.items():
            if model_name in model_key:
                pricing = model_pricing
                break

        if not pricing:
            # Default to GPT-3.5 pricing
            pricing = self._pricing["gpt-3.5-turbo"]

        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    async def list_available_models(self) -> List[str]:
        """List available OpenAI models."""
        try:
            if self._async_client:
                models = await self._async_client.models.list()
                return [model.id for model in models.data if "gpt" in model.id]
            else:
                models = await self._client.Model.alist()
                return [model.id for model in models.data if "gpt" in model.id]

        except Exception as e:
            logger.warning(f"Failed to list OpenAI models: {e}")
            return ["gpt-3.5-turbo", "gpt-4"]

    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        model = self.config.model.lower()
        return any(supported in model for supported in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return True

    def _handle_openai_error(self, error: Exception) -> None:
        """Handle OpenAI-specific errors."""
        error_str = str(error).lower()

        if "rate limit" in error_str or "429" in error_str:
            # Extract retry-after if available
            retry_after = None
            if hasattr(error, "response") and hasattr(error.response, "headers"):
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    retry_after = float(retry_after)

            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {error}", "openai", retry_after)

        elif "unauthorized" in error_str or "401" in error_str:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {error}", "openai")

        elif "quota" in error_str or "billing" in error_str:
            raise LLMQuotaExceededError(f"OpenAI quota exceeded: {error}", "openai")

        elif "timeout" in error_str:
            raise LLMTimeoutError(f"OpenAI request timed out: {error}", "openai")

        else:
            raise LLMError(f"OpenAI error: {error}", "openai")

    async def validate_connection(self) -> bool:
        """Validate connection to OpenAI."""
        try:
            # Try to list models as a connection test
            await self.list_available_models()
            self.set_available(True)
            return True

        except Exception as e:
            logger.warning(f"OpenAI connection validation failed: {e}")
            self.set_available(False)
            return False
