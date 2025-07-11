"""
Base LLM provider implementation for Coda.

This module provides the BaseLLMProvider class that all LLM providers should inherit from,
along with common utilities and exceptions.
"""

import asyncio
import logging
import time
from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from .interfaces import LLMProviderInterface
from .models import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    MessageRole,
    ProviderConfig,
)

logger = logging.getLogger("coda.llm.base")


class LLMError(Exception):
    """Base exception for LLM errors."""

    def __init__(self, message: str, provider: str = "", recoverable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.recoverable = recoverable


class LLMTimeoutError(LLMError):
    """Exception for LLM request timeouts."""

    def __init__(self, message: str, provider: str = "", timeout_seconds: float = 0):
        super().__init__(message, provider, recoverable=True)
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(LLMError):
    """Exception for rate limit errors."""

    def __init__(self, message: str, provider: str = "", retry_after: Optional[float] = None):
        super().__init__(message, provider, recoverable=True)
        self.retry_after = retry_after


class LLMAuthenticationError(LLMError):
    """Exception for authentication errors."""

    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider, recoverable=False)


class LLMQuotaExceededError(LLMError):
    """Exception for quota exceeded errors."""

    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider, recoverable=False)


class BaseLLMProvider(LLMProviderInterface):
    """
    Base class for all LLM providers.

    Provides common functionality including:
    - Configuration management
    - Error handling
    - Token estimation
    - Response formatting
    - Retry logic
    """

    def __init__(self, config: ProviderConfig):
        """Initialize the base provider."""
        self.config = config
        self._is_available = True
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._last_error: Optional[str] = None

        logger.info(
            f"Initialized {self.get_provider_name().value} provider with model: {config.model}"
        )

    @abstractmethod
    def get_provider_name(self) -> LLMProvider:
        """Get the provider name. Must be implemented by subclasses."""
        pass

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config.model

    @abstractmethod
    async def _make_request(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Make request to the LLM provider. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse provider response to LLMResponse. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _parse_streaming_response(self, response: Any) -> AsyncGenerator[LLMStreamChunk, None]:
        """Parse streaming response. Must be implemented by subclasses."""
        pass

    async def generate_response(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Generate a response from the LLM.

        Args:
            messages: List of conversation messages
            stream: Whether to stream the response
            functions: Available functions for function calling
            **kwargs: Additional parameters

        Returns:
            LLMResponse or AsyncGenerator for streaming

        Raises:
            LLMError: If generation fails
        """
        if not self.is_available():
            raise LLMError(
                f"Provider {self.get_provider_name().value} is not available",
                self.get_provider_name().value,
            )

        try:
            # Merge config parameters with kwargs
            params = self._prepare_parameters(**kwargs)

            # Execute with retry logic
            if stream:
                return self._generate_streaming_with_retries(messages, functions, **params)
            else:
                return await self._generate_with_retries(messages, functions, **params)

        except LLMError:
            # Re-raise LLM-specific errors
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            error_msg = f"LLM connection failed: {str(e)}"
            self._last_error = error_msg
            logger.error(f"Provider {self.get_provider_name().value} connection failed: {e}")
            raise LLMError(error_msg, self.get_provider_name().value)
        except (ValueError, TypeError) as e:
            error_msg = f"LLM parameter validation failed: {str(e)}"
            self._last_error = error_msg
            logger.error(f"Provider {self.get_provider_name().value} parameter error: {e}")
            raise LLMError(error_msg, self.get_provider_name().value)
        except Exception as e:
            error_msg = f"LLM generation failed: {str(e)}"
            self._last_error = error_msg
            logger.error(f"Provider {self.get_provider_name().value} unexpected error: {e}")
            raise LLMError(error_msg, self.get_provider_name().value)

    async def generate_streaming_response(
        self, messages: List[LLMMessage], functions: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate a streaming response from the LLM."""
        async for chunk in await self.generate_response(
            messages, stream=True, functions=functions, **kwargs
        ):
            yield chunk

    async def validate_connection(self) -> bool:
        """Validate connection to the LLM provider."""
        try:
            # Try a simple test request
            test_messages = [LLMMessage(role=MessageRole.USER, content="Hello")]

            await asyncio.wait_for(self._make_request(test_messages), timeout=10.0)

            self._is_available = True
            return True

        except Exception as e:
            logger.warning(
                f"Connection validation failed for {self.get_provider_name().value}: {e}"
            )
            self._is_available = False
            return False

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        This is a rough approximation. Subclasses should override
        with provider-specific tokenization if available.
        """
        # Rough approximation: 1 token ≈ 4 characters for English
        return len(text) // 4

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for token usage.

        Subclasses should override with provider-specific pricing.
        """
        # Default implementation returns 0
        return 0.0

    async def list_available_models(self) -> List[str]:
        """
        List available models for this provider.

        Subclasses should override with provider-specific implementation.
        """
        return [self.config.model]

    def supports_function_calling(self) -> bool:
        """
        Check if provider supports function calling.

        Subclasses should override based on capabilities.
        """
        return False

    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming.

        Subclasses should override based on capabilities.
        """
        return True

    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        logger.debug(f"Cleaning up provider: {self.get_provider_name().value}")
        # Base implementation does nothing
        pass

    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        return self._is_available

    def set_available(self, available: bool) -> None:
        """Set provider availability."""
        self._is_available = available
        if not available:
            logger.warning(f"Provider {self.get_provider_name().value} marked as unavailable")

    def get_request_count(self) -> int:
        """Get the number of requests made."""
        return self._request_count

    def get_total_tokens(self) -> int:
        """Get total tokens processed."""
        return self._total_tokens

    def get_total_cost(self) -> float:
        """Get total cost incurred."""
        return self._total_cost

    def get_last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error

    def reset_stats(self) -> None:
        """Reset provider statistics."""
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._last_error = None

    def _prepare_parameters(self, **kwargs) -> Dict[str, Any]:
        """Prepare parameters for the request."""
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }

        # Override with provided kwargs
        params.update(kwargs)

        # Add provider-specific extra params
        params.update(self.config.extra_params)

        return params

    async def _generate_with_retries(
        self, messages: List[LLMMessage], functions: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> LLMResponse:
        """Generate response with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                # Make the request
                response = await asyncio.wait_for(
                    self._make_request(messages, stream=False, functions=functions, **kwargs),
                    timeout=self.config.timeout,
                )

                # Parse the response
                llm_response = self._parse_response(response)
                llm_response.response_time_ms = (time.time() - start_time) * 1000

                # Update statistics
                self._request_count += 1
                self._total_tokens += llm_response.total_tokens
                self._total_cost += llm_response.metadata.get("cost", 0.0)
                self._last_error = None

                return llm_response

            except asyncio.TimeoutError:
                last_error = f"Request timed out after {self.config.timeout} seconds"
                logger.warning(
                    f"Provider {self.get_provider_name().value} timeout (attempt {attempt + 1}/{self.config.max_retries + 1})"
                )

                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2**attempt, 10))  # Exponential backoff

            except LLMRateLimitError as e:
                last_error = str(e)
                logger.warning(
                    f"Rate limit hit for {self.get_provider_name().value} (attempt {attempt + 1}/{self.config.max_retries + 1})"
                )

                if attempt < self.config.max_retries:
                    wait_time = e.retry_after or min(2**attempt, 30)
                    await asyncio.sleep(wait_time)

            except LLMError as e:
                last_error = str(e)
                if not e.recoverable:
                    break

                logger.warning(
                    f"Provider {self.get_provider_name().value} error: {e} (attempt {attempt + 1}/{self.config.max_retries + 1})"
                )

                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2**attempt, 10))

            except (ConnectionError, OSError) as e:
                last_error = f"Connection error: {str(e)}"
                logger.error(f"Connection error in provider {self.get_provider_name().value}: {e}")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2**attempt, 10))
                else:
                    break
            except (ValueError, TypeError) as e:
                last_error = f"Parameter error: {str(e)}"
                logger.error(f"Parameter error in provider {self.get_provider_name().value}: {e}")
                break  # Don't retry parameter errors
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error in provider {self.get_provider_name().value}: {e}")
                break

        # All attempts failed
        self._last_error = last_error
        raise LLMError(
            last_error or "Request failed after all retries", self.get_provider_name().value
        )

    async def _generate_streaming_with_retries(
        self, messages: List[LLMMessage], functions: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate streaming response with retry logic."""
        # For streaming, we typically don't retry as the stream may have started
        try:
            response = await asyncio.wait_for(
                self._make_request(messages, stream=True, functions=functions, **kwargs),
                timeout=self.config.timeout,
            )

            self._request_count += 1

            async for chunk in self._parse_streaming_response(response):
                yield chunk

        except Exception as e:
            error_msg = f"Streaming request failed: {str(e)}"
            self._last_error = error_msg
            raise LLMError(error_msg, self.get_provider_name().value)

    def _format_messages_for_api(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Format messages for API calls."""
        return [message.to_dict() for message in messages]

    def _create_response_id(self) -> str:
        """Create a unique response ID."""
        import uuid

        return str(uuid.uuid4())

    def _create_chunk_id(self) -> str:
        """Create a unique chunk ID."""
        import uuid

        return str(uuid.uuid4())
