"""
Ollama provider for Coda LLM system.

This module provides integration with Ollama for local model inference.
"""

import json
import logging
import aiohttp
from typing import Dict, List, Any, Optional, AsyncGenerator

from ..base_provider import (
    BaseLLMProvider,
    LLMError,
    LLMTimeoutError,
)
from ..models import (
    LLMMessage,
    LLMResponse,
    LLMStreamChunk,
    LLMProvider,
    ProviderConfig,
    MessageRole,
)

logger = logging.getLogger("coda.llm.ollama")


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local models.
    
    Features:
    - Support for local Ollama models
    - Streaming responses
    - No cost calculation (local inference)
    - Custom model support
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize Ollama provider."""
        super().__init__(config)
        
        # Default to local Ollama instance
        self.base_url = config.api_base or "http://localhost:11434"
        
        logger.info(f"Ollama provider initialized with base URL: {self.base_url}")
    
    def get_provider_name(self) -> LLMProvider:
        """Get the provider name."""
        return LLMProvider.OLLAMA
    
    async def _make_request(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """Make request to Ollama API."""
        try:
            # Format messages for Ollama
            formatted_messages = self._format_messages_for_api(messages)
            
            # Prepare request payload
            payload = {
                "model": self.config.model,
                "messages": formatted_messages,
                "stream": stream,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                }
            }
            
            # Add max_tokens if specified
            if "max_tokens" in kwargs or self.config.max_tokens:
                payload["options"]["num_predict"] = kwargs.get("max_tokens", self.config.max_tokens)
            
            # Make HTTP request
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/chat"
                
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Ollama API error: {response.status} - {error_text}", "ollama")
                    
                    if stream:
                        return response
                    else:
                        return await response.json()
                        
        except aiohttp.ClientTimeout:
            raise LLMTimeoutError(f"Ollama request timed out after {self.config.timeout} seconds", "ollama")
        except Exception as e:
            raise LLMError(f"Ollama request failed: {e}", "ollama")
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Ollama response to LLMResponse."""
        try:
            # Extract content
            content = response.get("message", {}).get("content", "")
            
            # Extract token usage if available
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens
            
            return LLMResponse(
                response_id=self._create_response_id(),
                content=content,
                provider=self.get_provider_name(),
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason="stop",
                metadata={
                    "cost": 0.0,  # Local inference is free
                    "model": self.config.model,
                    "eval_duration": response.get("eval_duration", 0),
                    "load_duration": response.get("load_duration", 0),
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            raise LLMError(f"Failed to parse response: {e}", "ollama")
    
    async def _parse_streaming_response(self, response: Any) -> AsyncGenerator[LLMStreamChunk, None]:
        """Parse Ollama streaming response."""
        try:
            accumulated_content = ""
            
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        
                        if "message" in data and "content" in data["message"]:
                            delta = data["message"]["content"]
                            accumulated_content += delta
                            
                            finish_reason = None
                            if data.get("done", False):
                                finish_reason = "stop"
                            
                            yield LLMStreamChunk(
                                chunk_id=self._create_chunk_id(),
                                content=accumulated_content,
                                delta=delta,
                                finish_reason=finish_reason,
                                metadata={
                                    "model": self.config.model,
                                    "provider": "ollama"
                                }
                            )
                            
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
                        
        except Exception as e:
            logger.error(f"Failed to parse Ollama streaming response: {e}")
            raise LLMError(f"Failed to parse streaming response: {e}", "ollama")
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for Ollama usage (always 0 for local inference)."""
        return 0.0
    
    async def list_available_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/tags"
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
                    else:
                        logger.warning(f"Failed to list Ollama models: {response.status}")
                        return [self.config.model]
                        
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return [self.config.model]
    
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        # Most Ollama models don't support function calling yet
        return False
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return True
    
    async def validate_connection(self) -> bool:
        """Validate connection to Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/tags"
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        self.set_available(True)
                        return True
                    else:
                        self.set_available(False)
                        return False
                        
        except Exception as e:
            logger.warning(f"Ollama connection validation failed: {e}")
            self.set_available(False)
            return False
