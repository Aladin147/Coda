"""
Local model provider for Coda LLM system.

This module provides integration with local models via transformers.
"""

import logging
from typing import Dict, List, Any, Optional, AsyncGenerator

from ..base_provider import (
    BaseLLMProvider,
    LLMError,
)
from ..models import (
    LLMMessage,
    LLMResponse,
    LLMStreamChunk,
    LLMProvider,
    ProviderConfig,
    MessageRole,
)

logger = logging.getLogger("coda.llm.local")


class LocalModelProvider(BaseLLMProvider):
    """
    Local model provider using transformers.
    
    Features:
    - Support for local Hugging Face models
    - GPU acceleration if available
    - No external API dependencies
    - No cost calculation (local inference)
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize local model provider."""
        super().__init__(config)
        
        self._model = None
        self._tokenizer = None
        self._device = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the local model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("Using GPU for local model inference")
            else:
                self._device = "cpu"
                logger.info("Using CPU for local model inference")
            
            # Load tokenizer and model
            logger.info(f"Loading local model: {self.config.model}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None
            )
            
            # Add padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            logger.info("Local model initialized successfully")
            
        except ImportError:
            raise LLMError("Transformers library not installed. Install with: pip install transformers torch", "local")
        except Exception as e:
            raise LLMError(f"Failed to initialize local model: {e}", "local")
    
    def get_provider_name(self) -> LLMProvider:
        """Get the provider name."""
        return LLMProvider.LOCAL
    
    async def _make_request(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """Make request to local model."""
        try:
            # Format messages into a single prompt
            prompt = self._format_messages_to_prompt(messages)
            
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Adjust based on model context
            )
            
            if self._device == "cuda":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens or 512),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "do_sample": True,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            
            # Generate response
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            return {
                "inputs": inputs,
                "outputs": outputs,
                "prompt_length": inputs["input_ids"].shape[1]
            }
            
        except Exception as e:
            raise LLMError(f"Local model inference failed: {e}", "local")
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse local model response to LLMResponse."""
        try:
            # Decode the generated text
            generated_ids = response["outputs"][0][response["prompt_length"]:]
            content = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Calculate token counts
            prompt_tokens = response["prompt_length"]
            completion_tokens = len(generated_ids)
            total_tokens = prompt_tokens + completion_tokens
            
            return LLMResponse(
                response_id=self._create_response_id(),
                content=content.strip(),
                provider=self.get_provider_name(),
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason="stop",
                metadata={
                    "cost": 0.0,  # Local inference is free
                    "model": self.config.model,
                    "device": self._device,
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse local model response: {e}")
            raise LLMError(f"Failed to parse response: {e}", "local")
    
    async def _parse_streaming_response(self, response: Any) -> AsyncGenerator[LLMStreamChunk, None]:
        """Parse local model streaming response."""
        # Local models don't typically support streaming in this implementation
        # This would require more complex implementation with text generation streaming
        
        # For now, just yield the complete response as a single chunk
        parsed_response = self._parse_response(response)
        
        yield LLMStreamChunk(
            chunk_id=self._create_chunk_id(),
            content=parsed_response.content,
            delta=parsed_response.content,
            finish_reason="stop",
            metadata={
                "model": self.config.model,
                "provider": "local"
            }
        )
    
    def _format_messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Format messages into a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {message.content}")
            elif message.role == MessageRole.USER:
                prompt_parts.append(f"User: {message.content}")
            elif message.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {message.content}")
        
        # Add assistant prompt for generation
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using the local tokenizer."""
        try:
            if self._tokenizer:
                tokens = self._tokenizer.encode(text)
                return len(tokens)
            else:
                return super().estimate_tokens(text)
        except Exception:
            return super().estimate_tokens(text)
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for local model usage (always 0)."""
        return 0.0
    
    async def list_available_models(self) -> List[str]:
        """List available local models."""
        # This would typically list models from a local directory or cache
        return [self.config.model]
    
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        # Most local models don't support structured function calling
        return False
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        # Basic implementation doesn't support true streaming
        return False
    
    async def validate_connection(self) -> bool:
        """Validate local model availability."""
        try:
            if self._model is not None and self._tokenizer is not None:
                self.set_available(True)
                return True
            else:
                self.set_available(False)
                return False
                
        except Exception as e:
            logger.warning(f"Local model validation failed: {e}")
            self.set_available(False)
            return False
    
    async def cleanup(self) -> None:
        """Cleanup local model resources."""
        try:
            if self._model is not None:
                # Move model to CPU to free GPU memory
                if hasattr(self._model, 'cpu'):
                    self._model.cpu()
                
                # Clear CUDA cache if using GPU
                if self._device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                
                logger.info("Local model resources cleaned up")
                
        except Exception as e:
            logger.warning(f"Error during local model cleanup: {e}")
        
        await super().cleanup()
