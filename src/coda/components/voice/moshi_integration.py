"""
Kyutai Moshi integration for Coda 2.0 voice system.

This module provides integration with Kyutai Moshi for real - time speech processing,
including client wrapper, WebSocket support, real - time audio streaming, and conversation management.
"""

import asyncio
import logging
import queue
import threading
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# Try to import real Moshi library
try:
    import moshi
    from moshi import models as moshi_models

    MOSHI_AVAILABLE = True
    logger = logging.getLogger("coda.voice.moshi_integration")
    logger.info("Real Moshi library available")
except ImportError:
    MOSHI_AVAILABLE = False
    logger = logging.getLogger("coda.voice.moshi_integration")
    logger.warning("Moshi library not available, using fallback implementation")

# Try to import transformers for model loading
try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")

from .conversation_state import ConversationPhase, ConversationStateManager
from .exceptions import (
    ComponentFailureError,
    ComponentNotInitializedError,
    ErrorCodes,
    ModelLoadingError,
    MoshiError,
    NetworkError,
    VRAMAllocationError,
    create_error,
    wrap_exception,
)
from .inner_monologue import ExtractedText, InnerMonologueProcessor
from .interfaces import MoshiInterface, VoiceProcessorInterface
from .models import (
    ConversationState,
    MoshiConfig,
    VoiceAnalytics,
    VoiceConfig,
    VoiceMessage,
    VoiceProcessingMode,
    VoiceResponse,
    VoiceStreamChunk,
)
from .resource_management import (
    CircuitBreaker,
    async_resource_cleanup,
    with_retry,
    with_timeout,
)
from .utils import LatencyTracker, track_latency
from .validation import (
    validate_audio_data,
    validate_conversation_id,
    validate_timeout,
    validate_voice_config,
)
from .vram_manager import get_vram_manager

# Update logger if not already set
if "logger" not in locals():
    logger = logging.getLogger(__name__)


class MoshiClient:
    """Client wrapper for Kyutai Moshi."""

    def __init__(self, config: MoshiConfig):
        """Initialize Moshi client."""
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.is_conversation_active = False
        self.current_conversation_id: Optional[str] = None

        # Performance tracking
        self.latency_tracker = LatencyTracker("moshi_processing")
        self.total_processed_chunks = 0

        # VRAM management
        self.vram_manager = get_vram_manager()
        self.vram_allocated = False

        # Inner monologue processor
        self.inner_monologue: Optional[InnerMonologueProcessor] = None

        logger.info(f"MoshiClient initialized for device: {self.device}")

    async def initialize(self) -> None:
        """Initialize Moshi model and components."""
        try:
            logger.info("Initializing Moshi model...")

            # Register VRAM allocation
            if self.vram_manager:
                vram_mb = float(self.config.vram_allocation.replace("GB", "")) * 1024
                success = self.vram_manager.register_component(
                    component_id="moshi_client",
                    max_mb=vram_mb,
                    priority=8,  # High priority
                    can_resize=False,  # Moshi needs fixed allocation
                )
                if not success:
                    raise create_error(
                        VRAMAllocationError,
                        f"Failed to register VRAM allocation for Moshi: {vram_mb}MB",
                        ErrorCodes.VRAM_ALLOCATION_FAILED,
                        requested_vram_mb=vram_mb,
                    )

            # Import Moshi components
            try:
                import os

                import moshi
                import moshi.models

                # Set cache directory to our workspace
                cache_dir = os.path.join(os.getcwd(), "models", "moshi")
                os.makedirs(cache_dir, exist_ok=True)

                # Set environment variables for model caching
                os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), "models", "torch")
                os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models", "huggingface")

                logger.info(f"Loading Moshi model to cache: {cache_dir}")

                # Load the language model and compression model
                # For Moshi, we need both the LM and the compression model (Mimi)
                model_path = self.config.model_path if self.config.model_path else None

                try:
                    self.lm_model = moshi.models.get_moshi_lm(
                        filename=model_path,
                        device=self.device,
                        dtype=(
                            torch.bfloat16 if self.config.optimization == "bf16" else torch.float32
                        ),
                    )
                except Exception as e:
                    raise create_error(
                        ModelLoadingError,
                        f"Failed to load Moshi language model: {str(e)}",
                        ErrorCodes.MODEL_LOAD_FAILED,
                        model_path=model_path,
                        device=self.device,
                        original_error=str(e),
                    )

                # Load the compression model (Mimi)
                try:
                    self.compression_model = moshi.models.get_mimi(
                        filename=None, device=self.device  # Use default Mimi model
                    )
                except Exception as e:
                    raise create_error(
                        ModelLoadingError,
                        f"Failed to load Mimi compression model: {str(e)}",
                        ErrorCodes.MODEL_LOAD_FAILED,
                        device=self.device,
                        original_error=str(e),
                    )

                # For compatibility, set self.model to the LM model
                self.model = self.lm_model

                # Allocate VRAM
                if self.vram_manager:
                    allocated = self.vram_manager.allocate("moshi_client", vram_mb)
                    if allocated:
                        self.vram_allocated = True
                        logger.info(f"Allocated {vram_mb:.0f}MB VRAM for Moshi")
                    else:
                        logger.warning("Failed to allocate VRAM for Moshi")

                # Move model to device
                self.model = self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode

                # Initialize tokenizer if available
                try:
                    # Check if the LM model has a tokenizer
                    if hasattr(self.lm_model, "tokenizer"):
                        self.tokenizer = self.lm_model.tokenizer
                        logger.info("Moshi tokenizer loaded from LM model")
                    else:
                        self.tokenizer = None
                        logger.info("No tokenizer found in LM model")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer: {e}")
                    self.tokenizer = None

                # Initialize inner monologue processor if enabled
                if self.config.inner_monologue_enabled:
                    self.inner_monologue = InnerMonologueProcessor(self.config)
                    await self.inner_monologue.initialize(self.model, self.tokenizer)
                    logger.info("Inner monologue processor initialized")

                self.is_initialized = True
                logger.info("Moshi model initialized successfully")

            except ImportError as e:
                raise create_error(
                    ModelLoadingError,
                    f"Moshi not properly installed: {str(e)}",
                    ErrorCodes.MODEL_NOT_FOUND,
                    original_error=str(e),
                )

        except Exception as e:
            logger.error(f"Failed to initialize Moshi: {e}")
            # Clean up VRAM allocation on failure
            if self.vram_manager and self.vram_allocated:
                self.vram_manager.deallocate("moshi_client")
                self.vram_allocated = False

            # Re-raise as appropriate error type
            if isinstance(e, (ModelLoadingError, VRAMAllocationError)):
                raise
            else:
                raise wrap_exception(
                    e,
                    ComponentFailureError,
                    "Failed to initialize Moshi client",
                    ErrorCodes.COMPONENT_INITIALIZATION_FAILED,
                )

    async def start_conversation(self, conversation_id: str) -> None:
        """Start a Moshi conversation."""
        # Validate inputs
        validate_conversation_id(conversation_id)

        if not self.is_initialized:
            raise create_error(
                ComponentNotInitializedError,
                "Moshi client not initialized",
                ErrorCodes.COMPONENT_NOT_INITIALIZED,
            )

        if self.is_conversation_active:
            logger.warning(f"Ending previous conversation {self.current_conversation_id}")
            await self.end_conversation(self.current_conversation_id)

        self.current_conversation_id = conversation_id
        self.is_conversation_active = True

        logger.info(f"Started Moshi conversation: {conversation_id}")

    async def process_audio(self, audio_data: bytes) -> bytes:
        """Process audio through Moshi."""
        if not self.is_initialized:
            raise RuntimeError("Moshi client not initialized")

        if not self.is_conversation_active:
            raise RuntimeError("No active conversation")

        with track_latency(self.latency_tracker) as tracker:
            try:
                # Convert audio bytes to tensor
                audio_tensor = self._bytes_to_tensor(audio_data)

                # Process through Moshi model
                with torch.no_grad():
                    # This is a simplified processing - actual Moshi API may differ
                    processed_audio = await self._process_audio_tensor(audio_tensor)

                # Convert back to bytes
                output_audio = self._tensor_to_bytes(processed_audio)

                self.total_processed_chunks += 1

                logger.debug(f"Processed audio chunk, latency: {tracker.get_latency():.1f}ms")
                return output_audio

            except Exception as e:
                logger.error(f"Audio processing failed: {e}")
                # Return original audio as fallback
                return audio_data

    async def process_audio_stream(
        self, audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """Process streaming audio through Moshi."""
        if not self.is_initialized:
            raise RuntimeError("Moshi client not initialized")

        if not self.is_conversation_active:
            raise RuntimeError("No active conversation")

        try:
            async for audio_chunk in audio_stream:
                processed_chunk = await self.process_audio(audio_chunk)
                yield processed_chunk

        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise

    async def extract_text(self, audio_data: bytes) -> str:
        """Extract text from Moshi's inner monologue."""
        if not self.is_initialized:
            raise RuntimeError("Moshi client not initialized")

        try:
            # Use inner monologue processor if available
            if self.inner_monologue:
                extracted_text = await self.inner_monologue.extract_text_from_audio(audio_data)
                if extracted_text:
                    logger.debug(
                        f"Extracted text: {extracted_text.text[:100]}... (confidence: {extracted_text.confidence:.3f})"
                    )
                    return extracted_text.text
                else:
                    logger.debug("No text extracted from inner monologue")
                    return ""
            else:
                # Fallback to basic extraction
                audio_tensor = self._bytes_to_tensor(audio_data)
                with torch.no_grad():
                    text = await self._extract_text_from_audio(audio_tensor)

                logger.debug(f"Extracted text (fallback): {text[:100]}...")
                return text

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

    async def inject_text(self, text: str) -> bytes:
        """Inject text into Moshi for speech synthesis."""
        if not self.is_initialized:
            raise RuntimeError("Moshi client not initialized")

        try:
            # Convert text to tokens if tokenizer available
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
            else:
                # Fallback: simple text processing
                tokens = text.encode("utf - 8")

            # Generate audio from text using Moshi
            with torch.no_grad():
                audio_tensor = await self._synthesize_from_text(tokens)

            # Convert to bytes
            audio_data = self._tensor_to_bytes(audio_tensor)

            logger.debug(f"Synthesized audio for text: {text[:50]}...")
            return audio_data

        except Exception as e:
            logger.error(f"Text injection failed: {e}")
            # Return silence as fallback
            return b"\x00" * 1600  # 100ms of silence at 16kHz

    async def get_conversation_state(self) -> Dict[str, Any]:
        """Get Moshi conversation state."""
        state = {
            "conversation_id": self.current_conversation_id,
            "is_active": self.is_conversation_active,
            "is_initialized": self.is_initialized,
            "device": str(self.device),
            "total_processed_chunks": self.total_processed_chunks,
            "latency_stats": self.latency_tracker.get_stats(),
            "vram_allocated": self.vram_allocated,
            "inner_monologue_enabled": self.config.inner_monologue_enabled,
        }

        # Add inner monologue statistics if available
        if self.inner_monologue:
            state["inner_monologue_stats"] = self.inner_monologue.get_statistics()

        return state

    async def end_conversation(self, conversation_id: str) -> None:
        """End Moshi conversation."""
        if self.current_conversation_id == conversation_id:
            self.is_conversation_active = False
            self.current_conversation_id = None
            logger.info(f"Ended Moshi conversation: {conversation_id}")
        else:
            logger.warning(
                f"Conversation ID mismatch: {conversation_id} vs {self.current_conversation_id}"
            )

    async def cleanup(self) -> None:
        """Cleanup Moshi client resources."""
        try:
            if self.is_conversation_active:
                await self.end_conversation(self.current_conversation_id)

            # Deallocate VRAM
            if self.vram_manager and self.vram_allocated:
                self.vram_manager.deallocate("moshi_client")
                self.vram_manager.unregister_component("moshi_client")
                self.vram_allocated = False

            # Clear model from memory
            if self.model is not None:
                del self.model
                self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_initialized = False
            logger.info("Moshi client cleaned up")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    # Private helper methods

    def _bytes_to_tensor(self, audio_data: bytes) -> torch.Tensor:
        """Convert audio bytes to tensor."""
        try:
            # Convert bytes to numpy array (assuming 16 - bit PCM)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # Convert to float32 and normalize
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Convert to tensor and move to device
            audio_tensor = torch.from_numpy(audio_float).to(self.device)

            # Ensure correct shape for Moshi (add batch dimension if needed)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            return audio_tensor

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            # Return empty tensor
            return torch.zeros(1, 1600).to(self.device)

    def _tensor_to_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert tensor to audio bytes."""
        try:
            # Move to CPU and convert to numpy
            audio_np = audio_tensor.cpu().numpy()

            # Remove batch dimension if present
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

            # Convert to 16 - bit PCM
            audio_int16 = (audio_np * 32767).astype(np.int16)

            return audio_int16.tobytes()

        except Exception as e:
            logger.error(f"Tensor conversion failed: {e}")
            return b"\x00" * 1600  # Return silence

    async def _process_audio_tensor(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Process audio tensor through Moshi model."""
        try:
            if not self.is_initialized or not self.model:
                logger.warning("Model not initialized, returning input audio")
                return audio_tensor

            # Ensure audio tensor is on the correct device and has correct shape
            audio_tensor = audio_tensor.to(self.device)

            # Moshi expects audio in specific format: [batch, channels, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dim

            # Ensure we have the right sample rate (24kHz for Moshi)
            expected_sample_rate = 24000
            if (
                hasattr(self.config, "sample_rate")
                and self.config.sample_rate != expected_sample_rate
            ):
                # Resample if needed
                current_sample_rate = self.config.sample_rate
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, orig_freq=current_sample_rate, new_freq=expected_sample_rate
                )

            with torch.no_grad():
                if (
                    MOSHI_AVAILABLE
                    and hasattr(self, "compression_model")
                    and self.compression_model
                ):
                    # Use real Moshi processing pipeline
                    try:
                        # First, compress audio using Mimi
                        compressed = self.compression_model.encode(audio_tensor)

                        # Process through Moshi LM if available
                        if hasattr(self, "lm_model") and self.lm_model:
                            # This would be the actual Moshi conversation processing
                            # The exact API depends on the Moshi implementation
                            processed = await self._run_moshi_conversation(compressed)
                        else:
                            processed = compressed

                        # Decompress back to audio
                        output_audio = self.compression_model.decode(processed)

                        logger.debug(f"Processed audio through real Moshi pipeline")
                        return output_audio.squeeze(0)  # Remove batch dim

                    except Exception as e:
                        logger.warning(f"Real Moshi processing failed, using fallback: {e}")
                        # Fall through to fallback processing

                # Fallback processing: apply some basic audio enhancement
                processed_audio = self._apply_audio_enhancement(audio_tensor)
                return processed_audio.squeeze(0)  # Remove batch dim

        except Exception as e:
            logger.error(f"Audio tensor processing failed: {e}")
            return audio_tensor

    async def _extract_text_from_audio(self, audio_tensor: torch.Tensor) -> str:
        """Extract text from audio using Moshi's inner monologue."""
        try:
            if not self.is_initialized or not self.model:
                logger.warning("Model not initialized for text extraction")
                return ""

            # Use inner monologue processor if available
            if self.inner_monologue:
                try:
                    # Convert tensor to bytes for inner monologue processor
                    audio_bytes = self._tensor_to_bytes(audio_tensor)
                    extracted_text = await self.inner_monologue.extract_text_from_audio(audio_bytes)

                    if extracted_text and extracted_text.text:
                        logger.debug(
                            f"Extracted text via inner monologue: {extracted_text.text[:100]}..."
                        )
                        return extracted_text.text
                    else:
                        logger.debug("No text extracted via inner monologue")

                except Exception as e:
                    logger.warning(f"Inner monologue extraction failed: {e}")

            # Fallback: Direct model-based text extraction
            if MOSHI_AVAILABLE and hasattr(self, "lm_model") and self.lm_model:
                try:
                    # Prepare audio for Moshi LM
                    audio_tensor = audio_tensor.to(self.device)
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                    elif audio_tensor.dim() == 2:
                        audio_tensor = audio_tensor.unsqueeze(0)

                    with torch.no_grad():
                        # Compress audio first if compression model available
                        if hasattr(self, "compression_model") and self.compression_model:
                            compressed = self.compression_model.encode(audio_tensor)
                        else:
                            compressed = audio_tensor

                        # Extract text using Moshi's inner monologue capability
                        # This is a simplified approach - actual API may differ
                        if hasattr(self.lm_model, "extract_text"):
                            extracted_text = self.lm_model.extract_text(compressed)
                        elif hasattr(self.lm_model, "decode_text"):
                            extracted_text = self.lm_model.decode_text(compressed)
                        else:
                            # Generic approach: use model to generate text tokens
                            text_tokens = self.lm_model.generate(
                                compressed, max_length=100, do_sample=True, temperature=0.7
                            )

                            # Decode tokens to text if tokenizer available
                            if self.tokenizer:
                                extracted_text = self.tokenizer.decode(
                                    text_tokens[0], skip_special_tokens=True
                                )
                            else:
                                extracted_text = f"Generated tokens: {text_tokens.shape}"

                        logger.debug(f"Extracted text via Moshi LM: {extracted_text[:100]}...")
                        return extracted_text

                except Exception as e:
                    logger.warning(f"Moshi LM text extraction failed: {e}")

            # Final fallback: return empty string
            logger.debug("No text extraction method available")
            return ""

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

    async def _synthesize_from_text(self, tokens: Any) -> torch.Tensor:
        """Synthesize audio from text tokens."""
        try:
            if not self.is_initialized or not self.model:
                logger.warning("Model not initialized for text synthesis")
                return torch.zeros(1, 1600).to(self.device)

            # Handle different input types
            if isinstance(tokens, str):
                text = tokens
                # Encode text to tokens if tokenizer available
                if self.tokenizer:
                    token_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                else:
                    # Fallback: use simple byte encoding
                    token_ids = (
                        torch.tensor([ord(c) for c in text[:100]], dtype=torch.long)
                        .unsqueeze(0)
                        .to(self.device)
                    )
            elif isinstance(tokens, bytes):
                # Decode bytes to string first
                text = tokens.decode("utf-8", errors="ignore")
                if self.tokenizer:
                    token_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                else:
                    token_ids = (
                        torch.tensor([ord(c) for c in text[:100]], dtype=torch.long)
                        .unsqueeze(0)
                        .to(self.device)
                    )
            else:
                # Assume it's already tensor tokens
                token_ids = tokens
                if not isinstance(token_ids, torch.Tensor):
                    token_ids = torch.tensor(token_ids, dtype=torch.long).to(self.device)
                if token_ids.dim() == 1:
                    token_ids = token_ids.unsqueeze(0)

            with torch.no_grad():
                if MOSHI_AVAILABLE and hasattr(self, "lm_model") and self.lm_model:
                    try:
                        # Use real Moshi text-to-speech synthesis
                        if hasattr(self.lm_model, "synthesize"):
                            # Direct synthesis method
                            audio_tensor = self.lm_model.synthesize(token_ids)
                        elif hasattr(self.lm_model, "generate_audio"):
                            # Alternative synthesis method
                            audio_tensor = self.lm_model.generate_audio(token_ids)
                        else:
                            # Generic generation approach
                            # Generate audio tokens/features from text tokens
                            audio_features = self.lm_model.generate(
                                token_ids,
                                max_length=token_ids.shape[1] + 200,  # Allow for audio expansion
                                do_sample=True,
                                temperature=0.8,
                                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
                            )

                            # Convert features to audio if compression model available
                            if hasattr(self, "compression_model") and self.compression_model:
                                # Assume audio_features are compressed audio tokens
                                audio_tensor = self.compression_model.decode(audio_features)
                            else:
                                # Fallback: create synthetic audio from features
                                audio_tensor = self._features_to_audio(audio_features)

                        # Ensure output is correct shape and on CPU for return
                        if audio_tensor.dim() > 2:
                            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dim if present

                        logger.debug(f"Synthesized audio via Moshi LM, shape: {audio_tensor.shape}")
                        return audio_tensor

                    except Exception as e:
                        logger.warning(f"Moshi synthesis failed, using fallback: {e}")

                # Fallback synthesis: generate synthetic speech-like audio
                audio_length = min(
                    len(token_ids[0]) * 100, 24000
                )  # ~100 samples per token, max 1 second
                audio_tensor = self._generate_synthetic_speech(token_ids, audio_length)

                logger.debug(f"Generated synthetic speech, length: {audio_length}")
                return audio_tensor

        except Exception as e:
            logger.error(f"Text synthesis failed: {e}")
            # Return silence as fallback
            return torch.zeros(1, 1600).to(self.device)

    async def _run_moshi_conversation(self, compressed_audio: torch.Tensor) -> torch.Tensor:
        """Run Moshi conversation processing on compressed audio."""
        try:
            if not hasattr(self, "lm_model") or not self.lm_model:
                return compressed_audio

            # This would implement the actual Moshi conversation flow
            # The exact implementation depends on Moshi's API
            with torch.no_grad():
                if hasattr(self.lm_model, "conversation_step"):
                    result = self.lm_model.conversation_step(compressed_audio)
                elif hasattr(self.lm_model, "forward"):
                    result = self.lm_model.forward(compressed_audio)
                else:
                    # Fallback: return input
                    result = compressed_audio

                return result

        except Exception as e:
            logger.warning(f"Moshi conversation processing failed: {e}")
            return compressed_audio

    def _apply_audio_enhancement(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply basic audio enhancement as fallback processing."""
        try:
            # Basic audio enhancement: normalize and apply slight filtering
            audio = audio_tensor.clone()

            # Normalize audio
            if audio.abs().max() > 0:
                audio = audio / audio.abs().max() * 0.8

            # Apply simple low-pass filter to reduce noise
            if audio.shape[-1] > 10:
                # Simple moving average filter
                kernel_size = 3
                kernel = torch.ones(1, 1, kernel_size) / kernel_size
                kernel = kernel.to(audio.device)

                # Pad audio for convolution
                padded_audio = F.pad(
                    audio.unsqueeze(0), (kernel_size // 2, kernel_size // 2), mode="reflect"
                )
                filtered_audio = F.conv1d(padded_audio, kernel, padding=0)
                audio = filtered_audio.squeeze(0)

            return audio

        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio_tensor

    def _features_to_audio(self, features: torch.Tensor) -> torch.Tensor:
        """Convert model features to audio tensor."""
        try:
            # This is a simplified conversion - real implementation would depend on model architecture
            if features.dim() > 2:
                features = features.squeeze(0)  # Remove batch dim

            # Convert features to audio samples
            # Assume features are in some latent space that needs to be converted to audio
            if features.shape[-1] > 1000:
                # If features are already audio-like, use directly
                audio = features[:, :24000]  # Limit to 1 second at 24kHz
            else:
                # Expand features to audio length
                repeat_factor = max(1, 24000 // features.shape[-1])
                audio = features.repeat(1, repeat_factor)[:, :24000]

            # Normalize
            if audio.abs().max() > 0:
                audio = audio / audio.abs().max() * 0.5

            return audio

        except Exception as e:
            logger.warning(f"Features to audio conversion failed: {e}")
            return torch.zeros(1, 1600).to(features.device)

    def _generate_synthetic_speech(
        self, token_ids: torch.Tensor, audio_length: int
    ) -> torch.Tensor:
        """Generate synthetic speech-like audio from tokens."""
        try:
            # Create synthetic speech by generating audio patterns based on tokens
            sample_rate = 24000
            duration = audio_length / sample_rate

            # Generate base frequency from tokens
            token_sum = token_ids.sum().item() if token_ids.numel() > 0 else 100
            base_freq = 80 + (token_sum % 200)  # Frequency between 80-280 Hz

            # Create time vector
            t = torch.linspace(0, duration, audio_length).to(self.device)

            # Generate synthetic speech with formants
            audio = torch.zeros(audio_length).to(self.device)

            # Add multiple harmonics for speech-like quality
            for i, token_id in enumerate(token_ids[0][:10]):  # Use first 10 tokens
                freq = base_freq * (1 + i * 0.1) + (token_id.item() % 50)
                amplitude = 0.1 / (i + 1)  # Decreasing amplitude for harmonics
                phase = (token_id.item() % 100) / 100 * 2 * np.pi

                # Add harmonic
                harmonic = amplitude * torch.sin(2 * np.pi * freq * t + phase)
                audio += harmonic

            # Apply envelope to make it more speech-like
            envelope = torch.exp(-t * 2)  # Exponential decay
            audio = audio * envelope

            # Normalize
            if audio.abs().max() > 0:
                audio = audio / audio.abs().max() * 0.3

            return audio.unsqueeze(0)  # Add channel dimension

        except Exception as e:
            logger.warning(f"Synthetic speech generation failed: {e}")
            return torch.zeros(1, audio_length).to(self.device)


class MoshiStreamingManager:
    """Manages real - time audio streaming for Moshi."""

    def __init__(self, moshi_client: MoshiClient, buffer_size: int = 10):
        """Initialize streaming manager."""
        self.moshi_client = moshi_client
        self.buffer_size = buffer_size

        # Input / output buffers
        self.input_buffer = queue.Queue(maxsize=buffer_size)
        self.output_buffer = queue.Queue(maxsize=buffer_size)

        # Streaming state
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.stream_lock = threading.Lock()

        # Callbacks
        self.output_callback: Optional[Callable[[bytes], None]] = None
        self.error_callback: Optional[Callable[[Exception], None]] = None

        # Statistics
        self.total_input_chunks = 0
        self.total_output_chunks = 0
        self.dropped_input_chunks = 0
        self.dropped_output_chunks = 0

        logger.info("MoshiStreamingManager initialized")

    def set_output_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for processed audio output."""
        self.output_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for streaming errors."""
        self.error_callback = callback

    async def start_streaming(self) -> None:
        """Start real - time audio streaming."""
        with self.stream_lock:
            if self.is_streaming:
                logger.warning("Streaming already active")
                return

            self.is_streaming = True

            # Start streaming thread
            self.stream_thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self.stream_thread.start()

            logger.info("Moshi streaming started")

    async def stop_streaming(self) -> None:
        """Stop real - time audio streaming."""
        with self.stream_lock:
            if not self.is_streaming:
                return

            self.is_streaming = False

            # Wait for thread to finish
            if self.stream_thread:
                self.stream_thread.join(timeout=5.0)
                if self.stream_thread.is_alive():
                    logger.warning("Streaming thread did not stop gracefully")

            # Clear buffers
            self._clear_buffers()

            logger.info("Moshi streaming stopped")

    def queue_input_audio(self, audio_data: bytes) -> bool:
        """Queue audio data for processing."""
        if not self.is_streaming:
            return False

        try:
            self.input_buffer.put_nowait(audio_data)
            self.total_input_chunks += 1
            return True
        except queue.Full:
            self.dropped_input_chunks += 1
            logger.warning("Input buffer full, dropping audio chunk")
            return False

    def get_output_audio(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get processed audio output."""
        if not self.is_streaming:
            return None

        try:
            return self.output_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    async def stream_input(self, audio_stream: AsyncGenerator[bytes, None]) -> None:
        """Stream input audio for processing."""
        try:
            async for audio_chunk in audio_stream:
                if not self.is_streaming:
                    break

                success = self.queue_input_audio(audio_chunk)
                if not success:
                    logger.warning("Failed to queue input audio chunk")

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Input streaming failed: {e}")
            if self.error_callback:
                self.error_callback(e)

    async def stream_output(self) -> AsyncGenerator[bytes, None]:
        """Stream processed audio output."""
        try:
            while self.is_streaming:
                output_audio = self.get_output_audio(timeout=0.1)
                if output_audio:
                    yield output_audio
                else:
                    # Small delay when no output available
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Output streaming failed: {e}")
            if self.error_callback:
                self.error_callback(e)

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "is_streaming": self.is_streaming,
            "total_input_chunks": self.total_input_chunks,
            "total_output_chunks": self.total_output_chunks,
            "dropped_input_chunks": self.dropped_input_chunks,
            "dropped_output_chunks": self.dropped_output_chunks,
            "input_buffer_size": self.input_buffer.qsize(),
            "output_buffer_size": self.output_buffer.qsize(),
            "drop_rate_input": self.dropped_input_chunks / max(self.total_input_chunks, 1),
            "drop_rate_output": self.dropped_output_chunks / max(self.total_output_chunks, 1),
        }

    def _streaming_loop(self) -> None:
        """Main streaming processing loop."""
        logger.info("Streaming loop started")

        while self.is_streaming:
            try:
                # Get input audio
                try:
                    input_audio = self.input_buffer.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process through Moshi (synchronous call in thread)
                try:
                    # Create event loop for async call
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    output_audio = loop.run_until_complete(
                        self.moshi_client.process_audio(input_audio)
                    )

                    loop.close()

                    # Queue output
                    try:
                        self.output_buffer.put_nowait(output_audio)
                        self.total_output_chunks += 1

                        # Call output callback if set
                        if self.output_callback:
                            self.output_callback(output_audio)

                    except queue.Full:
                        self.dropped_output_chunks += 1
                        logger.warning("Output buffer full, dropping processed audio")

                except Exception as e:
                    logger.error(f"Audio processing error in streaming loop: {e}")
                    if self.error_callback:
                        self.error_callback(e)

            except Exception as e:
                logger.error(f"Streaming loop error: {e}")
                if self.error_callback:
                    self.error_callback(e)

                # Small delay before retrying
                time.sleep(0.1)

        logger.info("Streaming loop ended")

    def _clear_buffers(self) -> None:
        """Clear input and output buffers."""
        while not self.input_buffer.empty():
            try:
                self.input_buffer.get_nowait()
            except queue.Empty:
                break

        while not self.output_buffer.empty():
            try:
                self.output_buffer.get_nowait()
            except queue.Empty:
                break


class MoshiWebSocketHandler:
    """WebSocket handler for Moshi real - time communication."""

    def __init__(self, moshi_client: MoshiClient):
        """Initialize WebSocket handler."""
        self.moshi_client = moshi_client
        self.streaming_manager = MoshiStreamingManager(moshi_client)
        self.websocket_connections: Dict[str, Any] = {}
        self.is_active = False

        logger.info("MoshiWebSocketHandler initialized")

    async def start_websocket_session(self, session_id: str, websocket: Any) -> None:
        """Start WebSocket session for real - time communication."""
        try:
            self.websocket_connections[session_id] = websocket
            self.is_active = True

            # Set up streaming callbacks
            self.streaming_manager.set_output_callback(
                lambda audio: self._send_audio_to_websocket(session_id, audio)
            )

            # Start streaming
            await self.streaming_manager.start_streaming()

            logger.info(f"Started WebSocket session: {session_id}")

        except Exception as e:
            logger.error(f"Failed to start WebSocket session: {e}")
            raise

    async def stop_websocket_session(self, session_id: str) -> None:
        """Stop WebSocket session."""
        try:
            if session_id in self.websocket_connections:
                del self.websocket_connections[session_id]

            if not self.websocket_connections:
                await self.streaming_manager.stop_streaming()
                self.is_active = False

            logger.info(f"Stopped WebSocket session: {session_id}")

        except Exception as e:
            logger.error(f"Failed to stop WebSocket session: {e}")

    async def handle_audio_input(self, session_id: str, audio_data: bytes) -> None:
        """Handle incoming audio from WebSocket."""
        if session_id not in self.websocket_connections:
            logger.warning(f"Unknown session: {session_id}")
            return

        # Queue audio for processing
        success = self.streaming_manager.queue_input_audio(audio_data)
        if not success:
            logger.warning(f"Failed to queue audio for session {session_id}")

    def _send_audio_to_websocket(self, session_id: str, audio_data: bytes) -> None:
        """Send processed audio to WebSocket."""
        if session_id not in self.websocket_connections:
            return

        try:
            websocket = self.websocket_connections[session_id]
            # This would be implemented based on the actual WebSocket library used
            # For now, it's a placeholder
            logger.debug(f"Sending audio to WebSocket session {session_id}")

        except Exception as e:
            logger.error(f"Failed to send audio to WebSocket: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get WebSocket session statistics."""
        return {
            "active_sessions": len(self.websocket_connections),
            "is_active": self.is_active,
            "streaming_stats": self.streaming_manager.get_streaming_stats(),
        }


class MoshiVoiceProcessor(VoiceProcessorInterface):
    """Voice processor using Kyutai Moshi."""

    def __init__(self):
        """Initialize Moshi voice processor."""
        self.config: Optional[VoiceConfig] = None
        self.moshi_client: Optional[MoshiClient] = None
        self.streaming_manager: Optional[MoshiStreamingManager] = None
        self.websocket_handler: Optional[MoshiWebSocketHandler] = None
        self.conversation_manager: Optional[ConversationStateManager] = None
        self.conversations: Dict[str, ConversationState] = {}

        logger.info("MoshiVoiceProcessor initialized")

    async def initialize(self, config: VoiceConfig) -> None:
        """Initialize the voice processor."""
        try:
            self.config = config

            # Create and initialize Moshi client
            self.moshi_client = MoshiClient(config.moshi)
            await self.moshi_client.initialize()

            # Create streaming manager
            self.streaming_manager = MoshiStreamingManager(self.moshi_client)

            # Create WebSocket handler if enabled
            if config.websocket_events_enabled:
                self.websocket_handler = MoshiWebSocketHandler(self.moshi_client)

            # Create conversation state manager
            self.conversation_manager = ConversationStateManager(max_conversations=50)
            await self.conversation_manager.start()

            logger.info("MoshiVoiceProcessor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MoshiVoiceProcessor: {e}")
            raise

    async def start_conversation(
        self, conversation_id: str, user_id: Optional[str] = None
    ) -> ConversationState:
        """Start a new voice conversation."""
        if not self.moshi_client or not self.conversation_manager:
            raise RuntimeError("Moshi client or conversation manager not initialized")

        # Start Moshi conversation
        await self.moshi_client.start_conversation(conversation_id)

        # Create conversation state using conversation manager
        conversation_state = self.conversation_manager.create_conversation(
            conversation_id=conversation_id, user_id=user_id, mode=VoiceProcessingMode.MOSHI_ONLY
        )

        # Update conversation phase
        self.conversation_manager.update_conversation_phase(
            conversation_id, ConversationPhase.LISTENING
        )

        # Store in local cache for quick access
        self.conversations[conversation_id] = conversation_state

        logger.info(f"Started Moshi conversation: {conversation_id}")
        return conversation_state

    async def process_audio(self, message: VoiceMessage) -> VoiceResponse:
        """Process audio input and generate response."""
        if not self.moshi_client or not self.conversation_manager:
            raise RuntimeError("Moshi client or conversation manager not initialized")

        try:
            # Update conversation phase to processing
            self.conversation_manager.update_conversation_phase(
                message.conversation_id, ConversationPhase.PROCESSING
            )

            # Process audio through Moshi
            start_time = time.time()
            response_audio = await self.moshi_client.process_audio(message.audio_data)
            processing_latency_ms = (time.time() - start_time) * 1000

            # Extract text if inner monologue is enabled
            text_content = ""
            confidence_score = None
            if self.config.moshi.inner_monologue_enabled:
                text_content = await self.moshi_client.extract_text(message.audio_data)

                # If we have inner monologue processor, get confidence and add to conversation
                if self.moshi_client.inner_monologue:
                    extracted_text = (
                        await self.moshi_client.inner_monologue.extract_text_from_audio(
                            message.audio_data
                        )
                    )
                    if extracted_text:
                        confidence_score = extracted_text.confidence
                        self.conversation_manager.add_inner_monologue_text(
                            message.conversation_id, extracted_text
                        )

            # Calculate audio duration
            audio_duration_ms = (
                len(message.audio_data) / (24000 * 2) * 1000
            )  # Assuming 24kHz 16 - bit

            # Record audio exchange in conversation manager
            self.conversation_manager.record_audio_exchange(
                conversation_id=message.conversation_id,
                audio_duration_ms=audio_duration_ms,
                processing_latency_ms=processing_latency_ms,
                confidence_score=confidence_score,
            )

            # Update conversation phase to responding
            self.conversation_manager.update_conversation_phase(
                message.conversation_id, ConversationPhase.RESPONDING
            )

            # Create response
            response = VoiceResponse(
                response_id=str(uuid.uuid4()),
                conversation_id=message.conversation_id,
                message_id=message.message_id,
                audio_data=response_audio,
                text_content=text_content,
                processing_mode=VoiceProcessingMode.MOSHI_ONLY,
                total_latency_ms=processing_latency_ms,
                moshi_latency_ms=processing_latency_ms,
                timestamp=datetime.now(),
            )

            # Update conversation phase back to listening
            self.conversation_manager.update_conversation_phase(
                message.conversation_id, ConversationPhase.LISTENING
            )

            return response

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise

    async def process_audio_stream(
        self, conversation_id: str, audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[VoiceStreamChunk, None]:
        """Process streaming audio input."""
        if not self.moshi_client or not self.streaming_manager:
            raise RuntimeError("Moshi client or streaming manager not initialized")

        chunk_index = 0

        try:
            # Start streaming if not already active
            if not self.streaming_manager.is_streaming:
                await self.streaming_manager.start_streaming()

            # Start input streaming task
            input_task = asyncio.create_task(self.streaming_manager.stream_input(audio_stream))

            # Stream output chunks
            async for processed_audio in self.streaming_manager.stream_output():
                chunk = VoiceStreamChunk(
                    conversation_id=conversation_id,
                    chunk_index=chunk_index,
                    audio_data=processed_audio,
                    timestamp=time.time(),
                    chunk_type="audio",
                )

                chunk_index += 1
                yield chunk

                # Check if input task is done
                if input_task.done():
                    break

            # Wait for input task to complete
            await input_task

        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise

    async def end_conversation(self, conversation_id: str) -> None:
        """End a voice conversation."""
        try:
            # Update conversation phase to ending
            if self.conversation_manager:
                self.conversation_manager.update_conversation_phase(
                    conversation_id, ConversationPhase.ENDING
                )

            # End Moshi conversation
            if self.moshi_client:
                await self.moshi_client.end_conversation(conversation_id)

            # End conversation in conversation manager and get final metrics
            if self.conversation_manager:
                final_metrics = self.conversation_manager.end_conversation(conversation_id)
                if final_metrics:
                    logger.info(
                        f"Conversation {conversation_id} ended with quality: {final_metrics.quality_level}"
                    )

            # Remove from local cache
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]

            logger.info(f"Ended Moshi conversation: {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to end conversation {conversation_id}: {e}")

    async def get_conversation_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Get current conversation state."""
        # Try to get from conversation manager first (most up - to - date)
        if self.conversation_manager:
            state = self.conversation_manager.get_conversation(conversation_id)
            if state:
                return state

        # Fallback to local cache
        return self.conversations.get(conversation_id)

    async def get_analytics(self) -> VoiceAnalytics:
        """Get voice processing analytics."""
        # Calculate basic stats
        total_conversations = len(self.conversations)
        total_messages = sum(
            len(conv.messages) if hasattr(conv, "messages") else 0
            for conv in self.conversations.values()
        )

        # Calculate total audio duration
        total_audio_duration = 0
        for conv in self.conversations.values():
            if hasattr(conv, "total_duration_ms") and conv.total_duration_ms:
                total_audio_duration += int(conv.total_duration_ms)

        # Get processing stats
        average_latency = 0.0
        moshi_latency = 0.0
        if self.moshi_client:
            moshi_state = await self.moshi_client.get_conversation_state()
            if "latency_stats" in moshi_state:
                average_latency = moshi_state["latency_stats"].get("avg", 0.0)
                moshi_latency = average_latency  # For now, same as average

        # Get VRAM usage as dict
        vram_usage = {}
        if self.moshi_client and hasattr(self.moshi_client, "vram_manager"):
            if self.moshi_client.vram_manager:
                allocated = self.moshi_client.vram_manager.get_total_allocated()
                vram_usage = {
                    "moshi_lm": f"{allocated * 0.7:.1f}GB",  # Estimate LM usage
                    "moshi_compression": f"{allocated * 0.3:.1f}GB",  # Estimate compression usage
                    "total": f"{allocated:.1f}GB",
                }

        # Create VoiceAnalytics object with correct fields
        analytics = VoiceAnalytics(
            total_conversations=total_conversations,
            total_messages=total_messages,
            total_audio_duration_ms=total_audio_duration,
            average_latency_ms=average_latency,
            moshi_latency_ms=moshi_latency,
            llm_latency_ms=0.0,  # No external LLM yet
            average_confidence=0.8,  # Default confidence
            audio_quality=0.8,  # Default audio quality
            vram_usage=vram_usage,
            cpu_usage=0.0,  # TODO: Track CPU usage
        )

        return analytics

    async def cleanup(self) -> None:
        """Cleanup voice processor resources."""
        try:
            # End all conversations
            for conversation_id in list(self.conversations.keys()):
                await self.end_conversation(conversation_id)

            # Stop streaming
            if self.streaming_manager:
                await self.streaming_manager.stop_streaming()
                self.streaming_manager = None

            # Cleanup WebSocket handler
            if self.websocket_handler:
                # Stop all WebSocket sessions
                for session_id in list(self.websocket_handler.websocket_connections.keys()):
                    await self.websocket_handler.stop_websocket_session(session_id)
                self.websocket_handler = None

            # Stop conversation manager
            if self.conversation_manager:
                await self.conversation_manager.stop()
                self.conversation_manager = None

            # Cleanup Moshi client
            if self.moshi_client:
                await self.moshi_client.cleanup()
                self.moshi_client = None

            logger.info("MoshiVoiceProcessor cleaned up")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
