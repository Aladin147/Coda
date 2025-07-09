"""
Inner monologue text extraction for Kyutai Moshi.

This module provides text extraction capabilities from Moshi's inner monologue,
enabling hybrid processing with external LLMs for enhanced reasoning.
"""

import asyncio
import logging
import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .models import VoiceConfig, MoshiConfig
from .utils import LatencyTracker, track_latency

logger = logging.getLogger(__name__)


class TextExtractionMode(str, Enum):
    """Text extraction modes."""
    CONTINUOUS = "continuous"
    ON_DEMAND = "on_demand"
    BUFFERED = "buffered"


@dataclass
class ExtractedText:
    """Extracted text with metadata."""
    text: str
    confidence: float
    timestamp: datetime
    audio_duration_ms: float
    extraction_latency_ms: float
    tokens: Optional[List[str]] = None
    raw_logits: Optional[torch.Tensor] = None


@dataclass
class InnerMonologueState:
    """State of the inner monologue processor."""
    is_active: bool
    mode: TextExtractionMode
    total_extractions: int
    total_audio_processed_ms: float
    average_confidence: float
    average_latency_ms: float
    last_extraction: Optional[ExtractedText]


class InnerMonologueProcessor:
    """Processes Moshi's inner monologue for text extraction."""
    
    def __init__(self, config: MoshiConfig):
        """Initialize inner monologue processor."""
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.moshi_model = None
        self.tokenizer = None
        self.text_decoder = None
        
        # Processing state
        self.is_initialized = False
        self.extraction_mode = TextExtractionMode.CONTINUOUS
        self.confidence_threshold = 0.7
        
        # Performance tracking
        self.latency_tracker = LatencyTracker("inner_monologue")
        self.state = InnerMonologueState(
            is_active=False,
            mode=self.extraction_mode,
            total_extractions=0,
            total_audio_processed_ms=0.0,
            average_confidence=0.0,
            average_latency_ms=0.0,
            last_extraction=None
        )
        
        # Text buffer for continuous extraction
        self.text_buffer: List[ExtractedText] = []
        self.max_buffer_size = 100
        
        logger.info("InnerMonologueProcessor initialized")
    
    async def initialize(self, moshi_model: Any, tokenizer: Any = None) -> None:
        """Initialize with Moshi model and tokenizer."""
        try:
            self.moshi_model = moshi_model
            self.tokenizer = tokenizer
            
            # Initialize text decoder if available
            if hasattr(moshi_model, 'text_decoder'):
                self.text_decoder = moshi_model.text_decoder
            elif hasattr(moshi_model, 'get_text_decoder'):
                self.text_decoder = moshi_model.get_text_decoder()
            else:
                logger.warning("No text decoder found in Moshi model")
            
            self.is_initialized = True
            logger.info("InnerMonologueProcessor initialized with Moshi model")
            
        except Exception as e:
            logger.error(f"Failed to initialize InnerMonologueProcessor: {e}")
            raise
    
    def set_extraction_mode(self, mode: TextExtractionMode) -> None:
        """Set text extraction mode."""
        self.extraction_mode = mode
        self.state.mode = mode
        logger.info(f"Text extraction mode set to: {mode}")
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for text extraction."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to: {self.confidence_threshold}")
    
    async def extract_text_from_audio(self, audio_data: bytes) -> Optional[ExtractedText]:
        """Extract text from audio using Moshi's inner monologue."""
        if not self.is_initialized:
            raise RuntimeError("InnerMonologueProcessor not initialized")
        
        with track_latency(self.latency_tracker) as tracker:
            try:
                # Convert audio to tensor
                audio_tensor = self._bytes_to_tensor(audio_data)
                audio_duration_ms = len(audio_data) / (24000 * 2) * 1000  # Assuming 24kHz 16-bit
                
                # Extract text using Moshi's inner monologue
                extracted_text = await self._process_audio_for_text(audio_tensor)
                
                if extracted_text and extracted_text.confidence >= self.confidence_threshold:
                    # Update statistics
                    self._update_statistics(extracted_text, audio_duration_ms)
                    
                    # Add to buffer if in continuous mode
                    if self.extraction_mode == TextExtractionMode.CONTINUOUS:
                        self._add_to_buffer(extracted_text)
                    
                    self.state.last_extraction = extracted_text
                    
                    logger.debug(f"Extracted text: '{extracted_text.text}' (confidence: {extracted_text.confidence:.3f})")
                    return extracted_text
                else:
                    logger.debug("Text extraction below confidence threshold or failed")
                    return None
                
            except Exception as e:
                logger.error(f"Text extraction failed: {e}")
                return None
    
    async def extract_text_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[ExtractedText, None]:
        """Extract text from streaming audio."""
        if not self.is_initialized:
            raise RuntimeError("InnerMonologueProcessor not initialized")
        
        self.state.is_active = True
        
        try:
            async for audio_chunk in audio_stream:
                extracted_text = await self.extract_text_from_audio(audio_chunk)
                if extracted_text:
                    yield extracted_text
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Stream text extraction failed: {e}")
            raise
        finally:
            self.state.is_active = False
    
    async def get_buffered_text(self, max_items: Optional[int] = None) -> List[ExtractedText]:
        """Get buffered extracted text."""
        if max_items is None:
            return self.text_buffer.copy()
        else:
            return self.text_buffer[-max_items:].copy()
    
    async def get_continuous_text(self, time_window_ms: Optional[float] = None) -> str:
        """Get continuous text from buffer within time window."""
        if not self.text_buffer:
            return ""
        
        if time_window_ms is None:
            # Return all buffered text
            return " ".join(item.text for item in self.text_buffer)
        
        # Filter by time window
        cutoff_time = datetime.now().timestamp() - (time_window_ms / 1000.0)
        recent_items = [
            item for item in self.text_buffer
            if item.timestamp.timestamp() >= cutoff_time
        ]
        
        return " ".join(item.text for item in recent_items)
    
    async def clear_buffer(self) -> None:
        """Clear the text buffer."""
        self.text_buffer.clear()
        logger.debug("Text buffer cleared")
    
    def get_state(self) -> InnerMonologueState:
        """Get current processor state."""
        return self.state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_extractions': self.state.total_extractions,
            'total_audio_processed_ms': self.state.total_audio_processed_ms,
            'average_confidence': self.state.average_confidence,
            'average_latency_ms': self.state.average_latency_ms,
            'buffer_size': len(self.text_buffer),
            'extraction_mode': self.extraction_mode.value,
            'confidence_threshold': self.confidence_threshold,
            'latency_stats': self.latency_tracker.get_stats()
        }
    
    # Private helper methods
    
    def _bytes_to_tensor(self, audio_data: bytes) -> torch.Tensor:
        """Convert audio bytes to tensor."""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Convert to tensor and move to device
            audio_tensor = torch.from_numpy(audio_float).to(self.device)
            
            # Ensure correct shape for Moshi
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            return audio_tensor
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return torch.zeros(1, 1600).to(self.device)
    
    async def _process_audio_for_text(self, audio_tensor: torch.Tensor) -> Optional[ExtractedText]:
        """Process audio tensor to extract text."""
        try:
            if self.moshi_model is None:
                return None
            
            # This is a placeholder implementation
            # The actual implementation will depend on Moshi's specific API for inner monologue
            
            with torch.no_grad():
                # Step 1: Get audio features from Moshi
                audio_features = await self._extract_audio_features(audio_tensor)
                
                # Step 2: Extract text tokens from features
                text_tokens = await self._extract_text_tokens(audio_features)
                
                # Step 3: Decode tokens to text
                text, confidence = await self._decode_tokens_to_text(text_tokens)
                
                if text and len(text.strip()) > 0:
                    return ExtractedText(
                        text=text.strip(),
                        confidence=confidence,
                        timestamp=datetime.now(),
                        audio_duration_ms=len(audio_tensor[0]) / 24000 * 1000,
                        extraction_latency_ms=self.latency_tracker.get_stats()['avg'],
                        tokens=text_tokens if isinstance(text_tokens, list) else None
                    )
                
            return None
            
        except Exception as e:
            logger.error(f"Audio processing for text failed: {e}")
            return None
    
    async def _extract_audio_features(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Extract audio features using Moshi model."""
        try:
            # For now, return a placeholder since Moshi's inner monologue API
            # requires the compression model (Mimi) to process raw audio first
            # and the LM model expects codebook tokens, not raw audio

            # TODO: Implement proper audio -> codebook -> features pipeline
            # This would require:
            # 1. Use Mimi compression model to encode audio to codebook tokens
            # 2. Use Moshi LM model to process codebook tokens
            # 3. Extract internal representations for text decoding

            logger.warning("Audio feature extraction not fully implemented - using placeholder")

            # Return placeholder features with correct device
            batch_size = audio_tensor.shape[0]
            feature_dim = 512  # Typical feature dimension
            return torch.zeros(batch_size, feature_dim).to(self.device)

        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return torch.zeros(1, 512).to(self.device)  # Placeholder
    
    async def _extract_text_tokens(self, audio_features: torch.Tensor) -> Optional[List[str]]:
        """Extract text tokens from audio features."""
        try:
            # This is a placeholder - actual implementation depends on Moshi's inner monologue API
            
            if self.text_decoder is not None:
                # Use dedicated text decoder
                token_logits = self.text_decoder(audio_features)
                tokens = torch.argmax(token_logits, dim=-1)
            else:
                # Fallback: try to extract from model output
                tokens = torch.randint(0, 1000, (1, 10))  # Placeholder
            
            # Convert token IDs to strings if tokenizer available
            if self.tokenizer is not None:
                token_strings = self.tokenizer.decode(tokens[0].cpu().numpy())
                return token_strings.split() if isinstance(token_strings, str) else []
            else:
                # Return token IDs as strings
                return [str(token.item()) for token in tokens[0]]
            
        except Exception as e:
            logger.error(f"Text token extraction failed: {e}")
            return None
    
    async def _decode_tokens_to_text(self, tokens: Optional[List[str]]) -> Tuple[str, float]:
        """Decode tokens to readable text."""
        try:
            if not tokens:
                return "", 0.0
            
            # This is a placeholder implementation
            # Actual implementation would use Moshi's tokenizer and text decoder
            
            if self.tokenizer is not None:
                # Use tokenizer to decode
                text = " ".join(tokens)
                confidence = 0.8  # Placeholder confidence
            else:
                # Fallback: simple concatenation
                text = " ".join(tokens)
                confidence = 0.5  # Lower confidence without proper tokenizer
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"Token decoding failed: {e}")
            return "", 0.0
    
    def _update_statistics(self, extracted_text: ExtractedText, audio_duration_ms: float) -> None:
        """Update processing statistics."""
        self.state.total_extractions += 1
        self.state.total_audio_processed_ms += audio_duration_ms
        
        # Update average confidence
        if self.state.total_extractions == 1:
            self.state.average_confidence = extracted_text.confidence
        else:
            self.state.average_confidence = (
                (self.state.average_confidence * (self.state.total_extractions - 1) + extracted_text.confidence)
                / self.state.total_extractions
            )
        
        # Update average latency
        if self.state.total_extractions == 1:
            self.state.average_latency_ms = extracted_text.extraction_latency_ms
        else:
            self.state.average_latency_ms = (
                (self.state.average_latency_ms * (self.state.total_extractions - 1) + extracted_text.extraction_latency_ms)
                / self.state.total_extractions
            )
    
    def _add_to_buffer(self, extracted_text: ExtractedText) -> None:
        """Add extracted text to buffer."""
        self.text_buffer.append(extracted_text)
        
        # Maintain buffer size
        if len(self.text_buffer) > self.max_buffer_size:
            self.text_buffer.pop(0)


class InnerMonologueManager:
    """Manages multiple inner monologue processors."""
    
    def __init__(self):
        """Initialize inner monologue manager."""
        self.processors: Dict[str, InnerMonologueProcessor] = {}
        self.default_config: Optional[MoshiConfig] = None
        
        logger.info("InnerMonologueManager initialized")
    
    def set_default_config(self, config: MoshiConfig) -> None:
        """Set default configuration."""
        self.default_config = config
    
    async def create_processor(
        self, 
        processor_id: str, 
        config: Optional[MoshiConfig] = None
    ) -> InnerMonologueProcessor:
        """Create a new inner monologue processor."""
        if processor_id in self.processors:
            raise ValueError(f"Processor {processor_id} already exists")
        
        processor_config = config or self.default_config
        if not processor_config:
            raise ValueError("No configuration provided and no default config set")
        
        processor = InnerMonologueProcessor(processor_config)
        self.processors[processor_id] = processor
        
        logger.info(f"Created inner monologue processor: {processor_id}")
        return processor
    
    async def initialize_processor(
        self, 
        processor_id: str, 
        moshi_model: Any, 
        tokenizer: Any = None
    ) -> None:
        """Initialize a processor with Moshi model."""
        if processor_id not in self.processors:
            raise ValueError(f"Processor {processor_id} not found")
        
        await self.processors[processor_id].initialize(moshi_model, tokenizer)
    
    def get_processor(self, processor_id: str) -> Optional[InnerMonologueProcessor]:
        """Get a processor by ID."""
        return self.processors.get(processor_id)
    
    def list_processors(self) -> List[str]:
        """List all processor IDs."""
        return list(self.processors.keys())
    
    async def remove_processor(self, processor_id: str) -> None:
        """Remove a processor."""
        if processor_id in self.processors:
            del self.processors[processor_id]
            logger.info(f"Removed inner monologue processor: {processor_id}")
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all processors."""
        return {
            processor_id: processor.get_statistics()
            for processor_id, processor in self.processors.items()
        }
    
    async def cleanup_all(self) -> None:
        """Cleanup all processors."""
        for processor_id in list(self.processors.keys()):
            await self.remove_processor(processor_id)
        
        logger.info("All inner monologue processors cleaned up")
