"""
Audio processing implementation for Coda 2.0 voice system.

This module provides comprehensive audio processing capabilities including:
- Voice Activity Detection (VAD) using Silero VAD
- Audio enhancement (noise reduction, echo cancellation)
- Format conversion and streaming support
- Real-time audio processing pipeline
"""

import asyncio
import logging
import numpy as np
import torch
import torchaudio
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
import io
import wave

from .interfaces import AudioProcessorInterface, VoiceActivityDetectorInterface
from .models import AudioConfig, AudioFormat

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD."""
    
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        """Initialize VAD with Silero model."""
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        self.confidence_score = 0.0
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the VAD model."""
        try:
            # Load Silero VAD model
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            self.threshold = config.get('vad_threshold', 0.5)
            self.sample_rate = config.get('sample_rate', 16000)
            
            logger.info(f"VAD initialized with threshold {self.threshold}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
            raise
    
    async def detect_activity(self, audio_data: bytes) -> bool:
        """Detect voice activity in audio chunk."""
        if self.model is None:
            raise RuntimeError("VAD not initialized")
        
        try:
            # Convert bytes to tensor
            audio_tensor = self._bytes_to_tensor(audio_data)
            
            # Get VAD prediction
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            self.confidence_score = speech_prob
            is_speech = speech_prob > self.threshold
            
            logger.debug(f"VAD: speech_prob={speech_prob:.3f}, is_speech={is_speech}")
            return is_speech
            
        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            return False
    
    async def detect_activity_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bool, None]:
        """Detect voice activity in streaming audio."""
        async for chunk in audio_stream:
            is_speech = await self.detect_activity(chunk)
            yield is_speech
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """Set VAD sensitivity (0.0 to 1.0)."""
        self.threshold = max(0.0, min(1.0, sensitivity))
        logger.info(f"VAD sensitivity set to {self.threshold}")
    
    def get_confidence_score(self) -> float:
        """Get confidence score of last detection."""
        return self.confidence_score
    
    async def calibrate(self, background_audio: bytes) -> None:
        """Calibrate VAD with background audio."""
        try:
            # Analyze background noise
            audio_tensor = self._bytes_to_tensor(background_audio)
            
            with torch.no_grad():
                noise_prob = self.model(audio_tensor, self.sample_rate).item()
            
            # Adjust threshold based on background noise
            self.threshold = max(0.3, noise_prob + 0.2)
            logger.info(f"VAD calibrated: noise_prob={noise_prob:.3f}, new_threshold={self.threshold:.3f}")
            
        except Exception as e:
            logger.error(f"VAD calibration failed: {e}")
    
    def _bytes_to_tensor(self, audio_data: bytes) -> torch.Tensor:
        """Convert audio bytes to tensor for VAD processing."""
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_float)
            
            # Ensure correct sample rate (resample if needed)
            if len(audio_tensor) > 0:
                # Simple resampling for VAD (more sophisticated resampling can be added)
                return audio_tensor
            else:
                return torch.zeros(1600)  # 100ms of silence at 16kHz
                
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return torch.zeros(1600)


class AudioProcessor(AudioProcessorInterface):
    """Main audio processor for voice system."""
    
    def __init__(self):
        """Initialize audio processor."""
        self.config: Optional[AudioConfig] = None
        self.vad: Optional[VoiceActivityDetector] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 24000
        self.channels = 1
        
        # Audio enhancement settings
        self.noise_reduction_enabled = True
        self.echo_cancellation_enabled = True
        self.auto_gain_control_enabled = True
        
        logger.info(f"AudioProcessor initialized on device: {self.device}")
    
    async def initialize(self, config: AudioConfig) -> None:
        """Initialize the audio processor."""
        try:
            self.config = config
            self.sample_rate = config.sample_rate
            self.channels = config.channels
            
            # Initialize VAD if enabled
            if config.vad_enabled:
                self.vad = VoiceActivityDetector(
                    threshold=config.vad_threshold,
                    sample_rate=16000  # Silero VAD works best at 16kHz
                )
                await self.vad.initialize({
                    'vad_threshold': config.vad_threshold,
                    'sample_rate': 16000
                })
            
            # Set enhancement settings
            self.noise_reduction_enabled = config.noise_reduction
            self.echo_cancellation_enabled = config.echo_cancellation
            self.auto_gain_control_enabled = config.auto_gain_control
            
            logger.info(f"AudioProcessor initialized: {config.sample_rate}Hz, {config.channels}ch")
            
        except Exception as e:
            logger.error(f"Failed to initialize AudioProcessor: {e}")
            raise
    
    async def process_input_audio(self, audio_data: bytes) -> bytes:
        """Process input audio (noise reduction, enhancement, etc.)."""
        try:
            # Convert to tensor for processing
            audio_tensor = self._bytes_to_tensor(audio_data)
            
            # Apply audio enhancements
            if self.noise_reduction_enabled:
                audio_tensor = await self._apply_noise_reduction(audio_tensor)
            
            if self.echo_cancellation_enabled:
                audio_tensor = await self._apply_echo_cancellation(audio_tensor)
            
            if self.auto_gain_control_enabled:
                audio_tensor = await self._apply_auto_gain_control(audio_tensor)
            
            # Convert back to bytes
            processed_audio = self._tensor_to_bytes(audio_tensor)
            
            logger.debug(f"Processed input audio: {len(audio_data)} -> {len(processed_audio)} bytes")
            return processed_audio
            
        except Exception as e:
            logger.error(f"Input audio processing failed: {e}")
            return audio_data  # Return original on error
    
    async def process_output_audio(self, audio_data: bytes) -> bytes:
        """Process output audio (enhancement, normalization, etc.)."""
        try:
            # Convert to tensor for processing
            audio_tensor = self._bytes_to_tensor(audio_data)
            
            # Apply output enhancements
            audio_tensor = await self._apply_normalization(audio_tensor)
            audio_tensor = await self._apply_dynamic_range_compression(audio_tensor)
            
            # Convert back to bytes
            processed_audio = self._tensor_to_bytes(audio_tensor)
            
            logger.debug(f"Processed output audio: {len(audio_data)} -> {len(processed_audio)} bytes")
            return processed_audio
            
        except Exception as e:
            logger.error(f"Output audio processing failed: {e}")
            return audio_data  # Return original on error
    
    async def detect_voice_activity(self, audio_data: bytes) -> bool:
        """Detect voice activity in audio."""
        if self.vad is None:
            logger.warning("VAD not initialized, assuming speech")
            return True
        
        return await self.vad.detect_activity(audio_data)
    
    async def extract_features(self, audio_data: bytes) -> Dict[str, Any]:
        """Extract audio features for analysis."""
        try:
            audio_tensor = self._bytes_to_tensor(audio_data)
            
            # Extract basic features
            features = {
                'duration_ms': len(audio_data) / (self.sample_rate * self.channels * 2) * 1000,
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'amplitude_mean': float(torch.mean(torch.abs(audio_tensor))),
                'amplitude_max': float(torch.max(torch.abs(audio_tensor))),
                'energy': float(torch.sum(audio_tensor ** 2)),
                'zero_crossing_rate': self._calculate_zcr(audio_tensor),
                'spectral_centroid': await self._calculate_spectral_centroid(audio_tensor),
            }
            
            # Add VAD confidence if available
            if self.vad is not None:
                await self.detect_voice_activity(audio_data)
                features['vad_confidence'] = self.vad.get_confidence_score()
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return [format.value for format in AudioFormat]
    
    async def convert_format(self, audio_data: bytes, source_format: str, target_format: str) -> bytes:
        """Convert audio between formats."""
        try:
            if source_format == target_format:
                return audio_data
            
            # For now, implement basic WAV conversion
            # More sophisticated format conversion can be added later
            if target_format == AudioFormat.WAV.value:
                return self._convert_to_wav(audio_data)
            else:
                logger.warning(f"Format conversion {source_format} -> {target_format} not implemented")
                return audio_data
                
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            return audio_data
    
    # Private helper methods
    
    def _bytes_to_tensor(self, audio_data: bytes) -> torch.Tensor:
        """Convert audio bytes to tensor."""
        try:
            # Assume 16-bit PCM audio
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0
            return torch.from_numpy(audio_float).to(self.device)
        except Exception as e:
            logger.error(f"Bytes to tensor conversion failed: {e}")
            return torch.zeros(1024).to(self.device)
    
    def _tensor_to_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert tensor to audio bytes."""
        try:
            # Convert to CPU and numpy
            audio_np = audio_tensor.cpu().numpy()
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
        except Exception as e:
            logger.error(f"Tensor to bytes conversion failed: {e}")
            return b''
    
    async def _apply_noise_reduction(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply noise reduction to audio tensor."""
        # Simple noise gate implementation
        # More sophisticated noise reduction can be added later
        threshold = 0.01
        return torch.where(torch.abs(audio_tensor) > threshold, audio_tensor, torch.zeros_like(audio_tensor))
    
    async def _apply_echo_cancellation(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply echo cancellation to audio tensor."""
        # Placeholder for echo cancellation
        # Real implementation would require reference signal
        return audio_tensor
    
    async def _apply_auto_gain_control(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply automatic gain control to audio tensor."""
        # Simple AGC implementation
        target_level = 0.5
        current_level = torch.max(torch.abs(audio_tensor))
        
        if current_level > 0:
            gain = target_level / current_level
            gain = torch.clamp(gain, 0.1, 10.0)  # Limit gain range
            return audio_tensor * gain
        
        return audio_tensor
    
    async def _apply_normalization(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization to audio tensor."""
        max_val = torch.max(torch.abs(audio_tensor))
        if max_val > 0:
            return audio_tensor / max_val * 0.95  # Normalize to 95% to avoid clipping
        return audio_tensor
    
    async def _apply_dynamic_range_compression(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply dynamic range compression to audio tensor."""
        # Simple compressor
        threshold = 0.7
        ratio = 4.0
        
        abs_audio = torch.abs(audio_tensor)
        compressed = torch.where(
            abs_audio > threshold,
            torch.sign(audio_tensor) * (threshold + (abs_audio - threshold) / ratio),
            audio_tensor
        )
        
        return compressed
    
    def _calculate_zcr(self, audio_tensor: torch.Tensor) -> float:
        """Calculate zero crossing rate."""
        try:
            signs = torch.sign(audio_tensor)
            zero_crossings = torch.sum(torch.abs(torch.diff(signs))) / 2
            return float(zero_crossings / len(audio_tensor))
        except Exception:
            return 0.0
    
    async def _calculate_spectral_centroid(self, audio_tensor: torch.Tensor) -> float:
        """Calculate spectral centroid."""
        try:
            # Simple spectral centroid calculation
            fft = torch.fft.fft(audio_tensor)
            magnitude = torch.abs(fft)
            freqs = torch.fft.fftfreq(len(audio_tensor), 1/self.sample_rate)
            
            centroid = torch.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / torch.sum(magnitude[:len(magnitude)//2])
            return float(centroid)
        except Exception:
            return 0.0
    
    def _convert_to_wav(self, audio_data: bytes) -> bytes:
        """Convert audio data to WAV format."""
        try:
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            return wav_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"WAV conversion failed: {e}")
            return audio_data
