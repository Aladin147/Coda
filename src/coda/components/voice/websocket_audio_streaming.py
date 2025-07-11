"""
WebSocket audio streaming for real-time voice processing.

This module provides bidirectional audio streaming capabilities over WebSocket,
with support for real-time audio chunk processing, format conversion,
and streaming optimization.
"""

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import numpy as np

from .audio_buffer_pool import OptimizedAudioProcessor, get_global_buffer_pool
from .models import VoiceMessage, VoiceStreamChunk
from .performance_profiler import get_performance_profiler
from .websocket_handler import ClientConnection, MessageType, VoiceWebSocketHandler

logger = logging.getLogger(__name__)


class AudioFormat(str, Enum):
    """Supported audio formats for streaming."""

    WAV = "wav"
    PCM = "pcm"
    OPUS = "opus"
    MP3 = "mp3"


class StreamingMode(str, Enum):
    """Audio streaming modes."""

    PUSH_TO_TALK = "push_to_talk"
    CONTINUOUS = "continuous"
    VOICE_ACTIVITY = "voice_activity"


@dataclass
class AudioStreamConfig:
    """Configuration for audio streaming."""

    sample_rate: int = 16000
    channels: int = 1
    format: AudioFormat = AudioFormat.WAV
    chunk_size_ms: int = 100  # Chunk size in milliseconds
    buffer_size_ms: int = 1000  # Buffer size in milliseconds
    enable_vad: bool = True  # Voice Activity Detection
    vad_threshold: float = 0.5
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = False
    streaming_mode: StreamingMode = StreamingMode.VOICE_ACTIVITY

    @property
    def chunk_size_samples(self) -> int:
        """Get chunk size in samples."""
        return int(self.sample_rate * self.channels * self.chunk_size_ms / 1000)

    @property
    def buffer_size_samples(self) -> int:
        """Get buffer size in samples."""
        return int(self.sample_rate * self.channels * self.buffer_size_ms / 1000)


class AudioStreamProcessor:
    """
    Real-time audio stream processor for WebSocket connections.

    Handles bidirectional audio streaming with real-time processing,
    format conversion, and optimization for low-latency communication.

    Features:
    - Real-time audio chunk processing
    - Multiple audio format support
    - Voice Activity Detection (VAD)
    - Audio enhancement and noise reduction
    - Adaptive streaming based on connection quality
    - Buffer management for smooth playback

    Example:
        >>> processor = AudioStreamProcessor(websocket_handler)
        >>> await processor.start_audio_stream(client_id, config)
    """

    def __init__(self, websocket_handler: VoiceWebSocketHandler, voice_manager=None):
        """
        Initialize audio stream processor.

        Args:
            websocket_handler: WebSocket handler for communication
            voice_manager: Voice manager for processing
        """
        self.websocket_handler = websocket_handler
        self.voice_manager = voice_manager

        # Audio processing components
        self.buffer_pool = get_global_buffer_pool()
        self.audio_processor = OptimizedAudioProcessor(self.buffer_pool)
        self.profiler = get_performance_profiler()

        # Active streams
        self.active_streams: Dict[str, Dict[str, Any]] = {}

        # Performance metrics
        self.metrics = {
            "streams_started": 0,
            "streams_ended": 0,
            "chunks_processed": 0,
            "total_audio_duration": 0.0,
            "average_latency": 0.0,
        }

        logger.info("AudioStreamProcessor initialized")

    async def start_audio_stream(
        self, client_id: str, config: AudioStreamConfig, conversation_id: Optional[str] = None
    ) -> bool:
        """
        Start audio streaming for a client.

        Args:
            client_id: Client identifier
            config: Audio streaming configuration
            conversation_id: Optional conversation ID

        Returns:
            True if stream started successfully
        """
        try:
            if client_id not in self.websocket_handler.connections:
                logger.warning(f"Cannot start stream for unknown client: {client_id}")
                return False

            connection = self.websocket_handler.connections[client_id]

            # Initialize stream state
            stream_state = {
                "config": config,
                "conversation_id": conversation_id or connection.conversation_id,
                "started_at": time.time(),
                "audio_buffer": np.array([], dtype=np.float32),
                "chunk_counter": 0,
                "last_activity": time.time(),
                "vad_state": False,
                "processing_task": None,
            }

            self.active_streams[client_id] = stream_state

            # Start processing task
            stream_state["processing_task"] = asyncio.create_task(
                self._process_audio_stream(client_id)
            )

            # Send stream start confirmation
            await self.websocket_handler._send_message(
                connection,
                MessageType.VOICE_START,
                {
                    "stream_id": client_id,
                    "config": {
                        "sample_rate": config.sample_rate,
                        "channels": config.channels,
                        "format": config.format.value,
                        "chunk_size_ms": config.chunk_size_ms,
                    },
                    "started_at": stream_state["started_at"],
                },
                conversation_id,
            )

            self.metrics["streams_started"] += 1
            logger.info(f"Started audio stream for client {client_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start audio stream for {client_id}: {e}")
            return False

    async def stop_audio_stream(self, client_id: str) -> bool:
        """
        Stop audio streaming for a client.

        Args:
            client_id: Client identifier

        Returns:
            True if stream stopped successfully
        """
        try:
            if client_id not in self.active_streams:
                return False

            stream_state = self.active_streams[client_id]

            # Cancel processing task
            if stream_state["processing_task"]:
                stream_state["processing_task"].cancel()
                try:
                    await stream_state["processing_task"]
                except asyncio.CancelledError:
                    pass

            # Send final audio if any remains in buffer
            if len(stream_state["audio_buffer"]) > 0:
                await self._process_audio_chunk(
                    client_id, stream_state["audio_buffer"], is_final=True
                )

            # Clean up
            del self.active_streams[client_id]

            # Send stream end confirmation
            if client_id in self.websocket_handler.connections:
                connection = self.websocket_handler.connections[client_id]
                await self.websocket_handler._send_message(
                    connection,
                    MessageType.VOICE_END,
                    {
                        "stream_id": client_id,
                        "ended_at": time.time(),
                        "duration": time.time() - stream_state["started_at"],
                    },
                    stream_state["conversation_id"],
                )

            self.metrics["streams_ended"] += 1
            logger.info(f"Stopped audio stream for client {client_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop audio stream for {client_id}: {e}")
            return False

    async def process_audio_chunk(
        self, client_id: str, audio_data: bytes, chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Process incoming audio chunk from client.

        Args:
            client_id: Client identifier
            audio_data: Raw audio data
            chunk_metadata: Optional metadata about the chunk

        Returns:
            True if chunk processed successfully
        """
        try:
            if client_id not in self.active_streams:
                logger.warning(f"Received audio chunk for inactive stream: {client_id}")
                return False

            stream_state = self.active_streams[client_id]
            config = stream_state["config"]

            # Convert audio data to numpy array
            audio_array = self._convert_audio_data(audio_data, config)

            # Add to buffer
            stream_state["audio_buffer"] = np.concatenate(
                [stream_state["audio_buffer"], audio_array]
            )

            stream_state["last_activity"] = time.time()
            stream_state["chunk_counter"] += 1

            # Process if buffer is large enough or if this is marked as final
            is_final = chunk_metadata and chunk_metadata.get("is_final", False)

            if len(stream_state["audio_buffer"]) >= config.chunk_size_samples or is_final:
                await self._process_buffered_audio(client_id, is_final)

            self.metrics["chunks_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to process audio chunk for {client_id}: {e}")
            return False

    async def _process_audio_stream(self, client_id: str) -> None:
        """Background task to process audio stream."""
        try:
            stream_state = self.active_streams[client_id]
            config = stream_state["config"]

            while client_id in self.active_streams:
                # Check for voice activity if VAD is enabled
                if config.enable_vad and len(stream_state["audio_buffer"]) > 0:
                    vad_result = self._detect_voice_activity(
                        stream_state["audio_buffer"], config.vad_threshold
                    )

                    # Process audio if voice is detected or mode is continuous
                    if (
                        vad_result
                        or config.streaming_mode == StreamingMode.CONTINUOUS
                        or config.streaming_mode == StreamingMode.PUSH_TO_TALK
                    ):

                        if len(stream_state["audio_buffer"]) >= config.chunk_size_samples:
                            await self._process_buffered_audio(client_id, is_final=False)

                # Check for stream timeout
                if time.time() - stream_state["last_activity"] > 30.0:  # 30 second timeout
                    logger.info(f"Audio stream timeout for client {client_id}")
                    break

                await asyncio.sleep(0.01)  # 10ms processing interval

        except asyncio.CancelledError:
            logger.debug(f"Audio stream processing cancelled for {client_id}")
        except Exception as e:
            logger.error(f"Audio stream processing error for {client_id}: {e}")

    async def _process_buffered_audio(self, client_id: str, is_final: bool = False) -> None:
        """Process buffered audio data."""
        stream_state = self.active_streams[client_id]
        config = stream_state["config"]

        if len(stream_state["audio_buffer"]) == 0:
            return

        # Extract chunk to process
        if is_final:
            chunk_data = stream_state["audio_buffer"]
            stream_state["audio_buffer"] = np.array([], dtype=np.float32)
        else:
            chunk_size = config.chunk_size_samples
            chunk_data = stream_state["audio_buffer"][:chunk_size]
            stream_state["audio_buffer"] = stream_state["audio_buffer"][chunk_size:]

        # Process the audio chunk
        await self._process_audio_chunk(client_id, chunk_data, is_final)

    async def _process_audio_chunk(
        self, client_id: str, audio_data: np.ndarray, is_final: bool = False
    ) -> None:
        """Process a single audio chunk."""
        try:
            with self.profiler.profile_operation("audio_streaming", "process_chunk"):
                stream_state = self.active_streams[client_id]
                config = stream_state["config"]

                # Apply audio enhancements
                enhanced_audio = await self._enhance_audio(audio_data, config)

                # Convert to bytes for voice processing
                audio_bytes = self._convert_to_bytes(enhanced_audio, config)

                # Create voice message
                voice_message = VoiceMessage(
                    conversation_id=stream_state["conversation_id"],
                    audio_data=audio_bytes,
                    speaker="user",
                    timestamp=time.time(),
                )

                # Process with voice manager if available
                if self.voice_manager and stream_state["conversation_id"]:
                    try:
                        if is_final:
                            # Process complete message
                            response = await self.voice_manager.process_voice_input(
                                conversation_id=stream_state["conversation_id"],
                                audio_data=audio_bytes,
                            )

                            # Send response back to client
                            await self._send_voice_response(client_id, response)
                        else:
                            # Process streaming chunk
                            async for chunk in self._process_streaming_chunk(voice_message):
                                await self._send_stream_chunk(client_id, chunk)
                    except Exception as e:
                        logger.error(f"Voice processing error in audio stream for {client_id}: {e}")
                        # Continue processing other chunks

        except Exception as e:
            logger.error(f"Error processing audio chunk for {client_id}: {e}")

    async def _enhance_audio(self, audio_data: np.ndarray, config: AudioStreamConfig) -> np.ndarray:
        """Apply audio enhancements."""
        enhanced = audio_data.copy()

        # Apply noise reduction if enabled
        if config.enable_noise_reduction:
            enhanced = self._apply_noise_reduction(enhanced)

        # Apply echo cancellation if enabled
        if config.enable_echo_cancellation:
            enhanced = self._apply_echo_cancellation(enhanced)

        # Normalize audio
        enhanced = self._normalize_audio(enhanced)

        return enhanced

    def _convert_audio_data(self, audio_data: bytes, config: AudioStreamConfig) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        if config.format == AudioFormat.WAV:
            # Parse WAV data
            wav_io = io.BytesIO(audio_data)
            with wave.open(wav_io, "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif config.format == AudioFormat.PCM:
            # Raw PCM data
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            # For other formats, assume PCM for now
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        return audio_array

    def _convert_to_bytes(self, audio_data: np.ndarray, config: AudioStreamConfig) -> bytes:
        """Convert numpy array to audio bytes."""
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        if config.format == AudioFormat.WAV:
            # Create WAV file
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(config.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(config.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            return wav_io.getvalue()
        else:
            # Return raw PCM
            return audio_int16.tobytes()

    def _detect_voice_activity(self, audio_data: np.ndarray, threshold: float) -> bool:
        """Simple voice activity detection."""
        if len(audio_data) == 0:
            return False

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data**2))
        return rms > threshold

    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simple noise reduction."""
        # Simple high-pass filter to remove low-frequency noise
        if len(audio_data) > 1:
            return np.diff(audio_data, prepend=audio_data[0])
        return audio_data

    def _apply_echo_cancellation(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply echo cancellation (placeholder)."""
        # Placeholder for echo cancellation algorithm
        return audio_data

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio levels."""
        if len(audio_data) == 0:
            return audio_data

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val * 0.8  # Normalize to 80% of max
        return audio_data

    async def _process_streaming_chunk(
        self, voice_message: VoiceMessage
    ) -> AsyncGenerator[VoiceStreamChunk, None]:
        """Process voice message as streaming chunks."""
        # Placeholder for streaming processing
        chunk = VoiceStreamChunk(
            conversation_id=voice_message.conversation_id,
            text_content="",  # Would be filled by actual processing
            audio_data=None,
            is_complete=False,
            chunk_index=0,
            timestamp=time.time(),
        )
        yield chunk

    async def _send_voice_response(self, client_id: str, response) -> None:
        """Send voice response to client."""
        if client_id in self.websocket_handler.connections:
            connection = self.websocket_handler.connections[client_id]
            await self.websocket_handler._send_voice_response(connection, response)

    async def _send_stream_chunk(self, client_id: str, chunk: VoiceStreamChunk) -> None:
        """Send stream chunk to client."""
        if client_id in self.websocket_handler.connections:
            connection = self.websocket_handler.connections[client_id]
            await self.websocket_handler._send_stream_chunk(connection, chunk)

    def get_stream_stats(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific stream."""
        if client_id not in self.active_streams:
            return None

        stream_state = self.active_streams[client_id]
        return {
            "client_id": client_id,
            "started_at": stream_state["started_at"],
            "duration": time.time() - stream_state["started_at"],
            "chunks_processed": stream_state["chunk_counter"],
            "buffer_size": len(stream_state["audio_buffer"]),
            "last_activity": stream_state["last_activity"],
            "config": {
                "sample_rate": stream_state["config"].sample_rate,
                "format": stream_state["config"].format.value,
                "streaming_mode": stream_state["config"].streaming_mode.value,
            },
        }

    def get_all_stream_stats(self) -> Dict[str, Any]:
        """Get statistics for all active streams."""
        return {
            "active_streams": len(self.active_streams),
            "metrics": self.metrics,
            "streams": {
                client_id: self.get_stream_stats(client_id) for client_id in self.active_streams
            },
        }
