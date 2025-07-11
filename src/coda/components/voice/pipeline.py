"""
Audio pipeline implementation for Coda 2.0 voice system.

This module provides comprehensive audio pipeline capabilities including:
- Real - time audio input / output processing
- Format conversion and streaming support
- Audio buffering and synchronization
- Pipeline orchestration and management
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from .audio_processor import AudioProcessor
from .interfaces import AudioProcessorInterface
from .models import AudioConfig, AudioFormat, VoiceStreamChunk
from .utils import LatencyTracker, StreamingUtils, track_latency

logger = logging.getLogger(__name__)


class PipelineState(str, Enum):
    """Audio pipeline states."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AudioChunk:
    """Audio chunk with metadata."""

    data: bytes
    timestamp: datetime
    sequence_number: int
    chunk_id: str
    format: AudioFormat
    sample_rate: int
    channels: int
    duration_ms: float


class AudioBuffer:
    """Thread - safe audio buffer for streaming."""

    def __init__(self, max_size: int = 100):
        """Initialize audio buffer."""
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
        self.total_chunks = 0
        self.dropped_chunks = 0

    def put(self, chunk: AudioChunk, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add audio chunk to buffer."""
        try:
            self.buffer.put(chunk, block=block, timeout=timeout)
            with self.lock:
                self.total_chunks += 1
            return True
        except queue.Full:
            with self.lock:
                self.dropped_chunks += 1
            logger.warning(f"Audio buffer full, dropped chunk {chunk.chunk_id}")
            return False

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """Get audio chunk from buffer."""
        try:
            return self.buffer.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.buffer.empty()

    def clear(self) -> None:
        """Clear the buffer."""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                "current_size": self.size(),
                "max_size": self.max_size,
                "total_chunks": self.total_chunks,
                "dropped_chunks": self.dropped_chunks,
                "drop_rate": self.dropped_chunks / max(self.total_chunks, 1),
            }


class AudioInputStream:
    """Audio input stream handler."""

    def __init__(self, config: AudioConfig, processor: AudioProcessorInterface):
        """Initialize audio input stream."""
        self.config = config
        self.processor = processor
        self.buffer = AudioBuffer(max_size=config.chunk_size * 2)
        self.is_running = False
        self.sequence_number = 0
        self.latency_tracker = LatencyTracker("audio_input")

    async def start(self) -> None:
        """Start audio input stream."""
        self.is_running = True
        logger.info("Audio input stream started")

    async def stop(self) -> None:
        """Stop audio input stream."""
        self.is_running = False
        self.buffer.clear()
        logger.info("Audio input stream stopped")

    async def process_chunk(self, audio_data: bytes) -> AudioChunk:
        """Process incoming audio chunk."""
        with track_latency(self.latency_tracker) as tracker:
            try:
                # Process audio through processor
                processed_data = await self.processor.process_input_audio(audio_data)

                # Create audio chunk
                chunk = AudioChunk(
                    data=processed_data,
                    timestamp=datetime.now(),
                    sequence_number=self.sequence_number,
                    chunk_id=f"input_{self.sequence_number}_{int(time.time() * 1000)}",
                    format=self.config.format,
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                    duration_ms=len(audio_data)
                    / (self.config.sample_rate * self.config.channels * 2)
                    * 1000,
                )

                self.sequence_number += 1

                # Add to buffer
                self.buffer.put(chunk, block=False, timeout=0.1)

                logger.debug(
                    f"Processed input chunk {chunk.chunk_id}, latency: {tracker.get_latency():.1f}ms"
                )
                return chunk

            except Exception as e:
                logger.error(f"Failed to process input chunk: {e}")
                raise

    async def get_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """Get processed audio chunk."""
        return self.buffer.get(block=True, timeout=timeout)

    async def stream_chunks(self) -> AsyncGenerator[AudioChunk, None]:
        """Stream processed audio chunks."""
        while self.is_running:
            chunk = await self.get_chunk(timeout=0.1)
            if chunk:
                yield chunk
            else:
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting

    def get_stats(self) -> Dict[str, Any]:
        """Get input stream statistics."""
        return {
            "is_running": self.is_running,
            "sequence_number": self.sequence_number,
            "buffer_stats": self.buffer.get_stats(),
            "latency_stats": self.latency_tracker.get_stats(),
        }


class AudioOutputStream:
    """Audio output stream handler."""

    def __init__(self, config: AudioConfig, processor: AudioProcessorInterface):
        """Initialize audio output stream."""
        self.config = config
        self.processor = processor
        self.buffer = AudioBuffer(max_size=config.chunk_size * 2)
        self.is_running = False
        self.sequence_number = 0
        self.latency_tracker = LatencyTracker("audio_output")
        self.output_callback: Optional[Callable[[AudioChunk], None]] = None

    async def start(self) -> None:
        """Start audio output stream."""
        self.is_running = True
        # Start output processing task
        asyncio.create_task(self._output_processor())
        logger.info("Audio output stream started")

    async def stop(self) -> None:
        """Stop audio output stream."""
        self.is_running = False
        self.buffer.clear()
        logger.info("Audio output stream stopped")

    def set_output_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Set callback for processed output chunks."""
        self.output_callback = callback

    async def queue_chunk(self, audio_data: bytes) -> bool:
        """Queue audio chunk for output processing."""
        try:
            chunk = AudioChunk(
                data=audio_data,
                timestamp=datetime.now(),
                sequence_number=self.sequence_number,
                chunk_id=f"output_{self.sequence_number}_{int(time.time() * 1000)}",
                format=self.config.format,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                duration_ms=len(audio_data)
                / (self.config.sample_rate * self.config.channels * 2)
                * 1000,
            )

            self.sequence_number += 1

            # Add to buffer
            success = self.buffer.put(chunk, block=False, timeout=0.1)

            if success:
                logger.debug(f"Queued output chunk {chunk.chunk_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to queue output chunk: {e}")
            return False

    async def _output_processor(self) -> None:
        """Process output audio chunks."""
        while self.is_running:
            try:
                chunk = self.buffer.get(block=True, timeout=0.1)
                if chunk:
                    await self._process_output_chunk(chunk)
            except Exception as e:
                logger.error(f"Output processing error: {e}")
                await asyncio.sleep(0.1)

    async def _process_output_chunk(self, chunk: AudioChunk) -> None:
        """Process single output chunk."""
        with track_latency(self.latency_tracker) as tracker:
            try:
                # Process audio through processor
                processed_data = await self.processor.process_output_audio(chunk.data)

                # Create processed chunk
                processed_chunk = AudioChunk(
                    data=processed_data,
                    timestamp=chunk.timestamp,
                    sequence_number=chunk.sequence_number,
                    chunk_id=chunk.chunk_id,
                    format=chunk.format,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    duration_ms=chunk.duration_ms,
                )

                # Call output callback if set
                if self.output_callback:
                    self.output_callback(processed_chunk)

                logger.debug(
                    f"Processed output chunk {chunk.chunk_id}, latency: {tracker.get_latency():.1f}ms"
                )

            except Exception as e:
                logger.error(f"Failed to process output chunk {chunk.chunk_id}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get output stream statistics."""
        return {
            "is_running": self.is_running,
            "sequence_number": self.sequence_number,
            "buffer_stats": self.buffer.get_stats(),
            "latency_stats": self.latency_tracker.get_stats(),
        }


class AudioPipeline:
    """Main audio processing pipeline."""

    def __init__(self, config: AudioConfig):
        """Initialize audio pipeline."""
        self.config = config
        self.processor = AudioProcessor()
        self.input_stream: Optional[AudioInputStream] = None
        self.output_stream: Optional[AudioOutputStream] = None
        self.state = PipelineState.IDLE
        self.start_time: Optional[datetime] = None

        # Pipeline callbacks
        self.input_callback: Optional[Callable[[AudioChunk], None]] = None
        self.output_callback: Optional[Callable[[AudioChunk], None]] = None

        logger.info("Audio pipeline initialized")

    async def initialize(self) -> None:
        """Initialize the audio pipeline."""
        try:
            self.state = PipelineState.STARTING

            # Initialize audio processor
            await self.processor.initialize(self.config)

            # Create input and output streams
            self.input_stream = AudioInputStream(self.config, self.processor)
            self.output_stream = AudioOutputStream(self.config, self.processor)

            # Set output callback
            if self.output_callback:
                self.output_stream.set_output_callback(self.output_callback)

            logger.info("Audio pipeline initialized successfully")

        except Exception as e:
            self.state = PipelineState.ERROR
            logger.error(f"Failed to initialize audio pipeline: {e}")
            raise

    async def start(self) -> None:
        """Start the audio pipeline."""
        try:
            if self.state != PipelineState.IDLE:
                raise RuntimeError(f"Cannot start pipeline in state {self.state}")

            await self.initialize()

            # Start streams
            if self.input_stream:
                await self.input_stream.start()

            if self.output_stream:
                await self.output_stream.start()

            self.state = PipelineState.RUNNING
            self.start_time = datetime.now()

            logger.info("Audio pipeline started")

        except Exception as e:
            self.state = PipelineState.ERROR
            logger.error(f"Failed to start audio pipeline: {e}")
            raise

    async def stop(self) -> None:
        """Stop the audio pipeline."""
        try:
            self.state = PipelineState.STOPPING

            # Stop streams
            if self.input_stream:
                await self.input_stream.stop()

            if self.output_stream:
                await self.output_stream.stop()

            self.state = PipelineState.IDLE

            logger.info("Audio pipeline stopped")

        except Exception as e:
            self.state = PipelineState.ERROR
            logger.error(f"Failed to stop audio pipeline: {e}")
            raise

    def set_input_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Set callback for processed input chunks."""
        self.input_callback = callback

    def set_output_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Set callback for processed output chunks."""
        self.output_callback = callback
        if self.output_stream:
            self.output_stream.set_output_callback(callback)

    async def process_input(self, audio_data: bytes) -> AudioChunk:
        """Process input audio data."""
        if not self.input_stream or self.state != PipelineState.RUNNING:
            raise RuntimeError("Pipeline not running")

        chunk = await self.input_stream.process_chunk(audio_data)

        # Call input callback if set
        if self.input_callback:
            self.input_callback(chunk)

        return chunk

    async def queue_output(self, audio_data: bytes) -> bool:
        """Queue audio data for output."""
        if not self.output_stream or self.state != PipelineState.RUNNING:
            raise RuntimeError("Pipeline not running")

        return await self.output_stream.queue_chunk(audio_data)

    async def stream_input(self) -> AsyncGenerator[AudioChunk, None]:
        """Stream processed input chunks."""
        if not self.input_stream or self.state != PipelineState.RUNNING:
            raise RuntimeError("Pipeline not running")

        async for chunk in self.input_stream.stream_chunks():
            yield chunk

    async def convert_format(
        self, audio_data: bytes, source_format: str, target_format: str
    ) -> bytes:
        """Convert audio format."""
        return await self.processor.convert_format(audio_data, source_format, target_format)

    async def detect_voice_activity(self, audio_data: bytes) -> bool:
        """Detect voice activity in audio."""
        return await self.processor.detect_voice_activity(audio_data)

    async def extract_features(self, audio_data: bytes) -> Dict[str, Any]:
        """Extract audio features."""
        return await self.processor.extract_features(audio_data)

    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "state": self.state.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (
                (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            ),
            "config": {
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "format": self.config.format.value,
                "chunk_size": self.config.chunk_size,
            },
        }

        if self.input_stream:
            stats["input_stream"] = self.input_stream.get_stats()

        if self.output_stream:
            stats["output_stream"] = self.output_stream.get_stats()

        return stats

    async def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        try:
            if self.state == PipelineState.RUNNING:
                await self.stop()

            self.input_stream = None
            self.output_stream = None

            logger.info("Audio pipeline cleaned up")

        except Exception as e:
            logger.error(f"Failed to cleanup audio pipeline: {e}")


class PipelineManager:
    """Manages multiple audio pipelines."""

    def __init__(self):
        """Initialize pipeline manager."""
        self.pipelines: Dict[str, AudioPipeline] = {}
        self.default_config: Optional[AudioConfig] = None

    def set_default_config(self, config: AudioConfig) -> None:
        """Set default audio configuration."""
        self.default_config = config

    async def create_pipeline(
        self, pipeline_id: str, config: Optional[AudioConfig] = None
    ) -> AudioPipeline:
        """Create a new audio pipeline."""
        if pipeline_id in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} already exists")

        pipeline_config = config or self.default_config
        if not pipeline_config:
            raise ValueError("No configuration provided and no default config set")

        pipeline = AudioPipeline(pipeline_config)
        self.pipelines[pipeline_id] = pipeline

        logger.info(f"Created audio pipeline: {pipeline_id}")
        return pipeline

    async def start_pipeline(self, pipeline_id: str) -> None:
        """Start an audio pipeline."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        await self.pipelines[pipeline_id].start()

    async def stop_pipeline(self, pipeline_id: str) -> None:
        """Stop an audio pipeline."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        await self.pipelines[pipeline_id].stop()

    async def remove_pipeline(self, pipeline_id: str) -> None:
        """Remove an audio pipeline."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        pipeline = self.pipelines[pipeline_id]
        await pipeline.cleanup()
        del self.pipelines[pipeline_id]

        logger.info(f"Removed audio pipeline: {pipeline_id}")

    def get_pipeline(self, pipeline_id: str) -> Optional[AudioPipeline]:
        """Get an audio pipeline."""
        return self.pipelines.get(pipeline_id)

    def list_pipelines(self) -> List[str]:
        """List all pipeline IDs."""
        return list(self.pipelines.keys())

    def get_pipeline_stats(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline statistics."""
        pipeline = self.pipelines.get(pipeline_id)
        return pipeline.get_stats() if pipeline else None

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pipelines."""
        return {
            pipeline_id: pipeline.get_stats() for pipeline_id, pipeline in self.pipelines.items()
        }

    async def cleanup_all(self) -> None:
        """Cleanup all pipelines."""
        for pipeline_id in list(self.pipelines.keys()):
            await self.remove_pipeline(pipeline_id)

        logger.info("All audio pipelines cleaned up")
