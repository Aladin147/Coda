"""
Utility functions and classes for Coda 2.0 voice system.

This module provides common utilities for voice processing including:
- Audio format conversion and validation
- Performance monitoring and metrics
- VRAM management and optimization
- Streaming utilities and helpers
"""

import asyncio
import logging
import time
import psutil
import torch
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring disabled")

from .models import AudioFormat, VoiceProcessingMode

logger = logging.getLogger(__name__)


class PerformanceLevel(str, Enum):
    """Performance levels for optimization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class PerformanceMetrics:
    """Performance metrics for voice processing."""
    latency_ms: float
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float
    gpu_memory_mb: float
    throughput_chunks_per_sec: float
    timestamp: datetime


@dataclass
class VRAMAllocation:
    """VRAM allocation information."""
    total_mb: float
    allocated_mb: float
    free_mb: float
    reserved_mb: float
    utilization_percent: float


class PerformanceMonitor:
    """Performance monitoring for voice processing."""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance monitor."""
        self.window_size = window_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.chunk_count = 0
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
    
    def record_chunk_processed(self) -> None:
        """Record that a chunk was processed."""
        self.chunk_count += 1
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        try:
            # CPU and memory metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_usage_mb = memory_info.used / 1024 / 1024
            
            # GPU metrics
            gpu_usage = 0.0
            gpu_memory_mb = 0.0
            
            if self.nvml_initialized:
                try:
                    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_usage = gpu_info.gpu
                    
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_memory_mb = memory_info.used / 1024 / 1024
                except Exception as e:
                    logger.debug(f"GPU metrics error: {e}")
            
            # Throughput calculation
            elapsed_time = time.time() - self.start_time
            throughput = self.chunk_count / elapsed_time if elapsed_time > 0 else 0.0
            
            # Latency (placeholder - should be measured per operation)
            latency_ms = 0.0
            if self.metrics_history:
                latency_ms = self.metrics_history[-1].latency_ms
            
            metrics = PerformanceMetrics(
                latency_ms=latency_ms,
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                gpu_usage=gpu_usage,
                gpu_memory_mb=gpu_memory_mb,
                throughput_chunks_per_sec=throughput,
                timestamp=datetime.now()
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.window_size:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(
                latency_ms=0.0,
                cpu_usage=0.0,
                memory_usage_mb=0.0,
                gpu_usage=0.0,
                gpu_memory_mb=0.0,
                throughput_chunks_per_sec=0.0,
                timestamp=datetime.now()
            )
    
    def get_average_metrics(self, window_minutes: int = 5) -> Optional[PerformanceMetrics]:
        """Get average metrics over a time window."""
        if not self.metrics_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        return PerformanceMetrics(
            latency_ms=sum(m.latency_ms for m in recent_metrics) / len(recent_metrics),
            cpu_usage=sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            memory_usage_mb=sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            gpu_usage=sum(m.gpu_usage for m in recent_metrics) / len(recent_metrics),
            gpu_memory_mb=sum(m.gpu_memory_mb for m in recent_metrics) / len(recent_metrics),
            throughput_chunks_per_sec=sum(m.throughput_chunks_per_sec for m in recent_metrics) / len(recent_metrics),
            timestamp=datetime.now()
        )
    
    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        if self.metrics_history:
            self.metrics_history[-1].latency_ms = latency_ms


class VRAMManager:
    """VRAM management and optimization."""
    
    def __init__(self):
        """Initialize VRAM manager."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_vram_mb = 0.0
        self.allocations: Dict[str, float] = {}
        
        if torch.cuda.is_available():
            self.total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            logger.info(f"VRAM Manager initialized: {self.total_vram_mb:.0f}MB total")
        else:
            logger.warning("CUDA not available, VRAM management disabled")
    
    def get_vram_info(self) -> VRAMAllocation:
        """Get current VRAM allocation information."""
        if not torch.cuda.is_available():
            return VRAMAllocation(0, 0, 0, 0, 0)
        
        try:
            allocated_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
            reserved_mb = torch.cuda.memory_reserved(0) / 1024 / 1024
            free_mb = self.total_vram_mb - reserved_mb
            utilization = (allocated_mb / self.total_vram_mb) * 100 if self.total_vram_mb > 0 else 0
            
            return VRAMAllocation(
                total_mb=self.total_vram_mb,
                allocated_mb=allocated_mb,
                free_mb=free_mb,
                reserved_mb=reserved_mb,
                utilization_percent=utilization
            )
        except Exception as e:
            logger.error(f"Failed to get VRAM info: {e}")
            return VRAMAllocation(0, 0, 0, 0, 0)
    
    def register_allocation(self, component: str, size_mb: float) -> None:
        """Register a VRAM allocation for a component."""
        self.allocations[component] = size_mb
        logger.info(f"Registered VRAM allocation: {component} = {size_mb:.0f}MB")
    
    def unregister_allocation(self, component: str) -> None:
        """Unregister a VRAM allocation."""
        if component in self.allocations:
            size_mb = self.allocations.pop(component)
            logger.info(f"Unregistered VRAM allocation: {component} = {size_mb:.0f}MB")
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of VRAM allocations."""
        vram_info = self.get_vram_info()
        
        return {
            'total_vram_mb': vram_info.total_mb,
            'allocated_mb': vram_info.allocated_mb,
            'free_mb': vram_info.free_mb,
            'utilization_percent': vram_info.utilization_percent,
            'component_allocations': self.allocations.copy(),
            'total_registered_mb': sum(self.allocations.values())
        }
    
    def optimize_memory(self) -> None:
        """Optimize VRAM usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("VRAM cache cleared")
    
    def check_available_space(self, required_mb: float) -> bool:
        """Check if enough VRAM is available for allocation."""
        vram_info = self.get_vram_info()
        available_mb = vram_info.free_mb - 1024  # Keep 1GB buffer
        
        return available_mb >= required_mb


class AudioUtils:
    """Audio processing utilities."""
    
    @staticmethod
    def validate_audio_format(format_str: str) -> bool:
        """Validate audio format string."""
        try:
            AudioFormat(format_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def calculate_audio_duration(audio_data: bytes, sample_rate: int, channels: int = 1, bit_depth: int = 16) -> float:
        """Calculate audio duration in seconds."""
        bytes_per_sample = bit_depth // 8
        total_samples = len(audio_data) // (channels * bytes_per_sample)
        return total_samples / sample_rate
    
    @staticmethod
    def calculate_audio_size(duration_seconds: float, sample_rate: int, channels: int = 1, bit_depth: int = 16) -> int:
        """Calculate audio size in bytes."""
        bytes_per_sample = bit_depth // 8
        total_samples = int(duration_seconds * sample_rate)
        return total_samples * channels * bytes_per_sample
    
    @staticmethod
    def resample_audio(audio_data: bytes, source_rate: int, target_rate: int) -> bytes:
        """Resample audio data (basic implementation)."""
        if source_rate == target_rate:
            return audio_data
        
        # Convert to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Simple linear interpolation resampling
        ratio = target_rate / source_rate
        new_length = int(len(audio_np) * ratio)
        
        # Create new indices
        old_indices = np.linspace(0, len(audio_np) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(old_indices, np.arange(len(audio_np)), audio_np)
        
        return resampled.astype(np.int16).tobytes()
    
    @staticmethod
    def normalize_audio(audio_data: bytes) -> bytes:
        """Normalize audio data."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        if len(audio_np) == 0:
            return audio_data
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            normalized = (audio_np / max_val * 32767 * 0.95).astype(np.int16)
            return normalized.tobytes()
        
        return audio_data


class StreamingUtils:
    """Utilities for streaming audio processing."""
    
    @staticmethod
    async def chunk_audio_stream(
        audio_stream: AsyncGenerator[bytes, None],
        chunk_size: int = 1024
    ) -> AsyncGenerator[bytes, None]:
        """Chunk audio stream into fixed-size pieces."""
        buffer = b''
        
        async for data in audio_stream:
            buffer += data
            
            while len(buffer) >= chunk_size:
                yield buffer[:chunk_size]
                buffer = buffer[chunk_size:]
        
        # Yield remaining data
        if buffer:
            yield buffer
    
    @staticmethod
    async def buffer_audio_stream(
        audio_stream: AsyncGenerator[bytes, None],
        buffer_duration_ms: int = 100,
        sample_rate: int = 24000
    ) -> AsyncGenerator[bytes, None]:
        """Buffer audio stream for specified duration."""
        buffer_size = AudioUtils.calculate_audio_size(
            buffer_duration_ms / 1000.0, sample_rate
        )
        
        async for chunk in StreamingUtils.chunk_audio_stream(audio_stream, buffer_size):
            yield chunk
    
    @staticmethod
    async def merge_audio_streams(
        *streams: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """Merge multiple audio streams."""
        # Simple implementation - more sophisticated mixing can be added
        async def stream_to_list(stream):
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            return chunks
        
        # Collect all streams
        all_chunks = await asyncio.gather(*[stream_to_list(stream) for stream in streams])
        
        # Yield chunks in order
        max_length = max(len(chunks) for chunks in all_chunks) if all_chunks else 0
        
        for i in range(max_length):
            merged_chunk = b''
            for chunks in all_chunks:
                if i < len(chunks):
                    # Simple concatenation - proper audio mixing would require sample-level operations
                    merged_chunk += chunks[i]
            
            if merged_chunk:
                yield merged_chunk


class LatencyTracker:
    """Track and analyze latency metrics."""
    
    def __init__(self, name: str):
        """Initialize latency tracker."""
        self.name = name
        self.start_time: Optional[float] = None
        self.measurements: List[float] = []
        self.max_measurements = 1000
    
    def start(self) -> None:
        """Start timing."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return latency in milliseconds."""
        if self.start_time is None:
            return 0.0
        
        latency_ms = (time.perf_counter() - self.start_time) * 1000
        self.measurements.append(latency_ms)
        
        # Keep only recent measurements
        if len(self.measurements) > self.max_measurements:
            self.measurements = self.measurements[-self.max_measurements:]
        
        self.start_time = None
        return latency_ms

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.measurements.append(latency_ms)

        # Keep only recent measurements
        if len(self.measurements) > self.max_measurements:
            self.measurements = self.measurements[-self.max_measurements:]

    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.measurements:
            return {'count': 0, 'avg': 0.0, 'min': 0.0, 'max': 0.0, 'p95': 0.0, 'p99': 0.0}
        
        measurements = sorted(self.measurements)
        count = len(measurements)
        
        return {
            'count': count,
            'avg': sum(measurements) / count,
            'min': measurements[0],
            'max': measurements[-1],
            'p95': measurements[int(count * 0.95)] if count > 0 else 0.0,
            'p99': measurements[int(count * 0.99)] if count > 0 else 0.0,
        }


# Context managers for performance tracking

class LatencyContext:
    """Context manager for latency tracking."""
    
    def __init__(self, tracker: LatencyTracker):
        """Initialize latency context."""
        self.tracker = tracker
        self.latency_ms = 0.0
    
    def __enter__(self) -> 'LatencyContext':
        """Enter context and start timing."""
        self.tracker.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and stop timing."""
        self.latency_ms = self.tracker.stop()
    
    def get_latency(self) -> float:
        """Get measured latency."""
        return self.latency_ms


class LatencyContext:
    """Context manager for tracking latency."""

    def __init__(self, tracker: LatencyTracker):
        """Initialize latency context."""
        self.tracker = tracker
        self.start_time = None
        self.latency = None

    def __enter__(self):
        """Enter context and start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record latency."""
        if self.start_time is not None:
            self.latency = (time.time() - self.start_time) * 1000  # Convert to ms
            self.tracker.record_latency(self.latency)

    def get_latency(self) -> float:
        """Get the recorded latency."""
        return self.latency or 0.0


def latency_tracker(name: str) -> LatencyTracker:
    """Create a latency tracker."""
    return LatencyTracker(name)


def track_latency(tracker: LatencyTracker) -> LatencyContext:
    """Create a latency tracking context."""
    return LatencyContext(tracker)
