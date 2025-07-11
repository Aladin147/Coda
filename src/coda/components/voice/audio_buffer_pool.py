"""
Optimized audio buffer pool for high-performance audio processing.

This module provides memory-efficient audio buffer management to reduce
allocations and improve processing performance.
"""

import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Reusable audio buffer with metadata."""

    def __init__(self, size: int, dtype: np.dtype = np.float32):
        """Initialize audio buffer."""
        self.size = size
        self.dtype = dtype
        self.data = np.zeros(size, dtype=dtype)
        self.tensor_data: Optional[torch.Tensor] = None
        self.last_used = time.time()
        self.ref_count = 0
        self.is_dirty = False

    def get_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Get tensor view of buffer data."""
        if self.tensor_data is None or self.tensor_data.device.type != device:
            self.tensor_data = torch.from_numpy(self.data).to(device)
        return self.tensor_data

    def update_from_tensor(self, tensor: torch.Tensor) -> None:
        """Update buffer data from tensor."""
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        self.data[:] = tensor.numpy()
        self.tensor_data = tensor
        self.is_dirty = True

    def resize(self, new_size: int) -> None:
        """Resize buffer if needed."""
        if new_size > self.size:
            self.data = np.zeros(new_size, dtype=self.dtype)
            self.size = new_size
            self.tensor_data = None

    def clear(self) -> None:
        """Clear buffer data."""
        self.data.fill(0)
        self.is_dirty = False
        if self.tensor_data is not None:
            self.tensor_data.zero_()


class AudioBufferPool:
    """High-performance audio buffer pool with automatic management."""

    def __init__(self, max_buffers: int = 50, cleanup_interval: float = 30.0):
        """Initialize audio buffer pool."""
        self.max_buffers = max_buffers
        self.cleanup_interval = cleanup_interval

        # Buffer storage by size
        self.buffers: Dict[int, deque] = {}
        self.active_buffers: Dict[int, AudioBuffer] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = {
            "total_allocations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "buffers_created": 0,
            "buffers_destroyed": 0,
            "peak_active_buffers": 0,
        }

        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

        logger.info(f"AudioBufferPool initialized with max_buffers={max_buffers}")

    def acquire_buffer(self, size: int, dtype: np.dtype = np.float32) -> AudioBuffer:
        """Acquire a buffer from the pool."""
        with self.lock:
            self.stats["total_allocations"] += 1

            # Try to get from pool
            if size in self.buffers and self.buffers[size]:
                buffer = self.buffers[size].popleft()
                buffer.last_used = time.time()
                buffer.ref_count += 1
                self.active_buffers[id(buffer)] = buffer
                self.stats["cache_hits"] += 1

                # Resize if needed
                if buffer.size < size:
                    buffer.resize(size)

                return buffer

            # Create new buffer
            buffer = AudioBuffer(size, dtype)
            buffer.ref_count = 1
            buffer.last_used = time.time()
            self.active_buffers[id(buffer)] = buffer
            self.stats["cache_misses"] += 1
            self.stats["buffers_created"] += 1

            # Update peak statistics
            active_count = len(self.active_buffers)
            if active_count > self.stats["peak_active_buffers"]:
                self.stats["peak_active_buffers"] = active_count

            return buffer

    def release_buffer(self, buffer: AudioBuffer) -> None:
        """Release a buffer back to the pool."""
        with self.lock:
            buffer_id = id(buffer)

            if buffer_id not in self.active_buffers:
                logger.warning("Attempting to release buffer not from this pool")
                return

            buffer.ref_count -= 1

            if buffer.ref_count <= 0:
                # Clear buffer data
                buffer.clear()
                buffer.last_used = time.time()

                # Return to pool if under limit
                total_pooled = sum(len(queue) for queue in self.buffers.values())
                if total_pooled < self.max_buffers:
                    if buffer.size not in self.buffers:
                        self.buffers[buffer.size] = deque()
                    self.buffers[buffer.size].append(buffer)
                else:
                    # Pool is full, destroy buffer
                    self.stats["buffers_destroyed"] += 1

                # Remove from active buffers
                del self.active_buffers[buffer_id]

    def acquire_numpy_buffer(self, size: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """Acquire numpy array buffer."""
        buffer = self.acquire_buffer(size, dtype)
        return buffer.data[:size]  # Return view of correct size

    def acquire_tensor_buffer(
        self, size: int, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Acquire tensor buffer."""
        np_dtype = np.float32 if dtype == torch.float32 else np.float64
        buffer = self.acquire_buffer(size, np_dtype)
        tensor = buffer.get_tensor(device)
        return tensor[:size]  # Return view of correct size

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats["active_buffers"] = len(self.active_buffers)
            stats["pooled_buffers"] = sum(len(queue) for queue in self.buffers.values())
            stats["cache_hit_rate"] = self.stats["cache_hits"] / max(
                1, self.stats["total_allocations"]
            )
            return stats

    def _cleanup_worker(self) -> None:
        """Background cleanup worker."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_old_buffers()
            except Exception as e:
                logger.error(f"Buffer pool cleanup error: {e}")

    def _cleanup_old_buffers(self) -> None:
        """Clean up old unused buffers."""
        current_time = time.time()
        cleanup_threshold = 60.0  # 1 minute

        with self.lock:
            for size, buffer_queue in list(self.buffers.items()):
                # Remove old buffers from queue
                while buffer_queue:
                    buffer = buffer_queue[0]
                    if current_time - buffer.last_used > cleanup_threshold:
                        buffer_queue.popleft()
                        self.stats["buffers_destroyed"] += 1
                    else:
                        break

                # Remove empty queues
                if not buffer_queue:
                    del self.buffers[size]

    def clear_pool(self) -> None:
        """Clear all buffers from pool."""
        with self.lock:
            destroyed_count = sum(len(queue) for queue in self.buffers.values())
            self.buffers.clear()
            self.stats["buffers_destroyed"] += destroyed_count
            logger.info(f"Cleared {destroyed_count} buffers from pool")


class OptimizedAudioProcessor:
    """Audio processor with optimized buffer management."""

    def __init__(self, buffer_pool: Optional[AudioBufferPool] = None):
        """Initialize optimized audio processor."""
        self.buffer_pool = buffer_pool or AudioBufferPool()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Processing cache for common operations
        self.operation_cache: Dict[str, torch.Tensor] = {}
        self.cache_lock = threading.Lock()

        logger.info(f"OptimizedAudioProcessor initialized on {self.device}")

    def process_audio_optimized(self, audio_data: bytes, operations: List[str] = None) -> bytes:
        """Process audio with optimized buffer management."""
        operations = operations or ["normalize", "denoise"]

        # Convert bytes to numpy array efficiently
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Acquire buffer from pool
        buffer = self.buffer_pool.acquire_buffer(len(audio_array))

        try:
            # Copy data to buffer
            buffer.data[: len(audio_array)] = audio_array

            # Get tensor view
            audio_tensor = buffer.get_tensor(self.device)

            # Apply operations in batch
            processed_tensor = self._apply_operations_batch(
                audio_tensor[: len(audio_array)], operations
            )

            # Update buffer from processed tensor
            buffer.update_from_tensor(processed_tensor)

            # Convert back to bytes
            processed_array = buffer.data[: len(audio_array)]
            processed_bytes = (processed_array * 32768.0).astype(np.int16).tobytes()

            return processed_bytes

        finally:
            # Always release buffer
            self.buffer_pool.release_buffer(buffer)

    def _apply_operations_batch(
        self, audio_tensor: torch.Tensor, operations: List[str]
    ) -> torch.Tensor:
        """Apply multiple audio operations in batch."""
        result = audio_tensor.clone()

        for operation in operations:
            if operation == "normalize":
                result = self._normalize_tensor(result)
            elif operation == "denoise":
                result = self._denoise_tensor(result)
            elif operation == "compress":
                result = self._compress_tensor(result)

        return result

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize audio tensor."""
        max_val = torch.max(torch.abs(tensor))
        if max_val > 0:
            return tensor / max_val
        return tensor

    def _denoise_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simple denoising (placeholder for more sophisticated algorithms)."""
        # Simple high-pass filter
        if len(tensor) > 1:
            diff = tensor[1:] - tensor[:-1]
            return torch.cat([tensor[:1], diff])
        return tensor

    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply dynamic range compression."""
        # Simple compression
        return torch.tanh(tensor * 2.0) * 0.8

    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics."""
        return {
            "buffer_pool_stats": self.buffer_pool.get_stats(),
            "device": self.device,
            "cache_size": len(self.operation_cache),
        }


# Global buffer pool instance
_global_buffer_pool: Optional[AudioBufferPool] = None


def get_global_buffer_pool() -> AudioBufferPool:
    """Get or create global buffer pool."""
    global _global_buffer_pool
    if _global_buffer_pool is None:
        _global_buffer_pool = AudioBufferPool()
    return _global_buffer_pool


def cleanup_global_buffer_pool() -> None:
    """Cleanup global buffer pool."""
    global _global_buffer_pool
    if _global_buffer_pool is not None:
        _global_buffer_pool.clear_pool()
        _global_buffer_pool = None
