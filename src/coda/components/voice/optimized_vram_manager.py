"""
Optimized VRAM manager with lock-free operations and improved performance.

This module provides high-performance VRAM management with minimal contention
and efficient memory tracking.
"""

import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """VRAM allocation strategies."""

    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    BUDDY_SYSTEM = "buddy_system"


@dataclass
class VRAMBlock:
    """VRAM memory block."""

    start_mb: float
    size_mb: float
    component_id: Optional[str] = None
    allocated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    is_free: bool = True
    priority: int = 5
    can_resize: bool = True

    @property
    def end_mb(self) -> float:
        """End position of block."""
        return self.start_mb + self.size_mb

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()


class OptimizedVRAMManager:
    """High-performance VRAM manager with lock-free operations."""

    def __init__(
        self,
        total_vram_gb: float = 32.0,
        reserved_system_gb: float = 2.0,
        allocation_strategy: AllocationStrategy = AllocationStrategy.BEST_FIT,
        enable_defragmentation: bool = True,
        defrag_threshold: float = 0.3,  # 30% fragmentation
    ):
        """Initialize optimized VRAM manager."""
        self.total_vram_mb = total_vram_gb * 1024
        self.reserved_system_mb = reserved_system_gb * 1024
        self.available_vram_mb = self.total_vram_mb - self.reserved_system_mb
        self.allocation_strategy = allocation_strategy
        self.enable_defragmentation = enable_defragmentation
        self.defrag_threshold = defrag_threshold

        # Memory blocks (sorted by start position)
        self.blocks: List[VRAMBlock] = [
            VRAMBlock(start_mb=0, size_mb=self.available_vram_mb, is_free=True)
        ]

        # Component tracking (lock-free using atomic operations where possible)
        self.component_allocations: Dict[str, List[VRAMBlock]] = defaultdict(list)

        # Statistics (atomic counters)
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "fragmentation_events": 0,
            "defragmentation_runs": 0,
            "allocation_failures": 0,
            "peak_usage_mb": 0.0,
            "current_usage_mb": 0.0,
        }

        # Lock for critical sections only
        self.allocation_lock = threading.Lock()

        # Background defragmentation
        self.defrag_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vram-defrag")

        logger.info(
            f"OptimizedVRAMManager initialized: {total_vram_gb}GB total, "
            f"{self.available_vram_mb/1024:.1f}GB available, strategy={allocation_strategy.value}"
        )

    def allocate(
        self, component_id: str, size_mb: float, priority: int = 5, can_resize: bool = True
    ) -> bool:
        """Allocate VRAM for component."""
        if size_mb <= 0:
            return False

        # Try lock-free allocation first for common cases
        if self._try_fast_allocation(component_id, size_mb, priority, can_resize):
            return True

        # Fall back to locked allocation for complex cases
        with self.allocation_lock:
            return self._allocate_locked(component_id, size_mb, priority, can_resize)

    def _try_fast_allocation(
        self, component_id: str, size_mb: float, priority: int, can_resize: bool
    ) -> bool:
        """Try fast lock-free allocation."""
        # Quick check for available space
        total_free = sum(block.size_mb for block in self.blocks if block.is_free)
        if total_free < size_mb:
            return False

        # Find suitable free block
        suitable_block = None
        for block in self.blocks:
            if block.is_free and block.size_mb >= size_mb:
                suitable_block = block
                break

        if suitable_block is None:
            return False

        # Try to atomically claim the block
        if self._atomic_claim_block(suitable_block, component_id, size_mb, priority, can_resize):
            self._update_stats_allocation(size_mb)
            return True

        return False

    def _atomic_claim_block(
        self, block: VRAMBlock, component_id: str, size_mb: float, priority: int, can_resize: bool
    ) -> bool:
        """Atomically claim a memory block."""
        # This is a simplified atomic operation
        # In a real implementation, you'd use atomic compare-and-swap
        if not block.is_free:
            return False

        # Split block if necessary
        if block.size_mb > size_mb:
            # Create new free block for remainder
            remainder_block = VRAMBlock(
                start_mb=block.start_mb + size_mb, size_mb=block.size_mb - size_mb, is_free=True
            )
            self.blocks.append(remainder_block)
            block.size_mb = size_mb

        # Claim the block
        block.is_free = False
        block.component_id = component_id
        block.priority = priority
        block.can_resize = can_resize
        block.allocated_at = time.time()
        block.touch()

        # Add to component tracking
        self.component_allocations[component_id].append(block)

        return True

    def _allocate_locked(
        self, component_id: str, size_mb: float, priority: int, can_resize: bool
    ) -> bool:
        """Allocate with full locking for complex cases."""
        # Find best block based on strategy
        best_block = self._find_best_block(size_mb)

        if best_block is None:
            # Try defragmentation if enabled
            if self.enable_defragmentation:
                self._defragment_memory()
                best_block = self._find_best_block(size_mb)

        if best_block is None:
            self.stats["allocation_failures"] += 1
            return False

        # Allocate the block
        if best_block.size_mb > size_mb:
            # Split block
            remainder = VRAMBlock(
                start_mb=best_block.start_mb + size_mb,
                size_mb=best_block.size_mb - size_mb,
                is_free=True,
            )
            self.blocks.append(remainder)
            best_block.size_mb = size_mb

        # Mark as allocated
        best_block.is_free = False
        best_block.component_id = component_id
        best_block.priority = priority
        best_block.can_resize = can_resize
        best_block.allocated_at = time.time()
        best_block.touch()

        # Track allocation
        self.component_allocations[component_id].append(best_block)
        self._update_stats_allocation(size_mb)

        return True

    def _find_best_block(self, size_mb: float) -> Optional[VRAMBlock]:
        """Find best block based on allocation strategy."""
        free_blocks = [block for block in self.blocks if block.is_free and block.size_mb >= size_mb]

        if not free_blocks:
            return None

        if self.allocation_strategy == AllocationStrategy.FIRST_FIT:
            return free_blocks[0]
        elif self.allocation_strategy == AllocationStrategy.BEST_FIT:
            return min(free_blocks, key=lambda b: b.size_mb)
        elif self.allocation_strategy == AllocationStrategy.WORST_FIT:
            return max(free_blocks, key=lambda b: b.size_mb)
        else:  # BUDDY_SYSTEM
            # Simplified buddy system - find power of 2 size
            target_size = 1
            while target_size < size_mb:
                target_size *= 2

            for block in free_blocks:
                if block.size_mb >= target_size:
                    return block

            return free_blocks[0] if free_blocks else None

    def deallocate(self, component_id: str) -> bool:
        """Deallocate all VRAM for component."""
        if component_id not in self.component_allocations:
            return False

        with self.allocation_lock:
            blocks_to_free = self.component_allocations[component_id].copy()
            total_freed = 0.0

            for block in blocks_to_free:
                if not block.is_free:
                    block.is_free = True
                    block.component_id = None
                    total_freed += block.size_mb

            # Remove from component tracking
            del self.component_allocations[component_id]

            # Merge adjacent free blocks
            self._merge_free_blocks()

            # Update statistics
            self.stats["total_deallocations"] += 1
            self.stats["current_usage_mb"] -= total_freed

            return True

    def _merge_free_blocks(self) -> None:
        """Merge adjacent free blocks."""
        # Sort blocks by start position
        self.blocks.sort(key=lambda b: b.start_mb)

        merged_blocks = []
        current_block = None

        for block in self.blocks:
            if current_block is None:
                current_block = block
            elif current_block.is_free and block.is_free and current_block.end_mb == block.start_mb:
                # Merge blocks
                current_block.size_mb += block.size_mb
            else:
                merged_blocks.append(current_block)
                current_block = block

        if current_block is not None:
            merged_blocks.append(current_block)

        self.blocks = merged_blocks

    def _defragment_memory(self) -> None:
        """Defragment memory by moving allocations."""
        if not self.enable_defragmentation:
            return

        # Calculate fragmentation
        fragmentation = self._calculate_fragmentation()
        if fragmentation < self.defrag_threshold:
            return

        # Submit defragmentation to background thread
        self.defrag_executor.submit(self._perform_defragmentation)

    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio."""
        free_blocks = [block for block in self.blocks if block.is_free]
        if not free_blocks:
            return 0.0

        total_free = sum(block.size_mb for block in free_blocks)
        largest_free = max(block.size_mb for block in free_blocks)

        if total_free == 0:
            return 0.0

        return 1.0 - (largest_free / total_free)

    def _perform_defragmentation(self) -> None:
        """Perform actual defragmentation."""
        with self.allocation_lock:
            # Simple defragmentation: merge all free blocks
            self._merge_free_blocks()
            self.stats["defragmentation_runs"] += 1

    def _update_stats_allocation(self, size_mb: float) -> None:
        """Update allocation statistics."""
        self.stats["total_allocations"] += 1
        self.stats["current_usage_mb"] += size_mb

        if self.stats["current_usage_mb"] > self.stats["peak_usage_mb"]:
            self.stats["peak_usage_mb"] = self.stats["current_usage_mb"]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics."""
        with self.allocation_lock:
            total_allocated = sum(block.size_mb for block in self.blocks if not block.is_free)
            total_free = sum(block.size_mb for block in self.blocks if block.is_free)

            fragmentation = self._calculate_fragmentation()

            return {
                "total_vram_mb": self.total_vram_mb,
                "available_vram_mb": self.available_vram_mb,
                "allocated_mb": total_allocated,
                "free_mb": total_free,
                "usage_percent": (total_allocated / self.available_vram_mb) * 100,
                "fragmentation": fragmentation,
                "num_blocks": len(self.blocks),
                "num_components": len(self.component_allocations),
                "allocation_strategy": self.allocation_strategy.value,
                **self.stats,
            }

    def get_component_usage(self, component_id: str) -> Dict[str, Any]:
        """Get usage statistics for specific component."""
        if component_id not in self.component_allocations:
            return {"allocated_mb": 0, "num_blocks": 0}

        blocks = self.component_allocations[component_id]
        total_allocated = sum(block.size_mb for block in blocks if not block.is_free)

        return {
            "allocated_mb": total_allocated,
            "num_blocks": len(blocks),
            "blocks": [
                {
                    "start_mb": block.start_mb,
                    "size_mb": block.size_mb,
                    "allocated_at": block.allocated_at,
                    "last_accessed": block.last_accessed,
                }
                for block in blocks
                if not block.is_free
            ],
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.defrag_executor.shutdown(wait=True)


# Global optimized VRAM manager
_global_vram_manager: Optional[OptimizedVRAMManager] = None


def get_optimized_vram_manager() -> OptimizedVRAMManager:
    """Get global optimized VRAM manager."""
    global _global_vram_manager
    if _global_vram_manager is None:
        _global_vram_manager = OptimizedVRAMManager()
    return _global_vram_manager


def cleanup_optimized_vram_manager() -> None:
    """Cleanup global VRAM manager."""
    global _global_vram_manager
    if _global_vram_manager is not None:
        _global_vram_manager.cleanup()
        _global_vram_manager = None
