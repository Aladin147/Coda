"""
VRAM management and optimization for Coda 2.0 voice system.

This module provides comprehensive VRAM management including:
- Dynamic VRAM allocation and monitoring
- Component-based memory tracking
- Optimization strategies and garbage collection
- Memory pressure detection and response
"""

import asyncio
import logging
import time
import torch
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import psutil

from .models import VoiceConfig

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available, advanced GPU monitoring disabled")

from .models import VoiceConfig
from .utils import VRAMAllocation

logger = logging.getLogger(__name__)


class MemoryPressure(str, Enum):
    """Memory pressure levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComponentAllocation:
    """VRAM allocation for a component."""
    component_id: str
    allocated_mb: float
    reserved_mb: float
    max_mb: float
    priority: int  # 1-10, higher is more important
    can_resize: bool
    last_used: datetime
    
    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        return (self.allocated_mb / self.max_mb) * 100 if self.max_mb > 0 else 0


@dataclass
class MemoryEvent:
    """Memory allocation/deallocation event."""
    timestamp: datetime
    event_type: str  # allocate, deallocate, resize, optimize
    component_id: str
    size_mb: float
    total_allocated_mb: float
    pressure_level: MemoryPressure


class VRAMMonitor:
    """VRAM monitoring and metrics collection."""
    
    def __init__(self, device_id: int = 0):
        """Initialize VRAM monitor."""
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[VRAMAllocation] = []
        self.max_history = 1000
        
        # Initialize NVML if available
        self.nvml_handle = None
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                logger.info("NVML initialized for advanced GPU monitoring")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
        
        # Get total VRAM
        self.total_vram_mb = 0.0
        if torch.cuda.is_available():
            self.total_vram_mb = torch.cuda.get_device_properties(device_id).total_memory / (1024 * 1024)
            logger.info(f"VRAM Monitor initialized: {self.total_vram_mb:.0f}MB total")
    
    def start_monitoring(self, interval_seconds: float = 1.0) -> None:
        """Start VRAM monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("VRAM monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop VRAM monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("VRAM monitoring stopped")
    
    def get_current_allocation(self) -> VRAMAllocation:
        """Get current VRAM allocation."""
        if not torch.cuda.is_available():
            return VRAMAllocation(0, 0, 0, 0, 0)
        
        try:
            allocated_mb = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved(self.device_id) / (1024 * 1024)
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
            logger.error(f"Failed to get VRAM allocation: {e}")
            return VRAMAllocation(0, 0, 0, 0, 0)
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """Get detailed GPU information."""
        info = {
            'basic': self.get_current_allocation().__dict__,
            'device_id': self.device_id,
            'device_name': torch.cuda.get_device_name(self.device_id) if torch.cuda.is_available() else "CPU",
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }
        
        # Add NVML info if available
        if self.nvml_handle:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0  # Convert to watts
                
                info['advanced'] = {
                    'gpu_utilization': gpu_util.gpu,
                    'memory_utilization': gpu_util.memory,
                    'temperature_c': temp,
                    'power_usage_w': power
                }
            except Exception as e:
                logger.debug(f"Failed to get advanced GPU info: {e}")
        
        return info
    
    def get_history(self, minutes: int = 10) -> List[VRAMAllocation]:
        """Get VRAM allocation history."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            allocation for allocation in self.metrics_history
            if hasattr(allocation, 'timestamp') and allocation.timestamp >= cutoff_time
        ]
    
    def _monitor_loop(self, interval_seconds: float) -> None:
        """VRAM monitoring loop."""
        while self.monitoring:
            try:
                allocation = self.get_current_allocation()
                allocation.timestamp = datetime.now()  # Add timestamp
                
                self.metrics_history.append(allocation)
                
                # Keep only recent history
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"VRAM monitoring error: {e}")
                time.sleep(interval_seconds)


class DynamicVRAMManager:
    """Dynamic VRAM allocation and management."""
    
    def __init__(self, config: VoiceConfig):
        """Initialize VRAM manager."""
        self.config = config
        self.device_id = 0
        self.monitor = VRAMMonitor(self.device_id)
        
        # Component allocations
        self.allocations: Dict[str, ComponentAllocation] = {}
        self.allocation_lock = threading.Lock()
        
        # Memory management settings
        self.total_vram_mb = float(config.total_vram.replace('GB', '')) * 1024
        self.reserved_system_mb = float(config.reserved_system.replace('GB', '')) * 1024
        self.available_vram_mb = self.total_vram_mb - self.reserved_system_mb
        
        # Memory pressure thresholds
        self.pressure_thresholds = {
            MemoryPressure.LOW: 0.6,      # 60% usage
            MemoryPressure.MEDIUM: 0.75,  # 75% usage
            MemoryPressure.HIGH: 0.85,    # 85% usage
            MemoryPressure.CRITICAL: 0.95 # 95% usage
        }
        
        # Event history
        self.events: List[MemoryEvent] = []
        self.max_events = 1000
        
        # Optimization callbacks
        self.pressure_callbacks: Dict[MemoryPressure, List[Callable]] = {
            level: [] for level in MemoryPressure
        }
        
        logger.info(f"VRAM Manager initialized: {self.total_vram_mb:.0f}MB total, {self.available_vram_mb:.0f}MB available")
    
    async def initialize(self) -> None:
        """Initialize VRAM manager."""
        try:
            # Start monitoring
            self.monitor.start_monitoring(interval_seconds=1.0)
            
            # Verify VRAM availability
            current = self.monitor.get_current_allocation()
            if current.total_mb < self.total_vram_mb * 0.9:
                logger.warning(f"Detected VRAM ({current.total_mb:.0f}MB) less than configured ({self.total_vram_mb:.0f}MB)")
                self.total_vram_mb = current.total_mb
                self.available_vram_mb = self.total_vram_mb - self.reserved_system_mb
            
            logger.info("VRAM Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VRAM Manager: {e}")
            raise
    
    def register_component(
        self,
        component_id: str,
        max_mb: float,
        priority: int = 5,
        can_resize: bool = True
    ) -> bool:
        """Register a component for VRAM allocation."""
        with self.allocation_lock:
            if component_id in self.allocations:
                logger.warning(f"Component {component_id} already registered")
                return False
            
            # Check if we have enough VRAM
            total_allocated = sum(alloc.max_mb for alloc in self.allocations.values())
            if total_allocated + max_mb > self.available_vram_mb:
                logger.error(f"Not enough VRAM for {component_id}: need {max_mb}MB, available {self.available_vram_mb - total_allocated}MB")
                return False
            
            allocation = ComponentAllocation(
                component_id=component_id,
                allocated_mb=0.0,
                reserved_mb=0.0,
                max_mb=max_mb,
                priority=priority,
                can_resize=can_resize,
                last_used=datetime.now()
            )
            
            self.allocations[component_id] = allocation
            
            self._record_event(
                event_type="register",
                component_id=component_id,
                size_mb=max_mb
            )
            
            logger.info(f"Registered component {component_id}: max {max_mb}MB, priority {priority}")
            return True
    
    def allocate(self, component_id: str, size_mb: float) -> bool:
        """Allocate VRAM for a component."""
        with self.allocation_lock:
            if component_id not in self.allocations:
                logger.error(f"Component {component_id} not registered")
                return False
            
            allocation = self.allocations[component_id]
            
            # Check if allocation exceeds maximum
            if allocation.allocated_mb + size_mb > allocation.max_mb:
                logger.error(f"Allocation exceeds maximum for {component_id}: {allocation.allocated_mb + size_mb}MB > {allocation.max_mb}MB")
                return False
            
            # Check memory pressure
            pressure = self.get_memory_pressure()
            if pressure == MemoryPressure.CRITICAL:
                logger.warning(f"Critical memory pressure, allocation for {component_id} may fail")
                # Try to free memory
                self._handle_memory_pressure(pressure)
            
            # Perform allocation
            try:
                allocation.allocated_mb += size_mb
                allocation.last_used = datetime.now()
                
                self._record_event(
                    event_type="allocate",
                    component_id=component_id,
                    size_mb=size_mb
                )
                
                logger.debug(f"Allocated {size_mb}MB for {component_id}, total: {allocation.allocated_mb}MB")
                return True
                
            except Exception as e:
                logger.error(f"Failed to allocate VRAM for {component_id}: {e}")
                return False
    
    def deallocate(self, component_id: str, size_mb: Optional[float] = None) -> bool:
        """Deallocate VRAM for a component."""
        with self.allocation_lock:
            if component_id not in self.allocations:
                logger.error(f"Component {component_id} not registered")
                return False
            
            allocation = self.allocations[component_id]
            
            # If no size specified, deallocate all
            if size_mb is None:
                size_mb = allocation.allocated_mb
            
            # Ensure we don't deallocate more than allocated
            size_mb = min(size_mb, allocation.allocated_mb)
            
            try:
                allocation.allocated_mb -= size_mb
                allocation.last_used = datetime.now()
                
                self._record_event(
                    event_type="deallocate",
                    component_id=component_id,
                    size_mb=size_mb
                )
                
                logger.debug(f"Deallocated {size_mb}MB for {component_id}, remaining: {allocation.allocated_mb}MB")
                return True
                
            except Exception as e:
                logger.error(f"Failed to deallocate VRAM for {component_id}: {e}")
                return False
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component and deallocate all its VRAM."""
        with self.allocation_lock:
            if component_id not in self.allocations:
                logger.warning(f"Component {component_id} not registered")
                return False
            
            # Deallocate all memory
            self.deallocate(component_id)
            
            # Remove from allocations
            del self.allocations[component_id]
            
            self._record_event(
                event_type="unregister",
                component_id=component_id,
                size_mb=0.0
            )
            
            logger.info(f"Unregistered component {component_id}")
            return True
    
    def get_memory_pressure(self) -> MemoryPressure:
        """Get current memory pressure level."""
        current = self.monitor.get_current_allocation()
        utilization = current.utilization_percent / 100.0
        
        for pressure, threshold in reversed(list(self.pressure_thresholds.items())):
            if utilization >= threshold:
                return pressure
        
        return MemoryPressure.LOW
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize VRAM usage."""
        optimization_results = {
            'freed_mb': 0.0,
            'actions_taken': [],
            'pressure_before': self.get_memory_pressure(),
            'pressure_after': None
        }
        
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimization_results['actions_taken'].append('cleared_pytorch_cache')
            
            # Handle memory pressure
            pressure = self.get_memory_pressure()
            if pressure != MemoryPressure.LOW:
                freed_mb = self._handle_memory_pressure(pressure)
                optimization_results['freed_mb'] += freed_mb
                optimization_results['actions_taken'].append(f'handled_{pressure.value}_pressure')
            
            optimization_results['pressure_after'] = self.get_memory_pressure()
            
            self._record_event(
                event_type="optimize",
                component_id="system",
                size_mb=optimization_results['freed_mb']
            )
            
            logger.info(f"Memory optimization completed: freed {optimization_results['freed_mb']:.1f}MB")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
        
        return optimization_results
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of all allocations."""
        with self.allocation_lock:
            current = self.monitor.get_current_allocation()
            
            component_summary = {}
            total_allocated = 0.0
            total_max = 0.0
            
            for comp_id, allocation in self.allocations.items():
                component_summary[comp_id] = {
                    'allocated_mb': allocation.allocated_mb,
                    'max_mb': allocation.max_mb,
                    'utilization': allocation.utilization,
                    'priority': allocation.priority,
                    'can_resize': allocation.can_resize,
                    'last_used': allocation.last_used.isoformat()
                }
                total_allocated += allocation.allocated_mb
                total_max += allocation.max_mb
            
            return {
                'system': {
                    'total_vram_mb': self.total_vram_mb,
                    'available_vram_mb': self.available_vram_mb,
                    'reserved_system_mb': self.reserved_system_mb,
                    'current_allocated_mb': current.allocated_mb,
                    'current_utilization': current.utilization_percent,
                    'memory_pressure': self.get_memory_pressure().value
                },
                'components': component_summary,
                'totals': {
                    'registered_allocated_mb': total_allocated,
                    'registered_max_mb': total_max,
                    'untracked_mb': max(0, current.allocated_mb - total_allocated)
                }
            }
    
    def add_pressure_callback(self, pressure: MemoryPressure, callback: Callable) -> None:
        """Add callback for memory pressure events."""
        self.pressure_callbacks[pressure].append(callback)
    
    def _handle_memory_pressure(self, pressure: MemoryPressure) -> float:
        """Handle memory pressure by freeing memory."""
        freed_mb = 0.0
        
        # Call pressure callbacks
        for callback in self.pressure_callbacks[pressure]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Pressure callback failed: {e}")
        
        # For high/critical pressure, try to resize components
        if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            with self.allocation_lock:
                # Sort components by priority (lower priority first) and last used
                components = sorted(
                    self.allocations.items(),
                    key=lambda x: (x[1].priority, x[1].last_used)
                )
                
                for comp_id, allocation in components:
                    if allocation.can_resize and allocation.allocated_mb > 0:
                        # Free some memory from low-priority components
                        reduction = allocation.allocated_mb * 0.2  # Reduce by 20%
                        if self.deallocate(comp_id, reduction):
                            freed_mb += reduction
                            logger.info(f"Reduced {comp_id} allocation by {reduction:.1f}MB due to memory pressure")
                        
                        # Stop if we've freed enough
                        if freed_mb > 1024:  # 1GB
                            break
        
        return freed_mb
    
    def _record_event(self, event_type: str, component_id: str, size_mb: float) -> None:
        """Record a memory event."""
        current = self.monitor.get_current_allocation()
        
        event = MemoryEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            component_id=component_id,
            size_mb=size_mb,
            total_allocated_mb=current.allocated_mb,
            pressure_level=self.get_memory_pressure()
        )
        
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    async def cleanup(self) -> None:
        """Cleanup VRAM manager."""
        try:
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Unregister all components
            with self.allocation_lock:
                for component_id in list(self.allocations.keys()):
                    self.unregister_component(component_id)
            
            # Final optimization
            self.optimize_memory()
            
            logger.info("VRAM Manager cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup VRAM Manager: {e}")


# Global VRAM manager instance
_vram_manager: Optional[DynamicVRAMManager] = None


def get_vram_manager() -> Optional[DynamicVRAMManager]:
    """Get global VRAM manager instance."""
    return _vram_manager


def initialize_vram_manager(config: VoiceConfig) -> DynamicVRAMManager:
    """Initialize global VRAM manager."""
    global _vram_manager
    _vram_manager = DynamicVRAMManager(config)
    return _vram_manager
