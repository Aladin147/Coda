"""
Voice Latency Optimization

This module provides comprehensive latency optimization for real-time voice conversations,
focusing on minimizing end-to-end response times while maintaining quality.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

from .models import VoiceMessage, VoiceResponse, VoiceProcessingMode
from .parallel_processor import ParallelVoiceProcessor, ParallelProcessingConfig, ProcessingPriority
from .mode_manager import ProcessingModeManager, ProcessingModeType

logger = logging.getLogger("coda.voice.latency_optimizer")


class LatencyTarget(str, Enum):
    """Latency optimization targets."""
    ULTRA_LOW = "ultra_low"      # < 100ms (gaming/real-time)
    LOW = "low"                  # < 200ms (conversational)
    MODERATE = "moderate"        # < 500ms (interactive)
    RELAXED = "relaxed"          # < 1000ms (batch processing)


class OptimizationTechnique(str, Enum):
    """Available optimization techniques."""
    PREDICTIVE_CACHING = "predictive_caching"
    SPECULATIVE_PROCESSING = "speculative_processing"
    PIPELINE_PARALLELIZATION = "pipeline_parallelization"
    MODEL_QUANTIZATION = "model_quantization"
    CONTEXT_COMPRESSION = "context_compression"
    EARLY_TERMINATION = "early_termination"
    BATCH_OPTIMIZATION = "batch_optimization"


@dataclass
class LatencyMeasurement:
    """Detailed latency measurement."""
    
    timestamp: float
    total_latency_ms: float
    
    # Component latencies
    audio_preprocessing_ms: float = 0.0
    moshi_processing_ms: float = 0.0
    llm_processing_ms: float = 0.0
    context_building_ms: float = 0.0
    response_generation_ms: float = 0.0
    audio_postprocessing_ms: float = 0.0
    
    # Network/IO latencies
    network_latency_ms: float = 0.0
    disk_io_ms: float = 0.0
    
    # Queue/waiting times
    queue_wait_ms: float = 0.0
    resource_wait_ms: float = 0.0
    
    # Metadata
    processing_mode: VoiceProcessingMode = VoiceProcessingMode.HYBRID
    message_length: int = 0
    complexity_score: float = 0.0


@dataclass
class LatencyOptimizationConfig:
    """Configuration for latency optimization."""
    
    # Target latency
    target_latency: LatencyTarget = LatencyTarget.LOW
    target_latency_ms: float = 200.0
    
    # Enabled optimizations
    enabled_techniques: List[OptimizationTechnique] = field(default_factory=lambda: [
        OptimizationTechnique.PREDICTIVE_CACHING,
        OptimizationTechnique.PIPELINE_PARALLELIZATION,
        OptimizationTechnique.CONTEXT_COMPRESSION
    ])
    
    # Measurement and monitoring
    measurement_window_size: int = 100
    latency_alert_threshold_ms: float = 500.0
    
    # Predictive caching
    cache_prediction_window: int = 10
    cache_hit_target_rate: float = 0.8
    
    # Speculative processing
    enable_speculative_processing: bool = False
    speculation_confidence_threshold: float = 0.7
    
    # Pipeline optimization
    enable_pipeline_parallelization: bool = True
    max_pipeline_stages: int = 4
    
    # Early termination
    enable_early_termination: bool = True
    early_termination_confidence: float = 0.9
    
    # Adaptive optimization
    enable_adaptive_optimization: bool = True
    adaptation_learning_rate: float = 0.1


class LatencyProfiler:
    """Profiles and measures latency across voice processing pipeline."""
    
    def __init__(self, config: LatencyOptimizationConfig):
        self.config = config
        self.measurements: deque = deque(maxlen=config.measurement_window_size)
        self.component_stats: Dict[str, List[float]] = {}
        
        # Real-time tracking
        self.current_measurements: Dict[str, float] = {}
        self.measurement_start_time: Optional[float] = None
        
    def start_measurement(self, message_id: str) -> None:
        """Start latency measurement for a request."""
        self.measurement_start_time = time.time()
        self.current_measurements = {"message_id": message_id}
        
    def record_component_latency(self, component: str, latency_ms: float) -> None:
        """Record latency for a specific component."""
        self.current_measurements[f"{component}_ms"] = latency_ms
        
        # Update component stats
        if component not in self.component_stats:
            self.component_stats[component] = []
        self.component_stats[component].append(latency_ms)
        
        # Keep only recent measurements
        if len(self.component_stats[component]) > self.config.measurement_window_size:
            self.component_stats[component] = self.component_stats[component][-self.config.measurement_window_size:]
    
    def finish_measurement(
        self,
        processing_mode: VoiceProcessingMode,
        message_length: int,
        complexity_score: float = 0.0
    ) -> LatencyMeasurement:
        """Finish measurement and create LatencyMeasurement."""
        
        if self.measurement_start_time is None:
            raise RuntimeError("Measurement not started")
        
        total_latency_ms = (time.time() - self.measurement_start_time) * 1000
        
        measurement = LatencyMeasurement(
            timestamp=time.time(),
            total_latency_ms=total_latency_ms,
            audio_preprocessing_ms=self.current_measurements.get("audio_preprocessing_ms", 0.0),
            moshi_processing_ms=self.current_measurements.get("moshi_processing_ms", 0.0),
            llm_processing_ms=self.current_measurements.get("llm_processing_ms", 0.0),
            context_building_ms=self.current_measurements.get("context_building_ms", 0.0),
            response_generation_ms=self.current_measurements.get("response_generation_ms", 0.0),
            audio_postprocessing_ms=self.current_measurements.get("audio_postprocessing_ms", 0.0),
            network_latency_ms=self.current_measurements.get("network_latency_ms", 0.0),
            disk_io_ms=self.current_measurements.get("disk_io_ms", 0.0),
            queue_wait_ms=self.current_measurements.get("queue_wait_ms", 0.0),
            resource_wait_ms=self.current_measurements.get("resource_wait_ms", 0.0),
            processing_mode=processing_mode,
            message_length=message_length,
            complexity_score=complexity_score
        )
        
        self.measurements.append(measurement)
        self.measurement_start_time = None
        self.current_measurements.clear()
        
        return measurement
    
    def get_latency_statistics(self) -> Dict[str, Any]:
        """Get comprehensive latency statistics."""
        
        if not self.measurements:
            return {"status": "no_data"}
        
        # Overall statistics
        total_latencies = [m.total_latency_ms for m in self.measurements]
        
        stats = {
            "overall": {
                "count": len(total_latencies),
                "mean_ms": statistics.mean(total_latencies),
                "median_ms": statistics.median(total_latencies),
                "min_ms": min(total_latencies),
                "max_ms": max(total_latencies),
                "std_dev_ms": statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0.0,
                "p95_ms": sorted(total_latencies)[int(len(total_latencies) * 0.95)] if total_latencies else 0.0,
                "p99_ms": sorted(total_latencies)[int(len(total_latencies) * 0.99)] if total_latencies else 0.0
            },
            "components": {}
        }
        
        # Component-wise statistics
        for component, latencies in self.component_stats.items():
            if latencies and total_latencies:
                mean_component = statistics.mean(latencies)
                mean_total = statistics.mean(total_latencies)
                contribution_percent = (mean_component / mean_total) * 100 if mean_total > 0 else 0.0

                stats["components"][component] = {
                    "mean_ms": mean_component,
                    "median_ms": statistics.median(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "contribution_percent": contribution_percent
                }
        
        # Processing mode breakdown
        mode_latencies = {}
        for measurement in self.measurements:
            mode = measurement.processing_mode.value
            if mode not in mode_latencies:
                mode_latencies[mode] = []
            mode_latencies[mode].append(measurement.total_latency_ms)
        
        stats["by_processing_mode"] = {}
        for mode, latencies in mode_latencies.items():
            stats["by_processing_mode"][mode] = {
                "count": len(latencies),
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies)
            }
        
        return stats


class PredictiveCache:
    """Predictive caching system for latency optimization."""
    
    def __init__(self, config: LatencyOptimizationConfig):
        self.config = config
        self.cache: Dict[str, Any] = {}
        self.access_patterns: deque = deque(maxlen=config.cache_prediction_window * 2)
        self.hit_rate_history: deque = deque(maxlen=50)
        
    def predict_next_requests(self, current_context: Dict[str, Any]) -> List[str]:
        """Predict likely next requests based on patterns."""
        
        # Simple pattern-based prediction
        predictions = []
        
        # Analyze recent access patterns
        if len(self.access_patterns) >= 3:
            recent_patterns = list(self.access_patterns)[-3:]
            
            # Look for common sequences
            for i in range(len(recent_patterns) - 1):
                pattern = tuple(recent_patterns[i:i+2])
                # This would be enhanced with ML-based prediction
                predictions.append(f"predicted_{pattern[1]}")
        
        return predictions[:3]  # Return top 3 predictions
    
    def preload_cache(self, predictions: List[str]) -> None:
        """Preload cache with predicted content."""
        
        for prediction in predictions:
            if prediction not in self.cache:
                # Simulate preloading (in real implementation, this would precompute responses)
                self.cache[prediction] = {
                    "preloaded": True,
                    "timestamp": time.time(),
                    "content": f"preloaded_content_for_{prediction}"
                }
    
    def get_cache_hit_rate(self) -> float:
        """Get current cache hit rate."""
        if not self.hit_rate_history:
            return 0.0
        return statistics.mean(self.hit_rate_history)
    
    def record_cache_access(self, key: str, hit: bool) -> None:
        """Record cache access for hit rate calculation."""
        self.access_patterns.append(key)
        self.hit_rate_history.append(1.0 if hit else 0.0)


class LatencyOptimizer:
    """
    Comprehensive latency optimization system for voice processing.
    
    Features:
    - Real-time latency profiling
    - Predictive caching
    - Pipeline optimization
    - Adaptive optimization
    - Component-level optimization
    """
    
    def __init__(
        self,
        config: LatencyOptimizationConfig,
        parallel_processor: Optional[ParallelVoiceProcessor] = None
    ):
        """Initialize the latency optimizer."""
        self.config = config
        self.parallel_processor = parallel_processor
        
        # Core components
        self.profiler = LatencyProfiler(config)
        self.predictive_cache = PredictiveCache(config)
        
        # Optimization state
        self.optimization_active = False
        self.adaptive_parameters: Dict[str, float] = {}
        
        # Performance tracking
        self.optimization_history: deque = deque(maxlen=100)
        self.current_optimizations: Dict[OptimizationTechnique, bool] = {
            technique: technique in config.enabled_techniques
            for technique in OptimizationTechnique
        }
        
        logger.info(f"LatencyOptimizer initialized with target: {config.target_latency} ({config.target_latency_ms}ms)")
    
    async def optimize_request(
        self,
        voice_message: VoiceMessage,
        processing_callback: Callable[[VoiceMessage], VoiceResponse]
    ) -> VoiceResponse:
        """
        Optimize processing of a voice request.
        
        Args:
            voice_message: The voice message to process
            processing_callback: Function to process the message
            
        Returns:
            Optimized voice response
        """
        
        # Start latency measurement
        self.profiler.start_measurement(voice_message.message_id)
        
        try:
            # Apply pre-processing optimizations
            optimized_message = await self._apply_preprocessing_optimizations(voice_message)
            
            # Check predictive cache
            cache_key = self._generate_cache_key(optimized_message)
            cached_response = self._check_predictive_cache(cache_key)
            
            if cached_response:
                self.predictive_cache.record_cache_access(cache_key, True)
                response = cached_response
            else:
                self.predictive_cache.record_cache_access(cache_key, False)
                
                # Apply processing optimizations
                if self.current_optimizations[OptimizationTechnique.PIPELINE_PARALLELIZATION]:
                    response = await self._process_with_pipeline_optimization(
                        optimized_message, processing_callback
                    )
                else:
                    response = await processing_callback(optimized_message)
                
                # Cache the response
                self._update_predictive_cache(cache_key, response)
            
            # Apply post-processing optimizations
            optimized_response = await self._apply_postprocessing_optimizations(response)
            
            # Finish measurement
            measurement = self.profiler.finish_measurement(
                optimized_response.processing_mode,
                len(voice_message.text_content or ""),
                self._calculate_complexity_score(voice_message)
            )
            
            # Adaptive optimization
            if self.config.enable_adaptive_optimization:
                await self._update_adaptive_optimization(measurement)
            
            # Trigger predictive caching for next requests
            if self.current_optimizations[OptimizationTechnique.PREDICTIVE_CACHING]:
                await self._trigger_predictive_caching(voice_message)
            
            return optimized_response
            
        except Exception as e:
            logger.error(f"Latency optimization failed: {e}")
            # Fallback to unoptimized processing
            return await processing_callback(voice_message)
    
    async def _apply_preprocessing_optimizations(self, voice_message: VoiceMessage) -> VoiceMessage:
        """Apply preprocessing optimizations."""
        
        start_time = time.time()
        
        # Context compression
        if self.current_optimizations[OptimizationTechnique.CONTEXT_COMPRESSION]:
            # Compress context to reduce processing time
            compressed_message = self._compress_message_context(voice_message)
        else:
            compressed_message = voice_message
        
        # Record preprocessing time
        preprocessing_time = (time.time() - start_time) * 1000
        self.profiler.record_component_latency("audio_preprocessing", preprocessing_time)
        
        return compressed_message
    
    async def _apply_postprocessing_optimizations(self, response: VoiceResponse) -> VoiceResponse:
        """Apply postprocessing optimizations."""
        
        start_time = time.time()
        
        # Early termination if confidence is high
        if (self.current_optimizations[OptimizationTechnique.EARLY_TERMINATION] and
            response.response_relevance and 
            response.response_relevance >= self.config.early_termination_confidence):
            
            # Skip additional processing
            pass
        
        # Record postprocessing time
        postprocessing_time = (time.time() - start_time) * 1000
        self.profiler.record_component_latency("audio_postprocessing", postprocessing_time)
        
        return response
    
    async def _process_with_pipeline_optimization(
        self,
        voice_message: VoiceMessage,
        processing_callback: Callable
    ) -> VoiceResponse:
        """Process with pipeline parallelization optimization."""
        
        # Submit to parallel processor if available
        if self.parallel_processor:
            request_id = await self.parallel_processor.submit_request(
                voice_message=voice_message,
                priority=ProcessingPriority.HIGH,
                timeout_seconds=self.config.target_latency_ms / 1000.0
            )
            
            return await self.parallel_processor.get_response(
                request_id,
                timeout=self.config.target_latency_ms / 1000.0
            )
        else:
            return await processing_callback(voice_message)
    
    def _check_predictive_cache(self, cache_key: str) -> Optional[VoiceResponse]:
        """Check predictive cache for response."""
        
        if cache_key in self.predictive_cache.cache:
            cached_data = self.predictive_cache.cache[cache_key]
            
            # Check if cache entry is still valid
            cache_age = time.time() - cached_data.get("timestamp", 0)
            if cache_age < 300:  # 5 minutes TTL
                return cached_data.get("response")
        
        return None
    
    def _update_predictive_cache(self, cache_key: str, response: VoiceResponse) -> None:
        """Update predictive cache with new response."""
        
        self.predictive_cache.cache[cache_key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1
        }
    
    def _generate_cache_key(self, voice_message: VoiceMessage) -> str:
        """Generate cache key for voice message."""
        
        # Simple cache key based on text content and conversation
        text_hash = hash(voice_message.text_content or "")
        return f"{voice_message.conversation_id}:{text_hash}"
    
    def _compress_message_context(self, voice_message: VoiceMessage) -> VoiceMessage:
        """Compress message context to reduce processing time."""
        
        # Simple context compression (in real implementation, this would be more sophisticated)
        if voice_message.text_content and len(voice_message.text_content) > 200:
            compressed_text = voice_message.text_content[:200] + "..."
            
            # Create new message with compressed context
            return VoiceMessage(
                message_id=voice_message.message_id,
                conversation_id=voice_message.conversation_id,
                audio_data=voice_message.audio_data,
                text_content=compressed_text,
                processing_mode=voice_message.processing_mode,
                timestamp=voice_message.timestamp
            )
        
        return voice_message
    
    def _calculate_complexity_score(self, voice_message: VoiceMessage) -> float:
        """Calculate complexity score for the message."""
        
        text_content = voice_message.text_content or ""
        
        # Simple complexity scoring
        word_count = len(text_content.split())
        char_count = len(text_content)
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (word_count / 100.0) + (char_count / 1000.0))
        
        return complexity
    
    async def _trigger_predictive_caching(self, voice_message: VoiceMessage) -> None:
        """Trigger predictive caching for likely next requests."""
        
        context = {
            "conversation_id": voice_message.conversation_id,
            "text_content": voice_message.text_content,
            "timestamp": voice_message.timestamp
        }
        
        predictions = self.predictive_cache.predict_next_requests(context)
        self.predictive_cache.preload_cache(predictions)
    
    async def _update_adaptive_optimization(self, measurement: LatencyMeasurement) -> None:
        """Update adaptive optimization based on measurement."""
        
        # Check if we're meeting latency targets
        if measurement.total_latency_ms > self.config.target_latency_ms:
            # Enable more aggressive optimizations
            if not self.current_optimizations[OptimizationTechnique.CONTEXT_COMPRESSION]:
                self.current_optimizations[OptimizationTechnique.CONTEXT_COMPRESSION] = True
                logger.info("Enabled context compression due to high latency")
            
            if not self.current_optimizations[OptimizationTechnique.EARLY_TERMINATION]:
                self.current_optimizations[OptimizationTechnique.EARLY_TERMINATION] = True
                logger.info("Enabled early termination due to high latency")
        
        elif measurement.total_latency_ms < self.config.target_latency_ms * 0.5:
            # We have headroom, can relax some optimizations for quality
            if self.current_optimizations[OptimizationTechnique.CONTEXT_COMPRESSION]:
                self.current_optimizations[OptimizationTechnique.CONTEXT_COMPRESSION] = False
                logger.info("Disabled context compression due to low latency")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        
        latency_stats = self.profiler.get_latency_statistics()
        cache_hit_rate = self.predictive_cache.get_cache_hit_rate()
        
        return {
            "target_latency_ms": self.config.target_latency_ms,
            "current_latency_stats": latency_stats,
            "cache_hit_rate": cache_hit_rate,
            "active_optimizations": [
                technique.value for technique, enabled in self.current_optimizations.items()
                if enabled
            ],
            "optimization_active": self.optimization_active,
            "adaptive_parameters": self.adaptive_parameters.copy()
        }
    
    async def cleanup(self) -> None:
        """Clean up optimizer resources."""
        
        try:
            # Clear caches
            self.predictive_cache.cache.clear()
            
            # Clear measurements
            self.profiler.measurements.clear()
            self.profiler.component_stats.clear()
            
            logger.info("LatencyOptimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
