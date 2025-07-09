"""
Processing Mode Manager

This module provides the ProcessingModeManager class that manages multiple
voice processing modes and handles mode selection and switching.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable

from .processing_modes import (
    ProcessingModeType, ProcessingModeInterface, ProcessingModeConfig,
    MoshiOnlyMode, LLMOnlyMode, HybridBalancedMode, HybridSpeedMode,
    HybridQualityMode, AdaptiveMode
)
from .models import VoiceMessage, VoiceResponse, ConversationState
from .moshi_client import MoshiConfig
from .context_integration import ContextConfig
from .llm_integration import VoiceLLMConfig
from .performance_optimizer import OptimizationConfig
from .hybrid_orchestrator import HybridConfig, ProcessingStrategy

logger = logging.getLogger("coda.voice.mode_manager")


class ProcessingModeManager:
    """
    Manages multiple processing modes and provides mode selection and switching.
    
    Features:
    - Multiple processing modes
    - Automatic mode selection
    - Dynamic mode switching
    - Performance monitoring
    - Mode comparison and analytics
    """
    
    def __init__(
        self,
        moshi_config: MoshiConfig,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        optimization_config: OptimizationConfig
    ):
        """Initialize the processing mode manager."""
        self.moshi_config = moshi_config
        self.context_config = context_config
        self.llm_config = llm_config
        self.optimization_config = optimization_config
        
        # Available processing modes
        self.available_modes: Dict[ProcessingModeType, ProcessingModeInterface] = {}
        self.current_mode: Optional[ProcessingModeType] = None
        
        # Performance tracking
        self.mode_performance: Dict[ProcessingModeType, List[Dict[str, float]]] = {}
        self.mode_usage_stats: Dict[ProcessingModeType, int] = {}
        
        # Auto-selection settings
        self.enable_auto_selection = True
        
        logger.info("ProcessingModeManager initialized")
    
    async def initialize(self) -> None:
        """Initialize all processing modes."""
        try:
            # Create hybrid config
            hybrid_config = HybridConfig(
                default_strategy=ProcessingStrategy.ADAPTIVE,
                enable_adaptive_learning=True,
                enable_parallel_processing=True
            )
            
            # Initialize all modes
            modes_to_init = [
                (ProcessingModeType.MOSHI_ONLY, MoshiOnlyMode(self.moshi_config)),
                (ProcessingModeType.LLM_ONLY, LLMOnlyMode(
                    self.context_config, self.llm_config, self.optimization_config
                )),
                (ProcessingModeType.HYBRID_BALANCED, HybridBalancedMode(
                    hybrid_config, self.moshi_config, self.context_config,
                    self.llm_config, self.optimization_config
                )),
                (ProcessingModeType.HYBRID_SPEED, HybridSpeedMode(
                    hybrid_config, self.moshi_config, self.context_config,
                    self.llm_config, self.optimization_config
                )),
                (ProcessingModeType.HYBRID_QUALITY, HybridQualityMode(
                    hybrid_config, self.moshi_config, self.context_config,
                    self.llm_config, self.optimization_config
                )),
                (ProcessingModeType.ADAPTIVE, AdaptiveMode(
                    hybrid_config, self.moshi_config, self.context_config,
                    self.llm_config, self.optimization_config
                ))
            ]
            
            # Initialize modes
            for mode_type, mode_instance in modes_to_init:
                try:
                    await mode_instance.initialize()
                    self.available_modes[mode_type] = mode_instance
                    self.mode_performance[mode_type] = []
                    self.mode_usage_stats[mode_type] = 0
                    logger.info(f"Initialized {mode_type} mode")
                except Exception as e:
                    logger.warning(f"Failed to initialize {mode_type} mode: {e}")
            
            # Set default mode
            if ProcessingModeType.ADAPTIVE in self.available_modes:
                self.current_mode = ProcessingModeType.ADAPTIVE
            elif ProcessingModeType.HYBRID_BALANCED in self.available_modes:
                self.current_mode = ProcessingModeType.HYBRID_BALANCED
            elif self.available_modes:
                self.current_mode = list(self.available_modes.keys())[0]
            
            logger.info(f"ProcessingModeManager initialized with {len(self.available_modes)} modes")
            logger.info(f"Default mode: {self.current_mode}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProcessingModeManager: {e}")
            raise
    
    async def process_voice_message(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState] = None,
        requested_mode: Optional[ProcessingModeType] = None
    ) -> VoiceResponse:
        """
        Process voice message using the appropriate mode.
        
        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state
            requested_mode: Specific mode to use (overrides auto-selection)
            
        Returns:
            Processed voice response
        """
        start_time = time.time()
        
        try:
            # Select processing mode
            if requested_mode and requested_mode in self.available_modes:
                selected_mode = requested_mode
            elif self.enable_auto_selection:
                selected_mode = await self._auto_select_mode(voice_message, conversation_state)
            else:
                selected_mode = self.current_mode
            
            if not selected_mode or selected_mode not in self.available_modes:
                raise RuntimeError(f"No valid processing mode available: {selected_mode}")
            
            # Process with selected mode
            mode_instance = self.available_modes[selected_mode]
            response = await mode_instance.process(voice_message, conversation_state)
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self._track_mode_performance(selected_mode, processing_time, response)
            
            # Update current mode if auto-selected
            if not requested_mode:
                self.current_mode = selected_mode
            
            logger.debug(f"Processed with {selected_mode} mode in {processing_time:.1f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Voice message processing failed: {e}")
            
            # Fallback to simplest available mode
            fallback_mode = self._get_fallback_mode()
            if fallback_mode and fallback_mode != selected_mode:
                logger.info(f"Falling back to {fallback_mode} mode")
                try:
                    mode_instance = self.available_modes[fallback_mode]
                    return await mode_instance.process(voice_message, conversation_state)
                except Exception as fallback_error:
                    logger.error(f"Fallback processing also failed: {fallback_error}")
            
            raise
    
    async def _auto_select_mode(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState]
    ) -> ProcessingModeType:
        """Automatically select the best processing mode."""
        
        # Analyze message characteristics
        text_content = voice_message.text_content or ""
        text_length = len(text_content)
        word_count = len(text_content.split())
        
        # Simple intent detection
        text_lower = text_content.lower()
        
        # Determine intent
        if any(word in text_lower for word in ["code", "program", "debug", "implement"]):
            intent = "coding"
        elif any(word in text_lower for word in ["explain", "how", "why", "what"]):
            intent = "explanation"
        elif any(word in text_lower for word in ["hello", "hi", "thanks", "bye"]):
            intent = "greeting"
        else:
            intent = "general"
        
        # Determine complexity
        if word_count > 50 or any(word in text_lower for word in ["complex", "detailed", "comprehensive"]):
            complexity = "high"
        elif word_count > 20 or intent in ["coding", "explanation"]:
            complexity = "medium"
        else:
            complexity = "low"
        
        # Mode selection logic
        if intent == "greeting" and complexity == "low":
            return ProcessingModeType.MOSHI_ONLY
        
        elif intent == "coding" or complexity == "high":
            return ProcessingModeType.HYBRID_QUALITY
        
        elif text_length < 20:
            return ProcessingModeType.HYBRID_SPEED
        
        elif intent == "explanation":
            return ProcessingModeType.LLM_ONLY
        
        else:
            # Default to adaptive or balanced
            if ProcessingModeType.ADAPTIVE in self.available_modes:
                return ProcessingModeType.ADAPTIVE
            else:
                return ProcessingModeType.HYBRID_BALANCED
    
    def _get_fallback_mode(self) -> Optional[ProcessingModeType]:
        """Get the simplest available mode for fallback."""
        
        fallback_priority = [
            ProcessingModeType.MOSHI_ONLY,
            ProcessingModeType.HYBRID_SPEED,
            ProcessingModeType.HYBRID_BALANCED,
            ProcessingModeType.LLM_ONLY,
            ProcessingModeType.ADAPTIVE
        ]
        
        for mode_type in fallback_priority:
            if mode_type in self.available_modes:
                return mode_type
        
        return None
    
    def _track_mode_performance(
        self,
        mode_type: ProcessingModeType,
        processing_time_ms: float,
        response: VoiceResponse
    ) -> None:
        """Track performance metrics for a mode."""
        
        performance_entry = {
            "timestamp": time.time(),
            "processing_time_ms": processing_time_ms,
            "total_latency_ms": response.total_latency_ms,
            "response_relevance": response.response_relevance or 0.5,
            "text_length": len(response.text_content or "")
        }
        
        self.mode_performance[mode_type].append(performance_entry)
        self.mode_usage_stats[mode_type] += 1
        
        # Keep only recent performance data
        if len(self.mode_performance[mode_type]) > 100:
            self.mode_performance[mode_type] = self.mode_performance[mode_type][-100:]
    
    async def switch_mode(self, mode_type: ProcessingModeType) -> bool:
        """Switch to a specific processing mode."""
        
        if mode_type not in self.available_modes:
            logger.warning(f"Mode {mode_type} not available")
            return False
        
        self.current_mode = mode_type
        logger.info(f"Switched to {mode_type} mode")
        return True
    
    def get_available_modes(self) -> List[ProcessingModeConfig]:
        """Get information about all available modes."""
        
        return [
            mode_instance.get_mode_info()
            for mode_instance in self.available_modes.values()
        ]
    
    def get_mode_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all modes."""
        
        stats = {}
        
        for mode_type, performance_data in self.mode_performance.items():
            if not performance_data:
                stats[mode_type.value] = {"usage_count": 0}
                continue
            
            recent_data = performance_data[-20:]  # Last 20 uses
            
            avg_processing_time = sum(p["processing_time_ms"] for p in recent_data) / len(recent_data)
            avg_latency = sum(p["total_latency_ms"] for p in recent_data) / len(recent_data)
            avg_relevance = sum(p["response_relevance"] for p in recent_data) / len(recent_data)
            
            stats[mode_type.value] = {
                "usage_count": self.mode_usage_stats[mode_type],
                "avg_processing_time_ms": avg_processing_time,
                "avg_total_latency_ms": avg_latency,
                "avg_relevance_score": avg_relevance,
                "recent_uses": len(recent_data)
            }
        
        return stats
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get current manager status."""
        
        return {
            "current_mode": self.current_mode.value if self.current_mode else None,
            "available_modes": [mode.value for mode in self.available_modes.keys()],
            "total_modes": len(self.available_modes),
            "auto_selection_enabled": self.enable_auto_selection,
            "performance_stats": self.get_mode_performance_stats()
        }
    
    async def cleanup(self) -> None:
        """Clean up all processing modes."""
        
        try:
            for mode_type, mode_instance in self.available_modes.items():
                try:
                    await mode_instance.cleanup()
                    logger.debug(f"Cleaned up {mode_type} mode")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {mode_type} mode: {e}")
            
            self.available_modes.clear()
            self.mode_performance.clear()
            self.mode_usage_stats.clear()
            
            logger.info("ProcessingModeManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
