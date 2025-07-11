"""
Voice Processing Modes

This module implements multiple processing modes for voice conversations,
providing different approaches for various use cases and performance requirements.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .context_integration import ContextConfig, VoiceContextManager
from .hybrid_orchestrator import (
    HybridConfig,
    HybridOrchestrator,
    ProcessingStrategy,
    ResponseSelection,
)
from .llm_integration import VoiceLLMConfig
from .models import ConversationState, VoiceMessage, VoiceProcessingMode, VoiceResponse
from .moshi_client import MoshiClient, MoshiConfig
from .performance_optimizer import OptimizationConfig, PerformanceOptimizer

logger = logging.getLogger("coda.voice.processing_modes")


class ProcessingModeType(str, Enum):
    """Types of processing modes."""

    MOSHI_ONLY = "moshi_only"  # Pure Moshi processing
    LLM_ONLY = "llm_only"  # Pure LLM processing
    HYBRID_BALANCED = "hybrid_balanced"  # Balanced hybrid processing
    HYBRID_SPEED = "hybrid_speed"  # Speed - optimized hybrid
    HYBRID_QUALITY = "hybrid_quality"  # Quality - optimized hybrid
    ADAPTIVE = "adaptive"  # Adaptive mode selection
    EXPERIMENTAL = "experimental"  # Experimental processing


@dataclass
class ProcessingModeConfig:
    """Configuration for a processing mode."""

    mode_type: ProcessingModeType
    name: str
    description: str

    # Performance characteristics
    target_latency_ms: float
    expected_quality_score: float
    resource_usage: str  # "low", "medium", "high"

    # Mode - specific settings
    settings: Dict[str, Any]

    # Conditions for automatic selection
    auto_select_conditions: Dict[str, Any]


class ProcessingModeInterface(ABC):
    """Abstract interface for processing modes."""

    @abstractmethod
    async def process(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """Process voice message in this mode."""
        pass

    @abstractmethod
    def get_mode_info(self) -> ProcessingModeConfig:
        """Get mode configuration information."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the processing mode."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up mode resources."""
        pass


class MoshiOnlyMode(ProcessingModeInterface):
    """Pure Moshi processing mode for maximum speed."""

    def __init__(self, moshi_config: MoshiConfig):
        self.moshi_config = moshi_config
        self.moshi_client: Optional[MoshiClient] = None

    async def initialize(self) -> None:
        """Initialize Moshi client."""
        self.moshi_client = MoshiClient(self.moshi_config)
        await self.moshi_client.initialize()

    async def process(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """Process with Moshi only."""
        if not self.moshi_client:
            raise RuntimeError("Moshi client not initialized")

        start_time = time.time()

        try:
            result = await self.moshi_client.process_audio(
                voice_message.audio_data, voice_message.conversation_id
            )

            latency_ms = (time.time() - start_time) * 1000

            return VoiceResponse(
                response_id=f"moshi_{voice_message.message_id}",
                conversation_id=voice_message.conversation_id,
                message_id=voice_message.message_id,
                text_content=result.get("text", ""),
                audio_data=result.get("audio", b""),
                processing_mode=VoiceProcessingMode.MOSHI_ONLY,
                total_latency_ms=latency_ms,
                moshi_latency_ms=latency_ms,
                response_relevance=0.8,  # Good baseline relevance
            )

        except Exception as e:
            logger.error(f"Moshi - only processing failed: {e}")
            raise

    def get_mode_info(self) -> ProcessingModeConfig:
        """Get mode configuration."""
        return ProcessingModeConfig(
            mode_type=ProcessingModeType.MOSHI_ONLY,
            name="Moshi Only",
            description="Pure Moshi processing for maximum speed and natural voice interaction",
            target_latency_ms=200.0,
            expected_quality_score=0.7,
            resource_usage="medium",
            settings={
                "enable_inner_monologue": self.moshi_config.enable_inner_monologue,
                "sample_rate": self.moshi_config.sample_rate,
            },
            auto_select_conditions={
                "text_length_max": 20,
                "complexity": "low",
                "intent": ["greeting", "general"],
            },
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.moshi_client:
            await self.moshi_client.cleanup()


class LLMOnlyMode(ProcessingModeInterface):
    """Pure LLM processing mode for maximum quality."""

    def __init__(
        self,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        optimization_config: OptimizationConfig,
    ):
        self.context_config = context_config
        self.llm_config = llm_config
        self.optimization_config = optimization_config
        self.context_manager: Optional[VoiceContextManager] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None

    async def initialize(self) -> None:
        """Initialize LLM components."""
        self.context_manager = VoiceContextManager(self.context_config, self.llm_config)
        await self.context_manager.initialize()

        self.performance_optimizer = PerformanceOptimizer(self.optimization_config)

    async def process(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """Process with LLM only."""
        if not self.context_manager or not self.performance_optimizer:
            raise RuntimeError("LLM components not initialized")

        try:
            response = await self.performance_optimizer.optimize_processing(
                voice_message, self.context_manager, conversation_state
            )

            # Ensure processing mode is set correctly
            response.processing_mode = VoiceProcessingMode.LLM_ENHANCED

            return response

        except Exception as e:
            logger.error(f"LLM - only processing failed: {e}")
            raise

    def get_mode_info(self) -> ProcessingModeConfig:
        """Get mode configuration."""
        return ProcessingModeConfig(
            mode_type=ProcessingModeType.LLM_ONLY,
            name="LLM Only",
            description="Pure LLM processing for maximum quality and reasoning capability",
            target_latency_ms=800.0,
            expected_quality_score=0.9,
            resource_usage="high",
            settings={
                "llm_model": self.llm_config.llm_model,
                "enable_streaming": self.llm_config.enable_streaming,
                "enable_context": True,
            },
            auto_select_conditions={
                "intent": ["coding", "reasoning", "explanation"],
                "complexity": ["medium", "high"],
                "text_length_min": 30,
            },
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.context_manager:
            await self.context_manager.cleanup()
        if self.performance_optimizer:
            await self.performance_optimizer.cleanup()


class HybridBalancedMode(ProcessingModeInterface):
    """Balanced hybrid processing mode."""

    def __init__(
        self,
        hybrid_config: HybridConfig,
        moshi_config: MoshiConfig,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        optimization_config: OptimizationConfig,
    ):
        self.configs = (
            hybrid_config,
            moshi_config,
            context_config,
            llm_config,
            optimization_config,
        )
        self.orchestrator: Optional[HybridOrchestrator] = None

        # Configure for balanced processing
        self.configs[0].default_strategy = ProcessingStrategy.HYBRID_SEQUENTIAL
        self.configs[0].response_selection = ResponseSelection.BALANCED

    async def initialize(self) -> None:
        """Initialize hybrid orchestrator."""
        self.orchestrator = HybridOrchestrator(*self.configs)
        await self.orchestrator.initialize()

    async def process(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """Process with balanced hybrid approach."""
        if not self.orchestrator:
            raise RuntimeError("Hybrid orchestrator not initialized")

        response = await self.orchestrator.process_voice_message(voice_message, conversation_state)
        response.processing_mode = VoiceProcessingMode.HYBRID
        return response

    def get_mode_info(self) -> ProcessingModeConfig:
        """Get mode configuration."""
        return ProcessingModeConfig(
            mode_type=ProcessingModeType.HYBRID_BALANCED,
            name="Hybrid Balanced",
            description="Balanced hybrid processing combining Moshi speed with LLM quality",
            target_latency_ms=500.0,
            expected_quality_score=0.85,
            resource_usage="high",
            settings={
                "strategy": "hybrid_sequential",
                "response_selection": "balanced",
                "enable_adaptive": False,
            },
            auto_select_conditions={"default": True},  # Good default choice
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()


class HybridSpeedMode(ProcessingModeInterface):
    """Speed - optimized hybrid processing mode."""

    def __init__(
        self,
        hybrid_config: HybridConfig,
        moshi_config: MoshiConfig,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        optimization_config: OptimizationConfig,
    ):
        self.configs = (
            hybrid_config,
            moshi_config,
            context_config,
            llm_config,
            optimization_config,
        )
        self.orchestrator: Optional[HybridOrchestrator] = None

        # Configure for speed
        self.configs[0].default_strategy = ProcessingStrategy.HYBRID_PARALLEL
        self.configs[0].response_selection = ResponseSelection.FASTEST
        self.configs[0].moshi_timeout_ms = 1000.0
        self.configs[0].llm_timeout_ms = 1500.0

    async def initialize(self) -> None:
        """Initialize speed - optimized orchestrator."""
        self.orchestrator = HybridOrchestrator(*self.configs)
        await self.orchestrator.initialize()

    async def process(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """Process with speed - optimized hybrid approach."""
        if not self.orchestrator:
            raise RuntimeError("Hybrid orchestrator not initialized")

        response = await self.orchestrator.process_voice_message(voice_message, conversation_state)
        response.processing_mode = VoiceProcessingMode.HYBRID
        return response

    def get_mode_info(self) -> ProcessingModeConfig:
        """Get mode configuration."""
        return ProcessingModeConfig(
            mode_type=ProcessingModeType.HYBRID_SPEED,
            name="Hybrid Speed",
            description="Speed - optimized hybrid processing prioritizing fast response times",
            target_latency_ms=300.0,
            expected_quality_score=0.75,
            resource_usage="high",
            settings={
                "strategy": "hybrid_parallel",
                "response_selection": "fastest",
                "timeouts_reduced": True,
            },
            auto_select_conditions={"latency_priority": True, "real_time_required": True},
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()


class HybridQualityMode(ProcessingModeInterface):
    """Quality - optimized hybrid processing mode."""

    def __init__(
        self,
        hybrid_config: HybridConfig,
        moshi_config: MoshiConfig,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        optimization_config: OptimizationConfig,
    ):
        self.configs = (
            hybrid_config,
            moshi_config,
            context_config,
            llm_config,
            optimization_config,
        )
        self.orchestrator: Optional[HybridOrchestrator] = None

        # Configure for quality
        self.configs[0].default_strategy = ProcessingStrategy.HYBRID_SEQUENTIAL
        self.configs[0].response_selection = ResponseSelection.HIGHEST_QUALITY
        self.configs[0].llm_timeout_ms = 5000.0  # Allow more time for quality
        self.configs[0].min_response_quality = 0.8

    async def initialize(self) -> None:
        """Initialize quality - optimized orchestrator."""
        self.orchestrator = HybridOrchestrator(*self.configs)
        await self.orchestrator.initialize()

    async def process(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """Process with quality - optimized hybrid approach."""
        if not self.orchestrator:
            raise RuntimeError("Hybrid orchestrator not initialized")

        response = await self.orchestrator.process_voice_message(voice_message, conversation_state)
        response.processing_mode = VoiceProcessingMode.HYBRID
        return response

    def get_mode_info(self) -> ProcessingModeConfig:
        """Get mode configuration."""
        return ProcessingModeConfig(
            mode_type=ProcessingModeType.HYBRID_QUALITY,
            name="Hybrid Quality",
            description="Quality - optimized hybrid processing for best possible responses",
            target_latency_ms=1000.0,
            expected_quality_score=0.95,
            resource_usage="high",
            settings={
                "strategy": "hybrid_sequential",
                "response_selection": "highest_quality",
                "quality_threshold": 0.8,
            },
            auto_select_conditions={"quality_priority": True, "complex_reasoning": True},
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()


class AdaptiveMode(ProcessingModeInterface):
    """Adaptive processing mode that selects optimal strategy dynamically."""

    def __init__(
        self,
        hybrid_config: HybridConfig,
        moshi_config: MoshiConfig,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        optimization_config: OptimizationConfig,
    ):
        self.configs = (
            hybrid_config,
            moshi_config,
            context_config,
            llm_config,
            optimization_config,
        )
        self.orchestrator: Optional[HybridOrchestrator] = None

        # Configure for adaptive processing
        self.configs[0].default_strategy = ProcessingStrategy.ADAPTIVE
        self.configs[0].enable_adaptive_learning = True

    async def initialize(self) -> None:
        """Initialize adaptive orchestrator."""
        self.orchestrator = HybridOrchestrator(*self.configs)
        await self.orchestrator.initialize()

    async def process(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """Process with adaptive strategy selection."""
        if not self.orchestrator:
            raise RuntimeError("Hybrid orchestrator not initialized")

        response = await self.orchestrator.process_voice_message(voice_message, conversation_state)
        response.processing_mode = VoiceProcessingMode.HYBRID
        return response

    def get_mode_info(self) -> ProcessingModeConfig:
        """Get mode configuration."""
        return ProcessingModeConfig(
            mode_type=ProcessingModeType.ADAPTIVE,
            name="Adaptive",
            description="Intelligent adaptive processing that learns and optimizes strategy selection",
            target_latency_ms=400.0,
            expected_quality_score=0.88,
            resource_usage="high",
            settings={"strategy": "adaptive", "enable_learning": True, "auto_optimization": True},
            auto_select_conditions={"learning_enabled": True, "variable_workload": True},
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()


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
        optimization_config: OptimizationConfig,
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

        # Auto - selection settings
        self.enable_auto_selection = True
        self.auto_selection_rules: List[Callable] = []

        logger.info("ProcessingModeManager initialized")
