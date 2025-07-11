"""
Hybrid Voice Processing Orchestrator

This module orchestrates hybrid processing between Moshi and external LLMs,
providing intelligent coordination for optimal voice conversation experiences.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..llm.models import LLMProvider
from .context_integration import ContextConfig, VoiceContextManager
from .llm_integration import VoiceLLMConfig
from .models import ConversationState, VoiceMessage, VoiceProcessingMode, VoiceResponse
from .moshi_client import MoshiClient, MoshiConfig
from .performance_optimizer import OptimizationConfig, PerformanceOptimizer

logger = logging.getLogger("coda.voice.hybrid_orchestrator")


class ProcessingStrategy(str, Enum):
    """Processing strategy for hybrid orchestration."""

    MOSHI_ONLY = "moshi_only"  # Use only Moshi for processing
    LLM_ONLY = "llm_only"  # Use only external LLM
    HYBRID_PARALLEL = "hybrid_parallel"  # Process with both in parallel, choose best
    HYBRID_SEQUENTIAL = "hybrid_sequential"  # Moshi first, then LLM enhancement
    ADAPTIVE = "adaptive"  # Dynamically choose strategy based on context


class ResponseSelection(str, Enum):
    """Response selection criteria for hybrid processing."""

    FASTEST = "fastest"  # Choose fastest response
    HIGHEST_QUALITY = "highest_quality"  # Choose highest quality response
    MOST_RELEVANT = "most_relevant"  # Choose most contextually relevant
    BALANCED = "balanced"  # Balance speed, quality, and relevance


@dataclass
class HybridConfig:
    """
    Configuration for hybrid processing orchestrator.

    This configuration class controls how the hybrid orchestrator coordinates
    between Moshi and LLM processing, including timing, quality thresholds,
    and adaptive behavior.

    Attributes:
        default_strategy: Default processing strategy to use
        response_selection: Criteria for selecting best response in hybrid mode
        moshi_timeout_ms: Maximum time to wait for Moshi response
        llm_timeout_ms: Maximum time to wait for LLM response
        hybrid_timeout_ms: Maximum total time for hybrid processing
        min_response_quality: Minimum quality score to accept response
        min_relevance_score: Minimum relevance score for response selection
        enable_adaptive_learning: Whether to learn from processing outcomes
        strategy_switch_threshold: Threshold for switching strategies
        performance_history_size: Number of recent operations to track
        enable_fallback: Whether to use fallback when primary processing fails
        fallback_strategy: Strategy to use as fallback
        enable_parallel_processing: Whether to process Moshi and LLM in parallel
        max_concurrent_requests: Maximum concurrent processing requests

    Example:
        >>> config = HybridConfig(
        ...     default_strategy=ProcessingStrategy.ADAPTIVE,
        ...     moshi_timeout_ms=1500.0,
        ...     llm_timeout_ms=2500.0,
        ...     enable_adaptive_learning=True
        ... )
    """

    # Processing strategy
    default_strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE
    response_selection: ResponseSelection = ResponseSelection.BALANCED

    # Timing thresholds (milliseconds)
    moshi_timeout_ms: float = 2000.0
    llm_timeout_ms: float = 3000.0
    hybrid_timeout_ms: float = 4000.0

    # Quality thresholds
    min_response_quality: float = 0.6
    min_relevance_score: float = 0.7

    # Adaptive strategy parameters
    enable_adaptive_learning: bool = True
    strategy_switch_threshold: float = 0.8
    performance_history_size: int = 100

    # Fallback configuration
    enable_fallback: bool = True
    fallback_strategy: ProcessingStrategy = ProcessingStrategy.MOSHI_ONLY

    # Parallel processing
    enable_parallel_processing: bool = True
    max_concurrent_requests: int = 3


@dataclass
class ProcessingResult:
    """Result from a processing strategy."""

    response: VoiceResponse
    strategy: ProcessingStrategy
    latency_ms: float
    quality_score: float
    relevance_score: float
    confidence: float
    metadata: Dict[str, Any]


class HybridOrchestrator:
    """
    Orchestrates hybrid processing between Moshi and external LLMs.

    Features:
    - Multiple processing strategies
    - Adaptive strategy selection
    - Performance optimization
    - Quality - based response selection
    - Graceful fallback handling
    """

    def __init__(
        self,
        hybrid_config: HybridConfig,
        moshi_config: MoshiConfig,
        context_config: ContextConfig,
        llm_config: VoiceLLMConfig,
        optimization_config: OptimizationConfig,
    ):
        """Initialize the hybrid orchestrator."""
        self.config = hybrid_config

        # Initialize components
        self.moshi_client: Optional[MoshiClient] = None
        self.context_manager: Optional[VoiceContextManager] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None

        # Store configurations
        self.moshi_config = moshi_config
        self.context_config = context_config
        self.llm_config = llm_config
        self.optimization_config = optimization_config

        # Performance tracking
        self.strategy_performance: Dict[ProcessingStrategy, List[float]] = {
            strategy: [] for strategy in ProcessingStrategy
        }
        self.current_strategy = hybrid_config.default_strategy

        # Active processing tracking
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.processing_stats = {
            "total_requests": 0,
            "strategy_usage": {strategy.value: 0 for strategy in ProcessingStrategy},
            "average_latency_ms": 0.0,
            "success_rate": 0.0,
        }

        logger.info(
            f"HybridOrchestrator initialized with {hybrid_config.default_strategy} strategy"
        )

    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            # Initialize Moshi client
            self.moshi_client = MoshiClient(self.moshi_config)
            await self.moshi_client.initialize()

            # Initialize context manager
            self.context_manager = VoiceContextManager(self.context_config, self.llm_config)
            await self.context_manager.initialize()

            # Initialize performance optimizer
            self.performance_optimizer = PerformanceOptimizer(self.optimization_config)

            logger.info("HybridOrchestrator initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize HybridOrchestrator: {e}")
            raise

    async def process_voice_message(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState] = None
    ) -> VoiceResponse:
        """
        Process voice message using hybrid orchestration.

        Args:
            voice_message: The voice message to process
            conversation_state: Current conversation state

        Returns:
            Best response from hybrid processing
        """
        start_time = time.time()
        request_id = f"req_{voice_message.message_id}"

        try:
            # Determine processing strategy
            strategy = await self._select_strategy(voice_message, conversation_state)

            logger.debug(f"Processing {request_id} with strategy: {strategy}")

            # Execute processing strategy
            result = await self._execute_strategy(
                strategy, voice_message, conversation_state, request_id
            )

            # Update performance tracking
            total_latency = (time.time() - start_time) * 1000
            self._update_performance_stats(strategy, total_latency, result)

            # Adaptive learning
            if self.config.enable_adaptive_learning:
                await self._update_adaptive_learning(strategy, result)

            return result.response

        except Exception as e:
            logger.error(f"Hybrid processing failed for {request_id}: {e}")

            # Fallback processing
            if self.config.enable_fallback:
                return await self._fallback_processing(voice_message, conversation_state)

            raise

        finally:
            # Clean up active request
            self.active_requests.pop(request_id, None)

    async def _select_strategy(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState]
    ) -> ProcessingStrategy:
        """Select the optimal processing strategy."""

        if self.config.default_strategy != ProcessingStrategy.ADAPTIVE:
            return self.config.default_strategy

        # Adaptive strategy selection based on context and performance
        context_factors = await self._analyze_context_factors(voice_message, conversation_state)

        # Simple adaptive logic (can be enhanced with ML)
        text_length = len(voice_message.text_content or "")

        if text_length < 20 and context_factors.get("complexity", "low") == "low":
            # Short, simple queries - use Moshi for speed
            return ProcessingStrategy.MOSHI_ONLY

        elif context_factors.get("intent", "general") in ["coding", "reasoning", "explanation"]:
            # Complex queries - use LLM for quality
            return ProcessingStrategy.LLM_ONLY

        elif self._has_good_performance(ProcessingStrategy.HYBRID_PARALLEL):
            # Use hybrid if it's performing well
            return ProcessingStrategy.HYBRID_PARALLEL

        else:
            # Default to sequential hybrid
            return ProcessingStrategy.HYBRID_SEQUENTIAL

    async def _execute_strategy(
        self,
        strategy: ProcessingStrategy,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState],
        request_id: str,
    ) -> ProcessingResult:
        """Execute the selected processing strategy."""

        if strategy == ProcessingStrategy.MOSHI_ONLY:
            return await self._process_moshi_only(voice_message, conversation_state)

        elif strategy == ProcessingStrategy.LLM_ONLY:
            return await self._process_llm_only(voice_message, conversation_state)

        elif strategy == ProcessingStrategy.HYBRID_PARALLEL:
            return await self._process_hybrid_parallel(
                voice_message, conversation_state, request_id
            )

        elif strategy == ProcessingStrategy.HYBRID_SEQUENTIAL:
            return await self._process_hybrid_sequential(voice_message, conversation_state)

        else:
            # Fallback to Moshi only
            return await self._process_moshi_only(voice_message, conversation_state)

    async def _process_moshi_only(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState]
    ) -> ProcessingResult:
        """Process using only Moshi."""
        start_time = time.time()

        try:
            # Process with Moshi
            response = await asyncio.wait_for(
                self.moshi_client.process_audio(voice_message.audio_data),
                timeout=self.config.moshi_timeout_ms / 1000.0,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Convert Moshi response to VoiceResponse
            voice_response = VoiceResponse(
                response_id=f"moshi_{voice_message.message_id}",
                conversation_id=voice_message.conversation_id,
                message_id=voice_message.message_id,
                text_content=response.get("text", ""),
                audio_data=response.get("audio", b""),
                processing_mode=VoiceProcessingMode.MOSHI_ONLY,
                total_latency_ms=latency_ms,
                moshi_latency_ms=latency_ms,
            )

            return ProcessingResult(
                response=voice_response,
                strategy=ProcessingStrategy.MOSHI_ONLY,
                latency_ms=latency_ms,
                quality_score=0.7,  # Moshi baseline quality
                relevance_score=0.8,  # Good relevance for voice
                confidence=0.8,
                metadata={"source": "moshi", "model": "moshi"},
            )

        except asyncio.TimeoutError:
            logger.warning("Moshi processing timeout")
            raise
        except Exception as e:
            logger.error(f"Moshi processing failed: {e}")
            raise

    async def _process_llm_only(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState]
    ) -> ProcessingResult:
        """Process using only external LLM."""
        start_time = time.time()

        try:
            # Process with optimized LLM
            response = await asyncio.wait_for(
                self.performance_optimizer.optimize_processing(
                    voice_message, self.context_manager, conversation_state
                ),
                timeout=self.config.llm_timeout_ms / 1000.0,
            )

            latency_ms = (time.time() - start_time) * 1000

            return ProcessingResult(
                response=response,
                strategy=ProcessingStrategy.LLM_ONLY,
                latency_ms=latency_ms,
                quality_score=0.9,  # High quality from LLM
                relevance_score=response.response_relevance or 0.8,
                confidence=0.9,
                metadata={"source": "llm", "model": self.llm_config.llm_model},
            )

        except asyncio.TimeoutError:
            logger.warning("LLM processing timeout")
            raise
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            raise

    async def _process_hybrid_parallel(
        self,
        voice_message: VoiceMessage,
        conversation_state: Optional[ConversationState],
        request_id: str,
    ) -> ProcessingResult:
        """Process using both Moshi and LLM in parallel."""
        start_time = time.time()

        try:
            # Start both processing tasks
            moshi_task = asyncio.create_task(
                self._process_moshi_only(voice_message, conversation_state)
            )
            llm_task = asyncio.create_task(
                self._process_llm_only(voice_message, conversation_state)
            )

            # Store active tasks
            self.active_requests[f"{request_id}_moshi"] = moshi_task
            self.active_requests[f"{request_id}_llm"] = llm_task

            # Wait for both with timeout
            done, pending = await asyncio.wait(
                [moshi_task, llm_task],
                timeout=self.config.hybrid_timeout_ms / 1000.0,
                return_when=asyncio.ALL_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Collect results
            results = []
            for task in done:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Parallel task failed: {e}")

            if not results:
                raise RuntimeError("All parallel processing tasks failed")

            # Select best result
            best_result = self._select_best_result(results)

            # Update metadata
            best_result.strategy = ProcessingStrategy.HYBRID_PARALLEL
            best_result.latency_ms = (time.time() - start_time) * 1000
            best_result.metadata["parallel_results"] = len(results)

            return best_result

        except asyncio.TimeoutError:
            logger.warning("Hybrid parallel processing timeout")
            raise
        except Exception as e:
            logger.error(f"Hybrid parallel processing failed: {e}")
            raise

    async def _process_hybrid_sequential(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState]
    ) -> ProcessingResult:
        """Process using Moshi first, then enhance with LLM."""
        start_time = time.time()

        try:
            # First, get Moshi response
            moshi_result = await self._process_moshi_only(voice_message, conversation_state)

            # Check if Moshi response is good enough
            if (
                moshi_result.quality_score >= self.config.min_response_quality
                and moshi_result.relevance_score >= self.config.min_relevance_score
            ):

                # Enhance with LLM context if time allows
                remaining_time = self.config.hybrid_timeout_ms - moshi_result.latency_ms

                if remaining_time > 500:  # At least 500ms for LLM enhancement
                    try:
                        # Create enhanced message with Moshi's inner monologue
                        enhanced_message = VoiceMessage(
                            message_id=f"enhanced_{voice_message.message_id}",
                            conversation_id=voice_message.conversation_id,
                            audio_data=voice_message.audio_data,
                            text_content=moshi_result.response.text_content,
                            processing_mode=VoiceProcessingMode.HYBRID,
                            timestamp=voice_message.timestamp,
                        )

                        llm_result = await asyncio.wait_for(
                            self._process_llm_only(enhanced_message, conversation_state),
                            timeout=remaining_time / 1000.0,
                        )

                        # Combine results
                        combined_response = self._combine_responses(
                            moshi_result.response, llm_result.response
                        )

                        return ProcessingResult(
                            response=combined_response,
                            strategy=ProcessingStrategy.HYBRID_SEQUENTIAL,
                            latency_ms=(time.time() - start_time) * 1000,
                            quality_score=max(moshi_result.quality_score, llm_result.quality_score),
                            relevance_score=max(
                                moshi_result.relevance_score, llm_result.relevance_score
                            ),
                            confidence=0.85,
                            metadata={
                                "moshi_latency": moshi_result.latency_ms,
                                "llm_latency": llm_result.latency_ms,
                                "enhanced": True,
                            },
                        )

                    except asyncio.TimeoutError:
                        logger.debug("LLM enhancement timeout, using Moshi result")
                        pass

            # Return Moshi result if enhancement failed or wasn't needed
            moshi_result.strategy = ProcessingStrategy.HYBRID_SEQUENTIAL
            moshi_result.metadata["enhanced"] = False
            return moshi_result

        except Exception as e:
            logger.error(f"Hybrid sequential processing failed: {e}")
            raise

    def _select_best_result(self, results: List[ProcessingResult]) -> ProcessingResult:
        """Select the best result from multiple processing results."""

        if len(results) == 1:
            return results[0]

        if self.config.response_selection == ResponseSelection.FASTEST:
            return min(results, key=lambda r: r.latency_ms)

        elif self.config.response_selection == ResponseSelection.HIGHEST_QUALITY:
            return max(results, key=lambda r: r.quality_score)

        elif self.config.response_selection == ResponseSelection.MOST_RELEVANT:
            return max(results, key=lambda r: r.relevance_score)

        else:  # BALANCED
            # Weighted score: 40% quality, 30% relevance, 20% speed, 10% confidence
            def balanced_score(result: ProcessingResult) -> float:
                speed_score = 1.0 - min(result.latency_ms / 3000.0, 1.0)  # Normalize to 3s max
                return (
                    result.quality_score * 0.4
                    + result.relevance_score * 0.3
                    + speed_score * 0.2
                    + result.confidence * 0.1
                )

            return max(results, key=balanced_score)

    def _combine_responses(
        self, moshi_response: VoiceResponse, llm_response: VoiceResponse
    ) -> VoiceResponse:
        """Combine Moshi and LLM responses intelligently."""

        # Use LLM text with Moshi audio if available
        combined_response = VoiceResponse(
            response_id=f"hybrid_{moshi_response.message_id}",
            conversation_id=moshi_response.conversation_id,
            message_id=moshi_response.message_id,
            text_content=llm_response.text_content or moshi_response.text_content,
            audio_data=moshi_response.audio_data,  # Prefer Moshi audio
            processing_mode=VoiceProcessingMode.HYBRID,
            total_latency_ms=moshi_response.total_latency_ms + (llm_response.total_latency_ms or 0),
            moshi_latency_ms=moshi_response.moshi_latency_ms,
            llm_latency_ms=llm_response.llm_latency_ms,
            response_relevance=max(
                moshi_response.response_relevance or 0.5, llm_response.response_relevance or 0.5
            ),
        )

        return combined_response

    async def _analyze_context_factors(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState]
    ) -> Dict[str, Any]:
        """Analyze context factors for strategy selection."""

        factors = {}

        # Text analysis
        text_content = voice_message.text_content or ""
        factors["text_length"] = len(text_content)
        factors["word_count"] = len(text_content.split())

        # Simple intent detection
        text_lower = text_content.lower()
        if any(word in text_lower for word in ["code", "program", "function", "debug"]):
            factors["intent"] = "coding"
        elif any(word in text_lower for word in ["explain", "how", "why", "what"]):
            factors["intent"] = "explanation"
        elif any(word in text_lower for word in ["solve", "calculate", "analyze"]):
            factors["intent"] = "reasoning"
        else:
            factors["intent"] = "general"

        # Complexity estimation
        complex_words = ["implement", "algorithm", "architecture", "optimization"]
        complexity_score = sum(1 for word in complex_words if word in text_lower)

        if complexity_score > 2 or factors["word_count"] > 50:
            factors["complexity"] = "high"
        elif complexity_score > 0 or factors["word_count"] > 20:
            factors["complexity"] = "medium"
        else:
            factors["complexity"] = "low"

        # Conversation context
        if conversation_state:
            factors["conversation_active"] = conversation_state.is_active
            factors["conversation_mode"] = conversation_state.current_mode.value

        return factors

    def _has_good_performance(self, strategy: ProcessingStrategy) -> bool:
        """Check if a strategy has good recent performance."""

        if strategy not in self.strategy_performance:
            return False

        recent_scores = self.strategy_performance[strategy][-10:]  # Last 10 results

        if len(recent_scores) < 3:
            return False

        avg_score = sum(recent_scores) / len(recent_scores)
        return avg_score >= self.config.strategy_switch_threshold

    async def _fallback_processing(
        self, voice_message: VoiceMessage, conversation_state: Optional[ConversationState]
    ) -> VoiceResponse:
        """Fallback processing when main strategy fails."""

        logger.info(f"Using fallback strategy: {self.config.fallback_strategy}")

        try:
            if self.config.fallback_strategy == ProcessingStrategy.MOSHI_ONLY:
                result = await self._process_moshi_only(voice_message, conversation_state)
                return result.response

            elif self.config.fallback_strategy == ProcessingStrategy.LLM_ONLY:
                result = await self._process_llm_only(voice_message, conversation_state)
                return result.response

            else:
                # Ultimate fallback - create basic response
                return VoiceResponse(
                    response_id=f"fallback_{voice_message.message_id}",
                    conversation_id=voice_message.conversation_id,
                    message_id=voice_message.message_id,
                    text_content="I'm having trouble processing that right now. Could you try again?",
                    audio_data=b"",
                    processing_mode=VoiceProcessingMode.MOSHI_ONLY,
                    total_latency_ms=0.0,
                )

        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")

            # Ultimate fallback
            return VoiceResponse(
                response_id=f"emergency_{voice_message.message_id}",
                conversation_id=voice_message.conversation_id,
                message_id=voice_message.message_id,
                text_content="I'm experiencing technical difficulties. Please try again later.",
                audio_data=b"",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY,
                total_latency_ms=0.0,
            )

    def _update_performance_stats(
        self, strategy: ProcessingStrategy, latency_ms: float, result: ProcessingResult
    ) -> None:
        """Update performance statistics."""

        # Update strategy usage
        self.processing_stats["strategy_usage"][strategy.value] += 1
        self.processing_stats["total_requests"] += 1

        # Update average latency
        current_avg = self.processing_stats["average_latency_ms"]
        total_requests = self.processing_stats["total_requests"]

        new_avg = ((current_avg * (total_requests - 1)) + latency_ms) / total_requests
        self.processing_stats["average_latency_ms"] = new_avg

        # Track strategy performance
        performance_score = (
            result.quality_score * 0.4 + result.relevance_score * 0.3 + result.confidence * 0.3
        )

        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []

        self.strategy_performance[strategy].append(performance_score)

        # Keep only recent history
        if len(self.strategy_performance[strategy]) > self.config.performance_history_size:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][
                -self.config.performance_history_size :
            ]

    async def _update_adaptive_learning(
        self, strategy: ProcessingStrategy, result: ProcessingResult
    ) -> None:
        """Update adaptive learning based on results."""

        # Simple adaptive learning - adjust current strategy based on performance
        performance_score = (
            result.quality_score * 0.4 + result.relevance_score * 0.3 + result.confidence * 0.3
        )

        # If current strategy is performing well, stick with it
        if performance_score >= self.config.strategy_switch_threshold:
            self.current_strategy = strategy

        # If performance is poor, consider switching
        elif performance_score < 0.6:
            # Find best performing strategy
            best_strategy = None
            best_avg = 0.0

            for strat, scores in self.strategy_performance.items():
                if len(scores) >= 3:
                    avg_score = sum(scores[-5:]) / min(len(scores), 5)
                    if avg_score > best_avg:
                        best_avg = avg_score
                        best_strategy = strat

            if best_strategy and best_avg > performance_score:
                logger.info(
                    f"Adaptive learning: switching from {self.current_strategy} to {best_strategy}"
                )
                self.current_strategy = best_strategy

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""

        return {
            "current_strategy": self.current_strategy.value,
            "processing_stats": self.processing_stats.copy(),
            "strategy_performance": {
                strategy.value: {
                    "recent_scores": scores[-5:] if scores else [],
                    "average_score": sum(scores) / len(scores) if scores else 0.0,
                    "total_uses": len(scores),
                }
                for strategy, scores in self.strategy_performance.items()
            },
            "active_requests": len(self.active_requests),
            "configuration": {
                "default_strategy": self.config.default_strategy.value,
                "response_selection": self.config.response_selection.value,
                "adaptive_learning": self.config.enable_adaptive_learning,
                "parallel_processing": self.config.enable_parallel_processing,
            },
        }

    async def set_strategy(self, strategy: ProcessingStrategy) -> None:
        """Manually set the processing strategy."""

        logger.info(f"Manually setting strategy to: {strategy}")
        self.current_strategy = strategy

        # Update config if it's not adaptive
        if self.config.default_strategy != ProcessingStrategy.ADAPTIVE:
            self.config.default_strategy = strategy

    async def cleanup(self) -> None:
        """Clean up resources."""

        try:
            # Cancel active requests
            for request_id, task in self.active_requests.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self.active_requests.clear()

            # Clean up components
            if self.moshi_client:
                await self.moshi_client.cleanup()

            if self.context_manager:
                await self.context_manager.cleanup()

            if self.performance_optimizer:
                await self.performance_optimizer.cleanup()

            logger.info("HybridOrchestrator cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
