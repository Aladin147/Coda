"""
Voice Fallback Manager

This module implements comprehensive fallback mechanisms for graceful degradation
when voice processing components fail, ensuring system reliability and continuity.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .exceptions import VoiceTimeoutError
from .mode_manager import ProcessingModeType
from .models import ConversationState, VoiceMessage, VoiceProcessingMode, VoiceResponse

logger = logging.getLogger("coda.voice.fallback_manager")


class FailureType(str, Enum):
    """Types of failures that can occur."""

    COMPONENT_FAILURE = "component_failure"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT = "rate_limit"
    INVALID_INPUT = "invalid_input"
    UNKNOWN_ERROR = "unknown_error"


class FallbackStrategy(str, Enum):
    """Available fallback strategies."""

    RETRY = "retry"  # Retry the same operation
    ALTERNATIVE_COMPONENT = "alternative"  # Use alternative component
    DEGRADED_MODE = "degraded"  # Use simpler processing
    CACHED_RESPONSE = "cached"  # Use cached response
    EMERGENCY_RESPONSE = "emergency"  # Use emergency response
    FAIL_FAST = "fail_fast"  # Fail immediately


class FallbackPriority(int, Enum):
    """Priority levels for fallback strategies."""

    HIGHEST = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    LOWEST = 5


@dataclass
class FailureEvent:
    """Represents a failure event."""

    timestamp: float
    failure_type: FailureType
    component: str
    error_message: str
    context: Dict[str, Any]

    # Recovery information
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_time_ms: Optional[float] = None
    fallback_strategy_used: Optional[FallbackStrategy] = None


@dataclass
class FallbackRule:
    """Defines a fallback rule for specific failure conditions."""

    rule_id: str
    failure_types: List[FailureType]
    components: List[str]  # Components this rule applies to
    strategy: FallbackStrategy
    priority: FallbackPriority

    # Conditions
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 10.0

    # Alternative configuration
    alternative_component: Optional[str] = None
    degraded_mode: Optional[ProcessingModeType] = None

    # Custom handler
    custom_handler: Optional[Callable] = None


@dataclass
class FallbackConfig:
    """Configuration for fallback management."""

    # Global settings
    enable_fallbacks: bool = True
    max_total_retries: int = 5
    failure_threshold: int = 3  # Failures before circuit breaker
    recovery_timeout_seconds: float = 60.0

    # Emergency response settings
    emergency_response_enabled: bool = True
    emergency_response_text: str = (
        "I'm experiencing technical difficulties. Please try again later."
    )

    # Monitoring and alerting
    enable_failure_monitoring: bool = True
    failure_alert_threshold: int = 5
    monitoring_window_minutes: int = 10

    # Circuit breaker settings
    enable_circuit_breaker: bool = True
    circuit_breaker_timeout_seconds: float = 30.0

    # Fallback rules
    fallback_rules: List[FallbackRule] = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker for component failure management."""

    def __init__(self, component: str, failure_threshold: int = 3, timeout_seconds: float = 30.0):
        self.component = component
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds

        # State tracking
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False

    def record_success(self) -> None:
        """Record a successful operation."""
        self.failure_count = 0
        self.is_open = False

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened for {self.component}")

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if not self.is_open:
            return True

        # Check if timeout has passed
        if self.last_failure_time and time.time() - self.last_failure_time > self.timeout_seconds:
            self.is_open = False
            self.failure_count = 0
            logger.info(f"Circuit breaker reset for {self.component}")
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "component": self.component,
            "is_open": self.is_open,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "timeout_seconds": self.timeout_seconds,
        }


class FallbackManager:
    """
    Comprehensive fallback management system for voice processing.

    Features:
    - Multiple fallback strategies
    - Circuit breaker pattern
    - Failure monitoring and alerting
    - Graceful degradation
    - Emergency response handling
    """

    def __init__(self, config: FallbackConfig):
        """Initialize the fallback manager."""
        self.config = config

        # Failure tracking
        self.failure_history: deque = deque(maxlen=1000)
        self.component_failures: Dict[str, List[FailureEvent]] = {}

        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Fallback rules
        self.fallback_rules = {rule.rule_id: rule for rule in config.fallback_rules}

        # Emergency response cache
        self.emergency_responses: Dict[str, VoiceResponse] = {}

        # Statistics
        self.fallback_stats = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "emergency_responses": 0,
            "circuit_breaker_activations": 0,
        }

        logger.info("FallbackManager initialized")

    async def execute_with_fallback(
        self,
        operation: Callable,
        component: str,
        voice_message: VoiceMessage,
        context: Optional[Dict[str, Any]] = None,
    ) -> VoiceResponse:
        """
        Execute an operation with comprehensive fallback handling.

        Args:
            operation: The operation to execute
            component: Component name for tracking
            voice_message: Voice message being processed
            context: Additional context

        Returns:
            Voice response (potentially from fallback)
        """

        if not self.config.enable_fallbacks:
            return await operation(voice_message)

        # Check circuit breaker
        circuit_breaker = self._get_circuit_breaker(component)
        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {component}, using fallback")
            return await self._handle_circuit_breaker_fallback(voice_message, component)

        # Attempt operation with retries
        last_exception = None

        for attempt in range(self.config.max_total_retries + 1):
            try:
                # Execute operation
                result = await self._execute_with_timeout(operation, voice_message)

                # Success - record and return
                circuit_breaker.record_success()
                return result

            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)

                # Record failure
                failure_event = FailureEvent(
                    timestamp=time.time(),
                    failure_type=failure_type,
                    component=component,
                    error_message=str(e),
                    context=context or {},
                )

                self._record_failure(failure_event)
                circuit_breaker.record_failure()

                # Find applicable fallback rule
                fallback_rule = self._find_fallback_rule(failure_type, component)

                if fallback_rule and attempt < self.config.max_total_retries:
                    # Apply fallback strategy
                    fallback_result = await self._apply_fallback_strategy(
                        fallback_rule, voice_message, failure_event, attempt
                    )

                    if fallback_result:
                        return fallback_result

                # Wait before retry (if not last attempt)
                if attempt < self.config.max_total_retries:
                    retry_delay = self._calculate_retry_delay(attempt, fallback_rule)
                    await asyncio.sleep(retry_delay)

        # All attempts failed - use emergency response
        logger.error(f"All fallback attempts failed for {component}: {last_exception}")
        return await self._generate_emergency_response(voice_message, last_exception)

    async def _execute_with_timeout(
        self,
        operation: Callable,
        voice_message: VoiceMessage,
        timeout_seconds: Optional[float] = None,
    ) -> VoiceResponse:
        """Execute operation with timeout."""

        timeout = timeout_seconds or self.config.recovery_timeout_seconds

        try:
            return await asyncio.wait_for(operation(voice_message), timeout=timeout)
        except asyncio.TimeoutError:
            raise VoiceTimeoutError(f"Operation timed out after {timeout} seconds")

    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception."""

        exception_type = type(exception).__name__
        exception_message = str(exception).lower()

        if isinstance(exception, asyncio.TimeoutError) or "timeout" in exception_message:
            return FailureType.TIMEOUT
        elif "network" in exception_message or "connection" in exception_message:
            return FailureType.NETWORK_ERROR
        elif "memory" in exception_message or "resource" in exception_message:
            return FailureType.RESOURCE_EXHAUSTION
        elif "auth" in exception_message or "permission" in exception_message:
            return FailureType.AUTHENTICATION_ERROR
        elif "rate" in exception_message or "limit" in exception_message:
            return FailureType.RATE_LIMIT
        elif "invalid" in exception_message or "bad" in exception_message:
            return FailureType.INVALID_INPUT
        else:
            return FailureType.UNKNOWN_ERROR

    def _find_fallback_rule(
        self, failure_type: FailureType, component: str
    ) -> Optional[FallbackRule]:
        """Find the best fallback rule for the failure."""

        applicable_rules = []

        for rule in self.fallback_rules.values():
            if failure_type in rule.failure_types and (
                not rule.components or component in rule.components
            ):
                applicable_rules.append(rule)

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority.value)

        return applicable_rules[0] if applicable_rules else None

    async def _apply_fallback_strategy(
        self,
        rule: FallbackRule,
        voice_message: VoiceMessage,
        failure_event: FailureEvent,
        attempt: int,
    ) -> Optional[VoiceResponse]:
        """Apply the specified fallback strategy."""

        logger.info(f"Applying fallback strategy: {rule.strategy} (attempt {attempt + 1})")

        try:
            if rule.strategy == FallbackStrategy.RETRY:
                # Retry will be handled by the main loop
                return None

            elif rule.strategy == FallbackStrategy.ALTERNATIVE_COMPONENT:
                return await self._use_alternative_component(rule, voice_message)

            elif rule.strategy == FallbackStrategy.DEGRADED_MODE:
                return await self._use_degraded_mode(rule, voice_message)

            elif rule.strategy == FallbackStrategy.CACHED_RESPONSE:
                return await self._use_cached_response(voice_message)

            elif rule.strategy == FallbackStrategy.EMERGENCY_RESPONSE:
                return await self._generate_emergency_response(voice_message, None)

            elif rule.strategy == FallbackStrategy.FAIL_FAST:
                raise RuntimeError("Fail - fast strategy triggered")

            else:
                logger.warning(f"Unknown fallback strategy: {rule.strategy}")
                return None

        except Exception as e:
            logger.error(f"Fallback strategy {rule.strategy} failed: {e}")
            return None

    async def _use_alternative_component(
        self, rule: FallbackRule, voice_message: VoiceMessage
    ) -> Optional[VoiceResponse]:
        """Use alternative component for processing."""

        if not rule.alternative_component:
            return None

        # This would integrate with the actual component system
        logger.info(f"Using alternative component: {rule.alternative_component}")

        # Mock alternative processing
        return VoiceResponse(
            response_id=f"fallback_{voice_message.message_id}",
            conversation_id=voice_message.conversation_id,
            message_id=voice_message.message_id,
            text_content=f"Alternative response: {voice_message.text_content[:50]}...",
            audio_data=b"",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,  # Simpler mode
            total_latency_ms=100.0,
            response_relevance=0.6,  # Lower quality
        )

    async def _use_degraded_mode(
        self, rule: FallbackRule, voice_message: VoiceMessage
    ) -> Optional[VoiceResponse]:
        """Use degraded processing mode."""

        if not rule.degraded_mode:
            return None

        logger.info(f"Using degraded mode: {rule.degraded_mode}")

        # Mock degraded processing
        return VoiceResponse(
            response_id=f"degraded_{voice_message.message_id}",
            conversation_id=voice_message.conversation_id,
            message_id=voice_message.message_id,
            text_content=f"Degraded response: {voice_message.text_content[:30]}...",
            audio_data=b"",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=50.0,
            response_relevance=0.5,
        )

    async def _use_cached_response(self, voice_message: VoiceMessage) -> Optional[VoiceResponse]:
        """Use cached response if available."""

        cache_key = f"{voice_message.conversation_id}:{hash(voice_message.text_content or '')}"

        if cache_key in self.emergency_responses:
            cached_response = self.emergency_responses[cache_key]
            logger.info("Using cached response")
            return cached_response

        return None

    async def _generate_emergency_response(
        self, voice_message: VoiceMessage, exception: Optional[Exception]
    ) -> VoiceResponse:
        """Generate emergency response."""

        self.fallback_stats["emergency_responses"] += 1

        emergency_response = VoiceResponse(
            response_id=f"emergency_{voice_message.message_id}_{int(time.time() * 1000)}",
            conversation_id=voice_message.conversation_id,
            message_id=voice_message.message_id,
            text_content=self.config.emergency_response_text,
            audio_data=b"",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=10.0,
            response_relevance=0.1,
        )

        # Cache for potential reuse
        cache_key = f"{voice_message.conversation_id}:emergency"
        self.emergency_responses[cache_key] = emergency_response

        logger.warning(f"Generated emergency response for {voice_message.message_id}")

        return emergency_response

    async def _handle_circuit_breaker_fallback(
        self, voice_message: VoiceMessage, component: str
    ) -> VoiceResponse:
        """Handle fallback when circuit breaker is open."""

        self.fallback_stats["circuit_breaker_activations"] += 1

        return VoiceResponse(
            response_id=f"circuit_breaker_{voice_message.message_id}",
            conversation_id=voice_message.conversation_id,
            message_id=voice_message.message_id,
            text_content=f"Service temporarily unavailable. Please try again in a moment.",
            audio_data=b"",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            total_latency_ms=5.0,
            response_relevance=0.2,
        )

    def _get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component."""

        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker(
                component=component,
                failure_threshold=self.config.failure_threshold,
                timeout_seconds=self.config.circuit_breaker_timeout_seconds,
            )

        return self.circuit_breakers[component]

    def _calculate_retry_delay(self, attempt: int, rule: Optional[FallbackRule]) -> float:
        """Calculate retry delay with exponential backoff."""

        base_delay = rule.retry_delay_seconds if rule else 1.0

        # Exponential backoff with jitter
        delay = base_delay * (2**attempt)
        jitter = delay * 0.1  # 10% jitter

        return min(delay + jitter, 10.0)  # Cap at 10 seconds

    def _record_failure(self, failure_event: FailureEvent) -> None:
        """Record failure event for monitoring."""

        self.failure_history.append(failure_event)
        self.fallback_stats["total_failures"] += 1

        # Track component - specific failures
        if failure_event.component not in self.component_failures:
            self.component_failures[failure_event.component] = []

        self.component_failures[failure_event.component].append(failure_event)

        # Keep only recent failures per component
        if len(self.component_failures[failure_event.component]) > 100:
            self.component_failures[failure_event.component] = self.component_failures[
                failure_event.component
            ][-100:]

    def add_fallback_rule(self, rule: FallbackRule) -> None:
        """Add a new fallback rule."""
        self.fallback_rules[rule.rule_id] = rule
        logger.info(f"Added fallback rule: {rule.rule_id}")

    def remove_fallback_rule(self, rule_id: str) -> bool:
        """Remove a fallback rule."""
        if rule_id in self.fallback_rules:
            del self.fallback_rules[rule_id]
            logger.info(f"Removed fallback rule: {rule_id}")
            return True
        return False

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics."""

        recent_failures = [
            f
            for f in self.failure_history
            if time.time() - f.timestamp < self.config.monitoring_window_minutes * 60
        ]

        failure_by_type = {}
        failure_by_component = {}

        for failure in recent_failures:
            # By type
            if failure.failure_type.value not in failure_by_type:
                failure_by_type[failure.failure_type.value] = 0
            failure_by_type[failure.failure_type.value] += 1

            # By component
            if failure.component not in failure_by_component:
                failure_by_component[failure.component] = 0
            failure_by_component[failure.component] += 1

        return {
            "total_failures": self.fallback_stats["total_failures"],
            "successful_recoveries": self.fallback_stats["successful_recoveries"],
            "emergency_responses": self.fallback_stats["emergency_responses"],
            "circuit_breaker_activations": self.fallback_stats["circuit_breaker_activations"],
            "recent_failures": len(recent_failures),
            "failure_by_type": failure_by_type,
            "failure_by_component": failure_by_component,
            "circuit_breaker_status": {
                component: cb.get_status() for component, cb in self.circuit_breakers.items()
            },
        }

    async def cleanup(self) -> None:
        """Clean up fallback manager resources."""

        try:
            # Clear failure history
            self.failure_history.clear()
            self.component_failures.clear()

            # Reset circuit breakers
            self.circuit_breakers.clear()

            # Clear emergency responses
            self.emergency_responses.clear()

            logger.info("FallbackManager cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
