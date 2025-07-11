"""
Custom exceptions for the voice processing system.

This module defines specific exception types for different error scenarios
in the voice processing pipeline, enabling better error handling and debugging.
"""

from typing import Any, Dict, Optional


class VoiceProcessingError(Exception):
    """Base exception for voice processing errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize voice processing error."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        """String representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class AudioProcessingError(VoiceProcessingError):
    """Exception for audio processing errors."""

    pass


class ModelLoadingError(VoiceProcessingError):
    """Exception for model loading errors."""

    pass


class VRAMAllocationError(VoiceProcessingError):
    """Exception for VRAM allocation errors."""

    pass


class ConversationError(VoiceProcessingError):
    """Exception for conversation management errors."""

    pass


class IntegrationError(VoiceProcessingError):
    """Exception for integration errors with other systems."""

    pass


class ConfigurationError(VoiceProcessingError):
    """Exception for configuration errors."""

    pass


class VoiceTimeoutError(VoiceProcessingError):
    """Exception for voice processing timeout errors."""

    pass


class ValidationError(VoiceProcessingError):
    """Exception for input validation errors."""

    pass


class NetworkError(VoiceProcessingError):
    """Exception for network-related errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        **kwargs,
    ):
        """Initialize network error."""
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.retry_after = retry_after


class MoshiError(VoiceProcessingError):
    """Exception for Moshi-specific errors."""

    pass


class LLMIntegrationError(VoiceProcessingError):
    """Exception for LLM integration errors."""

    pass


class StreamingError(VoiceProcessingError):
    """Exception for streaming errors."""

    pass


class ResourceExhaustionError(VoiceProcessingError):
    """Exception for resource exhaustion errors."""

    pass


class ComponentNotInitializedError(VoiceProcessingError):
    """Exception for using uninitialized components."""

    pass


class ComponentFailureError(VoiceProcessingError):
    """Exception for component failures."""

    def __init__(self, component_name: str, message: str, recoverable: bool = True, **kwargs):
        """Initialize component failure error."""
        super().__init__(f"{component_name}: {message}", **kwargs)
        self.component_name = component_name
        self.recoverable = recoverable


# Error code constants
class ErrorCodes:
    """Standard error codes for voice processing."""

    # Audio processing errors
    AUDIO_FORMAT_INVALID = "AUDIO_001"
    AUDIO_SIZE_EXCEEDED = "AUDIO_002"
    AUDIO_CORRUPTION = "AUDIO_003"
    AUDIO_ENCODING_FAILED = "AUDIO_004"

    # Model errors
    MODEL_NOT_FOUND = "MODEL_001"
    MODEL_LOAD_FAILED = "MODEL_002"
    MODEL_INCOMPATIBLE = "MODEL_003"
    MODEL_CORRUPTED = "MODEL_004"

    # VRAM errors
    VRAM_INSUFFICIENT = "VRAM_001"
    VRAM_ALLOCATION_FAILED = "VRAM_002"
    VRAM_FRAGMENTATION = "VRAM_003"

    # Conversation errors
    CONVERSATION_NOT_FOUND = "CONV_001"
    CONVERSATION_INVALID_STATE = "CONV_002"
    CONVERSATION_TIMEOUT = "CONV_003"

    # Integration errors
    MEMORY_INTEGRATION_FAILED = "INTEG_001"
    PERSONALITY_INTEGRATION_FAILED = "INTEG_002"
    TOOLS_INTEGRATION_FAILED = "INTEG_003"
    LLM_INTEGRATION_FAILED = "INTEG_004"

    # Processing errors
    PROCESSING_FAILED = "PROC_001"
    PROCESSING_TIMEOUT = "PROC_002"
    PROCESSING_INTERRUPTED = "PROC_003"
    PROCESSING_QUALITY_LOW = "PROC_004"

    # Configuration errors
    CONFIG_INVALID = "CONFIG_001"
    CONFIG_MISSING = "CONFIG_002"
    CONFIG_INCOMPATIBLE = "CONFIG_003"

    # Network errors
    NETWORK_TIMEOUT = "NET_001"
    NETWORK_CONNECTION_FAILED = "NET_002"
    NETWORK_RATE_LIMITED = "NET_003"
    NETWORK_UNAUTHORIZED = "NET_004"

    # Streaming errors
    STREAM_BUFFER_OVERFLOW = "STREAM_001"
    STREAM_BUFFER_UNDERFLOW = "STREAM_002"
    STREAM_CORRUPTION = "STREAM_003"
    STREAM_INTERRUPTED = "STREAM_004"

    # Resource errors
    RESOURCE_EXHAUSTED = "RES_001"
    RESOURCE_LOCKED = "RES_002"
    RESOURCE_CORRUPTED = "RES_003"

    # Component errors
    COMPONENT_NOT_INITIALIZED = "COMP_001"
    COMPONENT_INITIALIZATION_FAILED = "COMP_002"
    COMPONENT_SHUTDOWN_FAILED = "COMP_003"
    COMPONENT_HEALTH_CHECK_FAILED = "COMP_004"


def create_error(
    error_type: type, message: str, error_code: str, **context
) -> VoiceProcessingError:
    """Create a standardized error with code and context."""
    return error_type(message, error_code=error_code, context=context)


def wrap_exception(
    original_exception: Exception, error_type: type, message: str, error_code: str, **context
) -> VoiceProcessingError:
    """Wrap an original exception in a voice processing error."""
    context["original_exception"] = str(original_exception)
    context["original_type"] = type(original_exception).__name__
    return error_type(message, error_code=error_code, context=context)


# Additional exception classes for WebSocket and resource management
class WebSocketError(VoiceProcessingError):
    """Raised when WebSocket operations fail."""

    pass
