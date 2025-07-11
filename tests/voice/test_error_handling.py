"""
Tests for improved error handling in the voice processing system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.coda.components.voice.exceptions import (
    VoiceProcessingError, AudioProcessingError, ModelLoadingError,
    VRAMAllocationError, NetworkError, VoiceTimeoutError, ValidationError,
    ComponentNotInitializedError, ErrorCodes
)
from src.coda.components.voice.validation import (
    validate_audio_data, validate_voice_config, validate_conversation_id,
    validate_timeout, validate_file_path
)
from src.coda.components.voice.resource_management import (
    with_timeout, with_retry, CircuitBreaker, ResourcePool
)
from src.coda.components.voice.models import VoiceConfig, AudioConfig, MoshiConfig


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_voice_processing_error_basic(self):
        """Test basic VoiceProcessingError functionality."""
        error = VoiceProcessingError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.context == {}
    
    def test_voice_processing_error_with_code(self):
        """Test VoiceProcessingError with error code."""
        error = VoiceProcessingError(
            "Test error", 
            error_code="TEST_001",
            context={"key": "value"}
        )
        assert str(error) == "[TEST_001] Test error"
        assert error.error_code == "TEST_001"
        assert error.context == {"key": "value"}
    
    def test_network_error_with_status(self):
        """Test NetworkError with status code."""
        error = NetworkError(
            "Connection failed",
            status_code=500,
            retry_after=30.0
        )
        assert error.status_code == 500
        assert error.retry_after == 30.0
    
    def test_component_failure_error(self):
        """Test ComponentFailureError with component info."""
        error = ComponentFailureError(
            "moshi_client",
            "Failed to initialize",
            recoverable=False
        )
        assert error.component_name == "moshi_client"
        assert error.recoverable is False
        assert "moshi_client: Failed to initialize" in str(error)


class TestValidation:
    """Test input validation functions."""
    
    def test_validate_audio_data_empty(self):
        """Test validation of empty audio data."""
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(b"")
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_FORMAT_INVALID
        assert "empty" in str(exc_info.value)
    
    def test_validate_audio_data_too_small(self):
        """Test validation of too small audio data."""
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(b"small")
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_FORMAT_INVALID
        assert "too small" in str(exc_info.value)
    
    def test_validate_audio_data_too_large(self):
        """Test validation of too large audio data."""
        large_data = b"x" * (60 * 1024 * 1024)  # 60MB
        
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(large_data, max_size_mb=50)
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_SIZE_EXCEEDED
        assert "too large" in str(exc_info.value)
    
    def test_validate_conversation_id_empty(self):
        """Test validation of empty conversation ID."""
        with pytest.raises(ValidationError) as exc_info:
            validate_conversation_id("")
        
        assert exc_info.value.error_code == ErrorCodes.CONVERSATION_NOT_FOUND
    
    def test_validate_conversation_id_wrong_type(self):
        """Test validation of wrong type conversation ID."""
        with pytest.raises(ValidationError) as exc_info:
            validate_conversation_id(123)
        
        assert exc_info.value.error_code == ErrorCodes.CONVERSATION_INVALID_STATE
    
    def test_validate_conversation_id_too_long(self):
        """Test validation of too long conversation ID."""
        long_id = "x" * 150
        
        with pytest.raises(ValidationError) as exc_info:
            validate_conversation_id(long_id)
        
        assert exc_info.value.error_code == ErrorCodes.CONVERSATION_INVALID_STATE
    
    def test_validate_timeout_valid(self):
        """Test validation of valid timeout."""
        assert validate_timeout(10.0) == 10.0
        assert validate_timeout(None) == 300.0  # Default max
    
    def test_validate_timeout_too_small(self):
        """Test validation of too small timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout(0.05, min_timeout=0.1)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "too small" in str(exc_info.value)
    
    def test_validate_timeout_too_large(self):
        """Test validation of too large timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout(500.0, max_timeout=300.0)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "too large" in str(exc_info.value)


class TestResourceManagement:
    """Test resource management utilities."""
    
    @pytest.mark.asyncio
    async def test_timeout_decorator_success(self):
        """Test timeout decorator with successful operation."""
        @with_timeout(1.0)
        async def quick_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await quick_operation()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_timeout_decorator_timeout(self):
        """Test timeout decorator with timeout."""
        @with_timeout(0.1, "Custom timeout message")
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "success"
        
        with pytest.raises(VoiceTimeoutError) as exc_info:
            await slow_operation()
        
        assert "Custom timeout message" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCodes.NETWORK_TIMEOUT
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0
        
        @with_retry(max_retries=2, delay=0.01)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await flaky_operation()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_decorator_exhausted(self):
        """Test retry decorator with exhausted retries."""
        @with_retry(max_retries=1, delay=0.01, exceptions=(ValueError,))
        async def always_fail():
            raise ValueError("Always fails")
        
        with pytest.raises(ComponentFailureError) as exc_info:
            await always_fail()
        
        assert exc_info.value.error_code == ErrorCodes.COMPONENT_HEALTH_CHECK_FAILED
        assert "failed after 2 attempts" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self):
        """Test circuit breaker opening after failures."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        @circuit_breaker
        async def failing_operation():
            raise ValueError("Service unavailable")
        
        # First two failures should work normally
        with pytest.raises(ValueError):
            await failing_operation()
        
        with pytest.raises(ValueError):
            await failing_operation()
        
        # Third call should trigger circuit breaker
        with pytest.raises(ComponentFailureError) as exc_info:
            await failing_operation()
        
        assert "Circuit breaker is open" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCodes.COMPONENT_HEALTH_CHECK_FAILED
    
    @pytest.mark.asyncio
    async def test_resource_pool_acquire_release(self):
        """Test resource pool acquire and release."""
        pool = ResourcePool(max_size=2, timeout=1.0)
        
        async def create_resource():
            return Mock()
        
        # Acquire resources
        resource1 = await pool.acquire(create_resource)
        resource2 = await pool.acquire(create_resource)
        
        assert resource1 is not None
        assert resource2 is not None
        
        # Release resources
        await pool.release(resource1)
        await pool.release(resource2)
    
    @pytest.mark.asyncio
    async def test_resource_pool_exhaustion(self):
        """Test resource pool exhaustion."""
        pool = ResourcePool(max_size=1, timeout=0.1)
        
        async def create_resource():
            return Mock()
        
        # Acquire the only resource
        resource1 = await pool.acquire(create_resource)
        
        # Second acquire should timeout
        with pytest.raises(ResourceExhaustionError) as exc_info:
            await pool.acquire(create_resource)
        
        assert exc_info.value.error_code == ErrorCodes.RESOURCE_EXHAUSTED
        assert "pool exhausted" in str(exc_info.value)


class TestErrorIntegration:
    """Test error handling integration with voice components."""
    
    @pytest.mark.asyncio
    async def test_moshi_initialization_error_handling(self):
        """Test error handling in Moshi initialization."""
        # This would require mocking the actual Moshi components
        # For now, we'll test the error types are properly defined
        
        error = ModelLoadingError(
            "Failed to load Moshi model",
            error_code=ErrorCodes.MODEL_LOAD_FAILED,
            context={"model_path": "/path/to/model"}
        )
        
        assert error.error_code == ErrorCodes.MODEL_LOAD_FAILED
        assert error.context["model_path"] == "/path/to/model"
    
    def test_vram_allocation_error(self):
        """Test VRAM allocation error handling."""
        error = VRAMAllocationError(
            "Insufficient VRAM",
            error_code=ErrorCodes.VRAM_INSUFFICIENT,
            context={"requested_mb": 8000, "available_mb": 4000}
        )
        
        assert error.error_code == ErrorCodes.VRAM_INSUFFICIENT
        assert error.context["requested_mb"] == 8000
        assert error.context["available_mb"] == 4000


if __name__ == "__main__":
    pytest.main([__file__])
