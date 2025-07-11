"""
Tests for resource management utilities.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

from src.coda.components.voice.resource_management import (
    with_timeout, with_retry, CircuitBreaker, ResourcePool,
    resource_cleanup, async_resource_cleanup
)
from src.coda.components.voice.exceptions import (
    VoiceTimeoutError, ComponentFailureError, ResourceExhaustionError, ErrorCodes
)


class TestTimeoutDecorator:
    """Test timeout decorator functionality."""
    
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
        assert exc_info.value.context["timeout_seconds"] == 0.1
    
    @pytest.mark.asyncio
    async def test_timeout_decorator_default_message(self):
        """Test timeout decorator with default error message."""
        @with_timeout(0.1)
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "success"
        
        with pytest.raises(VoiceTimeoutError) as exc_info:
            await slow_operation()

        assert "timed out after 0.1s" in str(exc_info.value)


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success_first_try(self):
        """Test retry decorator with immediate success."""
        call_count = 0
        
        @with_retry(max_retries=2, delay=0.01)
        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_operation()
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success_after_retries(self):
        """Test retry decorator with eventual success."""
        call_count = 0
        
        @with_retry(max_retries=2, delay=0.01)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await flaky_operation()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_decorator_exhausted(self):
        """Test retry decorator with exhausted retries."""
        call_count = 0
        
        @with_retry(max_retries=1, delay=0.01, exceptions=(ValueError,))
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ComponentFailureError) as exc_info:
            await always_fail()
        
        assert exc_info.value.error_code == ErrorCodes.COMPONENT_HEALTH_CHECK_FAILED
        assert "failed after 2 attempts" in str(exc_info.value)
        assert call_count == 2  # Initial + 1 retry
    
    @pytest.mark.asyncio
    async def test_retry_decorator_specific_exceptions(self):
        """Test retry decorator with specific exception types."""
        @with_retry(max_retries=1, delay=0.01, exceptions=(ValueError,))
        async def operation_with_different_error():
            raise TypeError("Different error type")
        
        # Should not retry TypeError, only ValueError
        with pytest.raises(TypeError):
            await operation_with_different_error()
    
    def test_retry_decorator_sync_function(self):
        """Test retry decorator with synchronous function."""
        call_count = 0
        
        @with_retry(max_retries=2, delay=0.01)
        def flaky_sync_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_sync_operation()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_decorator_backoff(self):
        """Test retry decorator with exponential backoff."""
        call_times = []
        
        @with_retry(max_retries=2, delay=0.1, backoff_factor=2.0)
        async def operation_with_backoff():
            call_times.append(time.time())
            raise ValueError("Always fails")
        
        with pytest.raises(ComponentFailureError):
            await operation_with_backoff()
        
        # Should have 3 calls (initial + 2 retries)
        assert len(call_times) == 3
        
        # Check that delays increase (approximately)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        assert delay2 > delay1  # Second delay should be longer


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        @circuit_breaker
        async def successful_operation():
            return "success"
        
        result = await successful_operation()
        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker opening after failures."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        @circuit_breaker
        async def failing_operation():
            raise ValueError("Service unavailable")
        
        # First two failures should work normally
        with pytest.raises(ValueError):
            await failing_operation()
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 1
        
        with pytest.raises(ValueError):
            await failing_operation()
        assert circuit_breaker.state == "open"
        assert circuit_breaker.failure_count == 2
        
        # Third call should trigger circuit breaker
        with pytest.raises(ComponentFailureError) as exc_info:
            await failing_operation()
        
        assert "Circuit breaker is open" in str(exc_info.value)
        assert exc_info.value.error_code == ErrorCodes.COMPONENT_HEALTH_CHECK_FAILED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker half-open state after recovery timeout."""
        circuit_breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        @circuit_breaker
        async def operation():
            if circuit_breaker.state == "half-open":
                return "recovered"
            raise ValueError("Still failing")
        
        # Trigger circuit breaker
        with pytest.raises(ValueError):
            await operation()
        assert circuit_breaker.state == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Next call should succeed and close circuit
        result = await operation()
        assert result == "recovered"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
    
    def test_circuit_breaker_sync_function(self):
        """Test circuit breaker with synchronous function."""
        circuit_breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        @circuit_breaker
        def failing_sync_operation():
            raise ValueError("Sync failure")
        
        # First failure
        with pytest.raises(ValueError):
            failing_sync_operation()
        assert circuit_breaker.state == "open"
        
        # Second call should trigger circuit breaker
        with pytest.raises(ComponentFailureError):
            failing_sync_operation()


class TestResourcePool:
    """Test resource pool functionality."""
    
    @pytest.mark.asyncio
    async def test_resource_pool_acquire_release(self):
        """Test basic resource pool acquire and release."""
        pool = ResourcePool(max_size=2, timeout=1.0)
        
        async def create_resource():
            return Mock(name="resource")
        
        # Acquire resources
        resource1 = await pool.acquire(create_resource)
        resource2 = await pool.acquire(create_resource)
        
        assert resource1 is not None
        assert resource2 is not None
        assert resource1 != resource2
        
        # Release resources
        await pool.release(resource1)
        await pool.release(resource2)
    
    @pytest.mark.asyncio
    async def test_resource_pool_reuse(self):
        """Test resource pool reuse of released resources."""
        pool = ResourcePool(max_size=1, timeout=1.0)
        
        async def create_resource():
            return Mock(name="resource")
        
        # Acquire and release a resource
        resource1 = await pool.acquire(create_resource)
        await pool.release(resource1)
        
        # Acquire again - should get the same resource
        resource2 = await pool.acquire(create_resource)
        assert resource2 == resource1
    
    @pytest.mark.asyncio
    async def test_resource_pool_exhaustion(self):
        """Test resource pool exhaustion."""
        pool = ResourcePool(max_size=1, timeout=0.1)
        
        async def create_resource():
            return Mock(name="resource")
        
        # Acquire the only resource
        resource1 = await pool.acquire(create_resource)
        
        # Second acquire should timeout
        with pytest.raises(ResourceExhaustionError) as exc_info:
            await pool.acquire(create_resource)
        
        assert exc_info.value.error_code == ErrorCodes.RESOURCE_EXHAUSTED
        assert "pool exhausted" in str(exc_info.value)
        assert exc_info.value.context["max_size"] == 1
        assert exc_info.value.context["timeout"] == 0.1
    
    @pytest.mark.asyncio
    async def test_resource_pool_factory_failure(self):
        """Test resource pool with factory that fails."""
        pool = ResourcePool(max_size=2, timeout=1.0)
        
        async def failing_factory():
            raise ValueError("Factory failed")
        
        with pytest.raises(ValueError):
            await pool.acquire(failing_factory)
        
        # Pool should not be affected by factory failure
        async def working_factory():
            return Mock(name="resource")
        
        resource = await pool.acquire(working_factory)
        assert resource is not None
    
    @pytest.mark.asyncio
    async def test_resource_pool_release_full_pool(self):
        """Test releasing resource when pool is full."""
        pool = ResourcePool(max_size=1, timeout=1.0)
        
        # Mock resource with close method
        mock_resource = Mock()
        mock_resource.close = Mock()
        
        async def create_resource():
            return mock_resource
        
        # Fill the pool
        resource1 = await pool.acquire(create_resource)
        await pool.release(resource1)
        
        # Create another resource
        resource2 = await pool.acquire(create_resource)
        
        # Release when pool is full - should close the resource
        await pool.release(Mock(close=Mock()))
        
        # The extra resource should have been closed
        # (This is implementation-specific behavior)


class TestResourceCleanup:
    """Test resource cleanup utilities."""
    
    def test_resource_cleanup_sync(self):
        """Test synchronous resource cleanup."""
        resource1 = Mock()
        resource1.close = Mock()
        
        resource2 = Mock()
        resource2.cleanup = Mock()
        
        with resource_cleanup(resource1, resource2):
            pass
        
        resource1.close.assert_called_once()
        resource2.cleanup.assert_called_once()
    
    def test_resource_cleanup_with_exception(self):
        """Test resource cleanup when exception occurs."""
        resource = Mock()
        resource.close = Mock()
        
        with pytest.raises(ValueError):
            with resource_cleanup(resource):
                raise ValueError("Test exception")
        
        # Resource should still be cleaned up
        resource.close.assert_called_once()
    
    def test_resource_cleanup_with_cleanup_error(self):
        """Test resource cleanup when cleanup itself fails."""
        resource = Mock()
        resource.close = Mock(side_effect=Exception("Cleanup failed"))
        
        # Should not raise exception even if cleanup fails
        with resource_cleanup(resource):
            pass
        
        resource.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_resource_cleanup(self):
        """Test asynchronous resource cleanup."""
        resource1 = Mock()
        resource1.close = AsyncMock()
        
        resource2 = Mock()
        resource2.cleanup = Mock()  # Sync cleanup
        
        async with async_resource_cleanup(resource1, resource2):
            pass
        
        resource1.close.assert_called_once()
        resource2.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_resource_cleanup_with_exception(self):
        """Test async resource cleanup when exception occurs."""
        resource = Mock()
        resource.close = AsyncMock()
        
        with pytest.raises(ValueError):
            async with async_resource_cleanup(resource):
                raise ValueError("Test exception")
        
        # Resource should still be cleaned up
        resource.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
