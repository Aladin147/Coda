"""
Resource management utilities for the voice processing system.

This module provides context managers and utilities for proper resource
cleanup, timeout handling, and error recovery.
"""

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Callable, Generator, Optional, TypeVar

from .exceptions import (
    ComponentFailureError,
    ErrorCodes,
    ResourceExhaustionError,
    VoiceTimeoutError,
    create_error,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_timeout(timeout_seconds: float, error_message: Optional[str] = None):
    """
    Decorator to add timeout to async functions.

    Args:
        timeout_seconds: Timeout in seconds
        error_message: Custom error message

    Returns:
        Decorated function with timeout
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                msg = error_message or f"Operation timed out after {timeout_seconds}s"
                raise create_error(
                    VoiceTimeoutError,
                    msg,
                    ErrorCodes.NETWORK_TIMEOUT,
                    timeout_seconds=timeout_seconds,
                    function_name=func.__name__,
                )

        return wrapper

    return decorator


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to add retry logic to functions.

    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

            # All retries exhausted
            raise create_error(
                ComponentFailureError,
                f"Function {func.__name__} failed after {max_retries + 1} attempts",
                ErrorCodes.COMPONENT_HEALTH_CHECK_FAILED,
                max_retries=max_retries,
                last_error=str(last_exception),
                function_name=func.__name__,
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # All retries exhausted
            raise create_error(
                ComponentFailureError,
                f"Function {func.__name__} failed after {max_retries + 1} attempts",
                ErrorCodes.COMPONENT_HEALTH_CHECK_FAILED,
                max_retries=max_retries,
                last_error=str(last_exception),
                function_name=func.__name__,
            )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def resource_cleanup(*resources: Any) -> Generator[None, None, None]:
    """
    Context manager for automatic resource cleanup.

    Args:
        *resources: Resources that have a 'close' or 'cleanup' method

    Yields:
        None
    """
    try:
        yield
    finally:
        for resource in resources:
            try:
                if hasattr(resource, "close"):
                    if asyncio.iscoroutinefunction(resource.close):
                        # For async resources, we can't await here in sync context
                        logger.warning(f"Cannot close async resource {resource} in sync context")
                    else:
                        resource.close()
                elif hasattr(resource, "cleanup"):
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        logger.warning(f"Cannot cleanup async resource {resource} in sync context")
                    else:
                        resource.cleanup()
                elif hasattr(resource, "__exit__"):
                    resource.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up resource {resource}: {e}")


@asynccontextmanager
async def async_resource_cleanup(*resources: Any) -> AsyncGenerator[None, None]:
    """
    Async context manager for automatic resource cleanup.

    Args:
        *resources: Resources that have a 'close' or 'cleanup' method

    Yields:
        None
    """
    try:
        yield
    finally:
        for resource in resources:
            try:
                if hasattr(resource, "close"):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
                elif hasattr(resource, "cleanup"):
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        await resource.cleanup()
                    else:
                        resource.cleanup()
                elif hasattr(resource, "__aexit__"):
                    await resource.__aexit__(None, None, None)
                elif hasattr(resource, "__exit__"):
                    resource.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up resource {resource}: {e}")


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external service calls.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying again
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise create_error(
                        ComponentFailureError,
                        f"Circuit breaker is open for {func.__name__}",
                        ErrorCodes.COMPONENT_HEALTH_CHECK_FAILED,
                        state=self.state,
                        failure_count=self.failure_count,
                    )
                else:
                    self.state = "half-open"

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise create_error(
                        ComponentFailureError,
                        f"Circuit breaker is open for {func.__name__}",
                        ErrorCodes.COMPONENT_HEALTH_CHECK_FAILED,
                        state=self.state,
                        failure_count=self.failure_count,
                    )
                else:
                    self.state = "half-open"

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ResourcePool:
    """
    Generic resource pool for managing limited resources.
    """

    def __init__(self, max_size: int = 10, timeout: float = 30.0):
        """
        Initialize resource pool.

        Args:
            max_size: Maximum number of resources in pool
            timeout: Timeout for acquiring resources
        """
        self.max_size = max_size
        self.timeout = timeout
        self.pool = asyncio.Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = asyncio.Lock()

    async def acquire(self, factory: Callable[[], T]) -> T:
        """
        Acquire a resource from the pool.

        Args:
            factory: Function to create new resources

        Returns:
            Resource from pool or newly created

        Raises:
            ResourceExhaustionError: If pool is exhausted and timeout reached
        """
        try:
            # Try to get existing resource
            return await asyncio.wait_for(self.pool.get(), timeout=0.1)
        except asyncio.TimeoutError:
            pass

        # Try to create new resource if under limit
        async with self.lock:
            if self.created_count < self.max_size:
                self.created_count += 1
                try:
                    return await factory()
                except Exception:
                    self.created_count -= 1
                    raise

        # Wait for resource to become available
        try:
            return await asyncio.wait_for(self.pool.get(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise create_error(
                ResourceExhaustionError,
                f"Resource pool exhausted (max: {self.max_size}, timeout: {self.timeout}s)",
                ErrorCodes.RESOURCE_EXHAUSTED,
                max_size=self.max_size,
                timeout=self.timeout,
            )

    async def release(self, resource: T) -> None:
        """
        Release a resource back to the pool.

        Args:
            resource: Resource to release
        """
        try:
            await self.pool.put(resource)
        except asyncio.QueueFull:
            # Pool is full, discard resource
            if hasattr(resource, "close"):
                try:
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
                except Exception as e:
                    logger.error(f"Error closing discarded resource: {e}")

            async with self.lock:
                self.created_count -= 1
