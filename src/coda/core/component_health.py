"""
Component health tracking and error handling for the Coda assistant.

This module provides comprehensive error handling, health tracking, and recovery
mechanisms for component initialization and operation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component health status."""

    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


class ErrorCategory(Enum):
    """Categories of initialization errors."""

    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    NETWORK = "network"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


@dataclass
class ComponentError:
    """Represents an error that occurred during component initialization."""

    component: str
    category: ErrorCategory
    message: str
    exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    is_recoverable: bool = True


@dataclass
class ComponentHealth:
    """Tracks the health status of a component."""

    name: str
    status: ComponentStatus = ComponentStatus.NOT_INITIALIZED
    last_error: Optional[ComponentError] = None
    initialization_attempts: int = 0
    last_health_check: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    is_critical: bool = True


class ComponentHealthManager:
    """Manages component health tracking and error handling."""

    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self.error_history: List[ComponentError] = []
        self.max_retry_attempts = 3
        self.retry_delay_seconds = 2.0
        self.health_check_interval = 300  # 5 minutes

    def register_component(
        self, name: str, dependencies: List[str] = None, is_critical: bool = True
    ) -> None:
        """Register a component for health tracking."""
        self.components[name] = ComponentHealth(
            name=name, dependencies=dependencies or [], is_critical=is_critical
        )
        logger.debug(f"Registered component: {name}")

    def set_status(self, component: str, status: ComponentStatus) -> None:
        """Update component status."""
        if component in self.components:
            old_status = self.components[component].status
            self.components[component].status = status
            self.components[component].last_health_check = datetime.now()

            if old_status != status:
                logger.info(
                    f"Component {component} status changed: {old_status.value} -> {status.value}"
                )

    def record_error(
        self,
        component: str,
        error: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        is_recoverable: bool = True,
    ) -> ComponentError:
        """Record an error for a component."""
        component_error = ComponentError(
            component=component,
            category=category,
            message=str(error),
            exception=error,
            is_recoverable=is_recoverable,
        )

        self.error_history.append(component_error)

        if component in self.components:
            self.components[component].last_error = component_error
            self.components[component].initialization_attempts += 1
            self.set_status(component, ComponentStatus.FAILED)

        logger.error(f"Component {component} error ({category.value}): {error}")
        return component_error

    def should_retry(self, component: str) -> bool:
        """Determine if a component should be retried."""
        if component not in self.components:
            return False

        comp_health = self.components[component]

        # Don't retry if max attempts reached
        if comp_health.initialization_attempts >= self.max_retry_attempts:
            return False

        # Don't retry if last error was not recoverable
        if comp_health.last_error and not comp_health.last_error.is_recoverable:
            return False

        return True

    async def retry_component_initialization(
        self, component: str, init_func: Callable[[], Awaitable[Any]]
    ) -> bool:
        """Retry component initialization with exponential backoff."""
        if not self.should_retry(component):
            return False

        comp_health = self.components[component]
        retry_count = comp_health.initialization_attempts

        # Exponential backoff
        delay = self.retry_delay_seconds * (2**retry_count)
        logger.info(f"Retrying {component} initialization in {delay}s (attempt {retry_count + 1})")

        await asyncio.sleep(delay)

        try:
            self.set_status(component, ComponentStatus.INITIALIZING)
            result = await init_func()
            self.set_status(component, ComponentStatus.HEALTHY)
            logger.info(
                f"Component {component} initialization succeeded on retry {retry_count + 1}"
            )
            return True

        except Exception as e:
            self.record_error(component, e, self._categorize_error(e))
            return False

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()

        # Configuration errors
        if any(term in error_msg for term in ["config", "setting", "parameter", "invalid"]):
            return ErrorCategory.CONFIGURATION

        # Dependency errors
        if any(term in error_msg for term in ["import", "module", "dependency", "not found"]):
            return ErrorCategory.DEPENDENCY

        # Resource errors
        if any(term in error_msg for term in ["memory", "disk", "resource", "allocation"]):
            return ErrorCategory.RESOURCE

        # Network errors
        if any(term in error_msg for term in ["network", "connection", "timeout", "unreachable"]):
            return ErrorCategory.NETWORK

        # Permission errors
        if any(term in error_msg for term in ["permission", "access", "denied", "unauthorized"]):
            return ErrorCategory.PERMISSION

        return ErrorCategory.UNKNOWN

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        total_components = len(self.components)
        healthy_components = sum(
            1 for c in self.components.values() if c.status == ComponentStatus.HEALTHY
        )
        failed_components = sum(
            1 for c in self.components.values() if c.status == ComponentStatus.FAILED
        )
        critical_failed = sum(
            1
            for c in self.components.values()
            if c.status == ComponentStatus.FAILED and c.is_critical
        )

        overall_status = "healthy"
        if critical_failed > 0:
            overall_status = "critical"
        elif failed_components > 0:
            overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "total_components": total_components,
            "healthy_components": healthy_components,
            "failed_components": failed_components,
            "critical_failed": critical_failed,
            "components": {
                name: {
                    "status": comp.status.value,
                    "is_critical": comp.is_critical,
                    "initialization_attempts": comp.initialization_attempts,
                    "last_error": comp.last_error.message if comp.last_error else None,
                }
                for name, comp in self.components.items()
            },
        }

    def get_initialization_summary(self) -> str:
        """Get a human-readable initialization summary."""
        health = self.get_system_health()

        summary_lines = [
            f"System Status: {health['overall_status'].upper()}",
            f"Components: {health['healthy_components']}/{health['total_components']} healthy",
        ]

        if health["failed_components"] > 0:
            summary_lines.append(f"Failed Components: {health['failed_components']}")

            # List failed components
            failed_comps = [
                name
                for name, comp in self.components.items()
                if comp.status == ComponentStatus.FAILED
            ]
            for comp_name in failed_comps:
                comp = self.components[comp_name]
                error_msg = comp.last_error.message if comp.last_error else "Unknown error"
                critical_marker = " (CRITICAL)" if comp.is_critical else ""
                summary_lines.append(f"  - {comp_name}{critical_marker}: {error_msg}")

        return "\n".join(summary_lines)
