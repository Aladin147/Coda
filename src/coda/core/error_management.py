"""
Centralized Error Management System for Coda.

Provides unified error handling, categorization, recovery, and user-friendly messaging
across all system components.
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"  # Minor issues, system continues normally
    MEDIUM = "medium"  # Moderate issues, some functionality affected
    HIGH = "high"  # Major issues, significant functionality lost
    CRITICAL = "critical"  # System-threatening issues, immediate attention required


class ErrorCategory(Enum):
    """Error categories for classification."""

    CONFIGURATION = "configuration"  # Config file issues, missing settings
    DEPENDENCY = "dependency"  # Missing dependencies, import failures
    NETWORK = "network"  # Network connectivity, API failures
    RESOURCE = "resource"  # Memory, disk, GPU resource issues
    PERMISSION = "permission"  # File permissions, access rights
    VALIDATION = "validation"  # Input validation, data format issues
    COMPONENT = "component"  # Component initialization/operation failures
    INTEGRATION = "integration"  # Inter-component communication failures
    USER_INPUT = "user_input"  # User-provided invalid input
    SYSTEM = "system"  # OS-level, hardware issues
    UNKNOWN = "unknown"  # Unclassified errors


class RecoveryAction(Enum):
    """Available recovery actions."""

    RETRY = "retry"  # Retry the failed operation
    RESTART_COMPONENT = "restart_component"  # Restart the failed component
    FALLBACK = "fallback"  # Use fallback/degraded mode
    IGNORE = "ignore"  # Continue without the failed component
    ESCALATE = "escalate"  # Escalate to user/admin
    SHUTDOWN = "shutdown"  # Graceful system shutdown


@dataclass
class ErrorContext:
    """Context information for an error."""

    component: str
    operation: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodaError:
    """Standardized error representation."""

    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    message: str
    technical_details: str
    user_message: str
    error_code: str
    context: ErrorContext
    original_exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    is_recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ErrorPattern:
    """Pattern for matching and handling specific error types."""

    name: str
    category: ErrorCategory
    severity: ErrorSeverity
    match_conditions: Dict[str, Any]
    recovery_actions: List[RecoveryAction]
    user_message_template: str
    max_retries: int = 3
    cooldown_seconds: int = 60


class ErrorManager:
    """
    Centralized error management system.

    Handles error classification, recovery, user messaging, and reporting
    across all Coda components.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.errors: Dict[str, CodaError] = {}
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_counts: Dict[str, int] = {}
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self.user_message_templates: Dict[str, str] = {}

        # Error tracking
        self.error_history: List[CodaError] = []
        self.max_history_size = 1000

        # Recovery tracking
        self.recovery_attempts: Dict[str, List[datetime]] = {}
        self.component_restart_counts: Dict[str, int] = {}

        # Initialize default patterns and templates
        self._initialize_default_patterns()
        self._initialize_user_message_templates()

        logger.info("ErrorManager initialized")

    def _initialize_default_patterns(self):
        """Initialize default error patterns."""
        patterns = [
            ErrorPattern(
                name="ollama_connection_failed",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                match_conditions={"component": "llm", "message_contains": "connection"},
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.ESCALATE],
                user_message_template="Unable to connect to the AI service. Please check your connection.",
                max_retries=3,
                cooldown_seconds=30,
            ),
            ErrorPattern(
                name="memory_allocation_failed",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                match_conditions={"message_contains": ["memory", "allocation", "CUDA"]},
                recovery_actions=[RecoveryAction.RESTART_COMPONENT, RecoveryAction.FALLBACK],
                user_message_template="System is running low on memory. Some features may be temporarily unavailable.",
                max_retries=2,
                cooldown_seconds=60,
            ),
            ErrorPattern(
                name="voice_model_load_failed",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.MEDIUM,
                match_conditions={"component": "voice", "message_contains": "model"},
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK],
                user_message_template="Voice processing is temporarily unavailable. Text chat is still working.",
                max_retries=2,
                cooldown_seconds=120,
            ),
            ErrorPattern(
                name="websocket_connection_lost",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                match_conditions={"component": "websocket", "message_contains": "connection"},
                recovery_actions=[RecoveryAction.RETRY, RecoveryAction.RESTART_COMPONENT],
                user_message_template="Connection lost. Attempting to reconnect...",
                max_retries=5,
                cooldown_seconds=10,
            ),
            ErrorPattern(
                name="configuration_invalid",
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.CRITICAL,
                match_conditions={"category": ErrorCategory.CONFIGURATION},
                recovery_actions=[RecoveryAction.ESCALATE],
                user_message_template="System configuration error. Please check your settings.",
                max_retries=0,
                cooldown_seconds=0,
            ),
        ]

        for pattern in patterns:
            self.error_patterns[pattern.name] = pattern

    def _initialize_user_message_templates(self):
        """Initialize user-friendly message templates."""
        self.user_message_templates = {
            # Network errors
            "network_connection_failed": "Unable to connect to the service. Please check your internet connection.",
            "network_timeout": "The request timed out. Please try again.",
            "network_service_unavailable": "The service is temporarily unavailable. Please try again later.",
            # Resource errors
            "memory_insufficient": "System is running low on memory. Please close other applications.",
            "disk_space_low": "Insufficient disk space. Please free up some space.",
            "gpu_memory_full": "GPU memory is full. Some features may be temporarily unavailable.",
            # Component errors
            "component_initialization_failed": "A system component failed to start. Some features may be unavailable.",
            "component_crashed": "A system component has stopped working. Attempting to restart...",
            # User input errors
            "invalid_input_format": "The input format is not supported. Please try a different format.",
            "input_too_large": "The input is too large. Please try with smaller content.",
            # Configuration errors
            "config_file_missing": "Configuration file is missing. Please check your installation.",
            "config_invalid": "Configuration is invalid. Please check your settings.",
            # Generic messages
            "unknown_error": "An unexpected error occurred. Please try again.",
            "temporary_unavailable": "This feature is temporarily unavailable. Please try again later.",
            "operation_failed": "The operation failed. Please try again.",
        }

    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
    ) -> CodaError:
        """
        Handle an error with automatic classification and recovery.

        Args:
            exception: The original exception
            context: Error context information
            severity: Override severity classification
            category: Override category classification

        Returns:
            CodaError: Processed error object
        """
        # Generate unique error ID
        error_id = f"{context.component}_{int(time.time() * 1000)}"

        # Classify error if not provided
        if category is None:
            category = self._classify_error(exception, context)
        if severity is None:
            severity = self._determine_severity(exception, context, category)

        # Create error object
        error = CodaError(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=context.component,
            operation=context.operation,
            message=str(exception),
            technical_details=self._extract_technical_details(exception),
            user_message=self._generate_user_message(exception, context, category),
            error_code=self._generate_error_code(category, context.component),
            context=context,
            original_exception=exception,
            stack_trace=traceback.format_exc(),
            recovery_actions=self._determine_recovery_actions(exception, context, category),
            is_recoverable=self._is_recoverable(exception, category),
        )

        # Store error
        self.errors[error_id] = error
        self.error_history.append(error)

        # Trim history if needed
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size :]

        # Update error counts
        error_key = f"{context.component}_{category.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Log error
        self._log_error(error)

        # Attempt recovery if appropriate
        if error.is_recoverable and error.recovery_actions:
            await self._attempt_recovery(error)

        return error

    def _classify_error(self, exception: Exception, context: ErrorContext) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_message = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        # Network-related errors
        if any(
            keyword in error_message
            for keyword in ["connection", "network", "timeout", "unreachable"]
        ):
            return ErrorCategory.NETWORK

        # Resource-related errors
        if any(
            keyword in error_message
            for keyword in ["memory", "disk", "space", "allocation", "cuda", "gpu"]
        ):
            return ErrorCategory.RESOURCE

        # Permission errors
        if any(
            keyword in error_message for keyword in ["permission", "access", "denied", "forbidden"]
        ):
            return ErrorCategory.PERMISSION

        # Configuration errors
        if any(
            keyword in error_message for keyword in ["config", "setting", "parameter", "invalid"]
        ):
            return ErrorCategory.CONFIGURATION

        # Import/dependency errors
        if "import" in exception_type or "modulenotfound" in exception_type:
            return ErrorCategory.DEPENDENCY

        # Validation errors
        if any(
            keyword in error_message for keyword in ["validation", "invalid", "format", "parse"]
        ):
            return ErrorCategory.VALIDATION

        # Component-specific classification
        if context.component in ["llm", "memory", "voice", "tools", "personality"]:
            return ErrorCategory.COMPONENT

        return ErrorCategory.UNKNOWN

    def _determine_severity(
        self, exception: Exception, context: ErrorContext, category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity based on exception and context."""
        # Critical categories
        if category in [ErrorCategory.CONFIGURATION, ErrorCategory.DEPENDENCY]:
            return ErrorSeverity.CRITICAL

        # High severity for core components
        if context.component in ["llm", "memory"] and category == ErrorCategory.COMPONENT:
            return ErrorSeverity.HIGH

        # High severity for resource issues
        if category == ErrorCategory.RESOURCE:
            return ErrorSeverity.HIGH

        # Medium severity for network issues
        if category == ErrorCategory.NETWORK:
            return ErrorSeverity.MEDIUM

        # Low severity for validation and user input
        if category in [ErrorCategory.VALIDATION, ErrorCategory.USER_INPUT]:
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM

    def _extract_technical_details(self, exception: Exception) -> str:
        """Extract technical details from exception."""
        details = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "exception_args": getattr(exception, "args", []),
        }

        # Add specific details for known exception types
        if hasattr(exception, "error_code"):
            details["error_code"] = exception.error_code
        if hasattr(exception, "context"):
            details["exception_context"] = exception.context

        return json.dumps(details, indent=2)

    def _generate_user_message(
        self, exception: Exception, context: ErrorContext, category: ErrorCategory
    ) -> str:
        """Generate user-friendly error message."""
        # Check for pattern matches
        for pattern in self.error_patterns.values():
            if self._matches_pattern(exception, context, pattern):
                return pattern.user_message_template

        # Fallback to category-based messages
        category_messages = {
            ErrorCategory.NETWORK: "Connection issue detected. Please check your internet connection.",
            ErrorCategory.RESOURCE: "System resources are running low. Some features may be temporarily unavailable.",
            ErrorCategory.PERMISSION: "Permission error. Please check file permissions or run as administrator.",
            ErrorCategory.CONFIGURATION: "Configuration error. Please check your settings.",
            ErrorCategory.DEPENDENCY: "Missing system component. Please check your installation.",
            ErrorCategory.VALIDATION: "Invalid input provided. Please check your input and try again.",
            ErrorCategory.COMPONENT: f"{context.component.title()} component error. Some features may be temporarily unavailable.",
            ErrorCategory.USER_INPUT: "Invalid input. Please check your input and try again.",
        }

        return category_messages.get(category, "An unexpected error occurred. Please try again.")

    def _generate_error_code(self, category: ErrorCategory, component: str) -> str:
        """Generate standardized error code."""
        category_codes = {
            ErrorCategory.NETWORK: "NET",
            ErrorCategory.RESOURCE: "RES",
            ErrorCategory.PERMISSION: "PERM",
            ErrorCategory.CONFIGURATION: "CONF",
            ErrorCategory.DEPENDENCY: "DEP",
            ErrorCategory.VALIDATION: "VAL",
            ErrorCategory.COMPONENT: "COMP",
            ErrorCategory.INTEGRATION: "INT",
            ErrorCategory.USER_INPUT: "USER",
            ErrorCategory.SYSTEM: "SYS",
            ErrorCategory.UNKNOWN: "UNK",
        }

        component_codes = {
            "llm": "LLM",
            "memory": "MEM",
            "voice": "VOI",
            "tools": "TOL",
            "personality": "PER",
            "websocket": "WS",
            "dashboard": "DASH",
            "assistant": "AST",
        }

        category_code = category_codes.get(category, "UNK")
        component_code = component_codes.get(component, "GEN")
        timestamp_code = str(int(time.time()))[-6:]  # Last 6 digits of timestamp

        return f"{category_code}_{component_code}_{timestamp_code}"

    def _determine_recovery_actions(
        self, exception: Exception, context: ErrorContext, category: ErrorCategory
    ) -> List[RecoveryAction]:
        """Determine appropriate recovery actions for the error."""
        # Check for pattern matches first
        for pattern in self.error_patterns.values():
            if self._matches_pattern(exception, context, pattern):
                return pattern.recovery_actions.copy()

        # Default recovery actions based on category
        category_actions = {
            ErrorCategory.NETWORK: [RecoveryAction.RETRY, RecoveryAction.ESCALATE],
            ErrorCategory.RESOURCE: [RecoveryAction.RESTART_COMPONENT, RecoveryAction.FALLBACK],
            ErrorCategory.PERMISSION: [RecoveryAction.ESCALATE],
            ErrorCategory.CONFIGURATION: [RecoveryAction.ESCALATE],
            ErrorCategory.DEPENDENCY: [RecoveryAction.ESCALATE],
            ErrorCategory.VALIDATION: [RecoveryAction.IGNORE],
            ErrorCategory.COMPONENT: [RecoveryAction.RETRY, RecoveryAction.RESTART_COMPONENT],
            ErrorCategory.USER_INPUT: [RecoveryAction.IGNORE],
        }

        return category_actions.get(category, [RecoveryAction.RETRY])

    def _is_recoverable(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if error is recoverable."""
        # Non-recoverable categories
        if category in [
            ErrorCategory.CONFIGURATION,
            ErrorCategory.DEPENDENCY,
            ErrorCategory.PERMISSION,
        ]:
            return False

        # Check for specific non-recoverable exceptions
        non_recoverable_types = [
            "SystemExit",
            "KeyboardInterrupt",
            "ImportError",
            "ModuleNotFoundError",
        ]

        if type(exception).__name__ in non_recoverable_types:
            return False

        return True

    def _matches_pattern(
        self, exception: Exception, context: ErrorContext, pattern: ErrorPattern
    ) -> bool:
        """Check if error matches a specific pattern."""
        conditions = pattern.match_conditions

        # Check component match
        if "component" in conditions and context.component != conditions["component"]:
            return False

        # Check category match
        if (
            "category" in conditions
            and self._classify_error(exception, context) != conditions["category"]
        ):
            return False

        # Check message contains
        if "message_contains" in conditions:
            message = str(exception).lower()
            contains_list = conditions["message_contains"]
            if isinstance(contains_list, str):
                contains_list = [contains_list]

            if not any(keyword.lower() in message for keyword in contains_list):
                return False

        # Check exception type
        if "exception_type" in conditions:
            if type(exception).__name__ != conditions["exception_type"]:
                return False

        return True

    async def _attempt_recovery(self, error: CodaError) -> bool:
        """Attempt to recover from the error."""
        recovery_successful = False

        for action in error.recovery_actions:
            try:
                if action == RecoveryAction.RETRY:
                    recovery_successful = await self._retry_operation(error)
                elif action == RecoveryAction.RESTART_COMPONENT:
                    recovery_successful = await self._restart_component(error)
                elif action == RecoveryAction.FALLBACK:
                    recovery_successful = await self._enable_fallback(error)
                elif action == RecoveryAction.IGNORE:
                    recovery_successful = True  # Just ignore the error
                elif action == RecoveryAction.ESCALATE:
                    await self._escalate_error(error)
                    recovery_successful = False  # Escalation doesn't recover
                elif action == RecoveryAction.SHUTDOWN:
                    await self._initiate_shutdown(error)
                    recovery_successful = False

                if recovery_successful:
                    logger.info(
                        f"Recovery successful for error {error.error_id} using action {action.value}"
                    )
                    break

            except Exception as e:
                logger.error(
                    f"Recovery action {action.value} failed for error {error.error_id}: {e}"
                )

        return recovery_successful

    async def _retry_operation(self, error: CodaError) -> bool:
        """Retry the failed operation."""
        if error.retry_count >= error.max_retries:
            logger.warning(f"Max retries exceeded for error {error.error_id}")
            return False

        # Check cooldown
        component_key = f"{error.component}_{error.operation}"
        if component_key in self.recovery_attempts:
            last_attempts = self.recovery_attempts[component_key]
            recent_attempts = [
                attempt
                for attempt in last_attempts
                if datetime.now() - attempt < timedelta(minutes=5)
            ]

            if len(recent_attempts) >= 3:  # Too many recent attempts
                logger.warning(f"Too many recent retry attempts for {component_key}")
                return False

        # Record retry attempt
        if component_key not in self.recovery_attempts:
            self.recovery_attempts[component_key] = []
        self.recovery_attempts[component_key].append(datetime.now())

        error.retry_count += 1
        logger.info(f"Retrying operation for error {error.error_id} (attempt {error.retry_count})")

        # Wait before retry
        await asyncio.sleep(min(2**error.retry_count, 30))  # Exponential backoff, max 30s

        return True  # Indicate retry was attempted

    async def _restart_component(self, error: CodaError) -> bool:
        """Restart the failed component."""
        component = error.component

        # Check restart limits
        if self.component_restart_counts.get(component, 0) >= 3:
            logger.error(f"Component {component} has been restarted too many times")
            return False

        logger.info(f"Attempting to restart component: {component}")

        # Use registered recovery handler if available
        if RecoveryAction.RESTART_COMPONENT in self.recovery_handlers:
            handler = self.recovery_handlers[RecoveryAction.RESTART_COMPONENT]
            try:
                success = await handler(component, error)
                if success:
                    self.component_restart_counts[component] = (
                        self.component_restart_counts.get(component, 0) + 1
                    )
                return success
            except Exception as e:
                logger.error(f"Component restart handler failed: {e}")
                return False

        logger.warning(f"No restart handler registered for component {component}")
        return False

    async def _enable_fallback(self, error: CodaError) -> bool:
        """Enable fallback mode for the component."""
        component = error.component
        logger.info(f"Enabling fallback mode for component: {component}")

        # Use registered recovery handler if available
        if RecoveryAction.FALLBACK in self.recovery_handlers:
            handler = self.recovery_handlers[RecoveryAction.FALLBACK]
            try:
                return await handler(component, error)
            except Exception as e:
                logger.error(f"Fallback handler failed: {e}")
                return False

        logger.warning(f"No fallback handler registered for component {component}")
        return False

    async def _escalate_error(self, error: CodaError) -> None:
        """Escalate error to user/admin."""
        logger.critical(f"Escalating error {error.error_id}: {error.user_message}")

        # Use registered escalation handler if available
        if RecoveryAction.ESCALATE in self.recovery_handlers:
            handler = self.recovery_handlers[RecoveryAction.ESCALATE]
            try:
                await handler(error.component, error)
            except Exception as e:
                logger.error(f"Escalation handler failed: {e}")

    async def _initiate_shutdown(self, error: CodaError) -> None:
        """Initiate graceful system shutdown."""
        logger.critical(f"Initiating shutdown due to critical error {error.error_id}")

        # Use registered shutdown handler if available
        if RecoveryAction.SHUTDOWN in self.recovery_handlers:
            handler = self.recovery_handlers[RecoveryAction.SHUTDOWN]
            try:
                await handler(error.component, error)
            except Exception as e:
                logger.error(f"Shutdown handler failed: {e}")

    def _log_error(self, error: CodaError) -> None:
        """Log error with appropriate level and context."""
        log_message = f"[{error.error_code}] {error.component}.{error.operation}: {error.message}"

        context_info = {
            "error_id": error.error_id,
            "category": error.category.value,
            "severity": error.severity.value,
            "recoverable": error.is_recoverable,
            "session_id": error.context.session_id,
            "user_id": error.context.user_id,
        }

        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=context_info)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra=context_info)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra=context_info)
        else:
            logger.info(log_message, extra=context_info)

    def register_recovery_handler(
        self, action: RecoveryAction, handler: Callable[[str, CodaError], bool]
    ) -> None:
        """Register a recovery handler for a specific action."""
        self.recovery_handlers[action] = handler
        logger.info(f"Registered recovery handler for action: {action.value}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)

        recent_errors = [e for e in self.error_history if e.timestamp > last_hour]
        daily_errors = [e for e in self.error_history if e.timestamp > last_day]

        stats = {
            "total_errors": len(self.error_history),
            "errors_last_hour": len(recent_errors),
            "errors_last_day": len(daily_errors),
            "error_counts_by_component": {},
            "error_counts_by_category": {},
            "error_counts_by_severity": {},
            "most_common_errors": {},
            "component_health": self.component_health.copy(),
            "recovery_success_rate": self._calculate_recovery_success_rate(),
        }

        # Count by component
        for error in self.error_history:
            component = error.component
            stats["error_counts_by_component"][component] = (
                stats["error_counts_by_component"].get(component, 0) + 1
            )

            category = error.category.value
            stats["error_counts_by_category"][category] = (
                stats["error_counts_by_category"].get(category, 0) + 1
            )

            severity = error.severity.value
            stats["error_counts_by_severity"][severity] = (
                stats["error_counts_by_severity"].get(severity, 0) + 1
            )

        return stats

    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        recoverable_errors = [e for e in self.error_history if e.is_recoverable]
        if not recoverable_errors:
            return 1.0

        # This is a simplified calculation - in practice, you'd track actual recovery outcomes
        successful_recoveries = len([e for e in recoverable_errors if e.retry_count > 0])
        return successful_recoveries / len(recoverable_errors) if recoverable_errors else 1.0

    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component."""
        if component not in self.component_health:
            self.component_health[component] = {
                "status": "unknown",
                "last_error": None,
                "error_count": 0,
                "last_restart": None,
                "restart_count": 0,
            }

        return self.component_health[component].copy()

    def update_component_health(
        self, component: str, status: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update health status for a component."""
        if component not in self.component_health:
            self.component_health[component] = {}

        self.component_health[component]["status"] = status
        self.component_health[component]["last_updated"] = datetime.now()

        if additional_info:
            self.component_health[component].update(additional_info)

    def clear_error_history(self, older_than_days: int = 7) -> int:
        """Clear old errors from history."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        original_count = len(self.error_history)

        self.error_history = [e for e in self.error_history if e.timestamp > cutoff_date]

        cleared_count = original_count - len(self.error_history)
        logger.info(f"Cleared {cleared_count} old errors from history")

        return cleared_count

    def export_error_report(self, filepath: str) -> None:
        """Export error report to file."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_error_statistics(),
            "recent_errors": [
                {
                    "error_id": e.error_id,
                    "timestamp": e.timestamp.isoformat(),
                    "component": e.component,
                    "operation": e.operation,
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "message": e.message,
                    "user_message": e.user_message,
                    "error_code": e.error_code,
                    "is_recoverable": e.is_recoverable,
                    "retry_count": e.retry_count,
                }
                for e in self.error_history[-100:]  # Last 100 errors
            ],
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Error report exported to {filepath}")


# Global error manager instance
_error_manager: Optional[ErrorManager] = None


def get_error_manager() -> ErrorManager:
    """Get the global error manager instance."""
    global _error_manager
    if _error_manager is None:
        _error_manager = ErrorManager()
    return _error_manager


async def handle_error(
    exception: Exception,
    component: str,
    operation: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
) -> CodaError:
    """
    Convenience function to handle errors.

    Args:
        exception: The exception that occurred
        component: Component where error occurred
        operation: Operation that failed
        session_id: Optional session ID
        user_id: Optional user ID
        additional_context: Additional context data

    Returns:
        CodaError: Processed error object
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        session_id=session_id,
        user_id=user_id,
        additional_data=additional_context or {},
    )

    error_manager = get_error_manager()
    return await error_manager.handle_error(exception, context)
