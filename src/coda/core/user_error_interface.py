"""
User-Friendly Error Interface for Coda.

Provides user-friendly error messages, recovery suggestions, and
error reporting interface for end users.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .component_recovery import ComponentRecoveryManager, get_recovery_manager
from .error_management import (
    CodaError,
    ErrorCategory,
    ErrorManager,
    ErrorSeverity,
    get_error_manager,
)

logger = logging.getLogger(__name__)


class UserErrorLevel(Enum):
    """User-facing error levels."""

    INFO = "info"  # Informational messages
    WARNING = "warning"  # Warning messages that don't affect functionality
    ERROR = "error"  # Error messages that affect some functionality
    CRITICAL = "critical"  # Critical errors that affect core functionality


@dataclass
class UserErrorMessage:
    """User-friendly error message."""

    id: str
    level: UserErrorLevel
    title: str
    message: str
    suggestions: List[str]
    can_retry: bool = False
    can_ignore: bool = False
    technical_details: Optional[str] = None
    timestamp: datetime = None
    component: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class UserErrorInterface:
    """
    User-friendly error interface.

    Translates technical errors into user-friendly messages with
    actionable suggestions and recovery options.
    """

    def __init__(
        self,
        error_manager: Optional[ErrorManager] = None,
        recovery_manager: Optional[ComponentRecoveryManager] = None,
    ):
        self.error_manager = error_manager or get_error_manager()
        self.recovery_manager = recovery_manager or get_recovery_manager()

        # User message templates
        self.user_templates = self._initialize_user_templates()

        # Error callbacks for UI updates
        self.error_callbacks: List[callable] = []

        logger.info("UserErrorInterface initialized")

    def _initialize_user_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize user-friendly error message templates."""
        return {
            # Network errors
            "network_connection_failed": {
                "title": "Connection Problem",
                "message": "Unable to connect to the AI service. Please check your internet connection.",
                "suggestions": [
                    "Check your internet connection",
                    "Verify that Ollama is running (http://localhost:11434)",
                    "Try restarting the application",
                ],
                "can_retry": True,
                "level": UserErrorLevel.ERROR,
            },
            "network_timeout": {
                "title": "Request Timeout",
                "message": "The request took too long to complete.",
                "suggestions": [
                    "Try again with a shorter message",
                    "Check your internet connection",
                    "The AI service might be busy - please wait a moment",
                ],
                "can_retry": True,
                "level": UserErrorLevel.WARNING,
            },
            # Resource errors
            "memory_insufficient": {
                "title": "Low Memory",
                "message": "The system is running low on memory.",
                "suggestions": [
                    "Close other applications to free up memory",
                    "Restart the application",
                    "Consider using a smaller AI model",
                ],
                "can_retry": True,
                "level": UserErrorLevel.ERROR,
            },
            "gpu_memory_full": {
                "title": "GPU Memory Full",
                "message": "The graphics card memory is full. Voice processing may be temporarily unavailable.",
                "suggestions": [
                    "Text chat is still available",
                    "Close other GPU-intensive applications",
                    "Restart the application to free GPU memory",
                ],
                "can_retry": True,
                "level": UserErrorLevel.WARNING,
            },
            # Component errors
            "voice_unavailable": {
                "title": "Voice Processing Unavailable",
                "message": "Voice features are temporarily unavailable, but text chat is still working.",
                "suggestions": [
                    "Use text chat instead",
                    "Check if your microphone is connected",
                    "Try restarting the application",
                ],
                "can_retry": True,
                "can_ignore": True,
                "level": UserErrorLevel.WARNING,
            },
            "llm_unavailable": {
                "title": "AI Service Unavailable",
                "message": "The AI service is temporarily unavailable.",
                "suggestions": [
                    "Check if Ollama is running",
                    "Verify the AI model is installed",
                    "Try restarting Ollama service",
                ],
                "can_retry": True,
                "level": UserErrorLevel.CRITICAL,
            },
            "memory_system_error": {
                "title": "Memory System Error",
                "message": "There was an issue with the conversation memory system.",
                "suggestions": [
                    "Your current conversation may not be saved",
                    "Try starting a new conversation",
                    "Restart the application if the problem persists",
                ],
                "can_retry": True,
                "can_ignore": True,
                "level": UserErrorLevel.WARNING,
            },
            # Configuration errors
            "configuration_error": {
                "title": "Configuration Error",
                "message": "There's an issue with the application configuration.",
                "suggestions": [
                    "Check the configuration file",
                    "Reset to default settings",
                    "Contact support if the problem persists",
                ],
                "can_retry": False,
                "level": UserErrorLevel.CRITICAL,
            },
            # User input errors
            "input_too_large": {
                "title": "Input Too Large",
                "message": "Your message is too long to process.",
                "suggestions": [
                    "Try breaking your message into smaller parts",
                    "Remove unnecessary content",
                    "Use shorter sentences",
                ],
                "can_retry": True,
                "can_ignore": True,
                "level": UserErrorLevel.INFO,
            },
            "invalid_file_format": {
                "title": "Unsupported File Format",
                "message": "The file format you're trying to use is not supported.",
                "suggestions": [
                    "Use a supported file format (txt, pdf, doc)",
                    "Convert the file to a supported format",
                    "Try copying and pasting the content instead",
                ],
                "can_retry": True,
                "can_ignore": True,
                "level": UserErrorLevel.INFO,
            },
            # Generic fallback
            "unknown_error": {
                "title": "Unexpected Error",
                "message": "An unexpected error occurred.",
                "suggestions": [
                    "Try the operation again",
                    "Restart the application if the problem persists",
                    "Contact support if you continue to experience issues",
                ],
                "can_retry": True,
                "level": UserErrorLevel.ERROR,
            },
        }

    def create_user_error_from_coda_error(self, coda_error: CodaError) -> UserErrorMessage:
        """Convert a CodaError to a user-friendly error message."""
        # Try to find a specific template
        template = self._find_error_template(coda_error)

        if template:
            return UserErrorMessage(
                id=coda_error.error_id,
                level=template["level"],
                title=template["title"],
                message=template["message"],
                suggestions=template["suggestions"].copy(),
                can_retry=template.get("can_retry", False),
                can_ignore=template.get("can_ignore", False),
                technical_details=coda_error.technical_details,
                component=coda_error.component,
                timestamp=coda_error.timestamp,
            )
        else:
            # Fallback to generic message
            return self._create_generic_user_error(coda_error)

    def _find_error_template(self, coda_error: CodaError) -> Optional[Dict[str, Any]]:
        """Find the best matching error template."""
        error_message = coda_error.message.lower()
        component = coda_error.component.lower()
        category = coda_error.category

        # Component-specific templates
        if component == "llm" and any(
            keyword in error_message for keyword in ["connection", "network"]
        ):
            return self.user_templates["network_connection_failed"]
        elif component == "llm":
            return self.user_templates["llm_unavailable"]
        elif component == "voice":
            return self.user_templates["voice_unavailable"]
        elif component == "memory":
            return self.user_templates["memory_system_error"]

        # Category-based templates
        elif category == ErrorCategory.NETWORK:
            if "timeout" in error_message:
                return self.user_templates["network_timeout"]
            else:
                return self.user_templates["network_connection_failed"]
        elif category == ErrorCategory.RESOURCE:
            if any(keyword in error_message for keyword in ["gpu", "cuda", "vram"]):
                return self.user_templates["gpu_memory_full"]
            else:
                return self.user_templates["memory_insufficient"]
        elif category == ErrorCategory.CONFIGURATION:
            return self.user_templates["configuration_error"]
        elif category == ErrorCategory.USER_INPUT:
            if "too large" in error_message or "size" in error_message:
                return self.user_templates["input_too_large"]
            elif "format" in error_message:
                return self.user_templates["invalid_file_format"]

        return None

    def _create_generic_user_error(self, coda_error: CodaError) -> UserErrorMessage:
        """Create a generic user error message."""
        level_mapping = {
            ErrorSeverity.LOW: UserErrorLevel.INFO,
            ErrorSeverity.MEDIUM: UserErrorLevel.WARNING,
            ErrorSeverity.HIGH: UserErrorLevel.ERROR,
            ErrorSeverity.CRITICAL: UserErrorLevel.CRITICAL,
        }

        # Use the user message from the error if available, otherwise use template
        message = coda_error.user_message or self.user_templates["unknown_error"]["message"]

        return UserErrorMessage(
            id=coda_error.error_id,
            level=level_mapping.get(coda_error.severity, UserErrorLevel.ERROR),
            title=f"{coda_error.component.title()} Error",
            message=message,
            suggestions=self.user_templates["unknown_error"]["suggestions"].copy(),
            can_retry=coda_error.is_recoverable,
            can_ignore=coda_error.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM],
            technical_details=coda_error.technical_details,
            component=coda_error.component,
            timestamp=coda_error.timestamp,
        )

    def get_system_status_message(self) -> UserErrorMessage:
        """Get current system status as a user message."""
        health_summary = self.recovery_manager.get_system_health_summary()
        overall_health = health_summary["overall_health"]

        if overall_health == "healthy":
            return UserErrorMessage(
                id="system_status",
                level=UserErrorLevel.INFO,
                title="System Status",
                message="All systems are operating normally.",
                suggestions=[],
                can_ignore=True,
            )
        elif overall_health == "warning":
            degraded_count = health_summary["degraded_components"]
            return UserErrorMessage(
                id="system_status",
                level=UserErrorLevel.WARNING,
                title="System Warning",
                message=f"Some features may be running in limited mode ({degraded_count} components affected).",
                suggestions=[
                    "Core functionality is still available",
                    "Try restarting the application if needed",
                ],
                can_ignore=True,
            )
        elif overall_health == "degraded":
            failed_count = health_summary["failed_components"]
            return UserErrorMessage(
                id="system_status",
                level=UserErrorLevel.ERROR,
                title="System Issues",
                message=f"Some features are currently unavailable ({failed_count} components failed).",
                suggestions=[
                    "Some functionality may be limited",
                    "Try restarting the application",
                    "Check your system resources",
                ],
                can_retry=True,
            )
        else:  # critical
            return UserErrorMessage(
                id="system_status",
                level=UserErrorLevel.CRITICAL,
                title="System Critical",
                message="Critical system components have failed. The application may not function properly.",
                suggestions=[
                    "Restart the application immediately",
                    "Check system requirements",
                    "Contact support if the problem persists",
                ],
                can_retry=True,
            )

    def register_error_callback(self, callback: callable) -> None:
        """Register a callback to be notified of new errors."""
        self.error_callbacks.append(callback)

    def notify_error(self, user_error: UserErrorMessage) -> None:
        """Notify all registered callbacks of a new error."""
        for callback in self.error_callbacks:
            try:
                callback(user_error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    async def handle_user_error_action(self, error_id: str, action: str) -> bool:
        """Handle user action on an error (retry, ignore, etc.)."""
        if error_id not in self.error_manager.errors:
            return False

        coda_error = self.error_manager.errors[error_id]

        if action == "retry" and coda_error.is_recoverable:
            # Attempt recovery through the error manager
            return await self.error_manager._attempt_recovery(coda_error)
        elif action == "ignore":
            # Mark error as handled
            logger.info(f"User chose to ignore error {error_id}")
            return True
        elif action == "restart_component":
            # Force restart the component
            return await self.recovery_manager.force_restart_component(coda_error.component)

        return False


# Global user error interface instance
_user_error_interface: Optional[UserErrorInterface] = None


def get_user_error_interface() -> UserErrorInterface:
    """Get the global user error interface instance."""
    global _user_error_interface
    if _user_error_interface is None:
        _user_error_interface = UserErrorInterface()
    return _user_error_interface


async def create_user_error(
    exception: Exception,
    component: str,
    operation: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> UserErrorMessage:
    """
    Convenience function to create a user-friendly error message.

    Args:
        exception: The exception that occurred
        component: Component where error occurred
        operation: Operation that failed
        session_id: Optional session ID
        user_id: Optional user ID

    Returns:
        UserErrorMessage: User-friendly error message
    """
    from .error_management import handle_error

    # Handle the error through the error management system
    coda_error = await handle_error(exception, component, operation, session_id, user_id)

    # Convert to user-friendly message
    user_interface = get_user_error_interface()
    user_error = user_interface.create_user_error_from_coda_error(coda_error)

    # Notify callbacks
    user_interface.notify_error(user_error)

    return user_error
