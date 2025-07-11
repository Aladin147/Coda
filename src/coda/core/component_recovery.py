"""
Component Recovery Manager for Coda.

Handles component restart, health monitoring, and graceful degradation
when components fail or become unhealthy.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .error_management import CodaError, ErrorCategory, ErrorManager, ErrorSeverity, RecoveryAction, get_error_manager

logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """Component states."""

    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RESTARTING = "restarting"
    DISABLED = "disabled"


class HealthCheckResult(Enum):
    """Health check results."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class ComponentInfo:
    """Information about a component."""

    name: str
    state: ComponentState = ComponentState.UNKNOWN
    last_health_check: Optional[datetime] = None
    health_check_result: Optional[HealthCheckResult] = None
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    is_critical: bool = True
    restart_limit: int = 3
    restart_cooldown_minutes: int = 5


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""

    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2


class ComponentRecoveryManager:
    """
    Manages component recovery, health monitoring, and graceful degradation.

    Provides automatic component restart, health monitoring, dependency management,
    and fallback modes when components fail.
    """

    def __init__(self, error_manager: Optional[ErrorManager] = None):
        self.error_manager = error_manager or get_error_manager()
        self.components: Dict[str, ComponentInfo] = {}
        self.health_check_handlers: Dict[str, Callable[[], bool]] = {}
        self.restart_handlers: Dict[str, Callable[[], bool]] = {}
        self.fallback_handlers: Dict[str, Callable[[], bool]] = {}
        self.shutdown_handlers: Dict[str, Callable[[], bool]] = {}

        # Health monitoring
        self.health_check_config = HealthCheckConfig()
        self.health_check_task: Optional[asyncio.Task] = None
        self.running = False

        # Recovery tracking
        self.recovery_in_progress: Set[str] = set()
        self.failed_components: Set[str] = set()
        self.degraded_mode_components: Set[str] = set()

        # Register recovery handlers with error manager
        self._register_recovery_handlers()

        logger.info("ComponentRecoveryManager initialized")

    def _register_recovery_handlers(self):
        """Register recovery handlers with the error manager."""
        self.error_manager.register_recovery_handler(
            RecoveryAction.RESTART_COMPONENT, self._handle_component_restart
        )
        self.error_manager.register_recovery_handler(
            RecoveryAction.FALLBACK, self._handle_fallback_mode
        )
        self.error_manager.register_recovery_handler(
            RecoveryAction.ESCALATE, self._handle_escalation
        )
        self.error_manager.register_recovery_handler(RecoveryAction.SHUTDOWN, self._handle_shutdown)

    def register_component(
        self,
        name: str,
        dependencies: Optional[List[str]] = None,
        is_critical: bool = True,
        restart_limit: int = 3,
        restart_cooldown_minutes: int = 5,
    ) -> None:
        """Register a component for monitoring and recovery."""
        component = ComponentInfo(
            name=name,
            dependencies=set(dependencies or []),
            is_critical=is_critical,
            restart_limit=restart_limit,
            restart_cooldown_minutes=restart_cooldown_minutes,
        )

        self.components[name] = component

        # Update dependent relationships
        for dep in component.dependencies:
            if dep in self.components:
                self.components[dep].dependents.add(name)

        logger.info(f"Registered component: {name} (critical: {is_critical})")

    def register_health_check_handler(self, component: str, handler: Callable[[], bool]) -> None:
        """Register a health check handler for a component."""
        self.health_check_handlers[component] = handler
        logger.info(f"Registered health check handler for component: {component}")

    def register_restart_handler(self, component: str, handler: Callable[[], bool]) -> None:
        """Register a restart handler for a component."""
        self.restart_handlers[component] = handler
        logger.info(f"Registered restart handler for component: {component}")

    def register_fallback_handler(self, component: str, handler: Callable[[], bool]) -> None:
        """Register a fallback handler for a component."""
        self.fallback_handlers[component] = handler
        logger.info(f"Registered fallback handler for component: {component}")

    def register_shutdown_handler(self, component: str, handler: Callable[[], bool]) -> None:
        """Register a shutdown handler for a component."""
        self.shutdown_handlers[component] = handler
        logger.info(f"Registered shutdown handler for component: {component}")

    async def start_monitoring(self) -> None:
        """Start health monitoring for all components."""
        if self.running:
            logger.warning("Health monitoring is already running")
            return

        self.running = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started component health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.running = False

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped component health monitoring")

    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_config.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered components."""
        for component_name, component in self.components.items():
            if component.state == ComponentState.DISABLED:
                continue

            try:
                await self._check_component_health(component_name)
            except Exception as e:
                logger.error(f"Health check failed for component {component_name}: {e}")
                await self._handle_health_check_failure(component_name, e)

    async def _check_component_health(self, component_name: str) -> None:
        """Check health of a specific component."""
        component = self.components[component_name]

        if component_name not in self.health_check_handlers:
            # No health check handler - assume healthy if not in failed state
            if component.state != ComponentState.FAILED:
                component.state = ComponentState.HEALTHY
                component.health_check_result = HealthCheckResult.HEALTHY
            return

        handler = self.health_check_handlers[component_name]

        try:
            # Run health check with timeout
            is_healthy = await asyncio.wait_for(
                asyncio.create_task(self._run_health_check(handler)),
                timeout=self.health_check_config.timeout_seconds,
            )

            component.last_health_check = datetime.now()

            if is_healthy:
                if component.state in [ComponentState.FAILED, ComponentState.DEGRADED]:
                    logger.info(f"Component {component_name} recovered")
                    component.state = ComponentState.HEALTHY
                    self.failed_components.discard(component_name)
                    self.degraded_mode_components.discard(component_name)

                component.health_check_result = HealthCheckResult.HEALTHY
            else:
                await self._handle_unhealthy_component(component_name)

        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for component {component_name}")
            await self._handle_health_check_failure(
                component_name, TimeoutError("Health check timeout")
            )
        except Exception as e:
            logger.error(f"Health check error for component {component_name}: {e}")
            await self._handle_health_check_failure(component_name, e)

    async def _run_health_check(self, handler: Callable[[], bool]) -> bool:
        """Run a health check handler."""
        if asyncio.iscoroutinefunction(handler):
            return await handler()
        else:
            return handler()

    async def _handle_unhealthy_component(self, component_name: str) -> None:
        """Handle an unhealthy component."""
        component = self.components[component_name]
        component.health_check_result = HealthCheckResult.CRITICAL
        component.error_count += 1
        component.last_error = datetime.now()

        logger.warning(
            f"Component {component_name} is unhealthy (error count: {component.error_count})"
        )

        # Check if we should attempt recovery
        if component.error_count >= self.health_check_config.failure_threshold:
            if component.is_critical:
                await self._attempt_component_recovery(component_name)
            else:
                # Non-critical component - just mark as degraded
                component.state = ComponentState.DEGRADED
                self.degraded_mode_components.add(component_name)
                logger.info(f"Non-critical component {component_name} marked as degraded")

    async def _handle_health_check_failure(self, component_name: str, exception: Exception) -> None:
        """Handle health check failure."""
        component = self.components[component_name]
        component.error_count += 1
        component.last_error = datetime.now()
        component.health_check_result = HealthCheckResult.FAILED

        # Create error context and handle through error manager
        from .error_management import ErrorContext, handle_error

        context = ErrorContext(
            component=component_name,
            operation="health_check",
            additional_data={"error_count": component.error_count},
        )

        await handle_error(exception, component_name, "health_check")

    async def _attempt_component_recovery(self, component_name: str) -> bool:
        """Attempt to recover a failed component."""
        if component_name in self.recovery_in_progress:
            logger.info(f"Recovery already in progress for component {component_name}")
            return False

        component = self.components[component_name]

        # Check restart limits and cooldown
        if component.restart_count >= component.restart_limit:
            logger.error(f"Component {component_name} has exceeded restart limit")
            component.state = ComponentState.FAILED
            self.failed_components.add(component_name)
            return False

        if component.last_restart:
            cooldown_end = component.last_restart + timedelta(
                minutes=component.restart_cooldown_minutes
            )
            if datetime.now() < cooldown_end:
                logger.info(f"Component {component_name} is in restart cooldown")
                return False

        self.recovery_in_progress.add(component_name)

        try:
            # Try restart first
            if await self._restart_component(component_name):
                return True

            # If restart fails, try fallback mode
            if await self._enable_fallback_mode(component_name):
                return True

            # If all recovery attempts fail
            component.state = ComponentState.FAILED
            self.failed_components.add(component_name)
            logger.error(f"All recovery attempts failed for component {component_name}")
            return False

        finally:
            self.recovery_in_progress.discard(component_name)

    async def _restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        if component_name not in self.restart_handlers:
            logger.warning(f"No restart handler registered for component {component_name}")
            return False

        component = self.components[component_name]
        component.state = ComponentState.RESTARTING

        logger.info(f"Restarting component: {component_name}")

        try:
            handler = self.restart_handlers[component_name]
            success = await self._run_restart_handler(handler)

            if success:
                component.restart_count += 1
                component.last_restart = datetime.now()
                component.state = ComponentState.HEALTHY
                component.error_count = 0  # Reset error count on successful restart
                logger.info(f"Component {component_name} restarted successfully")
                return True
            else:
                component.state = ComponentState.FAILED
                logger.error(f"Failed to restart component {component_name}")
                return False

        except Exception as e:
            component.state = ComponentState.FAILED
            logger.error(f"Exception during restart of component {component_name}: {e}")
            return False

    async def _run_restart_handler(self, handler: Callable[[], bool]) -> bool:
        """Run a restart handler."""
        if asyncio.iscoroutinefunction(handler):
            return await handler()
        else:
            return handler()

    async def _enable_fallback_mode(self, component_name: str) -> bool:
        """Enable fallback mode for a component."""
        if component_name not in self.fallback_handlers:
            logger.warning(f"No fallback handler registered for component {component_name}")
            return False

        component = self.components[component_name]

        logger.info(f"Enabling fallback mode for component: {component_name}")

        try:
            handler = self.fallback_handlers[component_name]
            success = await self._run_fallback_handler(handler)

            if success:
                component.state = ComponentState.DEGRADED
                self.degraded_mode_components.add(component_name)
                logger.info(f"Fallback mode enabled for component {component_name}")
                return True
            else:
                logger.error(f"Failed to enable fallback mode for component {component_name}")
                return False

        except Exception as e:
            logger.error(
                f"Exception during fallback mode setup for component {component_name}: {e}"
            )
            return False

    async def _run_fallback_handler(self, handler: Callable[[], bool]) -> bool:
        """Run a fallback handler."""
        if asyncio.iscoroutinefunction(handler):
            return await handler()
        else:
            return handler()

    # Recovery handler implementations for error manager
    async def _handle_component_restart(self, component_name: str, error: CodaError) -> bool:
        """Handle component restart request from error manager."""
        return await self._restart_component(component_name)

    async def _handle_fallback_mode(self, component_name: str, error: CodaError) -> bool:
        """Handle fallback mode request from error manager."""
        return await self._enable_fallback_mode(component_name)

    async def _handle_escalation(self, component_name: str, error: CodaError) -> bool:
        """Handle error escalation."""
        logger.critical(f"Escalating error for component {component_name}: {error.user_message}")

        # Mark component as requiring attention
        if component_name in self.components:
            self.components[component_name].state = ComponentState.FAILED
            self.failed_components.add(component_name)

        # Implement comprehensive error escalation
        await self._execute_escalation_procedures(component_name, error)
        return True

    async def _execute_escalation_procedures(self, component_name: str, error: CodaError) -> None:
        """Execute comprehensive error escalation procedures."""
        try:
            escalation_data = {
                "timestamp": datetime.now().isoformat(),
                "component": component_name,
                "error_category": error.category.value if hasattr(error, 'category') else "unknown",
                "error_code": getattr(error, 'error_code', 'unknown'),
                "message": str(error),
                "severity": error.severity.value,
                "context": getattr(error, 'context', {}),
                "system_state": await self._get_system_state_summary()
            }

            # 1. Log escalation to structured log
            await self._log_escalation(escalation_data)

            # 2. Store escalation record for analysis
            await self._store_escalation_record(escalation_data)

            # 3. Trigger immediate notifications if critical
            if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
                await self._trigger_critical_notifications(escalation_data)

            # 4. Update system health metrics
            await self._update_health_metrics(component_name, error)

            # 5. Initiate automated remediation if available
            await self._attempt_automated_remediation(component_name, error)

            logger.info(f"Error escalation procedures completed for {component_name}")

        except Exception as e:
            logger.error(f"Error escalation procedures failed: {e}")

    async def _log_escalation(self, escalation_data: Dict[str, Any]) -> None:
        """Log escalation to structured logging system."""
        try:
            # Create structured log entry
            log_entry = {
                "event_type": "error_escalation",
                "severity": "ERROR",
                **escalation_data
            }

            # Log to file with structured format
            escalation_logger = logging.getLogger("coda.escalation")
            escalation_logger.error(json.dumps(log_entry, indent=2))

        except Exception as e:
            logger.error(f"Failed to log escalation: {e}")

    async def _store_escalation_record(self, escalation_data: Dict[str, Any]) -> None:
        """Store escalation record for analysis and trending."""
        try:
            # Store in data directory for analysis
            escalation_dir = Path("data/escalations")
            escalation_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped file
            timestamp = escalation_data["timestamp"].replace(":", "-")
            filename = f"escalation_{timestamp}_{escalation_data['component']}.json"
            filepath = escalation_dir / filename

            with open(filepath, 'w') as f:
                json.dump(escalation_data, f, indent=2)

            logger.debug(f"Escalation record stored: {filepath}")

        except Exception as e:
            logger.error(f"Failed to store escalation record: {e}")

    async def _trigger_critical_notifications(self, escalation_data: Dict[str, Any]) -> None:
        """Trigger immediate notifications for critical errors."""
        try:
            # For now, implement console notification and system logging
            # In production, this would integrate with notification services

            critical_message = (
                f"🚨 CRITICAL ERROR ESCALATION 🚨\n"
                f"Component: {escalation_data['component']}\n"
                f"Error: {escalation_data['message']}\n"
                f"Severity: {escalation_data['severity']}\n"
                f"Time: {escalation_data['timestamp']}\n"
                f"System requires immediate attention!"
            )

            # Log critical notification
            logger.critical(critical_message)

            # Print to console for immediate visibility
            print(f"\n{'='*60}")
            print(critical_message)
            print(f"{'='*60}\n")

            # Store critical alert
            alert_file = Path("data/critical_alerts.log")
            alert_file.parent.mkdir(parents=True, exist_ok=True)

            with open(alert_file, 'a') as f:
                f.write(f"{escalation_data['timestamp']}: {critical_message}\n\n")

        except Exception as e:
            logger.error(f"Failed to trigger critical notifications: {e}")

    async def _update_health_metrics(self, component_name: str, error: CodaError) -> None:
        """Update system health metrics based on error."""
        try:
            # Update component health tracking
            if hasattr(self, 'health_metrics'):
                if component_name not in self.health_metrics:
                    self.health_metrics[component_name] = {
                        "error_count": 0,
                        "last_error": None,
                        "health_score": 1.0
                    }

                metrics = self.health_metrics[component_name]
                metrics["error_count"] += 1
                metrics["last_error"] = datetime.now().isoformat()

                # Decrease health score based on error severity
                severity_impact = {
                    ErrorSeverity.LOW: 0.01,
                    ErrorSeverity.MEDIUM: 0.05,
                    ErrorSeverity.HIGH: 0.15,
                    ErrorSeverity.CRITICAL: 0.3,
                    ErrorSeverity.FATAL: 0.5
                }

                impact = severity_impact.get(error.severity, 0.1)
                metrics["health_score"] = max(0.0, metrics["health_score"] - impact)

        except Exception as e:
            logger.error(f"Failed to update health metrics: {e}")

    async def _attempt_automated_remediation(self, component_name: str, error: CodaError) -> None:
        """Attempt automated remediation based on error type."""
        try:
            # Define remediation strategies
            remediation_strategies = {
                ErrorCategory.NETWORK: self._remediate_network_error,
                ErrorCategory.RESOURCE: self._remediate_resource_error,
                ErrorCategory.CONFIGURATION: self._remediate_configuration_error,
                ErrorCategory.COMPONENT: self._remediate_component_error,
            }

            error_category = getattr(error, 'category', ErrorCategory.UNKNOWN)
            if error_category in remediation_strategies:
                strategy = remediation_strategies[error_category]
                await strategy(component_name, error)
                logger.info(f"Automated remediation attempted for {component_name}")
            else:
                logger.debug(f"No automated remediation available for error category: {error_category}")

        except Exception as e:
            logger.error(f"Automated remediation failed: {e}")

    async def _remediate_network_error(self, component_name: str, error: CodaError) -> None:
        """Remediate network errors."""
        # Attempt to restart component connections
        if component_name in self.components:
            component = self.components[component_name]
            if hasattr(component, 'reconnect'):
                await component.reconnect()

    async def _remediate_component_error(self, component_name: str, error: CodaError) -> None:
        """Remediate component errors."""
        # Attempt to restart the component
        if component_name in self.components:
            component = self.components[component_name]
            if hasattr(component, 'restart'):
                await component.restart()

    async def _remediate_resource_error(self, component_name: str, error: CodaError) -> None:
        """Remediate resource errors."""
        # Clear caches and free resources
        if component_name in self.components:
            component = self.components[component_name]
            if hasattr(component, 'clear_cache'):
                await component.clear_cache()

    async def _remediate_configuration_error(self, component_name: str, error: CodaError) -> None:
        """Remediate configuration errors."""
        # Reload configuration
        if component_name in self.components:
            component = self.components[component_name]
            if hasattr(component, 'reload_config'):
                await component.reload_config()

    async def _get_system_state_summary(self) -> Dict[str, Any]:
        """Get current system state summary."""
        try:
            return {
                "total_components": len(self.components),
                "healthy_components": len([c for c in self.components.values()
                                         if c.state == ComponentState.RUNNING]),
                "failed_components": len(self.failed_components),
                "recovery_attempts": len(self.recovery_attempts),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0
            }
        except Exception:
            return {"error": "Failed to get system state"}

    async def _handle_shutdown(self, component_name: str, error: CodaError) -> bool:
        """Handle shutdown request."""
        logger.critical(f"Shutdown requested due to critical error in component {component_name}")

        if component_name in self.shutdown_handlers:
            handler = self.shutdown_handlers[component_name]
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler()
                else:
                    return handler()
            except Exception as e:
                logger.error(f"Shutdown handler failed for component {component_name}: {e}")

        return False

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        healthy_count = sum(
            1 for c in self.components.values() if c.state == ComponentState.HEALTHY
        )
        degraded_count = len(self.degraded_mode_components)
        failed_count = len(self.failed_components)
        total_count = len(self.components)

        critical_failed = sum(
            1 for name in self.failed_components if self.components[name].is_critical
        )

        # Determine overall system health
        if critical_failed > 0:
            overall_health = "critical"
        elif failed_count > 0 or degraded_count > total_count * 0.3:
            overall_health = "degraded"
        elif degraded_count > 0:
            overall_health = "warning"
        else:
            overall_health = "healthy"

        return {
            "overall_health": overall_health,
            "total_components": total_count,
            "healthy_components": healthy_count,
            "degraded_components": degraded_count,
            "failed_components": failed_count,
            "critical_failures": critical_failed,
            "components": {
                name: {
                    "state": component.state.value,
                    "health_result": (
                        component.health_check_result.value
                        if component.health_check_result
                        else None
                    ),
                    "error_count": component.error_count,
                    "restart_count": component.restart_count,
                    "is_critical": component.is_critical,
                    "last_health_check": (
                        component.last_health_check.isoformat()
                        if component.last_health_check
                        else None
                    ),
                    "last_restart": (
                        component.last_restart.isoformat() if component.last_restart else None
                    ),
                }
                for name, component in self.components.items()
            },
            "degraded_mode_active": len(self.degraded_mode_components) > 0,
            "recovery_in_progress": list(self.recovery_in_progress),
        }

    def get_component_status(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific component."""
        if component_name not in self.components:
            return None

        component = self.components[component_name]

        return {
            "name": component_name,
            "state": component.state.value,
            "health_result": (
                component.health_check_result.value if component.health_check_result else None
            ),
            "error_count": component.error_count,
            "restart_count": component.restart_count,
            "restart_limit": component.restart_limit,
            "is_critical": component.is_critical,
            "dependencies": list(component.dependencies),
            "dependents": list(component.dependents),
            "last_health_check": (
                component.last_health_check.isoformat() if component.last_health_check else None
            ),
            "last_restart": component.last_restart.isoformat() if component.last_restart else None,
            "last_error": component.last_error.isoformat() if component.last_error else None,
            "in_degraded_mode": component_name in self.degraded_mode_components,
            "recovery_in_progress": component_name in self.recovery_in_progress,
        }

    async def force_restart_component(self, component_name: str) -> bool:
        """Force restart a component (bypasses cooldown and limits)."""
        if component_name not in self.components:
            logger.error(f"Component {component_name} not registered")
            return False

        logger.info(f"Force restarting component: {component_name}")
        return await self._restart_component(component_name)

    def disable_component(self, component_name: str) -> bool:
        """Disable a component (stops monitoring and recovery)."""
        if component_name not in self.components:
            return False

        self.components[component_name].state = ComponentState.DISABLED
        self.failed_components.discard(component_name)
        self.degraded_mode_components.discard(component_name)
        self.recovery_in_progress.discard(component_name)

        logger.info(f"Component {component_name} disabled")
        return True

    def enable_component(self, component_name: str) -> bool:
        """Re-enable a disabled component."""
        if component_name not in self.components:
            return False

        component = self.components[component_name]
        if component.state == ComponentState.DISABLED:
            component.state = ComponentState.UNKNOWN
            component.error_count = 0
            logger.info(f"Component {component_name} re-enabled")
            return True

        return False


# Global recovery manager instance
_recovery_manager: Optional[ComponentRecoveryManager] = None


def get_recovery_manager() -> ComponentRecoveryManager:
    """Get the global recovery manager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ComponentRecoveryManager()
    return _recovery_manager
