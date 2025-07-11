"""
WebSocket integration for the tool system.

This module provides WebSocket event broadcasting for real-time
tool monitoring and analytics.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .interfaces import WebSocketToolInterface
from .manager import ToolManager
from .models import (
    ToolCall,
    ToolConfig,
    ToolDefinition,
    ToolEvent,
    ToolExecution,
    ToolResult,
)

logger = logging.getLogger("coda.tools.websocket")


class WebSocketToolManager(ToolManager, WebSocketToolInterface):
    """
    Tool manager with WebSocket integration for real-time events.

    Features:
    - Real-time tool execution broadcasting
    - Tool registration/unregistration events
    - Performance metrics streaming
    - Error monitoring and alerts
    """

    def __init__(self, config: Optional[ToolConfig] = None):
        """
        Initialize the WebSocket tool manager.

        Args:
            config: Tool system configuration
        """
        super().__init__(config)

        self._websocket_integration: Optional[Any] = None
        self._event_queue: List[ToolEvent] = []

        logger.info("WebSocketToolManager initialized")

    async def set_websocket_integration(self, integration: Any) -> None:
        """
        Set WebSocket integration for event broadcasting.

        Args:
            integration: WebSocket integration instance
        """
        self._websocket_integration = integration

        # Broadcast initial state
        await self._broadcast_initial_state()

        logger.info("WebSocket integration set for tool manager")

    async def broadcast_tool_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast a tool event via WebSocket.

        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.config.websocket_events_enabled or not self._websocket_integration:
            return

        event = ToolEvent(event_type=event_type, data=data, timestamp=datetime.now())

        try:
            await self._websocket_integration.broadcast_event("tool_event", event.model_dump())
            logger.debug(f"Broadcasted tool event: {event_type}")
        except Exception as e:
            logger.warning(f"Failed to broadcast tool event: {e}")
            # Queue event for retry
            self._event_queue.append(event)

    async def broadcast_tool_execution(self, execution: ToolExecution, result: ToolResult) -> None:
        """
        Broadcast tool execution event.

        Args:
            execution: Tool execution context
            result: Tool execution result
        """
        await self.broadcast_tool_event(
            "tool_execution_completed",
            {
                "call_id": execution.call.call_id,
                "tool_name": execution.call.tool_name,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "attempts": execution.attempts,
                "error": result.error,
                "timestamp": result.timestamp.isoformat(),
            },
        )

    async def broadcast_tool_registration(self, tool_definition: ToolDefinition) -> None:
        """
        Broadcast tool registration event.

        Args:
            tool_definition: Registered tool definition
        """
        await self.broadcast_tool_event(
            "tool_registered",
            {
                "tool_name": tool_definition.name,
                "category": tool_definition.category.value,
                "description": tool_definition.description,
                "parameter_count": len(tool_definition.parameters),
                "is_dangerous": tool_definition.is_dangerous,
                "requires_auth": tool_definition.requires_auth,
                "tags": tool_definition.tags,
            },
        )

    async def process_function_call(self, function_call: Dict[str, Any]) -> ToolResult:
        """
        Process function call with WebSocket events.

        Args:
            function_call: Function call data

        Returns:
            Tool execution result
        """
        # Broadcast function call received
        await self.broadcast_tool_event(
            "function_call_received",
            {
                "tool_name": function_call.get("name", "unknown"),
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Process the function call
        result = await super().process_function_call(function_call)

        # Broadcast execution result
        await self.broadcast_tool_event(
            "function_call_completed",
            {
                "call_id": result.call_id,
                "tool_name": result.tool_name,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "error": result.error,
                "timestamp": result.timestamp.isoformat(),
            },
        )

        return result

    async def register_tool(self, tool: Any) -> bool:
        """
        Register tool with WebSocket event.

        Args:
            tool: Tool to register

        Returns:
            True if registration successful
        """
        success = await super().register_tool(tool)

        if success:
            await self.broadcast_tool_registration(tool.get_definition())

        return success

    async def broadcast_tool_analytics(self) -> None:
        """Broadcast comprehensive tool analytics."""
        analytics = self.get_tool_analytics()

        await self.broadcast_tool_event(
            "tool_analytics_update",
            {"analytics": analytics, "timestamp": datetime.now().isoformat()},
        )

    async def broadcast_performance_metrics(self) -> None:
        """Broadcast performance metrics."""
        metrics = self.executor.get_performance_metrics()

        await self.broadcast_tool_event(
            "tool_performance_metrics",
            {"metrics": metrics, "timestamp": datetime.now().isoformat()},
        )

    async def broadcast_registry_stats(self) -> None:
        """Broadcast registry statistics."""
        stats = self.registry.get_registry_stats()

        await self.broadcast_tool_event(
            "tool_registry_stats", {"stats": stats, "timestamp": datetime.now().isoformat()}
        )

    async def broadcast_execution_stats(self) -> None:
        """Broadcast execution statistics."""
        stats = self.executor.get_execution_stats()

        await self.broadcast_tool_event(
            "tool_execution_stats",
            {"stats": stats.model_dump(), "timestamp": datetime.now().isoformat()},
        )

    async def broadcast_tool_error(
        self, tool_name: str, error: str, call_id: Optional[str] = None
    ) -> None:
        """
        Broadcast tool error event.

        Args:
            tool_name: Name of the tool that errored
            error: Error message
            call_id: Optional call ID
        """
        await self.broadcast_tool_event(
            "tool_error",
            {
                "tool_name": tool_name,
                "error": error,
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def broadcast_tool_unregistered(self, tool_name: str) -> None:
        """
        Broadcast tool unregistration event.

        Args:
            tool_name: Name of unregistered tool
        """
        await self.broadcast_tool_event(
            "tool_unregistered", {"tool_name": tool_name, "timestamp": datetime.now().isoformat()}
        )

    async def broadcast_tool_availability_change(self, tool_name: str, available: bool) -> None:
        """
        Broadcast tool availability change.

        Args:
            tool_name: Name of the tool
            available: New availability status
        """
        await self.broadcast_tool_event(
            "tool_availability_changed",
            {
                "tool_name": tool_name,
                "available": available,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def _broadcast_initial_state(self) -> None:
        """Broadcast initial tool system state."""
        # Broadcast registry state
        await self.broadcast_registry_stats()

        # Broadcast available tools
        available_tools = self.get_available_tools()
        await self.broadcast_tool_event(
            "tools_available",
            {
                "tools": [
                    {
                        "name": tool.name,
                        "category": tool.category.value,
                        "description": tool.description,
                        "parameter_count": len(tool.parameters),
                        "is_dangerous": tool.is_dangerous,
                        "requires_auth": tool.requires_auth,
                    }
                    for tool in available_tools
                ],
                "total_count": len(available_tools),
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Broadcast execution stats
        await self.broadcast_execution_stats()

        logger.info("Broadcasted initial tool system state")

    async def _retry_queued_events(self) -> None:
        """Retry broadcasting queued events."""
        if not self._event_queue or not self._websocket_integration:
            return

        retry_queue = self._event_queue.copy()
        self._event_queue.clear()

        for event in retry_queue:
            try:
                await self._websocket_integration.broadcast_event("tool_event", event.model_dump())
                logger.debug(f"Retried broadcasting event: {event.event_type}")
            except Exception as e:
                logger.warning(f"Failed to retry event broadcast: {e}")
                # Re-queue if still failing
                self._event_queue.append(event)

    def get_websocket_stats(self) -> Dict[str, Any]:
        """Get WebSocket integration statistics."""
        return {
            "websocket_enabled": self.config.websocket_events_enabled,
            "integration_connected": self._websocket_integration is not None,
            "queued_events": len(self._event_queue),
            "events_broadcasted": True,  # Could track this with a counter
        }

    async def trigger_tool_snapshot(self) -> Dict[str, Any]:
        """Trigger a comprehensive tool system snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "registry": self.registry.get_registry_stats(),
            "execution": self.executor.get_execution_stats().model_dump(),
            "performance": self.executor.get_performance_metrics(),
            "websocket": self.get_websocket_stats(),
            "available_tools": [
                {"name": tool.name, "category": tool.category.value, "available": True}
                for tool in self.get_available_tools()
            ],
        }

        # Broadcast the snapshot
        await self.broadcast_tool_event("tool_system_snapshot", snapshot)

        return snapshot

    async def cleanup(self) -> None:
        """Cleanup WebSocket tool manager."""
        # Broadcast cleanup event
        if self._websocket_integration:
            await self.broadcast_tool_event(
                "tool_system_shutdown", {"timestamp": datetime.now().isoformat()}
            )

        # Clear event queue
        self._event_queue.clear()

        # Call parent cleanup
        await super().cleanup()

        logger.info("WebSocket tool manager cleanup completed")
