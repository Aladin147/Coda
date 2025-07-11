"""
WebSocket Integration for Component Integration Layer.

This module connects the ComponentIntegrationLayer events to the WebSocket system
for real-time monitoring of component status, health, and system operations.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..interfaces.websocket.events import EventType as WSEventType
from ..interfaces.websocket.events import (
    create_event,
)
from ..interfaces.websocket.server import CodaWebSocketServer
from .integration import (
    ComponentIntegrationLayer,
    IntegrationEvent,
)

logger = logging.getLogger("coda.core.websocket_integration")


class ComponentWebSocketIntegration:
    """
    Integration between ComponentIntegrationLayer and WebSocket server.

    This class bridges the component integration layer events with the WebSocket
    system, providing real-time visibility into component status, health, and
    system operations for connected clients.
    """

    def __init__(
        self,
        integration_layer: ComponentIntegrationLayer,
        websocket_server: Optional[CodaWebSocketServer] = None,
    ):
        """
        Initialize WebSocket integration.

        Args:
            integration_layer: The component integration layer to monitor
            websocket_server: Optional WebSocket server instance
        """
        self.integration_layer = integration_layer
        self.websocket_server = websocket_server
        self.is_active = False
        self.event_handlers_registered = False

        # Voice manager for audio streaming
        self.voice_manager = None

        # Event tracking
        self.component_events: List[Dict[str, Any]] = []
        self.max_event_history = 100
        self.sequence_number = 0

        logger.info("ComponentWebSocketIntegration initialized")

    def set_websocket_server(self, websocket_server: CodaWebSocketServer) -> None:
        """Set the WebSocket server instance."""
        self.websocket_server = websocket_server
        # Set this integration as the handler for chat messages
        websocket_server.set_integration_handler(self)

        # Connect voice manager if available
        if self.voice_manager:
            websocket_server.set_voice_manager(self.voice_manager)

        logger.info("WebSocket server set for component integration")

    def set_voice_manager(self, voice_manager) -> None:
        """Set the voice manager for audio streaming."""
        self.voice_manager = voice_manager

        # Connect to WebSocket server if available
        if self.websocket_server:
            self.websocket_server.set_voice_manager(voice_manager)

        logger.info("Voice manager set for WebSocket integration")

    async def start(self) -> None:
        """Start WebSocket integration."""
        if self.is_active:
            logger.warning("WebSocket integration already active")
            return

        if not self.websocket_server:
            logger.warning("No WebSocket server available, integration disabled")
            return

        # Register event handlers with integration layer
        await self._register_event_handlers()

        self.is_active = True
        logger.info("ComponentWebSocketIntegration started")

        # Broadcast initial system status
        await self._broadcast_system_status()

    async def stop(self) -> None:
        """Stop WebSocket integration."""
        if not self.is_active:
            return

        self.is_active = False
        logger.info("ComponentWebSocketIntegration stopped")

    async def _register_event_handlers(self) -> None:
        """Register event handlers with the integration layer."""
        if self.event_handlers_registered:
            return

        # Register handlers for different event types
        self.integration_layer.add_event_handler(
            "component_initialized", self._handle_component_initialized
        )
        self.integration_layer.add_event_handler(
            "component_shutdown", self._handle_component_shutdown
        )
        self.integration_layer.add_event_handler("component_failed", self._handle_component_failed)
        self.integration_layer.add_event_handler(
            "component_health_changed", self._handle_component_health_changed
        )

        self.event_handlers_registered = True
        logger.info("Event handlers registered with integration layer")

    async def _handle_component_initialized(self, event: IntegrationEvent) -> None:
        """Handle component initialization events."""
        try:
            component_metadata = self.integration_layer.get_component_metadata(event.component_type)

            event_data = {
                "component_type": event.component_type.value,
                "state": "initialized",
                "initialization_order": (
                    component_metadata.initialization_order if component_metadata else 0
                ),
                "dependencies": (
                    [dep.value for dep in component_metadata.dependencies]
                    if component_metadata
                    else []
                ),
            }

            await self._broadcast_component_event("component_status", event_data)
            await self._broadcast_system_status()

        except Exception as e:
            logger.error(f"Error handling component initialized event: {e}")

    async def _handle_component_shutdown(self, event: IntegrationEvent) -> None:
        """Handle component shutdown events."""
        try:
            event_data = {"component_type": event.component_type.value, "state": "shutdown"}

            await self._broadcast_component_event("component_status", event_data)
            await self._broadcast_system_status()

        except Exception as e:
            logger.error(f"Error handling component shutdown event: {e}")

    async def _handle_component_failed(self, event: IntegrationEvent) -> None:
        """Handle component failure events."""
        try:
            component_metadata = self.integration_layer.get_component_metadata(event.component_type)

            event_data = {
                "component_type": event.component_type.value,
                "error_message": (
                    str(component_metadata.last_error)
                    if component_metadata and component_metadata.last_error
                    else "Unknown error"
                ),
                "error_count": component_metadata.error_count if component_metadata else 0,
                "state": "failed",
            }

            await self._broadcast_component_event("component_error", event_data)
            await self._broadcast_system_status()

        except Exception as e:
            logger.error(f"Error handling component failed event: {e}")

    async def _handle_component_health_changed(self, event: IntegrationEvent) -> None:
        """Handle component health change events."""
        try:
            event_data = {
                "component_type": event.component_type.value,
                "health_status": event.data.get("health_status", {}),
            }

            await self._broadcast_component_event("component_health", event_data)

        except Exception as e:
            logger.error(f"Error handling component health changed event: {e}")

    async def _broadcast_component_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a component event via WebSocket."""
        if not self.is_active or not self.websocket_server:
            return

        try:
            # Get next sequence number
            self.sequence_number += 1

            # Map event type to WebSocket event type
            ws_event_type_map = {
                "component_status": WSEventType.COMPONENT_STATUS,
                "component_error": WSEventType.COMPONENT_ERROR,
                "component_health": WSEventType.COMPONENT_HEALTH,
                "system_status": WSEventType.SYSTEM_STATUS,
                "integration_metrics": WSEventType.INTEGRATION_METRICS,
            }

            ws_event_type = ws_event_type_map.get(event_type)
            if not ws_event_type:
                logger.warning(f"Unknown event type: {event_type}")
                return

            # Create WebSocket event
            ws_event = create_event(event_type=ws_event_type, seq=self.sequence_number, **data)

            # Broadcast to all connected clients
            await self.websocket_server.broadcast(ws_event)

            # Store in event history
            self._store_event(event_type, data)

            logger.debug(f"Broadcasted component event: {event_type}")

        except Exception as e:
            logger.error(f"Failed to broadcast component event: {e}")

    async def _broadcast_system_status(self) -> None:
        """Broadcast overall system status."""
        if not self.is_active or not self.websocket_server:
            return

        try:
            # Get integration status
            integration_status = self.integration_layer.get_integration_status()

            # Create system status event data
            system_status_data = {
                "total_components": integration_status["total_components"],
                "ready_components": integration_status["ready_components"],
                "failed_components": integration_status["failed_components"],
                "integration_health": integration_status["integration_health"],
                "components": integration_status["components"],
            }

            await self._broadcast_component_event("system_status", system_status_data)

            logger.debug("Broadcasted system status")

        except Exception as e:
            logger.error(f"Failed to broadcast system status: {e}")

    def _store_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Store event in history for replay to new clients."""
        event_record = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        self.component_events.append(event_record)

        # Maintain max history size
        if len(self.component_events) > self.max_event_history:
            self.component_events = self.component_events[-self.max_event_history :]

    async def get_component_status(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current component status.

        Args:
            component_type: Optional specific component type to get status for

        Returns:
            Component status information
        """
        try:
            integration_status = self.integration_layer.get_integration_status()

            if component_type:
                # Return specific component status
                component_info = integration_status["components"].get(component_type)
                if component_info:
                    return {
                        "component_type": component_type,
                        "status": component_info,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {
                        "error": f"Component {component_type} not found",
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                # Return all components status
                return {
                    "system_status": integration_status,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error getting component status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def get_event_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent component events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        return self.component_events[-limit:] if self.component_events else []

    async def replay_events_for_client(self, client_id: str) -> None:
        """
        Replay recent events for a new client.

        Args:
            client_id: ID of the client to replay events for
        """
        if not self.is_active or not self.websocket_server:
            return

        try:
            # Send current system status first
            await self._broadcast_system_status()

            # Replay recent component events
            recent_events = await self.get_event_history(20)  # Last 20 events

            for event_record in recent_events:
                # Replay the event using the same broadcast mechanism
                await self._broadcast_component_event(
                    event_record["event_type"], event_record["data"]
                )

            logger.info(f"Replayed {len(recent_events)} events for client {client_id}")

        except Exception as e:
            logger.error(f"Error replaying events for client {client_id}: {e}")

    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration metrics."""
        return {
            "is_active": self.is_active,
            "websocket_server_available": self.websocket_server is not None,
            "event_handlers_registered": self.event_handlers_registered,
            "total_events_processed": len(self.component_events),
            "max_event_history": self.max_event_history,
            "timestamp": datetime.now().isoformat(),
        }

    async def broadcast_chat_response(
        self, session_id: str, response: Dict[str, Any], processing_time: float = 0
    ) -> None:
        """Broadcast a chat response to connected clients."""
        try:
            await self._broadcast_component_event(
                "chat_response",
                {
                    "session_id": session_id,
                    "response": response,
                    "processing_time_ms": processing_time,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to broadcast chat response: {e}")

    async def broadcast_session_created(self, session_id: str) -> None:
        """Broadcast session creation to connected clients."""
        try:
            await self._broadcast_component_event(
                "session_created", {"session_id": session_id, "timestamp": time.time()}
            )
        except Exception as e:
            logger.error(f"Failed to broadcast session creation: {e}")

    async def handle_chat_message(self, websocket, message: Dict[str, Any]) -> None:
        """Handle incoming chat messages from WebSocket clients."""
        try:
            data = message.get("data", {})
            chat_message = data.get("message", "").strip()
            session_id = data.get("session_id")

            if not chat_message:
                await websocket.send_json(
                    {"type": "error", "message": "Chat message cannot be empty"}
                )
                return

            logger.info(f"Processing chat message via WebSocket: {chat_message[:50]}...")

            # Get the assistant component
            if self.integration_layer:
                from .integration import ComponentType

                assistant = self.integration_layer.get_component(ComponentType.ASSISTANT)

                if assistant:
                    # Process the message
                    result = await assistant.process_text_message(
                        message=chat_message,
                        session_id=session_id,
                        metadata={"source": "websocket_chat"},
                    )

                    # Broadcast response to all clients
                    await self.broadcast_chat_response(
                        session_id=result.get("session_id"),
                        response=result.get("response"),
                        processing_time=result.get("processing_time_ms", 0),
                    )
                else:
                    await self._broadcast_component_event(
                        "chat_response",
                        {
                            "response": {
                                "content": "Assistant not available. Please check system status."
                            },
                            "session_id": session_id or "error_session",
                            "error": True,
                        },
                    )
            else:
                await self._broadcast_component_event(
                    "chat_response",
                    {
                        "response": {"content": "Integration layer not available."},
                        "session_id": session_id or "error_session",
                        "error": True,
                    },
                )

        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            await websocket.send_json(
                {"type": "error", "message": f"Failed to process chat message: {str(e)}"}
            )
