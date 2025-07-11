"""
Dashboard Server for Coda.

This module provides a simple HTTP server to serve the dashboard interface
and handle dashboard-specific API endpoints.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from aiohttp import WSMsgType, web
from aiohttp.web_request import Request
from aiohttp.web_response import Response

logger = logging.getLogger("coda.dashboard.server")


class CodaDashboardServer:
    """
    HTTP server for the Coda dashboard interface.

    Serves static files and provides API endpoints for dashboard functionality.
    """

    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Initialize dashboard server.

        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.running = False

        # Dashboard directory
        self.dashboard_dir = Path(__file__).parent

        # Integration with Coda components
        self.integration_layer = None
        self.websocket_integration = None

        logger.info(f"CodaDashboardServer initialized on {host}:{port}")

    def set_integration_layer(self, integration_layer) -> None:
        """Set the component integration layer."""
        self.integration_layer = integration_layer
        logger.info("Integration layer set for dashboard server")

    def set_websocket_integration(self, websocket_integration) -> None:
        """Set the WebSocket integration."""
        self.websocket_integration = websocket_integration
        logger.info("WebSocket integration set for dashboard server")

    async def start(self) -> None:
        """Start the dashboard server."""
        if self.running:
            logger.warning("Dashboard server already running")
            return

        try:
            # Create aiohttp application
            self.app = web.Application()

            # Set up routes
            self._setup_routes()

            # Create runner and site
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            self.running = True
            logger.info(f"Dashboard server started at http://{self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the dashboard server."""
        if not self.running:
            return

        try:
            if self.site:
                await self.site.stop()

            if self.runner:
                await self.runner.cleanup()

            self.running = False
            logger.info("Dashboard server stopped")

        except Exception as e:
            logger.error(f"Error stopping dashboard server: {e}")

    def _setup_routes(self) -> None:
        """Set up HTTP routes."""
        # Static file routes
        self.app.router.add_get("/", self._serve_index)
        self.app.router.add_get("/index.html", self._serve_index)
        self.app.router.add_get("/dashboard.js", self._serve_dashboard_js)
        self.app.router.add_static("/static", self.dashboard_dir, name="static")

        # API routes
        self.app.router.add_get("/api/status", self._api_status)
        self.app.router.add_get("/api/components", self._api_components)
        self.app.router.add_get("/api/events", self._api_events)
        self.app.router.add_get("/api/metrics", self._api_metrics)
        self.app.router.add_post("/api/control/{action}", self._api_control)

        # Chat API routes
        self.app.router.add_post("/api/chat/message", self._api_chat_message)
        self.app.router.add_get("/api/chat/session", self._api_chat_session)
        self.app.router.add_post("/api/chat/session", self._api_create_chat_session)

        # Health check
        self.app.router.add_get("/health", self._health_check)

    async def _serve_index(self, request: Request) -> Response:
        """Serve the main dashboard HTML file."""
        try:
            index_path = self.dashboard_dir / "index.html"

            if not index_path.exists():
                return web.Response(
                    text="Dashboard index.html not found", status=404, content_type="text/plain"
                )

            async with aiofiles.open(index_path, "r", encoding="utf-8") as f:
                content = await f.read()

            return web.Response(text=content, content_type="text/html")

        except Exception as e:
            logger.error(f"Error serving index.html: {e}")
            return web.Response(
                text=f"Error loading dashboard: {e}", status=500, content_type="text/plain"
            )

    async def _serve_dashboard_js(self, request: Request) -> Response:
        """Serve the dashboard JavaScript file."""
        try:
            js_path = self.dashboard_dir / "dashboard.js"

            if not js_path.exists():
                return web.Response(
                    text="Dashboard JavaScript not found", status=404, content_type="text/plain"
                )

            async with aiofiles.open(js_path, "r", encoding="utf-8") as f:
                content = await f.read()

            return web.Response(text=content, content_type="application/javascript")

        except Exception as e:
            logger.error(f"Error serving dashboard.js: {e}")
            return web.Response(
                text=f"Error loading JavaScript: {e}", status=500, content_type="text/plain"
            )

    async def _api_status(self, request: Request) -> Response:
        """API endpoint for system status."""
        try:
            if not self.integration_layer:
                return web.json_response(
                    {
                        "error": "Integration layer not available",
                        "timestamp": datetime.now().isoformat(),
                    },
                    status=503,
                )

            status = self.integration_layer.get_integration_status()
            status["timestamp"] = datetime.now().isoformat()
            status["dashboard_server"] = {
                "running": self.running,
                "host": self.host,
                "port": self.port,
            }

            return web.json_response(status)

        except Exception as e:
            logger.error(f"Error in status API: {e}")
            return web.json_response(
                {"error": str(e), "timestamp": datetime.now().isoformat()}, status=500
            )

    async def _api_components(self, request: Request) -> Response:
        """API endpoint for component information."""
        try:
            if not self.integration_layer:
                return web.json_response({"error": "Integration layer not available"}, status=503)

            components = {}
            for comp_type, metadata in self.integration_layer.components.items():
                components[comp_type.value] = {
                    "state": metadata.state.value,
                    "dependencies": [dep.value for dep in metadata.dependencies],
                    "dependents": [dep.value for dep in metadata.dependents],
                    "initialization_order": metadata.initialization_order,
                    "error_count": metadata.error_count,
                    "last_error": str(metadata.last_error) if metadata.last_error else None,
                }

            return web.json_response(
                {"components": components, "timestamp": datetime.now().isoformat()}
            )

        except Exception as e:
            logger.error(f"Error in components API: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _api_events(self, request: Request) -> Response:
        """API endpoint for event history."""
        try:
            limit = int(request.query.get("limit", 50))

            if not self.websocket_integration:
                return web.json_response(
                    {"events": [], "message": "WebSocket integration not available"}
                )

            events = await self.websocket_integration.get_event_history(limit)

            return web.json_response(
                {"events": events, "count": len(events), "timestamp": datetime.now().isoformat()}
            )

        except Exception as e:
            logger.error(f"Error in events API: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _api_metrics(self, request: Request) -> Response:
        """API endpoint for dashboard metrics."""
        try:
            metrics = {
                "dashboard_server": {
                    "running": self.running,
                    "host": self.host,
                    "port": self.port,
                    "uptime_seconds": 0,  # TODO: Track actual uptime
                }
            }

            if self.websocket_integration:
                ws_metrics = self.websocket_integration.get_integration_metrics()
                metrics["websocket_integration"] = ws_metrics

            if self.integration_layer:
                integration_status = self.integration_layer.get_integration_status()
                metrics["integration_layer"] = {
                    "total_components": integration_status["total_components"],
                    "ready_components": integration_status["ready_components"],
                    "failed_components": integration_status["failed_components"],
                    "integration_health": integration_status["integration_health"],
                }

            metrics["timestamp"] = datetime.now().isoformat()

            return web.json_response(metrics)

        except Exception as e:
            logger.error(f"Error in metrics API: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _api_control(self, request: Request) -> Response:
        """API endpoint for dashboard control actions."""
        try:
            action = request.match_info["action"]

            if action == "refresh":
                # Trigger system refresh
                result = {"action": "refresh", "status": "completed"}

            elif action == "restart_component":
                # TODO: Implement component restart
                result = {"action": "restart_component", "status": "not_implemented"}

            elif action == "clear_events":
                # Clear event history
                if self.websocket_integration:
                    self.websocket_integration.component_events.clear()
                result = {"action": "clear_events", "status": "completed"}

            else:
                return web.json_response({"error": f"Unknown action: {action}"}, status=400)

            result["timestamp"] = datetime.now().isoformat()
            return web.json_response(result)

        except Exception as e:
            logger.error(f"Error in control API: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _health_check(self, request: Request) -> Response:
        """Health check endpoint."""
        health_status = {
            "status": "healthy" if self.running else "unhealthy",
            "dashboard_server": {"running": self.running, "host": self.host, "port": self.port},
            "integrations": {
                "integration_layer": self.integration_layer is not None,
                "websocket_integration": self.websocket_integration is not None,
            },
            "timestamp": datetime.now().isoformat(),
        }

        status_code = 200 if self.running else 503
        return web.json_response(health_status, status=status_code)

    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.host}:{self.port}"

    def get_status(self) -> Dict[str, Any]:
        """Get dashboard server status."""
        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "dashboard_url": self.get_dashboard_url(),
            "integrations": {
                "integration_layer": self.integration_layer is not None,
                "websocket_integration": self.websocket_integration is not None,
            },
        }

    async def _api_chat_message(self, request: Request) -> Response:
        """Handle chat message API endpoint."""
        try:
            data = await request.json()
            message = data.get("message", "").strip()
            session_id = data.get("session_id")

            if not message:
                return web.json_response({"error": "Message cannot be empty"}, status=400)

            logger.info(f"Received chat message: {message[:50]}...")

            # Process message through CodaAssistant if available
            if self.integration_layer and hasattr(self.integration_layer, "get_component"):
                try:
                    # Get the assistant component
                    from ...core.integration import ComponentType

                    assistant = self.integration_layer.get_component(ComponentType.ASSISTANT)

                    if assistant:
                        # Process the message
                        result = await assistant.process_text_message(
                            message=message,
                            session_id=session_id,
                            metadata={"source": "dashboard_chat"},
                        )

                        # Broadcast response via WebSocket if available
                        if self.websocket_integration:
                            await self.websocket_integration.broadcast_chat_response(
                                session_id=result.get("session_id"),
                                response=result.get("response"),
                                processing_time=result.get("processing_time_ms", 0),
                            )

                        return web.json_response(
                            {
                                "status": "success",
                                "response": result.get("response", {}).get(
                                    "content", "No response"
                                ),
                                "session_id": result.get("session_id"),
                                "processing_time_ms": result.get("processing_time_ms", 0),
                            }
                        )
                    else:
                        return web.json_response({"error": "Assistant not available"}, status=503)

                except Exception as e:
                    logger.error(f"Error processing chat message: {e}")
                    return web.json_response({"error": f"Processing failed: {str(e)}"}, status=500)
            else:
                # Fallback response when integration layer is not available
                return web.json_response(
                    {
                        "status": "success",
                        "response": "I'm currently not connected to the main assistant. Please check the system status.",
                        "session_id": session_id or "fallback_session",
                    }
                )

        except Exception as e:
            logger.error(f"Error in chat message API: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    async def _api_chat_session(self, request: Request) -> Response:
        """Get current chat session information."""
        try:
            if self.integration_layer and hasattr(self.integration_layer, "get_component"):
                from ...core.integration import ComponentType

                assistant = self.integration_layer.get_component(ComponentType.ASSISTANT)

                if assistant and hasattr(assistant, "current_session_id"):
                    return web.json_response(
                        {
                            "session_id": assistant.current_session_id,
                            "status": "active" if assistant.conversation_active else "inactive",
                        }
                    )

            return web.json_response({"session_id": None, "status": "no_session"})

        except Exception as e:
            logger.error(f"Error getting chat session: {e}")
            return web.json_response({"error": "Failed to get session info"}, status=500)

    async def _api_create_chat_session(self, request: Request) -> Response:
        """Create a new chat session."""
        try:
            if self.integration_layer and hasattr(self.integration_layer, "get_component"):
                from ...core.integration import ComponentType

                assistant = self.integration_layer.get_component(ComponentType.ASSISTANT)

                if assistant:
                    session_id = await assistant.start_conversation()

                    # Broadcast session creation via WebSocket if available
                    if self.websocket_integration:
                        await self.websocket_integration.broadcast_session_created(session_id)

                    return web.json_response({"session_id": session_id, "status": "created"})
                else:
                    return web.json_response({"error": "Assistant not available"}, status=503)
            else:
                return web.json_response({"error": "Integration layer not available"}, status=503)

        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            return web.json_response({"error": "Failed to create session"}, status=500)
