"""
Complete WebSocket server for voice processing.

This module provides a complete WebSocket server implementation that
integrates all voice processing components for real-time communication.
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .manager import VoiceManager
from .models import AudioConfig, MoshiConfig, VoiceConfig
from .performance_profiler import get_performance_profiler
from .websocket_audio_streaming import AudioStreamProcessor
from .websocket_events import VoiceEventBroadcaster
from .websocket_handler import VoiceWebSocketHandler
from .websocket_monitoring import WebSocketMonitor

logger = logging.getLogger(__name__)


class VoiceWebSocketServer:
    """
    Complete WebSocket server for voice processing.

    Integrates all voice processing components into a single server
    that provides real-time voice communication capabilities.

    Features:
    - Real-time voice processing over WebSocket
    - Audio streaming and processing
    - Event broadcasting and monitoring
    - Performance monitoring and analytics
    - Graceful shutdown and resource cleanup

    Example:
        >>> server = VoiceWebSocketServer()
        >>> await server.start()
    """

    def __init__(
        self,
        voice_config: Optional[VoiceConfig] = None,
        host: str = "localhost",
        port: int = 8765,
        max_connections: int = 100,
        auth_required: bool = False,
    ):
        """
        Initialize WebSocket server.

        Args:
            voice_config: Voice processing configuration
            host: Server host address
            port: Server port
            max_connections: Maximum concurrent connections
            auth_required: Whether authentication is required
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.auth_required = auth_required

        # Create default voice config if not provided
        if voice_config is None:
            voice_config = VoiceConfig(
                audio=AudioConfig(sample_rate=16000, channels=1, format="wav"),
                moshi=MoshiConfig(device="cpu", vram_allocation="2GB"),
            )

        self.voice_config = voice_config

        # Initialize components
        self.voice_manager: Optional[VoiceManager] = None
        self.websocket_handler: Optional[VoiceWebSocketHandler] = None
        self.event_broadcaster: Optional[VoiceEventBroadcaster] = None
        self.audio_processor: Optional[AudioStreamProcessor] = None
        self.monitor: Optional[WebSocketMonitor] = None

        # Server state
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        logger.info(f"VoiceWebSocketServer initialized on {host}:{port}")

    async def start(self) -> None:
        """Start the WebSocket server and all components."""
        try:
            logger.info("Starting voice WebSocket server...")

            # Initialize voice manager
            self.voice_manager = VoiceManager(self.voice_config)
            await self.voice_manager.initialize(self.voice_config)

            # Initialize WebSocket handler
            self.websocket_handler = VoiceWebSocketHandler(
                voice_manager=self.voice_manager,
                host=self.host,
                port=self.port,
                max_connections=self.max_connections,
                auth_required=self.auth_required,
            )

            # Initialize event broadcaster
            self.event_broadcaster = VoiceEventBroadcaster(websocket_handler=self.websocket_handler)

            # Initialize audio stream processor
            self.audio_processor = AudioStreamProcessor(
                websocket_handler=self.websocket_handler, voice_manager=self.voice_manager
            )

            # Initialize monitoring
            self.monitor = WebSocketMonitor(
                websocket_handler=self.websocket_handler, event_broadcaster=self.event_broadcaster
            )

            # Start WebSocket server
            await self.websocket_handler.start_server()

            # Start monitoring
            await self.monitor.start_monitoring()

            self.is_running = True

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            logger.info(f"Voice WebSocket server started on {self.host}:{self.port}")
            logger.info(f"Max connections: {self.max_connections}")
            logger.info(f"Authentication required: {self.auth_required}")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the WebSocket server and cleanup resources."""
        if not self.is_running:
            return

        logger.info("Stopping voice WebSocket server...")

        self.is_running = False

        # Stop monitoring
        if self.monitor:
            await self.monitor.stop_monitoring()

        # Stop WebSocket server
        if self.websocket_handler:
            await self.websocket_handler.stop_server()

        # Cleanup voice manager
        if self.voice_manager:
            await self.voice_manager.cleanup()

        # Signal shutdown complete
        self.shutdown_event.set()

        logger.info("Voice WebSocket server stopped")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics."""
        stats = {
            "server": {
                "host": self.host,
                "port": self.port,
                "is_running": self.is_running,
                "max_connections": self.max_connections,
                "auth_required": self.auth_required,
            }
        }

        if self.websocket_handler:
            stats["websocket"] = self.websocket_handler.get_connection_stats()

        if self.monitor:
            stats["monitoring"] = self.monitor.get_real_time_metrics()

        if self.audio_processor:
            stats["audio_streaming"] = self.audio_processor.get_all_stream_stats()

        if self.voice_manager:
            stats["voice_processing"] = {
                "active_conversations": len(getattr(self.voice_manager, "conversations", {})),
                "analytics": getattr(self.voice_manager, "analytics", {}),
            }

        return stats

    async def broadcast_system_message(self, message: str, message_type: str = "info") -> int:
        """Broadcast a system message to all connected clients."""
        if not self.event_broadcaster:
            return 0

        from .websocket_events import EventType

        return await self.event_broadcaster.broadcast_event(
            EventType.SYSTEM_STATUS,
            {"message": message, "type": message_type, "timestamp": time.time()},
        )


async def main():
    """Main function to run the WebSocket server."""
    import argparse

    parser = argparse.ArgumentParser(description="Coda Voice WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--max-connections", type=int, default=100, help="Maximum connections")
    parser.add_argument("--auth-required", action="store_true", help="Require authentication")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and start server
    server = VoiceWebSocketServer(
        host=args.host,
        port=args.port,
        max_connections=args.max_connections,
        auth_required=args.auth_required,
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
