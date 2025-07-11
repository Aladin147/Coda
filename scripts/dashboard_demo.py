#!/usr/bin/env python3
"""
Dashboard Demo Script

This script demonstrates the Coda Dashboard functionality by:
1. Starting the dashboard server
2. Simulating component events
3. Providing a web interface for monitoring

Usage:
    python scripts/dashboard_demo.py
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coda.interfaces.dashboard.server import CodaDashboardServer
from coda.core.integration import ComponentIntegrationLayer, ComponentType
from coda.core.websocket_integration import ComponentWebSocketIntegration
from coda.interfaces.websocket.server import CodaWebSocketServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dashboard_demo")


class DashboardDemo:
    """Demo class for dashboard functionality."""
    
    def __init__(self):
        self.running = False
        self.dashboard_server = None
        self.websocket_server = None
        self.integration_layer = None
        self.websocket_integration = None
        
        # Demo components
        self.demo_components = [
            ComponentType.MEMORY,
            ComponentType.LLM,
            ComponentType.PERSONALITY,
            ComponentType.TOOLS
        ]
    
    async def start(self):
        """Start the demo."""
        logger.info("🚀 Starting Coda Dashboard Demo...")
        
        try:
            # Create integration layer
            self.integration_layer = ComponentIntegrationLayer()
            
            # Create WebSocket server
            self.websocket_server = CodaWebSocketServer(host="localhost", port=8765)
            await self.websocket_server.start()
            
            # Create WebSocket integration
            self.websocket_integration = ComponentWebSocketIntegration()
            self.websocket_integration.set_websocket_server(self.websocket_server)
            await self.websocket_integration.start()
            
            # Create dashboard server
            self.dashboard_server = CodaDashboardServer(host="localhost", port=8081)
            self.dashboard_server.set_integration_layer(self.integration_layer)
            self.dashboard_server.set_websocket_integration(self.websocket_integration)
            await self.dashboard_server.start()
            
            self.running = True
            
            logger.info("✅ Demo started successfully!")
            logger.info(f"📊 Dashboard: {self.dashboard_server.get_dashboard_url()}")
            logger.info(f"🔌 WebSocket: ws://localhost:8765")
            logger.info("📝 Open the dashboard URL in your browser to see real-time monitoring")
            
            # Start demo simulation
            await self.run_demo_simulation()
            
        except Exception as e:
            logger.error(f"❌ Failed to start demo: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the demo."""
        if not self.running:
            return
        
        logger.info("🛑 Stopping dashboard demo...")
        
        try:
            if self.dashboard_server:
                await self.dashboard_server.stop()
            
            if self.websocket_integration:
                await self.websocket_integration.stop()
            
            if self.websocket_server:
                await self.websocket_server.stop()
            
            self.running = False
            logger.info("✅ Demo stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping demo: {e}")
    
    async def run_demo_simulation(self):
        """Run demo simulation with component events."""
        logger.info("🎭 Starting demo simulation...")
        
        # Simulate component initialization
        for i, component_type in enumerate(self.demo_components):
            await asyncio.sleep(2)
            
            # Register component
            self.integration_layer.register_component(component_type, dependencies=[])
            
            # Simulate initialization
            await self.integration_layer.initialize_component(component_type)
            
            logger.info(f"✅ Initialized {component_type.value}")
        
        # Simulate some activity
        event_count = 0
        while self.running:
            await asyncio.sleep(5)
            
            # Simulate random component events
            import random
            component = random.choice(self.demo_components)
            
            if event_count % 4 == 0:
                # Simulate component status update
                await self.websocket_integration._broadcast_component_event(
                    "component_status",
                    {
                        "component_type": component.value,
                        "state": "ready",
                        "initialization_order": event_count
                    }
                )
                logger.info(f"📊 Status update: {component.value} -> ready")
                
            elif event_count % 7 == 0:
                # Simulate component health check
                await self.websocket_integration._broadcast_component_event(
                    "component_health",
                    {
                        "component_type": component.value,
                        "health_status": {"status": "healthy", "last_check": "2024-01-01T12:00:00"}
                    }
                )
                logger.info(f"💚 Health check: {component.value} -> healthy")
                
            elif event_count % 10 == 0:
                # Simulate occasional error
                await self.websocket_integration._broadcast_component_event(
                    "component_error",
                    {
                        "component_type": component.value,
                        "error_message": f"Simulated error #{event_count}",
                        "error_count": 1
                    }
                )
                logger.info(f"⚠️ Error simulation: {component.value} -> error")
            
            # Broadcast system status
            await self.websocket_integration._broadcast_system_status()
            
            event_count += 1
            
            # Show periodic status
            if event_count % 5 == 0:
                logger.info(f"📈 Demo running... {event_count} events generated")


async def main():
    """Main demo function."""
    demo = DashboardDemo()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler():
        logger.info("🔄 Received shutdown signal...")
        asyncio.create_task(demo.stop())
    
    # Handle Ctrl+C gracefully
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    try:
        await demo.start()
        
        # Keep running until interrupted
        while demo.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🔄 Keyboard interrupt received...")
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
    finally:
        await demo.stop()


if __name__ == "__main__":
    print("🎯 Coda Dashboard Demo")
    print("=" * 50)
    print("This demo will start:")
    print("  📊 Dashboard Server (http://localhost:8081)")
    print("  🔌 WebSocket Server (ws://localhost:8765)")
    print("  🎭 Component Event Simulation")
    print()
    print("Open http://localhost:8081 in your browser to see the dashboard!")
    print("Press Ctrl+C to stop the demo")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
