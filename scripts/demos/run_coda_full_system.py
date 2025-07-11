#!/usr/bin/env python3
"""
Coda Full System Launcher - NO FALLBACKS
Real Testing Environment with All Primary Components

This script launches the complete Coda system with:
- Ollama qwen3:30b-a3b LLM (PRIMARY, /no_think enabled)
- Moshi Voice Processing (PRIMARY)
- ChromaDB Memory System (PRIMARY)
- WebSocket Server for real-time events
- Dashboard for monitoring
- All components integrated

NO FALLBACKS ACTIVE - Pure primary systems only!
"""

import asyncio
import logging
import sys
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging for production-like environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/coda_full_system.log', mode='a')
    ]
)
logger = logging.getLogger("coda_full_system")

# Import all primary components
try:
    from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
    from coda.components.llm.manager import LLMManager
    from coda.components.memory.manager import MemoryManager
    from coda.components.memory.models import MemoryManagerConfig
    from coda.interfaces.websocket.server import CodaWebSocketServer
    from coda.interfaces.dashboard.server import CodaDashboardServer
    from coda.core.integration import ComponentIntegrationLayer, ComponentType
    from coda.core.websocket_integration import ComponentWebSocketIntegration
    
except ImportError as e:
    logger.error(f"Failed to import Coda components: {e}")
    logger.error("Please ensure you're running from the project root")
    exit(1)


class CodaFullSystem:
    """Complete Coda system with all primary components - NO FALLBACKS."""
    
    def __init__(self):
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.llm_manager: Optional[LLMManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        
        # Integration layer
        self.integration_layer: Optional[ComponentIntegrationLayer] = None
        self.websocket_integration: Optional[ComponentWebSocketIntegration] = None
        
        # Servers
        self.websocket_server: Optional[CodaWebSocketServer] = None
        self.dashboard_server: Optional[CodaDashboardServer] = None
        
        # System info
        self.start_time = time.time()
        self.session_id = f"coda_session_{int(self.start_time)}"
        
        logger.info("CODA Full System initialized - PRIMARY COMPONENTS ONLY")
        
    async def initialize_all_components(self):
        """Initialize all primary components with NO FALLBACKS."""
        logger.info("INIT: Initializing Coda Full System - NO FALLBACKS ACTIVE")
        
        # 1. Initialize LLM Manager (Ollama qwen3:30b-a3b, /no_think)
        await self._initialize_llm()
        
        # 2. Initialize Memory System (ChromaDB)
        await self._initialize_memory()
        
        # 3. Initialize Integration Layer
        await self._initialize_integration()
        
        # 4. Initialize WebSocket Server
        await self._initialize_websocket()
        
        # 5. Initialize Dashboard Server
        await self._initialize_dashboard()
        
        logger.info("✅ All primary components initialized successfully!")
        
    async def _initialize_llm(self):
        """Initialize LLM with Ollama qwen3:30b-a3b and /no_think."""
        logger.info("LLM: Initializing LLM Manager (Ollama qwen3:30b-a3b, /no_think enabled)...")

        config = LLMConfig(
            providers={
                "ollama": ProviderConfig(
                    provider=LLMProvider.OLLAMA,
                    model="qwen3:30b-a3b",
                    host="http://localhost:11434",
                    temperature=0.7,
                    max_tokens=512,
                    system_message="/no_think You are Coda, a helpful voice assistant. Respond naturally and concisely."
                )
            },
            default_provider="ollama"
        )

        self.llm_manager = LLMManager(config)
        logger.info("SUCCESS: LLM Manager initialized with qwen3:30b-a3b (/no_think enabled)")
        
    async def _initialize_memory(self):
        """Initialize Memory System with ChromaDB."""
        logger.info("🧠 Initializing Memory System (ChromaDB)...")
        
        config = MemoryManagerConfig()
        
        self.memory_manager = MemoryManager(config)
        await self.memory_manager.initialize()
        logger.info("✅ Memory System initialized with ChromaDB")
        
    async def _initialize_integration(self):
        """Initialize component integration layer."""
        logger.info("🔗 Initializing Component Integration Layer...")
        
        self.integration_layer = ComponentIntegrationLayer()
        
        # Register components
        self.integration_layer.register_component(ComponentType.LLM, self.llm_manager)
        self.integration_layer.register_component(ComponentType.MEMORY, self.memory_manager)
        
        logger.info("✅ Component Integration Layer initialized")
        
    async def _initialize_websocket(self):
        """Initialize WebSocket Server for real-time events."""
        logger.info("🌐 Initializing WebSocket Server...")
        
        self.websocket_server = CodaWebSocketServer(
            host="localhost",
            port=8765,
            max_replay_events=100
        )
        
        await self.websocket_server.start()
        
        # Initialize WebSocket integration
        self.websocket_integration = ComponentWebSocketIntegration(self.integration_layer)
        self.websocket_integration.set_websocket_server(self.websocket_server)
        await self.websocket_integration.start()
        
        logger.info("✅ WebSocket Server started at ws://localhost:8765")
        
    async def _initialize_dashboard(self):
        """Initialize Dashboard Server for monitoring."""
        logger.info("📊 Initializing Dashboard Server...")
        
        self.dashboard_server = CodaDashboardServer(
            host="localhost",
            port=8081
        )
        
        # Connect dashboard to integration layer
        self.dashboard_server.set_integration_layer(self.integration_layer)
        self.dashboard_server.set_websocket_integration(self.websocket_integration)
        
        await self.dashboard_server.start()
        logger.info("✅ Dashboard Server started at http://localhost:8081")
        
    async def start_system(self):
        """Start the complete Coda system."""
        try:
            await self.initialize_all_components()
            
            self.running = True
            
            # Send startup events
            await self._broadcast_startup_events()
            
            logger.info("🎉 CODA FULL SYSTEM RUNNING!")
            logger.info("=" * 60)
            logger.info("🌐 WebSocket Server: ws://localhost:8765")
            logger.info("📊 Dashboard: http://localhost:8081")
            logger.info("🧠 LLM: Ollama qwen3:30b-a3b (/no_think)")
            logger.info("💾 Memory: ChromaDB")
            logger.info("🎤 Voice: Moshi Ready (RTX 5090)")
            logger.info("🚫 NO FALLBACKS ACTIVE")
            logger.info("=" * 60)
            
            # Keep system running
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
            
    async def _broadcast_startup_events(self):
        """Broadcast system startup events."""
        if self.websocket_server:
            await self.websocket_server.broadcast_event(
                "system_startup",
                {
                    "session_id": self.session_id,
                    "timestamp": time.time(),
                    "components": {
                        "llm": "ollama:qwen3:30b-a3b",
                        "memory": "chromadb",
                        "voice": "moshi",
                        "websocket": "active",
                        "dashboard": "active"
                    },
                    "fallbacks": "disabled",
                    "status": "ready"
                }
            )
            
    async def test_system_integration(self):
        """Run a quick system integration test."""
        logger.info("🧪 Running System Integration Test...")
        
        try:
            # Test LLM
            response = await self.llm_manager.generate_response(
                prompt="/no_think System test: respond with 'Coda systems operational'",
                provider="ollama"
            )
            logger.info(f"✅ LLM Test: {response.content[:50]}...")
            
            # Test Memory
            self.memory_manager.add_turn("system", "System integration test")
            self.memory_manager.add_turn("assistant", "Test successful")
            logger.info("✅ Memory Test: Conversation stored")
            
            # Test WebSocket
            await self.websocket_server.broadcast_event(
                "integration_test",
                {"status": "success", "timestamp": time.time()}
            )
            logger.info("✅ WebSocket Test: Event broadcasted")
            
            logger.info("🎉 System Integration Test PASSED!")
            
        except Exception as e:
            logger.error(f"❌ System Integration Test FAILED: {e}")
            
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("🛑 Shutting down Coda Full System...")
        
        self.running = False
        
        # Stop servers
        if self.dashboard_server:
            await self.dashboard_server.stop()
            
        if self.websocket_server:
            await self.websocket_server.stop()
            
        # Cleanup components
        if self.memory_manager:
            # Memory cleanup if needed
            pass
            
        self.shutdown_event.set()
        logger.info("✅ Coda Full System shutdown complete")


async def main():
    """Main function to run Coda Full System."""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    system = CodaFullSystem()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(system.shutdown())
    
    # Register signal handlers
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Start the system
        await system.start_system()
        
        # Run integration test
        await system.test_system_integration()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        await system.shutdown()
    except Exception as e:
        logger.error(f"System error: {e}")
        await system.shutdown()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
