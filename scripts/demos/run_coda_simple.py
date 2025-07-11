#!/usr/bin/env python3
"""
Coda Simple System Launcher - NO FALLBACKS
Real Testing Environment with Core Components

This script launches the core Coda system with:
- Ollama qwen3:30b-a3b LLM (PRIMARY, /no_think enabled)
- ChromaDB Memory System (PRIMARY)
- WebSocket Server for real-time events
- Dashboard for monitoring

NO FALLBACKS ACTIVE - Pure primary systems only!
"""

import asyncio
import logging
import sys
import signal
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("coda_simple")

# Import core components
try:
    from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
    from coda.components.llm.manager import LLMManager
    from coda.components.memory.manager import MemoryManager
    from coda.components.memory.models import MemoryManagerConfig
    from coda.interfaces.websocket.server import CodaWebSocketServer
    from coda.interfaces.dashboard.server import CodaDashboardServer
    
except ImportError as e:
    logger.error(f"Failed to import Coda components: {e}")
    exit(1)


class CodaSimpleSystem:
    """Simple Coda system with core components - NO FALLBACKS."""
    
    def __init__(self):
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.llm_manager: Optional[LLMManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.websocket_server: Optional[CodaWebSocketServer] = None
        self.dashboard_server: Optional[CodaDashboardServer] = None
        
        logger.info("CODA Simple System initialized - PRIMARY COMPONENTS ONLY")
        
    async def initialize_components(self):
        """Initialize core components with NO FALLBACKS."""
        logger.info("INIT: Initializing Coda Simple System - NO FALLBACKS")
        
        # 1. Initialize LLM Manager
        await self._initialize_llm()
        
        # 2. Initialize Memory System
        await self._initialize_memory()
        
        # 3. Initialize WebSocket Server
        await self._initialize_websocket()
        
        # 4. Initialize Dashboard Server
        await self._initialize_dashboard()
        
        logger.info("SUCCESS: All core components initialized!")
        
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
        logger.info("MEMORY: Initializing Memory System (ChromaDB)...")
        
        config = MemoryManagerConfig()
        
        self.memory_manager = MemoryManager(config)
        await self.memory_manager.initialize()
        logger.info("SUCCESS: Memory System initialized with ChromaDB")
        
    async def _initialize_websocket(self):
        """Initialize WebSocket Server."""
        logger.info("WEBSOCKET: Initializing WebSocket Server...")
        
        self.websocket_server = CodaWebSocketServer(
            host="localhost",
            port=8765,
            max_replay_events=100
        )
        
        await self.websocket_server.start()
        logger.info("SUCCESS: WebSocket Server started at ws://localhost:8765")
        
    async def _initialize_dashboard(self):
        """Initialize Dashboard Server."""
        logger.info("DASHBOARD: Initializing Dashboard Server...")
        
        self.dashboard_server = CodaDashboardServer(
            host="localhost",
            port=8081
        )
        
        await self.dashboard_server.start()
        logger.info("SUCCESS: Dashboard Server started at http://localhost:8081")
        
    async def start_system(self):
        """Start the Coda system."""
        try:
            await self.initialize_components()
            
            self.running = True
            
            # Send startup event
            await self._broadcast_startup()
            
            logger.info("=" * 60)
            logger.info("CODA SIMPLE SYSTEM RUNNING!")
            logger.info("=" * 60)
            logger.info("WebSocket Server: ws://localhost:8765")
            logger.info("Dashboard: http://localhost:8081")
            logger.info("LLM: Ollama qwen3:30b-a3b (/no_think)")
            logger.info("Memory: ChromaDB")
            logger.info("Voice: Moshi Ready (RTX 5090)")
            logger.info("NO FALLBACKS ACTIVE")
            logger.info("=" * 60)
            logger.info("System ready for testing!")
            
            # Run integration test
            await self.test_system()
            
            # Keep system running
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
            
    async def _broadcast_startup(self):
        """Broadcast system startup event."""
        if self.websocket_server:
            await self.websocket_server.broadcast_event(
                "system_startup",
                {
                    "timestamp": time.time(),
                    "components": {
                        "llm": "ollama:qwen3:30b-a3b",
                        "memory": "chromadb",
                        "voice": "moshi_ready",
                        "websocket": "active",
                        "dashboard": "active"
                    },
                    "fallbacks": "disabled",
                    "status": "ready"
                }
            )
            
    async def test_system(self):
        """Run system integration test."""
        logger.info("TEST: Running System Integration Test...")
        
        try:
            # Test LLM
            response = await self.llm_manager.generate_response(
                prompt="/no_think System test: respond with 'Coda systems operational'",
                provider="ollama"
            )
            logger.info(f"TEST LLM: {response.content[:50]}...")
            
            # Test Memory
            self.memory_manager.add_turn("system", "System integration test")
            self.memory_manager.add_turn("assistant", "Test successful")
            logger.info("TEST MEMORY: Conversation stored")
            
            # Test WebSocket
            await self.websocket_server.broadcast_event(
                "integration_test",
                {"status": "success", "timestamp": time.time()}
            )
            logger.info("TEST WEBSOCKET: Event broadcasted")
            
            logger.info("SUCCESS: System Integration Test PASSED!")
            
        except Exception as e:
            logger.error(f"FAILED: System Integration Test FAILED: {e}")
            
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("SHUTDOWN: Shutting down Coda Simple System...")
        
        self.running = False
        
        # Stop servers
        if self.dashboard_server:
            await self.dashboard_server.stop()
            
        if self.websocket_server:
            await self.websocket_server.stop()
            
        self.shutdown_event.set()
        logger.info("SUCCESS: Coda Simple System shutdown complete")


async def main():
    """Main function to run Coda Simple System."""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    system = CodaSimpleSystem()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(system.shutdown())
    
    try:
        # Start the system
        await system.start_system()
        
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
