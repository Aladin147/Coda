#!/usr/bin/env python3
"""
WebSocket system demonstration script.

This script demonstrates the WebSocket server functionality by:
1. Starting a WebSocket server
2. Simulating various Coda events
3. Showing real-time event broadcasting
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coda.interfaces.websocket.server import CodaWebSocketServer
from coda.interfaces.websocket.integration import CodaWebSocketIntegration
from coda.interfaces.websocket.performance import WebSocketPerfIntegration
from coda.interfaces.websocket.events import EventType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("websocket_demo")


async def simulate_conversation_flow(integration: CodaWebSocketIntegration):
    """Simulate a complete conversation flow with events."""
    logger.info("🎬 Starting conversation simulation...")
    
    # Start conversation
    conversation_id = "demo-conversation-001"
    await integration.conversation_start(conversation_id)
    await asyncio.sleep(1)
    
    # STT phase
    await integration.stt_start("push_to_talk")
    await asyncio.sleep(0.5)
    
    await integration.stt_interim_result("Hello", 0.8)
    await asyncio.sleep(0.3)
    
    await integration.stt_interim_result("Hello Coda", 0.9)
    await asyncio.sleep(0.3)
    
    await integration.stt_final_result("Hello Coda, what time is it?", 0.95, 2500.0, "en")
    await asyncio.sleep(0.5)
    
    # Memory retrieval
    await integration.memory_retrieve("time query", 3, [0.9, 0.7, 0.5])
    await asyncio.sleep(0.3)
    
    # LLM processing
    await integration.llm_start("Hello Coda, what time is it?", "llama3", 0.7)
    await asyncio.sleep(0.5)
    
    # Simulate token streaming
    tokens = ["The", " current", " time", " is", " 3", ":", "45", " PM", "."]
    cumulative = ""
    for token in tokens:
        cumulative += token
        await integration.llm_token(token, cumulative)
        await asyncio.sleep(0.1)
    
    await integration.llm_result(cumulative, 1200.0, len(tokens), 7.5)
    await asyncio.sleep(0.5)
    
    # Tool call
    tool_call_id = "time-001"
    await integration.tool_call("get_time", {}, tool_call_id)
    await asyncio.sleep(0.3)
    
    await integration.tool_result("get_time", tool_call_id, "15:45:00", 50.0)
    await asyncio.sleep(0.3)
    
    # TTS phase
    await integration.tts_start("The current time is 3:45 PM.", "default", "elevenlabs")
    await asyncio.sleep(0.2)
    
    for progress in [25, 50, 75, 100]:
        await integration.tts_progress(progress, 2000.0)
        await asyncio.sleep(0.2)
    
    await integration.tts_result(800.0, 2000.0, True)
    await asyncio.sleep(0.5)
    
    # Memory storage
    await integration.memory_store(
        "User asked for current time at 3:45 PM",
        "conversation",
        0.7,
        "mem-001"
    )
    await asyncio.sleep(0.3)
    
    # Conversation turn
    await integration.conversation_turn(
        conversation_id,
        1,
        "Hello Coda, what time is it?",
        "The current time is 3:45 PM."
    )
    await asyncio.sleep(1)
    
    # End conversation
    await integration.conversation_end(conversation_id, 1, 8.5)
    
    logger.info("✅ Conversation simulation completed")


async def simulate_error_scenarios(integration: CodaWebSocketIntegration):
    """Simulate various error scenarios."""
    logger.info("⚠️ Simulating error scenarios...")
    
    await integration.stt_error("Microphone not found", {"device": "default"})
    await asyncio.sleep(0.5)
    
    await integration.llm_error("Model timeout", {"model": "llama3", "timeout": 30})
    await asyncio.sleep(0.5)
    
    await integration.tts_error("Voice not available", {"voice_id": "invalid"})
    await asyncio.sleep(0.5)
    
    await integration.tool_error("weather", "weather-001", "API key missing")
    await asyncio.sleep(0.5)
    
    await integration.system_error("warning", "High memory usage detected", {"memory_mb": 2048})
    
    logger.info("✅ Error simulation completed")


async def simulate_performance_tracking(perf_integration: WebSocketPerfIntegration):
    """Simulate performance tracking."""
    logger.info("📊 Starting performance tracking...")
    
    await perf_integration.start()
    
    # Simulate some operations
    await perf_integration.track_operation("stt", "transcribe", 1500.0, {"model": "whisper"})
    await asyncio.sleep(0.5)
    
    await perf_integration.track_operation("llm", "generate", 2300.0, {"tokens": 45})
    await asyncio.sleep(0.5)
    
    await perf_integration.track_operation("tts", "synthesize", 800.0, {"voice": "default"})
    await asyncio.sleep(0.5)
    
    # Track component timings
    await perf_integration.track_component_timings("memory", {
        "encode": 150.0,
        "search": 75.0,
        "store": 200.0
    })
    
    # Broadcast performance summary
    await perf_integration.broadcast_performance_summary()
    
    logger.info("✅ Performance tracking completed")
    
    await perf_integration.stop()


async def client_connection_handler(client_id: str):
    """Handle client connections."""
    logger.info(f"🔌 Client connected: {client_id}")


async def client_disconnection_handler(client_id: str):
    """Handle client disconnections."""
    logger.info(f"🔌 Client disconnected: {client_id}")


async def main():
    """Main demonstration function."""
    logger.info("🚀 Starting WebSocket demonstration...")
    
    # Create server
    server = CodaWebSocketServer(host="localhost", port=8765)
    
    # Add connection handlers
    server.add_connect_handler(client_connection_handler)
    server.add_disconnect_handler(client_disconnection_handler)
    
    # Create integrations
    integration = CodaWebSocketIntegration(server)
    perf_integration = WebSocketPerfIntegration(server, metrics_interval=3.0)
    
    try:
        # Start server
        await server.start()
        logger.info(f"🌐 WebSocket server running at ws://{server.host}:{server.port}")
        logger.info("💡 Connect with a WebSocket client to see events in real-time")
        logger.info("   Example: wscat -c ws://localhost:8765")
        
        # Send startup event
        await integration.system_startup("2.0.0-alpha", {
            "voice": {"stt": "whisper", "tts": "elevenlabs"},
            "memory": {"enabled": True},
            "tools": {"count": 5}
        })
        
        # Wait a moment for clients to connect
        logger.info("⏳ Waiting 5 seconds for clients to connect...")
        await asyncio.sleep(5)
        
        # Run demonstrations
        await simulate_conversation_flow(integration)
        await asyncio.sleep(2)
        
        await simulate_error_scenarios(integration)
        await asyncio.sleep(2)
        
        await simulate_performance_tracking(perf_integration)
        await asyncio.sleep(2)
        
        # Send shutdown event
        await integration.system_shutdown()
        
        # Show final stats
        stats = server.get_stats()
        logger.info(f"📊 Final server stats: {stats}")
        
        logger.info("✅ Demonstration completed successfully!")
        logger.info("⏳ Server will stop in 5 seconds...")
        await asyncio.sleep(5)
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
    finally:
        await server.stop()
        logger.info("👋 WebSocket server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
