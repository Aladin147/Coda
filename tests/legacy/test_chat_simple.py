#!/usr/bin/env python3
"""
Simple test for the Interactive Chat Interface.

This script tests just the dashboard chat API without WebSocket complexity.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_chat_simple")

# Import components
from coda.interfaces.dashboard.server import CodaDashboardServer
from coda.core.integration import ComponentIntegrationLayer, ComponentType
from coda.core.assistant import CodaAssistant
from coda.core.config import CodaConfig


class SimpleChatTester:
    """Simple test for chat functionality."""
    
    def __init__(self):
        self.dashboard_server = None
        self.integration_layer = None
        self.assistant = None
        
    async def setup(self):
        """Set up test environment."""
        logger.info("🔧 Setting up simple test environment...")
        
        try:
            # Create config with Ollama LLM
            config = CodaConfig()

            # Configure Ollama LLM
            from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
            config.llm = LLMConfig(
                providers={
                    "ollama": ProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model="qwen3:30b-a3b",
                        host="http://localhost:11434",
                        temperature=0.7,
                        max_tokens=100,
                        system_message="/no_think Respond briefly for testing."
                    )
                },
                default_provider="ollama"
            )

            # Create and initialize assistant
            self.assistant = CodaAssistant(config)
            await self.assistant.initialize()

            # Get the integration layer from the assistant
            self.integration_layer = self.assistant.integration_layer

            # Register the assistant itself with the integration layer
            self.integration_layer.register_component(ComponentType.ASSISTANT, self.assistant)

            # Create dashboard server
            self.dashboard_server = CodaDashboardServer(host="localhost", port=8081)
            self.dashboard_server.set_integration_layer(self.integration_layer)
            await self.dashboard_server.start()

            logger.info("✅ Simple test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    async def test_dashboard_chat_api(self):
        """Test the dashboard chat API endpoint."""
        logger.info("🧪 Testing Dashboard Chat API...")
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Test chat message API
                chat_data = {
                    "message": "Hello, this is a simple test message",
                    "session_id": None
                }
                
                async with session.post(
                    "http://localhost:8081/api/chat/message",
                    json=chat_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_text = await response.text()
                    logger.info(f"Response status: {response.status}")
                    logger.info(f"Response text: {response_text}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Chat API response: {result.get('response', 'No response')}")
                        return True
                    else:
                        logger.error(f"❌ Chat API failed with status {response.status}: {response_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Dashboard Chat API test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test environment."""
        logger.info("🧹 Cleaning up test environment...")
        
        try:
            if self.dashboard_server:
                await self.dashboard_server.stop()
            if self.assistant:
                await self.assistant.shutdown()

            logger.info("✅ Cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    async def run_test(self):
        """Run simple chat test."""
        logger.info("🚀 Starting Simple Chat Test")
        logger.info("=" * 40)
        
        # Setup
        if not await self.setup():
            logger.error("❌ Setup failed, aborting test")
            return False
        
        # Wait a moment for server to be ready
        await asyncio.sleep(1)
        
        # Run test
        success = await self.test_dashboard_chat_api()
        
        # Cleanup
        await self.cleanup()
        
        if success:
            logger.info("🎉 SIMPLE CHAT TEST PASSED!")
        else:
            logger.error("❌ SIMPLE CHAT TEST FAILED")
        
        return success


async def main():
    """Main test function."""
    tester = SimpleChatTester()
    success = await tester.run_test()
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
