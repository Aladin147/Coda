#!/usr/bin/env python3
"""
Test script for the Interactive Chat Interface.

This script validates that the chat interface works correctly with:
1. WebSocket message handling
2. Chat API endpoints
3. Real-time communication
4. Session management
"""

import asyncio
import json
import logging
import sys
import time
import websockets
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_interactive_chat")

# Import components
from coda.interfaces.dashboard.server import CodaDashboardServer
from coda.interfaces.websocket.server import CodaWebSocketServer
from coda.core.websocket_integration import ComponentWebSocketIntegration
from coda.core.integration import ComponentIntegrationLayer


class InteractiveChatTester:
    """Test the interactive chat interface functionality."""
    
    def __init__(self):
        self.dashboard_server = None
        self.websocket_server = None
        self.websocket_integration = None
        self.integration_layer = None
        self.test_results = []
        
    async def setup(self):
        """Set up test environment."""
        logger.info("🔧 Setting up test environment...")
        
        try:
            # Create integration layer
            self.integration_layer = ComponentIntegrationLayer()
            
            # Create WebSocket server
            self.websocket_server = CodaWebSocketServer(host="localhost", port=8765)
            await self.websocket_server.start()
            
            # Create WebSocket integration
            self.websocket_integration = ComponentWebSocketIntegration(self.integration_layer)
            self.websocket_integration.set_websocket_server(self.websocket_server)
            await self.websocket_integration.start()
            
            # Create dashboard server
            self.dashboard_server = CodaDashboardServer(host="localhost", port=8081)
            self.dashboard_server.set_integration_layer(self.integration_layer)
            self.dashboard_server.set_websocket_integration(self.websocket_integration)
            await self.dashboard_server.start()
            
            logger.info("✅ Test environment setup complete")
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
                    "message": "Hello, this is a test message",
                    "session_id": None
                }
                
                async with session.post(
                    "http://localhost:8081/api/chat/message",
                    json=chat_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Chat API response: {result.get('response', 'No response')[:50]}...")
                        self.test_results.append(("Dashboard Chat API", True, "API responded successfully"))
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Chat API failed with status {response.status}: {error_text}")
                        self.test_results.append(("Dashboard Chat API", False, f"Status {response.status}"))
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Dashboard Chat API test failed: {e}")
            self.test_results.append(("Dashboard Chat API", False, str(e)))
            return False
    
    async def test_websocket_chat(self):
        """Test WebSocket chat functionality."""
        logger.info("🧪 Testing WebSocket Chat...")
        
        try:
            # Connect to WebSocket
            uri = "ws://localhost:8765"
            async with websockets.connect(uri) as websocket:
                logger.info("✅ Connected to WebSocket server")
                
                # Send a chat message
                chat_message = {
                    "type": "chat_message",
                    "data": {
                        "message": "Hello via WebSocket!",
                        "session_id": None
                    },
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(chat_message))
                logger.info("📤 Sent chat message via WebSocket")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") == "chat_response":
                        logger.info(f"✅ Received chat response: {response_data.get('data', {}).get('response', {}).get('content', 'No content')[:50]}...")
                        self.test_results.append(("WebSocket Chat", True, "Chat response received"))
                        return True
                    else:
                        logger.warning(f"⚠️ Unexpected response type: {response_data.get('type')}")
                        self.test_results.append(("WebSocket Chat", False, f"Unexpected response: {response_data.get('type')}"))
                        return False
                        
                except asyncio.TimeoutError:
                    logger.error("❌ Timeout waiting for WebSocket chat response")
                    self.test_results.append(("WebSocket Chat", False, "Timeout waiting for response"))
                    return False
                    
        except Exception as e:
            logger.error(f"❌ WebSocket chat test failed: {e}")
            self.test_results.append(("WebSocket Chat", False, str(e)))
            return False
    
    async def test_dashboard_accessibility(self):
        """Test that the dashboard is accessible."""
        logger.info("🧪 Testing Dashboard Accessibility...")
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8081/",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        if "Interactive Chat" in content:
                            logger.info("✅ Dashboard accessible with chat interface")
                            self.test_results.append(("Dashboard Accessibility", True, "Dashboard loaded with chat interface"))
                            return True
                        else:
                            logger.warning("⚠️ Dashboard accessible but chat interface not found")
                            self.test_results.append(("Dashboard Accessibility", False, "Chat interface not found in HTML"))
                            return False
                    else:
                        logger.error(f"❌ Dashboard not accessible: status {response.status}")
                        self.test_results.append(("Dashboard Accessibility", False, f"Status {response.status}"))
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Dashboard accessibility test failed: {e}")
            self.test_results.append(("Dashboard Accessibility", False, str(e)))
            return False
    
    async def cleanup(self):
        """Clean up test environment."""
        logger.info("🧹 Cleaning up test environment...")
        
        try:
            if self.dashboard_server:
                await self.dashboard_server.stop()
            if self.websocket_integration:
                await self.websocket_integration.stop()
            if self.websocket_server:
                await self.websocket_server.stop()
                
            logger.info("✅ Cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    async def run_tests(self):
        """Run all interactive chat tests."""
        logger.info("🚀 Starting Interactive Chat Interface Tests")
        logger.info("=" * 60)
        
        # Setup
        if not await self.setup():
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        # Wait a moment for servers to be ready
        await asyncio.sleep(2)
        
        # Run tests
        tests = [
            ("Dashboard Accessibility", self.test_dashboard_accessibility),
            ("Dashboard Chat API", self.test_dashboard_chat_api),
            ("WebSocket Chat", self.test_websocket_chat),
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*30}")
            logger.info(f"Running: {test_name}")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"❌ {test_name} failed with exception: {e}")
                self.test_results.append((test_name, False, str(e)))
        
        # Cleanup
        await self.cleanup()
        
        # Results
        self.print_results()
        
        # Return overall success
        return all(result[1] for result in self.test_results)
    
    def print_results(self):
        """Print test results summary."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 INTERACTIVE CHAT INTERFACE TEST RESULTS")
        logger.info("=" * 60)
        
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        total_tests = len(self.test_results)
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        for test_name, passed, details in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {status} {test_name}: {details}")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL INTERACTIVE CHAT TESTS PASSED!")
        else:
            logger.error(f"\n❌ {total_tests - passed_tests} TESTS FAILED")


async def main():
    """Main test function."""
    tester = InteractiveChatTester()
    success = await tester.run_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
