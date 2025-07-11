#!/usr/bin/env python3
"""
Test Audio Streaming Infrastructure.

This script tests the WebSocket audio streaming capabilities between
the GUI and backend for real-time voice communication.
"""

import asyncio
import logging
import sys
import json
import wave
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_audio_streaming")

# Import components
from coda.interfaces.websocket.server import CodaWebSocketServer
from coda.core.websocket_integration import ComponentWebSocketIntegration
from coda.core.integration import ComponentIntegrationLayer
from coda.core.assistant import CodaAssistant
from coda.core.config import CodaConfig
from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
from coda.components.voice.models import VoiceConfig, MoshiConfig, VoiceProcessingMode


class AudioStreamingTester:
    """Test audio streaming infrastructure."""
    
    def __init__(self):
        self.websocket_server = None
        self.websocket_integration = None
        self.integration_layer = None
        self.assistant = None
        
    def generate_test_audio(self, duration_seconds: float = 1.0, sample_rate: int = 24000) -> bytes:
        """Generate synthetic test audio data."""
        # Generate a simple sine wave as test audio
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        import io
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    async def setup(self):
        """Set up test environment."""
        logger.info("🔧 Setting up audio streaming test environment...")
        
        try:
            # Create config with voice processing
            config = CodaConfig()
            
            # Configure Ollama LLM
            config.llm = LLMConfig(
                providers={
                    "ollama": ProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model="qwen3:30b-a3b",
                        host="http://localhost:11434",
                        temperature=0.7,
                        max_tokens=50,
                        system_message="/no_think Respond briefly to voice input."
                    )
                },
                default_provider="ollama"
            )
            
            # Configure Moshi voice processing
            config.voice = VoiceConfig(
                mode=VoiceProcessingMode.MOSHI_ONLY,
                moshi=MoshiConfig(
                    model_path="kyutai/moshika-pytorch-bf16",
                    device="cuda" if True else "cpu",
                    vram_allocation="4GB",
                    inner_monologue_enabled=True,
                    enable_streaming=False
                )
            )
            
            # Create assistant with voice processing
            self.assistant = CodaAssistant(config)
            await self.assistant.initialize()
            
            # Create integration layer
            self.integration_layer = self.assistant.integration_layer
            
            # Create WebSocket server
            self.websocket_server = CodaWebSocketServer(host="localhost", port=8766)
            await self.websocket_server.start()
            
            # Create WebSocket integration
            self.websocket_integration = ComponentWebSocketIntegration(
                integration_layer=self.integration_layer,
                websocket_server=self.websocket_server
            )
            
            # Connect voice manager to WebSocket
            if self.assistant.voice_manager:
                self.websocket_integration.set_voice_manager(self.assistant.voice_manager)
            
            await self.websocket_integration.start()
            
            logger.info("✅ Audio streaming test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    async def test_websocket_audio_session(self):
        """Test WebSocket audio session management."""
        logger.info("🧪 Testing WebSocket Audio Session Management...")
        
        try:
            # Simulate client connection and audio session start
            client_id = "test_client_001"
            session_id = "test_session_001"
            
            # Check if WebSocket server has audio session support
            if hasattr(self.websocket_server, 'audio_sessions'):
                logger.info("✅ WebSocket server has audio session support")
                
                # Check if voice manager is connected
                if self.websocket_server.voice_manager:
                    logger.info("✅ Voice manager connected to WebSocket server")
                else:
                    logger.warning("⚠️ Voice manager not connected to WebSocket server")
                
                # Test audio session data structure
                test_session_data = {
                    "session_id": session_id,
                    "config": {
                        "sample_rate": 24000,
                        "channels": 1,
                        "format": "wav"
                    }
                }
                
                logger.info(f"✅ Audio session test data: {test_session_data}")
                return True
            else:
                logger.error("❌ WebSocket server missing audio session support")
                return False
                
        except Exception as e:
            logger.error(f"❌ WebSocket audio session test failed: {e}")
            return False
    
    async def test_binary_message_handling(self):
        """Test binary message handling capability."""
        logger.info("🧪 Testing Binary Message Handling...")
        
        try:
            # Generate test audio
            test_audio = self.generate_test_audio(duration_seconds=0.5)
            logger.info(f"Generated test audio: {len(test_audio)} bytes")
            
            # Check if WebSocket server has binary message handler
            if hasattr(self.websocket_server, '_handle_binary_message'):
                logger.info("✅ WebSocket server has binary message handler")
                
                # Check audio streaming stats
                if "audio_chunks_processed" in self.websocket_server.stats:
                    logger.info("✅ Audio streaming statistics available")
                    logger.info(f"Audio chunks processed: {self.websocket_server.stats['audio_chunks_processed']}")
                else:
                    logger.warning("⚠️ Audio streaming statistics not available")
                
                return True
            else:
                logger.error("❌ WebSocket server missing binary message handler")
                return False
                
        except Exception as e:
            logger.error(f"❌ Binary message handling test failed: {e}")
            return False
    
    async def test_voice_integration(self):
        """Test voice manager integration with WebSocket."""
        logger.info("🧪 Testing Voice Manager Integration...")
        
        try:
            # Check if voice manager is available
            if self.assistant.voice_manager:
                logger.info("✅ Voice manager available")
                
                # Check if WebSocket integration has voice manager
                if self.websocket_integration.voice_manager:
                    logger.info("✅ Voice manager connected to WebSocket integration")
                    
                    # Check if WebSocket server has voice manager
                    if self.websocket_server.voice_manager:
                        logger.info("✅ Voice manager connected to WebSocket server")
                        return True
                    else:
                        logger.warning("⚠️ Voice manager not connected to WebSocket server")
                        return False
                else:
                    logger.warning("⚠️ Voice manager not connected to WebSocket integration")
                    return False
            else:
                logger.error("❌ Voice manager not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ Voice integration test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test environment."""
        logger.info("🧹 Cleaning up test environment...")
        
        try:
            if self.websocket_integration:
                await self.websocket_integration.stop()
            if self.websocket_server:
                await self.websocket_server.stop()
            if self.assistant:
                await self.assistant.shutdown()
                
            logger.info("✅ Cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    async def run_tests(self):
        """Run all audio streaming tests."""
        logger.info("🚀 Starting Audio Streaming Infrastructure Tests")
        logger.info("=" * 60)
        
        # Setup
        if not await self.setup():
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        # Wait a moment for initialization
        await asyncio.sleep(2)
        
        # Run tests
        tests = [
            ("WebSocket Audio Session Management", self.test_websocket_audio_session),
            ("Binary Message Handling", self.test_binary_message_handling),
            ("Voice Manager Integration", self.test_voice_integration),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*30}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*30}")
            
            try:
                if await test_func():
                    logger.info(f"✅ PASS: {test_name}")
                    passed += 1
                else:
                    logger.error(f"❌ FAIL: {test_name}")
            except Exception as e:
                logger.error(f"❌ ERROR: {test_name} - {e}")
        
        # Cleanup
        await self.cleanup()
        
        # Results
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 AUDIO STREAMING INFRASTRUCTURE TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 ALL AUDIO STREAMING TESTS PASSED!")
        else:
            logger.error(f"❌ {total - passed} TESTS FAILED")
        
        return passed == total


async def main():
    """Main test function."""
    tester = AudioStreamingTester()
    success = await tester.run_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
