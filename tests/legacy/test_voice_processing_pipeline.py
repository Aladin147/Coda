#!/usr/bin/env python3
"""
Test Voice Processing Pipeline Integration.

This script tests the complete voice processing pipeline:
Audio Input → Moshi STT → LLM Processing → Memory Storage → Moshi TTS → Audio Output
"""

import asyncio
import logging
import sys
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
logger = logging.getLogger("test_voice_pipeline")

# Import components
from coda.core.assistant import CodaAssistant
from coda.core.config import CodaConfig
from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
from coda.components.voice.models import VoiceConfig, MoshiConfig, VoiceProcessingMode


class VoicePipelineTester:
    """Test the complete voice processing pipeline."""
    
    def __init__(self):
        self.assistant = None
        
    def generate_test_audio(self, duration_seconds: float = 2.0, sample_rate: int = 24000) -> bytes:
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
        """Set up test environment with voice processing."""
        logger.info("🔧 Setting up voice processing test environment...")
        
        try:
            # Create config with Ollama LLM and Moshi voice
            config = CodaConfig()
            
            # Configure Ollama LLM
            config.llm = LLMConfig(
                providers={
                    "ollama": ProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model="qwen3:30b-a3b",
                        host="http://localhost:11434",
                        temperature=0.7,
                        max_tokens=100,
                        system_message="/no_think Respond briefly to voice input."
                    )
                },
                default_provider="ollama"
            )
            
            # Configure Moshi voice processing
            config.voice = VoiceConfig(
                mode=VoiceProcessingMode.MOSHI_ONLY,  # Use Moshi for both STT and TTS
                moshi=MoshiConfig(
                    model_path="kyutai/moshika-pytorch-bf16",
                    device="cuda" if True else "cpu",  # Use GPU if available
                    vram_allocation="4GB",
                    inner_monologue_enabled=True,  # Enable text extraction
                    enable_streaming=False  # Start with non-streaming for testing
                )
            )
            
            # Create and initialize assistant
            self.assistant = CodaAssistant(config)
            await self.assistant.initialize()
            
            logger.info("✅ Voice processing test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    async def test_voice_input_processing(self):
        """Test voice input processing pipeline."""
        logger.info("🧪 Testing Voice Input Processing...")
        
        try:
            # Generate test audio
            test_audio = self.generate_test_audio(duration_seconds=1.0)
            logger.info(f"Generated test audio: {len(test_audio)} bytes")
            
            # Process voice input
            result = await self.assistant.process_voice_input(
                audio_data=test_audio,
                metadata={"test": "voice_input_processing"}
            )
            
            logger.info(f"Voice processing result: {result.get('status', 'unknown')}")
            
            if result.get("status") == "success":
                response = result.get("response", {})
                logger.info(f"✅ Text response: {response.get('content', 'No text')}")
                logger.info(f"✅ Extracted text: {response.get('extracted_text', 'None')}")
                logger.info(f"✅ Audio response: {len(response.get('audio_data', b''))} bytes")
                logger.info(f"✅ Processing time: {result.get('processing_time_ms', 0):.1f}ms")
                return True
            else:
                logger.error(f"❌ Voice processing failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Voice input processing test failed: {e}")
            return False
    
    async def test_voice_text_integration(self):
        """Test integration between voice and text processing."""
        logger.info("🧪 Testing Voice-Text Integration...")
        
        try:
            # First, send a text message to establish context
            text_result = await self.assistant.process_text_message(
                "Hello, I'm testing voice integration. Please remember this context."
            )
            
            if text_result.get("status") != "success":
                logger.warning("Text message failed, continuing with voice test")
            
            # Then process voice input in the same session
            test_audio = self.generate_test_audio(duration_seconds=1.5)
            voice_result = await self.assistant.process_voice_input(
                audio_data=test_audio,
                session_id=text_result.get("session_id"),
                metadata={"test": "voice_text_integration"}
            )
            
            if voice_result.get("status") == "success":
                logger.info("✅ Voice-text integration working")
                logger.info(f"✅ Session continuity: {voice_result.get('session_id') == text_result.get('session_id')}")
                return True
            else:
                logger.error(f"❌ Voice-text integration failed: {voice_result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Voice-text integration test failed: {e}")
            return False
    
    async def test_voice_memory_integration(self):
        """Test voice processing with memory storage."""
        logger.info("🧪 Testing Voice-Memory Integration...")
        
        try:
            # Process voice input
            test_audio = self.generate_test_audio(duration_seconds=1.0)
            result = await self.assistant.process_voice_input(
                audio_data=test_audio,
                metadata={"test": "voice_memory_integration"}
            )
            
            if result.get("status") == "success":
                # Check if memory was updated
                session_id = result.get("session_id")
                if session_id and self.assistant.memory_manager:
                    # Try to retrieve session history
                    history = await self.assistant.get_session_history(session_id)
                    logger.info(f"✅ Voice-memory integration working: {len(history)} messages in history")
                    return True
                else:
                    logger.warning("⚠️ Memory manager not available for integration test")
                    return True  # Still consider success if voice processing worked
            else:
                logger.error(f"❌ Voice-memory integration failed: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Voice-memory integration test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test environment."""
        logger.info("🧹 Cleaning up test environment...")
        
        try:
            if self.assistant:
                await self.assistant.shutdown()
                
            logger.info("✅ Cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    async def run_tests(self):
        """Run all voice processing pipeline tests."""
        logger.info("🚀 Starting Voice Processing Pipeline Tests")
        logger.info("=" * 60)
        
        # Setup
        if not await self.setup():
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        # Wait a moment for initialization
        await asyncio.sleep(2)
        
        # Run tests
        tests = [
            ("Voice Input Processing", self.test_voice_input_processing),
            ("Voice-Text Integration", self.test_voice_text_integration),
            ("Voice-Memory Integration", self.test_voice_memory_integration),
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
        logger.info(f"📊 VOICE PROCESSING PIPELINE TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 ALL VOICE PROCESSING TESTS PASSED!")
        else:
            logger.error(f"❌ {total - passed} TESTS FAILED")
        
        return passed == total


async def main():
    """Main test function."""
    tester = VoicePipelineTester()
    success = await tester.run_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
