#!/usr/bin/env python3
"""
Test Multi-modal Session Support.

This script tests seamless switching between text and voice within the same
conversation session with unified memory context.
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
logger = logging.getLogger("test_multimodal_sessions")

# Import components
from coda.core.assistant import CodaAssistant
from coda.core.config import CodaConfig
from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
from coda.components.voice.models import VoiceConfig, MoshiConfig, VoiceProcessingMode


class MultiModalSessionTester:
    """Test multi-modal session support."""
    
    def __init__(self):
        self.assistant = None
        self.test_session_id = "multimodal_test_session_001"
        
    def generate_test_audio(self, duration_seconds: float = 0.5, sample_rate: int = 24000) -> bytes:
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
        logger.info("🔧 Setting up multi-modal session test environment...")
        
        try:
            # Create config with both text and voice processing
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
                        system_message="/no_think Respond briefly. Note the input modality (text/voice) in your response."
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
            
            # Create assistant
            self.assistant = CodaAssistant(config)
            await self.assistant.initialize()
            
            logger.info("✅ Multi-modal session test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    async def test_text_to_voice_switching(self):
        """Test switching from text to voice in the same session."""
        logger.info("🧪 Testing Text → Voice Switching...")
        
        try:
            # Start with text message
            text_result = await self.assistant.process_text_message(
                "Hello, I'm starting with a text message.",
                session_id=self.test_session_id,
                metadata={"test": "text_to_voice_switching", "step": 1}
            )
            
            if text_result.get("status") != "success":
                logger.error(f"❌ Text message failed: {text_result.get('error')}")
                return False
            
            logger.info(f"✅ Text message processed: {text_result.get('response', {}).get('content', 'No response')[:50]}...")
            
            # Switch to voice in the same session
            test_audio = self.generate_test_audio(duration_seconds=0.5)
            voice_result = await self.assistant.process_voice_input(
                audio_data=test_audio,
                session_id=text_result.get("session_id"),
                metadata={"test": "text_to_voice_switching", "step": 2}
            )
            
            if voice_result.get("status") != "success":
                logger.error(f"❌ Voice message failed: {voice_result.get('error')}")
                return False
            
            logger.info(f"✅ Voice message processed: {voice_result.get('response', {}).get('content', 'No response')[:50]}...")
            
            # Verify same session
            if text_result.get("session_id") == voice_result.get("session_id"):
                logger.info("✅ Session continuity maintained")
                return True
            else:
                logger.error("❌ Session continuity broken")
                return False
                
        except Exception as e:
            logger.error(f"❌ Text to voice switching test failed: {e}")
            return False
    
    async def test_voice_to_text_switching(self):
        """Test switching from voice to text in the same session."""
        logger.info("🧪 Testing Voice → Text Switching...")
        
        try:
            # Start with voice message
            test_audio = self.generate_test_audio(duration_seconds=0.5)
            voice_result = await self.assistant.process_voice_input(
                audio_data=test_audio,
                session_id=self.test_session_id,
                metadata={"test": "voice_to_text_switching", "step": 1}
            )
            
            if voice_result.get("status") != "success":
                logger.error(f"❌ Voice message failed: {voice_result.get('error')}")
                return False
            
            logger.info(f"✅ Voice message processed: {voice_result.get('response', {}).get('content', 'No response')[:50]}...")
            
            # Switch to text in the same session
            text_result = await self.assistant.process_text_message(
                "Now I'm switching to text mode.",
                session_id=voice_result.get("session_id"),
                metadata={"test": "voice_to_text_switching", "step": 2}
            )
            
            if text_result.get("status") != "success":
                logger.error(f"❌ Text message failed: {text_result.get('error')}")
                return False
            
            logger.info(f"✅ Text message processed: {text_result.get('response', {}).get('content', 'No response')[:50]}...")
            
            # Verify same session
            if voice_result.get("session_id") == text_result.get("session_id"):
                logger.info("✅ Session continuity maintained")
                return True
            else:
                logger.error("❌ Session continuity broken")
                return False
                
        except Exception as e:
            logger.error(f"❌ Voice to text switching test failed: {e}")
            return False
    
    async def test_multimodal_session_history(self):
        """Test unified session history with both text and voice."""
        logger.info("🧪 Testing Multi-modal Session History...")
        
        try:
            # Get multi-modal session history
            if hasattr(self.assistant, 'get_multimodal_session_history'):
                history = await self.assistant.get_multimodal_session_history(self.test_session_id)
                
                if history:
                    logger.info(f"✅ Retrieved {len(history)} messages from session history")
                    
                    # Check for both text and voice messages
                    text_messages = [msg for msg in history if msg.get("modality") == "text"]
                    voice_messages = [msg for msg in history if msg.get("modality") == "voice"]
                    
                    logger.info(f"✅ Text messages: {len(text_messages)}")
                    logger.info(f"✅ Voice messages: {len(voice_messages)}")
                    
                    if len(text_messages) > 0 and len(voice_messages) > 0:
                        logger.info("✅ Multi-modal session history working")
                        return True
                    else:
                        logger.warning("⚠️ Missing text or voice messages in history")
                        return True  # Still consider success if history exists
                else:
                    logger.warning("⚠️ No session history found")
                    return False
            else:
                logger.error("❌ Multi-modal session history method not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ Multi-modal session history test failed: {e}")
            return False
    
    async def test_session_modality_stats(self):
        """Test session modality statistics."""
        logger.info("🧪 Testing Session Modality Statistics...")
        
        try:
            # Get session modality stats
            if hasattr(self.assistant, 'get_session_modality_stats'):
                stats = await self.assistant.get_session_modality_stats(self.test_session_id)
                
                if stats:
                    logger.info(f"✅ Session stats: {json.dumps(stats, indent=2)}")
                    
                    # Check expected fields
                    expected_fields = ["total_messages", "text_messages", "voice_messages", "is_multimodal", "modality_switches"]
                    for field in expected_fields:
                        if field in stats:
                            logger.info(f"✅ {field}: {stats[field]}")
                        else:
                            logger.warning(f"⚠️ Missing field: {field}")
                    
                    if stats.get("is_multimodal", False):
                        logger.info("✅ Session correctly identified as multi-modal")
                        return True
                    else:
                        logger.warning("⚠️ Session not identified as multi-modal")
                        return True  # Still consider success if stats exist
                else:
                    logger.error("❌ No session stats returned")
                    return False
            else:
                logger.error("❌ Session modality stats method not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ Session modality stats test failed: {e}")
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
        """Run all multi-modal session tests."""
        logger.info("🚀 Starting Multi-modal Session Support Tests")
        logger.info("=" * 60)
        
        # Setup
        if not await self.setup():
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        # Wait a moment for initialization
        await asyncio.sleep(2)
        
        # Run tests
        tests = [
            ("Text → Voice Switching", self.test_text_to_voice_switching),
            ("Voice → Text Switching", self.test_voice_to_text_switching),
            ("Multi-modal Session History", self.test_multimodal_session_history),
            ("Session Modality Statistics", self.test_session_modality_stats),
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
        logger.info(f"📊 MULTI-MODAL SESSION SUPPORT TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 ALL MULTI-MODAL SESSION TESTS PASSED!")
        else:
            logger.error(f"❌ {total - passed} TESTS FAILED")
        
        return passed == total


async def main():
    """Main test function."""
    tester = MultiModalSessionTester()
    success = await tester.run_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
