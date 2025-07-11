#!/usr/bin/env python3
"""
Basic Voice Chat Example

This example demonstrates how to create a simple voice chat application
using Coda's voice processing capabilities.

Features demonstrated:
- Voice input processing
- Real-time speech-to-speech conversation
- Multiple processing modes
- Basic conversation management

Usage:
    python examples/basic_voice_chat.py [--mode moshi-only|hybrid|adaptive]
"""

import asyncio
import logging
import argparse
import wave
import io
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice import (
        VoiceManager, VoiceConfig, VoiceMessage, VoiceProcessingMode
    )
    from src.coda.components.voice.config import AudioConfig, MoshiConfig
except ImportError as e:
    logger.error(f"Failed to import voice components: {e}")
    logger.error("Please ensure voice dependencies are installed: pip install -e '.[voice]'")
    exit(1)


class BasicVoiceChat:
    """Simple voice chat application using Coda's voice processing."""
    
    def __init__(self, processing_mode: VoiceProcessingMode = VoiceProcessingMode.ADAPTIVE):
        self.processing_mode = processing_mode
        self.voice_manager: Optional[VoiceManager] = None
        self.conversation_id: Optional[str] = None
        self.is_running = False
    
    async def initialize(self) -> None:
        """Initialize the voice chat system."""
        logger.info("🎙️ Initializing voice chat system...")
        
        try:
            # Configure voice system
            config = VoiceConfig(
                audio=AudioConfig(
                    sample_rate=24000,
                    channels=1,
                    format="wav",
                    enable_vad=True,
                    enable_noise_reduction=True
                ),
                moshi=MoshiConfig(
                    device="cuda" if self._check_gpu_available() else "cpu",
                    model_size="small"
                ),
                default_mode=self.processing_mode,
                enable_streaming=True,
                enable_memory_integration=False,  # Keep it simple for this example
                enable_personality_integration=False,
                enable_tools_integration=False
            )
            
            # Initialize voice manager
            self.voice_manager = VoiceManager(config)
            await self.voice_manager.initialize()
            
            # Start conversation
            self.conversation_id = await self.voice_manager.start_conversation()
            
            logger.info(f"✅ Voice chat initialized (mode: {self.processing_mode})")
            logger.info(f"📞 Conversation started: {self.conversation_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize voice chat: {e}")
            raise
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for voice processing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def start_chat(self) -> None:
        """Start the voice chat loop."""
        if not self.voice_manager or not self.conversation_id:
            raise RuntimeError("Voice chat not initialized. Call initialize() first.")
        
        self.is_running = True
        logger.info("\n🎉 Voice chat started!")
        logger.info("💡 Tips:")
        logger.info("   - Speak clearly into your microphone")
        logger.info("   - Say 'exit' or 'quit' to end the conversation")
        logger.info("   - Press Ctrl+C to force quit")
        logger.info("\n🎤 Listening for your voice input...")
        
        try:
            while self.is_running:
                # In a real implementation, you would capture audio from microphone
                # For this example, we'll simulate with text input
                user_input = await self._get_user_input()
                
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    logger.info("👋 Ending conversation...")
                    break
                
                # Process voice input (simulated)
                response = await self._process_voice_input(user_input)
                
                # Display response
                logger.info(f"🤖 Coda: {response}")
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Chat interrupted by user")
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
        finally:
            await self._cleanup()
    
    async def _get_user_input(self) -> str:
        """Get user input (simulated voice input for this example)."""
        # In a real implementation, this would capture audio from microphone
        # For demo purposes, we'll use text input
        print("\n🎤 You (speak or type): ", end="", flush=True)
        
        # Simulate async input
        loop = asyncio.get_event_loop()
        user_input = await loop.run_in_executor(None, input)
        
        return user_input.strip()
    
    async def _process_voice_input(self, text_input: str) -> str:
        """Process voice input and return response."""
        try:
            # Create simulated audio data (in real app, this would be actual audio)
            audio_data = self._text_to_mock_audio(text_input)
            
            # Process through voice manager
            response = await self.voice_manager.process_voice_input(
                conversation_id=self.conversation_id,
                audio_data=audio_data
            )
            
            # In a real implementation, you would also play the audio response
            # For this example, we'll just return the text
            return response.text_content
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return "I'm sorry, I had trouble processing your voice input. Could you try again?"
    
    def _text_to_mock_audio(self, text: str) -> bytes:
        """Create mock audio data for demonstration purposes."""
        # In a real implementation, this would be actual audio data from microphone
        # For demo, we create a simple WAV header with minimal data
        
        sample_rate = 24000
        duration = len(text) * 0.1  # Rough estimate
        frames = int(sample_rate * duration)
        
        # Create a simple WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Write silence (in real app, this would be actual audio)
            silence = b'\x00\x00' * frames
            wav_file.writeframes(silence)
        
        return buffer.getvalue()
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("🧹 Cleaning up...")
        
        if self.voice_manager and self.conversation_id:
            try:
                await self.voice_manager.end_conversation(self.conversation_id)
            except Exception as e:
                logger.warning(f"Error ending conversation: {e}")
        
        if self.voice_manager:
            try:
                await self.voice_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up voice manager: {e}")
        
        logger.info("✅ Cleanup completed")


async def main():
    """Main function to run the voice chat example."""
    parser = argparse.ArgumentParser(description="Basic Voice Chat Example")
    parser.add_argument(
        "--mode",
        choices=["moshi-only", "hybrid", "adaptive"],
        default="adaptive",
        help="Voice processing mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Map string to enum
    mode_map = {
        "moshi-only": VoiceProcessingMode.MOSHI_ONLY,
        "hybrid": VoiceProcessingMode.HYBRID,
        "adaptive": VoiceProcessingMode.ADAPTIVE
    }
    
    processing_mode = mode_map[args.mode]
    
    # Create and run voice chat
    chat = BasicVoiceChat(processing_mode)
    
    try:
        await chat.initialize()
        await chat.start_chat()
    except Exception as e:
        logger.error(f"Voice chat failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
