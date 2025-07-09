#!/usr/bin/env python3
"""
Test Voice-LLM Integration

This script tests the integration between voice processing and external LLMs.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice.llm_integration import VoiceLLMProcessor, VoiceLLMConfig
    from src.coda.components.voice.models import VoiceMessage, VoiceProcessingMode
    from src.coda.components.llm.models import LLMProvider
    logger.info("✓ Successfully imported voice-LLM integration components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_config() -> VoiceLLMConfig:
    """Create test configuration for voice-LLM integration."""
    return VoiceLLMConfig(
        llm_provider=LLMProvider.OLLAMA,
        llm_model="gemma3:1b",  # Use fast model for testing
        llm_temperature=0.7,
        llm_max_tokens=256,
        enable_streaming=True,
        llm_timeout_seconds=10.0,
        fallback_to_moshi=True
    )


def create_test_voice_message(conversation_id: str, text_content: str) -> VoiceMessage:
    """Create a test voice message."""
    # Create dummy audio data
    audio_data = np.random.randint(-32768, 32767, 24000, dtype=np.int16).tobytes()
    
    return VoiceMessage(
        message_id=f"test_msg_{datetime.now().strftime('%H%M%S')}",
        conversation_id=conversation_id,
        audio_data=audio_data,
        text_content=text_content,  # Simulated from Moshi inner monologue
        processing_mode=VoiceProcessingMode.HYBRID,
        timestamp=datetime.now()
    )


async def test_voice_llm_initialization():
    """Test voice-LLM processor initialization."""
    logger.info("=== Testing Voice-LLM Initialization ===")
    
    try:
        config = create_test_config()
        processor = VoiceLLMProcessor(config)
        logger.info("✓ VoiceLLMProcessor created")
        
        await processor.initialize()
        logger.info("✓ VoiceLLMProcessor initialized successfully")
        
        await processor.cleanup()
        logger.info("✓ VoiceLLMProcessor cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return False


async def test_voice_llm_processing():
    """Test voice message processing with LLM."""
    logger.info("=== Testing Voice-LLM Processing ===")
    
    try:
        config = create_test_config()
        processor = VoiceLLMProcessor(config)
        await processor.initialize()
        
        # Test simple conversation
        conversation_id = "test_conversation_llm_001"
        
        # Test message 1
        voice_message1 = create_test_voice_message(
            conversation_id,
            "Hello! What's the weather like today?"
        )
        
        logger.info("Processing first voice message...")
        response1 = await processor.process_voice_message(voice_message1)
        logger.info(f"✓ First response: {response1.text_content[:100]}...")
        logger.info(f"  Latency: {response1.total_latency_ms:.1f}ms")
        logger.info(f"  Processing mode: {response1.processing_mode}")

        # Test message 2 (with context)
        voice_message2 = create_test_voice_message(
            conversation_id,
            "Can you tell me a short joke?"
        )

        logger.info("Processing second voice message...")
        response2 = await processor.process_voice_message(voice_message2)
        logger.info(f"✓ Second response: {response2.text_content[:100]}...")
        logger.info(f"  Latency: {response2.total_latency_ms:.1f}ms")
        
        # Test performance metrics
        metrics = processor.get_performance_metrics()
        logger.info(f"✓ Performance metrics: {metrics}")
        
        await processor.cleanup()
        logger.info("✓ Voice-LLM processing test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_voice_llm_streaming():
    """Test streaming voice-LLM processing."""
    logger.info("=== Testing Voice-LLM Streaming ===")
    
    try:
        config = create_test_config()
        config.enable_streaming = True
        
        processor = VoiceLLMProcessor(config)
        await processor.initialize()
        
        conversation_id = "test_conversation_streaming"
        voice_message = create_test_voice_message(
            conversation_id,
            "Please write a very short story about a robot learning to paint."
        )
        
        logger.info("Processing streaming voice message...")
        start_time = asyncio.get_event_loop().time()
        
        response = await processor.process_voice_message(voice_message)
        
        end_time = asyncio.get_event_loop().time()
        total_time = (end_time - start_time) * 1000
        
        logger.info(f"✓ Streaming response: {response.text_content[:150]}...")
        logger.info(f"  Total time: {total_time:.1f}ms")
        logger.info(f"  Reported latency: {response.total_latency_ms:.1f}ms")
        logger.info(f"  Processing mode: {response.processing_mode}")
        
        await processor.cleanup()
        logger.info("✓ Streaming test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_voice_llm_fallback():
    """Test fallback behavior when LLM fails."""
    logger.info("=== Testing Voice-LLM Fallback ===")
    
    try:
        config = create_test_config()
        config.llm_timeout_seconds = 0.1  # Very short timeout to trigger fallback
        
        processor = VoiceLLMProcessor(config)
        await processor.initialize()
        
        conversation_id = "test_conversation_fallback"
        voice_message = create_test_voice_message(
            conversation_id,
            ""  # Empty text content should trigger fallback
        )
        
        logger.info("Processing message with empty text (should fallback)...")
        response = await processor.process_voice_message(voice_message)
        
        logger.info(f"✓ Fallback response: {response.text_content}")
        logger.info(f"  Processing mode: {response.processing_mode}")
        logger.info(f"  Response relevance: {response.response_relevance}")
        
        await processor.cleanup()
        logger.info("✓ Fallback test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback test failed: {e}")
        return False


async def test_voice_llm_context_injection():
    """Test context injection (personality, memory)."""
    logger.info("=== Testing Voice-LLM Context Injection ===")
    
    try:
        config = create_test_config()
        config.personality_injection = True
        config.memory_injection = True
        
        processor = VoiceLLMProcessor(config)
        await processor.initialize()
        
        conversation_id = "test_conversation_context"
        voice_message = create_test_voice_message(
            conversation_id,
            "What should I know about you?"
        )
        
        # Mock context data
        context = {
            "personality": {
                "description": "You are a helpful and friendly AI assistant who loves science and technology."
            },
            "memory": {
                "summary": "The user previously asked about machine learning and showed interest in AI development."
            }
        }
        
        logger.info("Processing message with context injection...")
        response = await processor.process_voice_message(voice_message, context=context)
        
        logger.info(f"✓ Context-aware response: {response.text_content[:150]}...")
        logger.info(f"  Processing mode: {response.processing_mode}")
        
        await processor.cleanup()
        logger.info("✓ Context injection test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Context injection test failed: {e}")
        return False


async def main():
    """Run all voice-LLM integration tests."""
    logger.info("🚀 Starting Voice-LLM Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Initialization", test_voice_llm_initialization),
        ("Processing", test_voice_llm_processing),
        ("Streaming", test_voice_llm_streaming),
        ("Fallback", test_voice_llm_fallback),
        ("Context Injection", test_voice_llm_context_injection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "❌ CRASHED"
    
    # Print results summary
    logger.info("=" * 50)
    logger.info("🏁 Voice-LLM Integration Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<20}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice-LLM integration tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
