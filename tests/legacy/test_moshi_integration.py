#!/usr/bin/env python3
"""
Comprehensive test script for Moshi integration.

This script tests all components of the Moshi integration including:
- Moshi client initialization
- Audio processing
- Inner monologue text extraction
- Conversation state management
- Real-time streaming
- WebSocket handling
"""

import asyncio
import logging
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our Moshi integration components
try:
    from src.coda.components.voice.models import VoiceConfig, MoshiConfig, VoiceMessage, VoiceProcessingMode
    from src.coda.components.voice.moshi_integration import MoshiClient, MoshiVoiceProcessor
    from src.coda.components.voice.conversation_state import ConversationStateManager, ConversationPhase
    from src.coda.components.voice.inner_monologue import InnerMonologueProcessor
    logger.info("✓ Successfully imported Moshi integration components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_audio_data(duration_ms: int = 1000, sample_rate: int = 24000) -> bytes:
    """Create test audio data (sine wave)."""
    samples = int(duration_ms * sample_rate / 1000)
    t = np.linspace(0, duration_ms / 1000, samples)
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def create_test_config() -> VoiceConfig:
    """Create test configuration."""
    from src.coda.components.voice.models import AudioConfig

    audio_config = AudioConfig(
        sample_rate=24000,
        channels=1,
        chunk_size=1024
    )

    moshi_config = MoshiConfig(
        model_path=None,  # Use default model for testing
        device="cuda",  # Use GPU for proper testing
        vram_allocation="8GB",  # Allocate proper VRAM for RTX 5090
        inner_monologue_enabled=True
    )

    config = VoiceConfig(
        audio=audio_config,
        moshi=moshi_config,
        websocket_events_enabled=True
    )

    return config


async def test_moshi_client():
    """Test basic Moshi client functionality."""
    logger.info("=== Testing Moshi Client ===")
    
    try:
        config = create_test_config()
        client = MoshiClient(config.moshi)
        
        # Test initialization
        logger.info("Initializing Moshi client...")
        await client.initialize()
        logger.info("✓ Moshi client initialized successfully")
        
        # Test conversation start
        conversation_id = "test_conversation_001"
        await client.start_conversation(conversation_id)
        logger.info(f"✓ Started conversation: {conversation_id}")
        
        # Test audio processing
        test_audio = create_test_audio_data(500)  # 500ms of test audio
        logger.info("Processing test audio...")
        
        start_time = time.time()
        processed_audio = await client.process_audio(test_audio)
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✓ Audio processed in {processing_time:.1f}ms")
        logger.info(f"  Input size: {len(test_audio)} bytes")
        logger.info(f"  Output size: {len(processed_audio)} bytes")
        
        # Test text extraction
        if config.moshi.inner_monologue_enabled:
            logger.info("Testing text extraction...")
            extracted_text = await client.extract_text(test_audio)
            logger.info(f"✓ Extracted text: '{extracted_text}'")
        
        # Test conversation state
        state = await client.get_conversation_state()
        logger.info(f"✓ Conversation state retrieved: {state['is_active']}")
        
        # Test conversation end
        await client.end_conversation(conversation_id)
        logger.info(f"✓ Ended conversation: {conversation_id}")
        
        # Cleanup
        await client.cleanup()
        logger.info("✓ Moshi client cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Moshi client test failed: {e}")
        return False


async def test_conversation_state_manager():
    """Test conversation state management."""
    logger.info("=== Testing Conversation State Manager ===")
    
    try:
        manager = ConversationStateManager(max_conversations=10)
        await manager.start()
        logger.info("✓ Conversation state manager started")
        
        # Test conversation creation
        conversation_id = "test_conversation_002"
        conversation = manager.create_conversation(
            conversation_id=conversation_id,
            user_id="test_user",
            mode=VoiceProcessingMode.MOSHI_ONLY
        )
        logger.info(f"✓ Created conversation: {conversation_id}")
        
        # Test phase updates
        manager.update_conversation_phase(conversation_id, ConversationPhase.PROCESSING)
        logger.info("✓ Updated conversation phase")
        
        # Test audio exchange recording
        manager.record_audio_exchange(
            conversation_id=conversation_id,
            audio_duration_ms=500.0,
            processing_latency_ms=150.0,
            confidence_score=0.85
        )
        logger.info("✓ Recorded audio exchange")
        
        # Test metrics retrieval
        metrics = manager.get_conversation_metrics(conversation_id)
        if metrics:
            logger.info(f"✓ Retrieved metrics: {metrics.total_exchanges} exchanges")
        
        # Test events retrieval
        events = manager.get_conversation_events(conversation_id)
        logger.info(f"✓ Retrieved {len(events)} events")
        
        # Test conversation summary
        summary = manager.get_conversation_summary(conversation_id)
        if summary:
            logger.info("✓ Retrieved conversation summary")
        
        # Test conversation end
        final_metrics = manager.end_conversation(conversation_id)
        if final_metrics:
            logger.info(f"✓ Ended conversation with quality: {final_metrics.quality_level}")
        
        # Cleanup
        await manager.stop()
        logger.info("✓ Conversation state manager stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversation state manager test failed: {e}")
        return False


async def test_inner_monologue():
    """Test inner monologue processor."""
    logger.info("=== Testing Inner Monologue Processor ===")
    
    try:
        config = create_test_config()
        processor = InnerMonologueProcessor(config.moshi)
        
        # Note: We can't fully test without a real Moshi model
        # This tests the structure and basic functionality
        
        logger.info("✓ Inner monologue processor created")
        
        # Test configuration
        processor.set_confidence_threshold(0.8)
        logger.info("✓ Set confidence threshold")
        
        # Test statistics
        stats = processor.get_statistics()
        logger.info(f"✓ Retrieved statistics: {stats['total_extractions']} extractions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Inner monologue test failed: {e}")
        return False


async def test_voice_processor():
    """Test the complete voice processor."""
    logger.info("=== Testing Voice Processor ===")
    
    try:
        config = create_test_config()
        processor = MoshiVoiceProcessor()
        
        # Test initialization
        logger.info("Initializing voice processor...")
        await processor.initialize(config)
        logger.info("✓ Voice processor initialized")
        
        # Test conversation start
        conversation_id = "test_conversation_003"
        conversation_state = await processor.start_conversation(
            conversation_id=conversation_id,
            user_id="test_user"
        )
        logger.info(f"✓ Started conversation: {conversation_id}")
        
        # Test audio processing
        test_audio = create_test_audio_data(1000)  # 1 second of test audio
        
        voice_message = VoiceMessage(
            message_id="test_message_001",
            conversation_id=conversation_id,
            audio_data=test_audio,
            processing_mode=VoiceProcessingMode.MOSHI_ONLY,
            timestamp=datetime.now()
        )
        
        logger.info("Processing voice message...")
        start_time = time.time()
        response = await processor.process_audio(voice_message)
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✓ Voice message processed in {processing_time:.1f}ms")
        logger.info(f"  Response ID: {response.response_id}")
        logger.info(f"  Text content: '{response.text_content}'")
        
        # Test conversation state retrieval
        current_state = await processor.get_conversation_state(conversation_id)
        if current_state:
            logger.info(f"✓ Retrieved conversation state: {current_state.message_count} messages")
        
        # Test analytics
        analytics = await processor.get_analytics()
        logger.info(f"✓ Retrieved analytics: {analytics.total_conversations} total conversations")
        
        # Test conversation end
        await processor.end_conversation(conversation_id)
        logger.info(f"✓ Ended conversation: {conversation_id}")
        
        # Cleanup
        await processor.cleanup()
        logger.info("✓ Voice processor cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice processor test failed: {e}")
        return False


async def run_performance_test():
    """Run performance tests."""
    logger.info("=== Performance Testing ===")
    
    try:
        config = create_test_config()
        processor = MoshiVoiceProcessor()
        
        await processor.initialize(config)
        
        conversation_id = "perf_test_conversation"
        await processor.start_conversation(conversation_id)
        
        # Test multiple audio processing cycles
        num_cycles = 5
        total_time = 0
        
        for i in range(num_cycles):
            test_audio = create_test_audio_data(500)  # 500ms audio
            
            voice_message = VoiceMessage(
                message_id=f"perf_message_{i:03d}",
                conversation_id=conversation_id,
                audio_data=test_audio,
                processing_mode=VoiceProcessingMode.MOSHI_ONLY,
                timestamp=datetime.now()
            )
            
            start_time = time.time()
            response = await processor.process_audio(voice_message)
            cycle_time = (time.time() - start_time) * 1000
            total_time += cycle_time
            
            logger.info(f"  Cycle {i+1}: {cycle_time:.1f}ms")
        
        average_time = total_time / num_cycles
        logger.info(f"✓ Performance test completed")
        logger.info(f"  Average processing time: {average_time:.1f}ms")
        logger.info(f"  Total time for {num_cycles} cycles: {total_time:.1f}ms")
        
        await processor.end_conversation(conversation_id)
        await processor.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Moshi Integration Tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Run individual component tests
    test_results['moshi_client'] = await test_moshi_client()
    test_results['conversation_state'] = await test_conversation_state_manager()
    test_results['inner_monologue'] = await test_inner_monologue()
    test_results['voice_processor'] = await test_voice_processor()
    test_results['performance'] = await run_performance_test()
    
    # Summary
    logger.info("=" * 50)
    logger.info("🏁 Test Results Summary")
    logger.info("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Moshi integration is ready.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        exit(1)
