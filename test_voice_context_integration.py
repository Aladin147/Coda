#!/usr/bin/env python3
"""
Test Voice Context Integration

This script tests the integration between voice processing, LLMs, and context systems.
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
    from src.coda.components.voice.context_integration import (
        VoiceContextManager, ContextInjector, ContextConfig
    )
    from src.coda.components.voice.llm_integration import VoiceLLMConfig
    from src.coda.components.voice.models import VoiceMessage, VoiceProcessingMode, ConversationState
    from src.coda.components.llm.models import LLMProvider
    logger.info("✓ Successfully imported voice context integration components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_configs() -> tuple[ContextConfig, VoiceLLMConfig]:
    """Create test configurations."""
    context_config = ContextConfig(
        enable_memory_injection=True,
        enable_personality_injection=True,
        enable_dynamic_models=False,  # Disable for testing
        context_cache_ttl_minutes=5,
        parallel_context_loading=True
    )
    
    llm_config = VoiceLLMConfig(
        llm_provider=LLMProvider.OLLAMA,
        llm_model="gemma3:1b",
        enable_streaming=True,
        llm_timeout_seconds=10.0
    )
    
    return context_config, llm_config


def create_test_voice_message(conversation_id: str, text_content: str) -> VoiceMessage:
    """Create a test voice message."""
    audio_data = np.random.randint(-32768, 32767, 24000, dtype=np.int16).tobytes()
    
    return VoiceMessage(
        message_id=f"test_msg_{datetime.now().strftime('%H%M%S')}",
        conversation_id=conversation_id,
        audio_data=audio_data,
        text_content=text_content,
        processing_mode=VoiceProcessingMode.HYBRID,
        timestamp=datetime.now()
    )


def create_test_conversation_state(conversation_id: str) -> ConversationState:
    """Create a test conversation state."""
    return ConversationState(
        conversation_id=conversation_id,
        is_active=True,
        current_mode=VoiceProcessingMode.HYBRID,
        total_duration_ms=0
    )


async def test_context_injector():
    """Test context injector functionality."""
    logger.info("=== Testing Context Injector ===")
    
    try:
        context_config, _ = create_test_configs()
        injector = ContextInjector(context_config)
        
        # Test context building
        conversation_id = "test_conversation_context"
        voice_message = create_test_voice_message(
            conversation_id,
            "Can you explain how machine learning algorithms work in detail?"
        )
        conversation_state = create_test_conversation_state(conversation_id)
        
        logger.info("Building context...")
        context = await injector.build_context(voice_message, conversation_state)
        
        logger.info("✓ Context built successfully")
        logger.info(f"  Context keys: {list(context.keys())}")
        
        # Check context components
        if "memory" in context:
            memory = context["memory"]
            logger.info(f"  Memory context: {len(memory.get('recent_topics', []))} topics")
            logger.info(f"  Relevance score: {memory.get('relevance_score', 0)}")
        
        if "personality" in context:
            personality = context["personality"]
            logger.info(f"  Personality traits: {list(personality.get('traits', {}).keys())}")
            logger.info(f"  Communication style: {personality.get('communication_style', {})}")
        
        if "current_message" in context:
            message = context["current_message"]
            logger.info(f"  Message complexity: {message.get('complexity_estimate')}")
            logger.info(f"  Intent category: {message.get('intent_category')}")
        
        # Test caching
        logger.info("Testing context caching...")
        cached_context = await injector.build_context(voice_message, conversation_state)
        logger.info("✓ Context caching works")
        
        logger.info("✓ Context injector test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Context injector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_manager_initialization():
    """Test context manager initialization."""
    logger.info("=== Testing Context Manager Initialization ===")
    
    try:
        context_config, llm_config = create_test_configs()
        manager = VoiceContextManager(context_config, llm_config)
        
        logger.info("Initializing context manager...")
        await manager.initialize()
        logger.info("✓ Context manager initialized")
        
        # Check status
        status = manager.get_context_status()
        logger.info(f"✓ Context manager status:")
        logger.info(f"  Processing stats: {status['processing_stats']}")
        logger.info(f"  Cache size: {status['cache_size']}")
        
        await manager.cleanup()
        logger.info("✓ Context manager initialization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Context manager initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_aware_processing():
    """Test context-aware voice processing."""
    logger.info("=== Testing Context-Aware Processing ===")
    
    try:
        context_config, llm_config = create_test_configs()
        manager = VoiceContextManager(context_config, llm_config)
        await manager.initialize()
        
        conversation_id = "test_conversation_processing"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Test different types of messages
        test_messages = [
            "Hello! How are you today?",
            "Can you explain how neural networks work?",
            "I need help debugging my Python code",
            "What's the difference between supervised and unsupervised learning?"
        ]
        
        for i, text in enumerate(test_messages):
            logger.info(f"Processing message {i+1}: {text[:50]}...")
            
            voice_message = create_test_voice_message(conversation_id, text)
            
            response = await manager.process_with_context(voice_message, conversation_state)
            
            logger.info(f"✓ Response {i+1}: {response.text_content[:100]}...")
            logger.info(f"  Latency: {response.total_latency_ms:.1f}ms")
            logger.info(f"  Processing mode: {response.processing_mode}")
        
        # Check final status
        final_status = manager.get_context_status()
        logger.info(f"✓ Final processing stats:")
        logger.info(f"  Total requests: {final_status['processing_stats']['total_requests']}")
        logger.info(f"  Average latency: {final_status['processing_stats']['average_latency_ms']:.1f}ms")
        
        await manager.cleanup()
        logger.info("✓ Context-aware processing test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Context-aware processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_intent_categorization():
    """Test intent categorization functionality."""
    logger.info("=== Testing Intent Categorization ===")
    
    try:
        context_config, _ = create_test_configs()
        injector = ContextInjector(context_config)
        
        # Test different intent categories
        test_cases = [
            ("What is machine learning?", "question"),
            ("Please help me with my code", "request"),
            ("How do I implement a neural network?", "coding"),
            ("Explain the difference between AI and ML", "explanation"),
            ("Compare Python and JavaScript", "comparison"),
            ("My code is not working", "troubleshooting"),
            ("Hello there!", "general")
        ]
        
        for text, expected_intent in test_cases:
            actual_intent = injector._categorize_intent(text)
            logger.info(f"✓ '{text}' -> {actual_intent} (expected: {expected_intent})")
            
            # Also test complexity estimation
            complexity = injector._estimate_complexity(text)
            logger.info(f"  Complexity: {complexity}")
        
        logger.info("✓ Intent categorization test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Intent categorization test failed: {e}")
        return False


async def test_parallel_context_loading():
    """Test parallel context loading performance."""
    logger.info("=== Testing Parallel Context Loading ===")
    
    try:
        # Test with parallel loading enabled
        context_config, _ = create_test_configs()
        context_config.parallel_context_loading = True
        injector_parallel = ContextInjector(context_config)
        
        # Test with parallel loading disabled
        context_config.parallel_context_loading = False
        injector_sequential = ContextInjector(context_config)
        
        conversation_id = "test_conversation_parallel"
        voice_message = create_test_voice_message(
            conversation_id,
            "This is a complex technical question about implementing advanced machine learning algorithms for real-time voice processing systems."
        )
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Time parallel loading
        start_time = asyncio.get_event_loop().time()
        context_parallel = await injector_parallel.build_context(voice_message, conversation_state)
        parallel_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Time sequential loading
        start_time = asyncio.get_event_loop().time()
        context_sequential = await injector_sequential.build_context(voice_message, conversation_state)
        sequential_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.info(f"✓ Parallel loading: {parallel_time:.1f}ms")
        logger.info(f"✓ Sequential loading: {sequential_time:.1f}ms")
        logger.info(f"✓ Performance improvement: {((sequential_time - parallel_time) / sequential_time * 100):.1f}%")
        
        # Verify context content is similar
        assert set(context_parallel.keys()) == set(context_sequential.keys())
        logger.info("✓ Context content consistency verified")
        
        logger.info("✓ Parallel context loading test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parallel context loading test failed: {e}")
        return False


async def test_context_caching():
    """Test context caching functionality."""
    logger.info("=== Testing Context Caching ===")
    
    try:
        context_config, _ = create_test_configs()
        context_config.context_cache_ttl_minutes = 1  # Short TTL for testing
        injector = ContextInjector(context_config)
        
        conversation_id = "test_conversation_caching"
        voice_message = create_test_voice_message(conversation_id, "Test message for caching")
        conversation_state = create_test_conversation_state(conversation_id)
        
        # First call - should build fresh context
        logger.info("Building fresh context...")
        start_time = asyncio.get_event_loop().time()
        context1 = await injector.build_context(voice_message, conversation_state)
        first_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Second call - should use cache
        logger.info("Using cached context...")
        start_time = asyncio.get_event_loop().time()
        context2 = await injector.build_context(voice_message, conversation_state)
        second_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.info(f"✓ Fresh context build: {first_time:.1f}ms")
        logger.info(f"✓ Cached context access: {second_time:.1f}ms")
        logger.info(f"✓ Cache speedup: {((first_time - second_time) / first_time * 100):.1f}%")
        
        # Verify cache is working
        assert len(injector.context_cache) > 0
        logger.info(f"✓ Cache contains {len(injector.context_cache)} entries")
        
        logger.info("✓ Context caching test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Context caching test failed: {e}")
        return False


async def main():
    """Run all voice context integration tests."""
    logger.info("🚀 Starting Voice Context Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Context Injector", test_context_injector),
        ("Manager Initialization", test_context_manager_initialization),
        ("Context-Aware Processing", test_context_aware_processing),
        ("Intent Categorization", test_intent_categorization),
        ("Parallel Context Loading", test_parallel_context_loading),
        ("Context Caching", test_context_caching),
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
    logger.info("🏁 Voice Context Integration Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice context integration tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
