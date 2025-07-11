#!/usr/bin/env python3
"""
Test Voice Performance Optimizer

This script tests the performance optimization capabilities for voice-LLM processing.
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

try:
    from src.coda.components.voice.performance_optimizer import (
        PerformanceOptimizer, OptimizationConfig, OptimizationLevel, PerformanceMetrics
    )
    from src.coda.components.voice.context_integration import VoiceContextManager, ContextConfig
    from src.coda.components.voice.llm_integration import VoiceLLMConfig
    from src.coda.components.voice.models import VoiceMessage, VoiceProcessingMode, ConversationState
    from src.coda.components.llm.models import LLMProvider
    logger.info("✓ Successfully imported voice performance optimizer components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_configs() -> tuple[OptimizationConfig, ContextConfig, VoiceLLMConfig]:
    """Create test configurations."""
    opt_config = OptimizationConfig(
        optimization_level=OptimizationLevel.BALANCED,
        target_total_latency_ms=500.0,
        target_llm_latency_ms=300.0,
        enable_response_caching=True,
        enable_parallel_processing=True,
        max_queue_depth=5
    )
    
    context_config = ContextConfig(
        enable_memory_injection=True,
        enable_personality_injection=True,
        enable_dynamic_models=False,
        parallel_context_loading=True
    )
    
    llm_config = VoiceLLMConfig(
        llm_provider=LLMProvider.OLLAMA,
        llm_model="gemma3:1b",
        enable_streaming=True,
        llm_timeout_seconds=10.0
    )
    
    return opt_config, context_config, llm_config


def create_test_voice_message(conversation_id: str, text_content: str) -> VoiceMessage:
    """Create a test voice message."""
    audio_data = np.random.randint(-32768, 32767, 24000, dtype=np.int16).tobytes()
    
    return VoiceMessage(
        message_id=f"test_msg_{datetime.now().strftime('%H%M%S%f')}",
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


async def test_optimizer_initialization():
    """Test performance optimizer initialization."""
    logger.info("=== Testing Performance Optimizer Initialization ===")
    
    try:
        opt_config, _, _ = create_test_configs()
        optimizer = PerformanceOptimizer(opt_config)
        
        logger.info("✓ Performance optimizer created")
        logger.info(f"  Optimization level: {opt_config.optimization_level}")
        logger.info(f"  Target latency: {opt_config.target_total_latency_ms}ms")
        logger.info(f"  Max queue depth: {opt_config.max_queue_depth}")
        
        # Check initial metrics
        metrics = optimizer.get_performance_metrics()
        logger.info(f"✓ Initial metrics: latency={metrics.total_latency_ms}ms, queue={metrics.queue_depth}")
        
        # Check performance summary
        summary = optimizer.get_performance_summary()
        logger.info(f"✓ Performance summary: {summary['optimization_status']['level']}")
        
        await optimizer.cleanup()
        logger.info("✓ Performance optimizer initialization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_basic_optimization():
    """Test basic optimization functionality."""
    logger.info("=== Testing Basic Optimization ===")
    
    try:
        opt_config, context_config, llm_config = create_test_configs()
        optimizer = PerformanceOptimizer(opt_config)
        context_manager = VoiceContextManager(context_config, llm_config)
        await context_manager.initialize()
        
        conversation_id = "test_conversation_optimization"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Test single message optimization
        voice_message = create_test_voice_message(
            conversation_id,
            "Hello! Can you help me with a quick question?"
        )
        
        logger.info("Processing message with optimization...")
        start_time = time.time()
        
        response = await optimizer.optimize_processing(
            voice_message, context_manager, conversation_state
        )
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        logger.info(f"✓ Optimized response: {response.text_content[:100]}...")
        logger.info(f"  Processing time: {processing_time:.1f}ms")
        logger.info(f"  Response latency: {response.total_latency_ms:.1f}ms")
        
        # Check updated metrics
        metrics = optimizer.get_performance_metrics()
        logger.info(f"✓ Updated metrics: latency={metrics.total_latency_ms:.1f}ms")
        
        await context_manager.cleanup()
        await optimizer.cleanup()
        logger.info("✓ Basic optimization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_caching_optimization():
    """Test response caching optimization."""
    logger.info("=== Testing Caching Optimization ===")
    
    try:
        opt_config, context_config, llm_config = create_test_configs()
        opt_config.enable_response_caching = True
        
        optimizer = PerformanceOptimizer(opt_config)
        context_manager = VoiceContextManager(context_config, llm_config)
        await context_manager.initialize()
        
        conversation_id = "test_conversation_caching"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Same message for caching test
        text_content = "What is machine learning?"
        
        # First request - should be processed normally
        voice_message1 = create_test_voice_message(conversation_id, text_content)
        
        logger.info("Processing first request (cache miss)...")
        start_time = time.time()
        response1 = await optimizer.optimize_processing(
            voice_message1, context_manager, conversation_state
        )
        first_time = (time.time() - start_time) * 1000
        
        # Second request - should use cache
        voice_message2 = create_test_voice_message(conversation_id, text_content)
        
        logger.info("Processing second request (cache hit)...")
        start_time = time.time()
        response2 = await optimizer.optimize_processing(
            voice_message2, context_manager, conversation_state
        )
        second_time = (time.time() - start_time) * 1000
        
        logger.info(f"✓ First request: {first_time:.1f}ms")
        logger.info(f"✓ Second request: {second_time:.1f}ms")
        if first_time > 0:
            logger.info(f"✓ Cache speedup: {((first_time - second_time) / first_time * 100):.1f}%")
        else:
            logger.info("✓ Cache speedup: N/A (processing too fast to measure)")
        
        # Check cache stats
        summary = optimizer.get_performance_summary()
        cache_stats = summary.get('cache_stats', {})
        logger.info(f"✓ Cache stats: {cache_stats}")
        
        await context_manager.cleanup()
        await optimizer.cleanup()
        logger.info("✓ Caching optimization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Caching optimization test failed: {e}")
        return False


async def test_parallel_optimization():
    """Test parallel processing optimization."""
    logger.info("=== Testing Parallel Optimization ===")
    
    try:
        opt_config, context_config, llm_config = create_test_configs()
        opt_config.enable_parallel_processing = True
        
        optimizer = PerformanceOptimizer(opt_config)
        context_manager = VoiceContextManager(context_config, llm_config)
        await context_manager.initialize()
        
        conversation_id = "test_conversation_parallel"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Test multiple concurrent requests
        messages = [
            "Hello, how are you?",
            "What's the weather like?",
            "Can you help me with coding?",
            "Tell me about AI"
        ]
        
        voice_messages = [
            create_test_voice_message(conversation_id, text)
            for text in messages
        ]
        
        logger.info("Processing multiple requests in parallel...")
        start_time = time.time()
        
        # Process all messages concurrently
        tasks = [
            optimizer.optimize_processing(msg, context_manager, conversation_state)
            for msg in voice_messages
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        
        logger.info(f"✓ Processed {len(successful_responses)}/{len(messages)} messages")
        logger.info(f"✓ Total time: {total_time:.1f}ms")
        logger.info(f"✓ Average per message: {total_time / len(messages):.1f}ms")
        
        # Check final metrics
        metrics = optimizer.get_performance_metrics()
        logger.info(f"✓ Final queue depth: {metrics.queue_depth}")
        
        await context_manager.cleanup()
        await optimizer.cleanup()
        logger.info("✓ Parallel optimization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Parallel optimization test failed: {e}")
        return False


async def test_adaptive_optimization():
    """Test adaptive optimization functionality."""
    logger.info("=== Testing Adaptive Optimization ===")
    
    try:
        opt_config, context_config, llm_config = create_test_configs()
        opt_config.target_total_latency_ms = 100.0  # Very aggressive target
        
        optimizer = PerformanceOptimizer(opt_config)
        optimizer.enable_adaptive_optimization()
        
        context_manager = VoiceContextManager(context_config, llm_config)
        await context_manager.initialize()
        
        conversation_id = "test_conversation_adaptive"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Process several messages to trigger adaptive optimization
        for i in range(3):
            voice_message = create_test_voice_message(
                conversation_id,
                f"This is test message {i+1} for adaptive optimization"
            )
            
            logger.info(f"Processing adaptive message {i+1}...")
            response = await optimizer.optimize_processing(
                voice_message, context_manager, conversation_state
            )
            
            logger.info(f"  Response {i+1}: {response.total_latency_ms:.1f}ms")
        
        # Check if adaptive adjustments were made
        summary = optimizer.get_performance_summary()
        adjustments = summary['optimization_status']['adjustments']
        
        logger.info(f"✓ Adaptive optimization active: {summary['optimization_status']['adaptive_active']}")
        logger.info(f"✓ Adjustments made: {adjustments}")
        
        optimizer.disable_adaptive_optimization()
        logger.info("✓ Adaptive optimization disabled")
        
        await context_manager.cleanup()
        await optimizer.cleanup()
        logger.info("✓ Adaptive optimization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive optimization test failed: {e}")
        return False


async def test_optimization_levels():
    """Test different optimization levels."""
    logger.info("=== Testing Optimization Levels ===")
    
    try:
        _, context_config, llm_config = create_test_configs()
        
        levels = [
            OptimizationLevel.MINIMAL,
            OptimizationLevel.BALANCED,
            OptimizationLevel.AGGRESSIVE
        ]
        
        results = {}
        
        for level in levels:
            logger.info(f"Testing {level} optimization...")
            
            opt_config = OptimizationConfig(optimization_level=level)
            optimizer = PerformanceOptimizer(opt_config)
            context_manager = VoiceContextManager(context_config, llm_config)
            await context_manager.initialize()
            
            conversation_id = f"test_conversation_{level}"
            conversation_state = create_test_conversation_state(conversation_id)
            
            voice_message = create_test_voice_message(
                conversation_id,
                "Test message for optimization level comparison"
            )
            
            start_time = time.time()
            response = await optimizer.optimize_processing(
                voice_message, context_manager, conversation_state
            )
            processing_time = (time.time() - start_time) * 1000
            
            results[level] = {
                "processing_time_ms": processing_time,
                "response_latency_ms": response.total_latency_ms
            }
            
            logger.info(f"  {level}: {processing_time:.1f}ms")
            
            await context_manager.cleanup()
            await optimizer.cleanup()
        
        # Compare results
        logger.info("✓ Optimization level comparison:")
        for level, metrics in results.items():
            logger.info(f"  {level}: {metrics['processing_time_ms']:.1f}ms")
        
        logger.info("✓ Optimization levels test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization levels test failed: {e}")
        return False


async def main():
    """Run all voice performance optimizer tests."""
    logger.info("🚀 Starting Voice Performance Optimizer Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Optimizer Initialization", test_optimizer_initialization),
        ("Basic Optimization", test_basic_optimization),
        ("Caching Optimization", test_caching_optimization),
        ("Parallel Optimization", test_parallel_optimization),
        ("Adaptive Optimization", test_adaptive_optimization),
        ("Optimization Levels", test_optimization_levels),
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
    logger.info("🏁 Voice Performance Optimizer Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All voice performance optimizer tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
