#!/usr/bin/env python3
"""
Test Latency Optimizer

This script tests the latency optimization system for voice conversations.
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
    from src.coda.components.voice.latency_optimizer import (
        LatencyOptimizer, LatencyOptimizationConfig, LatencyTarget, OptimizationTechnique,
        LatencyProfiler, PredictiveCache, LatencyMeasurement
    )
    from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode
    logger.info("✓ Successfully imported latency optimizer components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_config() -> LatencyOptimizationConfig:
    """Create test configuration for latency optimization."""
    return LatencyOptimizationConfig(
        target_latency=LatencyTarget.LOW,
        target_latency_ms=200.0,
        enabled_techniques=[
            OptimizationTechnique.PREDICTIVE_CACHING,
            OptimizationTechnique.PIPELINE_PARALLELIZATION,
            OptimizationTechnique.CONTEXT_COMPRESSION,
            OptimizationTechnique.EARLY_TERMINATION
        ],
        enable_adaptive_optimization=True
    )


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


async def mock_processing_callback(voice_message: VoiceMessage) -> VoiceResponse:
    """Mock processing callback for testing."""
    
    # Simulate processing time based on message complexity
    text_length = len(voice_message.text_content or "")
    processing_time = 0.05 + (text_length / 1000.0)  # Base time + complexity
    
    await asyncio.sleep(processing_time)
    
    return VoiceResponse(
        response_id=f"mock_{voice_message.message_id}",
        conversation_id=voice_message.conversation_id,
        message_id=voice_message.message_id,
        text_content=f"Mock response to: {voice_message.text_content[:50]}...",
        audio_data=b"",
        processing_mode=VoiceProcessingMode.HYBRID,
        total_latency_ms=processing_time * 1000,
        response_relevance=0.8
    )


async def test_latency_profiler():
    """Test latency profiler functionality."""
    logger.info("=== Testing Latency Profiler ===")
    
    try:
        config = create_test_config()
        profiler = LatencyProfiler(config)
        
        # Test measurement cycle
        profiler.start_measurement("test_message_1")
        
        # Simulate component latencies
        profiler.record_component_latency("audio_preprocessing", 10.0)
        profiler.record_component_latency("moshi_processing", 50.0)
        profiler.record_component_latency("llm_processing", 80.0)
        profiler.record_component_latency("context_building", 15.0)
        
        # Finish measurement
        measurement = profiler.finish_measurement(
            VoiceProcessingMode.HYBRID,
            message_length=100,
            complexity_score=0.5
        )
        
        logger.info(f"✓ Measurement completed:")
        logger.info(f"  Total latency: {measurement.total_latency_ms:.1f}ms")
        logger.info(f"  Audio preprocessing: {measurement.audio_preprocessing_ms:.1f}ms")
        logger.info(f"  Moshi processing: {measurement.moshi_processing_ms:.1f}ms")
        logger.info(f"  LLM processing: {measurement.llm_processing_ms:.1f}ms")
        logger.info(f"  Context building: {measurement.context_building_ms:.1f}ms")
        
        # Add more measurements for statistics
        for i in range(5):
            profiler.start_measurement(f"test_message_{i+2}")
            profiler.record_component_latency("moshi_processing", 40.0 + i * 5)
            profiler.record_component_latency("llm_processing", 70.0 + i * 10)
            profiler.finish_measurement(VoiceProcessingMode.HYBRID, 80 + i * 10, 0.4 + i * 0.1)
        
        # Get statistics
        stats = profiler.get_latency_statistics()
        logger.info(f"✓ Latency statistics:")
        logger.info(f"  Mean latency: {stats['overall']['mean_ms']:.1f}ms")
        logger.info(f"  Median latency: {stats['overall']['median_ms']:.1f}ms")
        logger.info(f"  P95 latency: {stats['overall']['p95_ms']:.1f}ms")
        logger.info(f"  Component count: {len(stats['components'])}")
        
        logger.info("✓ Latency profiler test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Latency profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_predictive_cache():
    """Test predictive cache functionality."""
    logger.info("=== Testing Predictive Cache ===")
    
    try:
        config = create_test_config()
        cache = PredictiveCache(config)
        
        # Test prediction
        context = {
            "conversation_id": "test_conversation",
            "recent_messages": ["hello", "how are you", "what's the weather"]
        }
        
        predictions = cache.predict_next_requests(context)
        logger.info(f"✓ Generated {len(predictions)} predictions")
        
        # Test preloading
        cache.preload_cache(predictions)
        logger.info(f"✓ Preloaded cache with {len(predictions)} items")
        logger.info(f"  Cache size: {len(cache.cache)}")
        
        # Test cache access tracking
        for i in range(10):
            hit = i % 3 == 0  # 33% hit rate
            cache.record_cache_access(f"key_{i}", hit)
        
        hit_rate = cache.get_cache_hit_rate()
        logger.info(f"✓ Cache hit rate: {hit_rate:.2f}")
        
        logger.info("✓ Predictive cache test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Predictive cache test failed: {e}")
        return False


async def test_latency_optimizer_initialization():
    """Test latency optimizer initialization."""
    logger.info("=== Testing Latency Optimizer Initialization ===")
    
    try:
        config = create_test_config()
        optimizer = LatencyOptimizer(config)
        
        logger.info("✓ Latency optimizer created")
        logger.info(f"  Target latency: {config.target_latency} ({config.target_latency_ms}ms)")
        logger.info(f"  Enabled techniques: {len(config.enabled_techniques)}")
        
        # Check optimization status
        status = optimizer.get_optimization_status()
        logger.info(f"✓ Optimization status:")
        logger.info(f"  Target latency: {status['target_latency_ms']}ms")
        logger.info(f"  Active optimizations: {len(status['active_optimizations'])}")
        logger.info(f"  Cache hit rate: {status['cache_hit_rate']:.2f}")
        
        await optimizer.cleanup()
        logger.info("✓ Latency optimizer initialization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return False


async def test_optimization_techniques():
    """Test different optimization techniques."""
    logger.info("=== Testing Optimization Techniques ===")
    
    try:
        config = create_test_config()
        optimizer = LatencyOptimizer(config)
        
        conversation_id = "test_conversation_optimization"
        
        # Test different message types
        test_messages = [
            "Hello there!",  # Simple
            "Can you explain how machine learning works?",  # Medium
            "I need a comprehensive analysis of the current state of artificial intelligence research, including recent breakthroughs in large language models, computer vision, and robotics, with specific focus on their practical applications and potential societal impacts."  # Complex
        ]
        
        for i, text_content in enumerate(test_messages):
            voice_message = create_test_voice_message(conversation_id, text_content)
            
            logger.info(f"Processing message {i+1}: {text_content[:50]}...")
            
            start_time = time.time()
            response = await optimizer.optimize_request(voice_message, mock_processing_callback)
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"✓ Optimized processing completed in {processing_time:.1f}ms")
            logger.info(f"  Response: {response.text_content[:50]}...")
            logger.info(f"  Total latency: {response.total_latency_ms:.1f}ms")
        
        # Check final optimization status
        final_status = optimizer.get_optimization_status()
        logger.info(f"✓ Final optimization status:")
        logger.info(f"  Cache hit rate: {final_status['cache_hit_rate']:.2f}")
        logger.info(f"  Active optimizations: {final_status['active_optimizations']}")
        
        await optimizer.cleanup()
        logger.info("✓ Optimization techniques test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization techniques test failed: {e}")
        return False


async def test_adaptive_optimization():
    """Test adaptive optimization functionality."""
    logger.info("=== Testing Adaptive Optimization ===")
    
    try:
        config = create_test_config()
        config.enable_adaptive_optimization = True
        config.target_latency_ms = 100.0  # Aggressive target
        
        optimizer = LatencyOptimizer(config)
        
        conversation_id = "test_conversation_adaptive"
        
        # Process several messages to trigger adaptive behavior
        for i in range(5):
            # Create increasingly complex messages
            complexity = "simple" if i < 2 else "complex"
            text_content = f"This is a {complexity} test message number {i+1}" + (" with additional complexity" * i)
            
            voice_message = create_test_voice_message(conversation_id, text_content)
            
            response = await optimizer.optimize_request(voice_message, mock_processing_callback)
            
            logger.info(f"✓ Adaptive processing {i+1}: {response.total_latency_ms:.1f}ms")
        
        # Check if adaptive optimizations were triggered
        status = optimizer.get_optimization_status()
        logger.info(f"✓ Adaptive optimization results:")
        logger.info(f"  Active optimizations: {status['active_optimizations']}")
        logger.info(f"  Adaptive parameters: {status['adaptive_parameters']}")
        
        await optimizer.cleanup()
        logger.info("✓ Adaptive optimization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive optimization test failed: {e}")
        return False


async def test_latency_targets():
    """Test different latency targets."""
    logger.info("=== Testing Latency Targets ===")
    
    try:
        targets = [
            (LatencyTarget.ULTRA_LOW, 100.0),
            (LatencyTarget.LOW, 200.0),
            (LatencyTarget.MODERATE, 500.0),
            (LatencyTarget.RELAXED, 1000.0)
        ]
        
        for target, target_ms in targets:
            config = LatencyOptimizationConfig(
                target_latency=target,
                target_latency_ms=target_ms,
                enabled_techniques=[OptimizationTechnique.PREDICTIVE_CACHING]
            )
            
            optimizer = LatencyOptimizer(config)
            
            voice_message = create_test_voice_message(
                "test_conversation",
                f"Test message for {target} latency target"
            )
            
            response = await optimizer.optimize_request(voice_message, mock_processing_callback)
            
            logger.info(f"✓ {target} target ({target_ms}ms):")
            logger.info(f"  Actual latency: {response.total_latency_ms:.1f}ms")
            logger.info(f"  Target met: {'✓' if response.total_latency_ms <= target_ms else '✗'}")
            
            await optimizer.cleanup()
        
        logger.info("✓ Latency targets test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Latency targets test failed: {e}")
        return False


async def test_performance_comparison():
    """Test performance comparison with and without optimization."""
    logger.info("=== Testing Performance Comparison ===")
    
    try:
        conversation_id = "test_conversation_comparison"
        voice_message = create_test_voice_message(
            conversation_id,
            "This is a test message for performance comparison between optimized and unoptimized processing"
        )
        
        # Test without optimization
        start_time = time.time()
        unoptimized_response = await mock_processing_callback(voice_message)
        unoptimized_time = (time.time() - start_time) * 1000
        
        # Test with optimization
        config = create_test_config()
        optimizer = LatencyOptimizer(config)
        
        start_time = time.time()
        optimized_response = await optimizer.optimize_request(voice_message, mock_processing_callback)
        optimized_time = (time.time() - start_time) * 1000
        
        # Compare results
        improvement = ((unoptimized_time - optimized_time) / unoptimized_time) * 100
        
        logger.info(f"✓ Performance comparison:")
        logger.info(f"  Unoptimized: {unoptimized_time:.1f}ms")
        logger.info(f"  Optimized: {optimized_time:.1f}ms")
        logger.info(f"  Improvement: {improvement:.1f}%")
        
        await optimizer.cleanup()
        logger.info("✓ Performance comparison test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {e}")
        return False


async def main():
    """Run all latency optimizer tests."""
    logger.info("🚀 Starting Latency Optimizer Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Latency Profiler", test_latency_profiler),
        ("Predictive Cache", test_predictive_cache),
        ("Optimizer Initialization", test_latency_optimizer_initialization),
        ("Optimization Techniques", test_optimization_techniques),
        ("Adaptive Optimization", test_adaptive_optimization),
        ("Latency Targets", test_latency_targets),
        ("Performance Comparison", test_performance_comparison),
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
    logger.info("🏁 Latency Optimizer Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All latency optimizer tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
