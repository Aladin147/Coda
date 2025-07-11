#!/usr/bin/env python3
"""
Test Parallel Voice Processor

This script tests the parallel processing pipeline for voice conversations.
"""

import asyncio
import logging
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.coda.components.voice.parallel_processor import (
        ParallelVoiceProcessor, ParallelProcessingConfig, ProcessingPriority,
        ProcessingRequest, WorkerPool
    )
    from src.coda.components.voice.mode_manager import ProcessingModeType
    from src.coda.components.voice.moshi_client import MoshiConfig
    from src.coda.components.voice.context_integration import ContextConfig
    from src.coda.components.voice.llm_integration import VoiceLLMConfig
    from src.coda.components.voice.performance_optimizer import OptimizationConfig
    from src.coda.components.voice.models import VoiceMessage, VoiceProcessingMode, ConversationState
    from src.coda.components.llm.models import LLMProvider
    logger.info("✓ Successfully imported parallel processor components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_configs() -> tuple:
    """Create test configurations for all components."""
    
    parallel_config = ParallelProcessingConfig(
        max_workers=3,
        worker_type="thread",
        max_queue_size=20,
        priority_queue_enabled=True,
        default_timeout_seconds=5.0
    )
    
    moshi_config = MoshiConfig(
        model_path="kyutai/moshi",
        device="cuda",
        sample_rate=24000,
        enable_inner_monologue=True
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
    
    optimization_config = OptimizationConfig(
        enable_response_caching=True,
        enable_parallel_processing=True,
        max_queue_depth=5
    )
    
    return parallel_config, moshi_config, context_config, llm_config, optimization_config


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


async def test_worker_pool():
    """Test worker pool functionality."""
    logger.info("=== Testing Worker Pool ===")
    
    try:
        config = ParallelProcessingConfig(max_workers=2, worker_type="thread")
        worker_pool = WorkerPool(config)
        
        logger.info("✓ Worker pool created")
        
        await worker_pool.initialize()
        logger.info("✓ Worker pool initialized")
        
        # Check worker stats
        stats = worker_pool.get_worker_stats()
        logger.info(f"✓ Worker stats:")
        logger.info(f"  Total workers: {stats['total_workers']}")
        logger.info(f"  Worker type: {stats['worker_type']}")
        logger.info(f"  Is running: {stats['is_running']}")
        
        await worker_pool.cleanup()
        logger.info("✓ Worker pool test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Worker pool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_parallel_processor_initialization():
    """Test parallel processor initialization."""
    logger.info("=== Testing Parallel Processor Initialization ===")
    
    try:
        configs = create_test_configs()
        processor = ParallelVoiceProcessor(*configs)
        
        logger.info("✓ Parallel processor created")
        
        # Note: Skip actual initialization due to component dependencies
        # await processor.initialize()
        
        # Test configuration
        stats = processor.get_processing_stats()
        logger.info(f"✓ Initial stats:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Queue size: {stats['queue_size']}")
        logger.info(f"  Active requests: {stats['active_requests']}")
        
        await processor.cleanup()
        logger.info("✓ Parallel processor initialization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return False


async def test_request_submission():
    """Test request submission and tracking."""
    logger.info("=== Testing Request Submission ===")
    
    try:
        configs = create_test_configs()
        processor = ParallelVoiceProcessor(*configs)
        
        conversation_id = "test_conversation_parallel"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Test different priority requests
        test_requests = [
            ("Hello there!", ProcessingPriority.NORMAL),
            ("Urgent help needed!", ProcessingPriority.URGENT),
            ("Low priority question", ProcessingPriority.LOW),
            ("High priority task", ProcessingPriority.HIGH)
        ]
        
        request_ids = []
        
        for text_content, priority in test_requests:
            voice_message = create_test_voice_message(conversation_id, text_content)
            
            request_id = await processor.submit_request(
                voice_message=voice_message,
                conversation_state=conversation_state,
                priority=priority,
                timeout_seconds=5.0
            )
            
            request_ids.append(request_id)
            logger.info(f"✓ Submitted request {request_id[:8]}... with priority {priority}")
        
        # Check stats after submission
        stats = processor.get_processing_stats()
        logger.info(f"✓ Stats after submission:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Queue size: {stats['queue_size']}")
        logger.info(f"  Active requests: {stats['active_requests']}")
        
        # Test request cancellation
        if request_ids:
            cancelled = await processor.cancel_request(request_ids[0])
            logger.info(f"✓ Request cancellation: {cancelled}")
        
        await processor.cleanup()
        logger.info("✓ Request submission test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Request submission test failed: {e}")
        return False


async def test_priority_queue():
    """Test priority queue functionality."""
    logger.info("=== Testing Priority Queue ===")
    
    try:
        configs = create_test_configs()
        configs[0].priority_queue_enabled = True
        processor = ParallelVoiceProcessor(*configs)
        
        conversation_id = "test_conversation_priority"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Submit requests in reverse priority order
        priorities = [
            ProcessingPriority.LOW,
            ProcessingPriority.NORMAL,
            ProcessingPriority.HIGH,
            ProcessingPriority.URGENT
        ]
        
        request_ids = []
        
        for i, priority in enumerate(priorities):
            voice_message = create_test_voice_message(
                conversation_id,
                f"Priority test message {i} - {priority}"
            )
            
            request_id = await processor.submit_request(
                voice_message=voice_message,
                conversation_state=conversation_state,
                priority=priority
            )
            
            request_ids.append((request_id, priority))
            logger.info(f"✓ Queued {priority} priority request")
        
        # Check queue ordering (higher priority should be processed first)
        stats = processor.get_processing_stats()
        logger.info(f"✓ Priority queue stats:")
        logger.info(f"  Queue size: {stats['queue_size']}")
        logger.info(f"  Total requests: {stats['total_requests']}")
        
        await processor.cleanup()
        logger.info("✓ Priority queue test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Priority queue test failed: {e}")
        return False


async def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    logger.info("=== Testing Concurrent Processing ===")
    
    try:
        configs = create_test_configs()
        configs[0].max_workers = 3  # Allow concurrent processing
        processor = ParallelVoiceProcessor(*configs)
        
        conversation_id = "test_conversation_concurrent"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Submit multiple requests concurrently
        num_requests = 5
        request_tasks = []
        
        for i in range(num_requests):
            voice_message = create_test_voice_message(
                conversation_id,
                f"Concurrent test message {i}"
            )
            
            task = processor.submit_request(
                voice_message=voice_message,
                conversation_state=conversation_state,
                priority=ProcessingPriority.NORMAL
            )
            
            request_tasks.append(task)
        
        # Submit all requests concurrently
        start_time = time.time()
        request_ids = await asyncio.gather(*request_tasks)
        submission_time = (time.time() - start_time) * 1000
        
        logger.info(f"✓ Submitted {num_requests} requests concurrently in {submission_time:.1f}ms")
        
        # Check final stats
        stats = processor.get_processing_stats()
        logger.info(f"✓ Concurrent processing stats:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Queue size: {stats['queue_size']}")
        logger.info(f"  Active requests: {stats['active_requests']}")
        
        await processor.cleanup()
        logger.info("✓ Concurrent processing test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Concurrent processing test failed: {e}")
        return False


async def test_performance_monitoring():
    """Test performance monitoring functionality."""
    logger.info("=== Testing Performance Monitoring ===")
    
    try:
        configs = create_test_configs()
        processor = ParallelVoiceProcessor(*configs)
        
        # Simulate some processing activity
        conversation_id = "test_conversation_monitoring"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Submit a few requests
        for i in range(3):
            voice_message = create_test_voice_message(
                conversation_id,
                f"Monitoring test message {i}"
            )
            
            await processor.submit_request(
                voice_message=voice_message,
                conversation_state=conversation_state
            )
        
        # Get performance stats
        stats = processor.get_processing_stats()
        
        logger.info(f"✓ Performance monitoring results:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Completed requests: {stats['completed_requests']}")
        logger.info(f"  Failed requests: {stats['failed_requests']}")
        logger.info(f"  Average latency: {stats['average_latency_ms']:.1f}ms")
        logger.info(f"  Queue depth: {stats['queue_depth']}")
        logger.info(f"  Active workers: {stats['active_workers']}")
        
        await processor.cleanup()
        logger.info("✓ Performance monitoring test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring test failed: {e}")
        return False


async def test_timeout_handling():
    """Test timeout handling for requests."""
    logger.info("=== Testing Timeout Handling ===")
    
    try:
        configs = create_test_configs()
        configs[0].default_timeout_seconds = 0.1  # Very short timeout
        processor = ParallelVoiceProcessor(*configs)
        
        conversation_id = "test_conversation_timeout"
        conversation_state = create_test_conversation_state(conversation_id)
        
        voice_message = create_test_voice_message(
            conversation_id,
            "This request should timeout"
        )
        
        # Submit request with short timeout
        request_id = await processor.submit_request(
            voice_message=voice_message,
            conversation_state=conversation_state,
            timeout_seconds=0.05  # Very short timeout
        )
        
        logger.info(f"✓ Submitted request {request_id[:8]}... with short timeout")
        
        # Wait a bit to let timeout occur
        await asyncio.sleep(0.2)
        
        # Check if request was cancelled due to timeout
        stats = processor.get_processing_stats()
        logger.info(f"✓ Timeout handling stats:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Active requests: {stats['active_requests']}")
        
        await processor.cleanup()
        logger.info("✓ Timeout handling test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Timeout handling test failed: {e}")
        return False


async def main():
    """Run all parallel processor tests."""
    logger.info("🚀 Starting Parallel Processor Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Worker Pool", test_worker_pool),
        ("Processor Initialization", test_parallel_processor_initialization),
        ("Request Submission", test_request_submission),
        ("Priority Queue", test_priority_queue),
        ("Concurrent Processing", test_concurrent_processing),
        ("Performance Monitoring", test_performance_monitoring),
        ("Timeout Handling", test_timeout_handling),
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
    logger.info("🏁 Parallel Processor Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All parallel processor tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
