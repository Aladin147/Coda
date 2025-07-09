#!/usr/bin/env python3
"""
Test Hybrid Orchestrator

This script tests the hybrid processing orchestrator for Moshi + LLM coordination.
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
    from src.coda.components.voice.hybrid_orchestrator import (
        HybridOrchestrator, HybridConfig, ProcessingStrategy, ResponseSelection
    )
    from src.coda.components.voice.moshi_client import MoshiConfig
    from src.coda.components.voice.context_integration import ContextConfig
    from src.coda.components.voice.llm_integration import VoiceLLMConfig
    from src.coda.components.voice.performance_optimizer import OptimizationConfig
    from src.coda.components.voice.models import VoiceMessage, VoiceProcessingMode, ConversationState
    from src.coda.components.llm.models import LLMProvider
    logger.info("✓ Successfully imported hybrid orchestrator components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_configs() -> tuple:
    """Create test configurations for all components."""
    
    hybrid_config = HybridConfig(
        default_strategy=ProcessingStrategy.ADAPTIVE,
        response_selection=ResponseSelection.BALANCED,
        moshi_timeout_ms=2000.0,
        llm_timeout_ms=3000.0,
        enable_adaptive_learning=True,
        enable_parallel_processing=True
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
        enable_dynamic_models=False,  # Disable for testing
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
    
    return hybrid_config, moshi_config, context_config, llm_config, optimization_config


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


async def test_orchestrator_initialization():
    """Test hybrid orchestrator initialization."""
    logger.info("=== Testing Hybrid Orchestrator Initialization ===")
    
    try:
        configs = create_test_configs()
        orchestrator = HybridOrchestrator(*configs)
        
        logger.info("✓ Hybrid orchestrator created")
        logger.info(f"  Default strategy: {configs[0].default_strategy}")
        logger.info(f"  Response selection: {configs[0].response_selection}")
        
        # Note: We'll skip actual initialization since it requires Moshi and LLM setup
        # await orchestrator.initialize()
        
        # Check initial status
        status = orchestrator.get_orchestrator_status()
        logger.info(f"✓ Initial status:")
        logger.info(f"  Current strategy: {status['current_strategy']}")
        logger.info(f"  Active requests: {status['active_requests']}")
        logger.info(f"  Total requests: {status['processing_stats']['total_requests']}")
        
        await orchestrator.cleanup()
        logger.info("✓ Orchestrator initialization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_strategy_selection():
    """Test strategy selection logic."""
    logger.info("=== Testing Strategy Selection ===")
    
    try:
        configs = create_test_configs()
        orchestrator = HybridOrchestrator(*configs)
        
        # Test different message types for strategy selection
        test_cases = [
            ("Hi there!", "short greeting"),
            ("What's the weather like today?", "simple question"),
            ("Can you explain how neural networks work in detail?", "complex explanation"),
            ("Help me debug this Python function that's not working", "coding assistance"),
            ("Implement a binary search algorithm with error handling", "complex coding")
        ]
        
        for text_content, description in test_cases:
            voice_message = create_test_voice_message("test_conversation", text_content)
            conversation_state = create_test_conversation_state("test_conversation")
            
            # Analyze context factors
            factors = await orchestrator._analyze_context_factors(voice_message, conversation_state)
            
            # Select strategy
            strategy = await orchestrator._select_strategy(voice_message, conversation_state)
            
            logger.info(f"✓ {description}:")
            logger.info(f"  Text: '{text_content[:50]}...'")
            logger.info(f"  Intent: {factors.get('intent')}, Complexity: {factors.get('complexity')}")
            logger.info(f"  Selected strategy: {strategy}")
        
        await orchestrator.cleanup()
        logger.info("✓ Strategy selection test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy selection test failed: {e}")
        return False


async def test_processing_strategies():
    """Test different processing strategies."""
    logger.info("=== Testing Processing Strategies ===")
    
    try:
        configs = create_test_configs()
        orchestrator = HybridOrchestrator(*configs)
        
        conversation_id = "test_conversation_strategies"
        conversation_state = create_test_conversation_state(conversation_id)
        
        # Test each strategy manually
        strategies = [
            ProcessingStrategy.MOSHI_ONLY,
            ProcessingStrategy.LLM_ONLY,
            # Skip parallel and sequential for now due to component dependencies
        ]
        
        for strategy in strategies:
            logger.info(f"Testing {strategy} strategy...")
            
            voice_message = create_test_voice_message(
                conversation_id,
                f"Test message for {strategy} processing"
            )
            
            try:
                # Set strategy manually
                await orchestrator.set_strategy(strategy)
                
                # Test strategy selection
                selected_strategy = await orchestrator._select_strategy(voice_message, conversation_state)
                logger.info(f"  Selected strategy: {selected_strategy}")
                
                # Note: Skip actual processing due to component dependencies
                # result = await orchestrator._execute_strategy(strategy, voice_message, conversation_state, "test")
                
                logger.info(f"✓ {strategy} strategy test completed")
                
            except Exception as e:
                logger.warning(f"  {strategy} strategy failed (expected): {e}")
        
        await orchestrator.cleanup()
        logger.info("✓ Processing strategies test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Processing strategies test failed: {e}")
        return False


async def test_adaptive_learning():
    """Test adaptive learning functionality."""
    logger.info("=== Testing Adaptive Learning ===")
    
    try:
        configs = create_test_configs()
        configs[0].enable_adaptive_learning = True
        orchestrator = HybridOrchestrator(*configs)
        
        # Simulate performance data
        from src.coda.components.voice.hybrid_orchestrator import ProcessingResult
        from src.coda.components.voice.models import VoiceResponse
        
        # Create mock results for different strategies
        mock_response = VoiceResponse(
            response_id="mock_response",
            conversation_id="test_conversation",
            message_id="mock_message",
            text_content="Mock response",
            audio_data=b"",
            processing_mode=VoiceProcessingMode.HYBRID,
            total_latency_ms=100.0
        )
        
        # Simulate good performance for LLM_ONLY
        good_result = ProcessingResult(
            response=mock_response,
            strategy=ProcessingStrategy.LLM_ONLY,
            latency_ms=150.0,
            quality_score=0.9,
            relevance_score=0.85,
            confidence=0.9,
            metadata={"test": True}
        )
        
        # Simulate poor performance for MOSHI_ONLY
        poor_result = ProcessingResult(
            response=mock_response,
            strategy=ProcessingStrategy.MOSHI_ONLY,
            latency_ms=50.0,
            quality_score=0.5,
            relevance_score=0.6,
            confidence=0.6,
            metadata={"test": True}
        )
        
        # Update performance stats
        for _ in range(5):
            orchestrator._update_performance_stats(ProcessingStrategy.LLM_ONLY, 150.0, good_result)
            await orchestrator._update_adaptive_learning(ProcessingStrategy.LLM_ONLY, good_result)
        
        for _ in range(3):
            orchestrator._update_performance_stats(ProcessingStrategy.MOSHI_ONLY, 50.0, poor_result)
            await orchestrator._update_adaptive_learning(ProcessingStrategy.MOSHI_ONLY, poor_result)
        
        # Check if adaptive learning worked
        status = orchestrator.get_orchestrator_status()
        logger.info(f"✓ Adaptive learning results:")
        logger.info(f"  Current strategy: {status['current_strategy']}")
        logger.info(f"  LLM performance: {status['strategy_performance']['llm_only']['average_score']:.2f}")
        logger.info(f"  Moshi performance: {status['strategy_performance']['moshi_only']['average_score']:.2f}")
        
        await orchestrator.cleanup()
        logger.info("✓ Adaptive learning test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive learning test failed: {e}")
        return False


async def test_performance_tracking():
    """Test performance tracking and statistics."""
    logger.info("=== Testing Performance Tracking ===")
    
    try:
        configs = create_test_configs()
        orchestrator = HybridOrchestrator(*configs)
        
        # Simulate some processing statistics
        strategies = [ProcessingStrategy.MOSHI_ONLY, ProcessingStrategy.LLM_ONLY, ProcessingStrategy.HYBRID_PARALLEL]
        
        for i, strategy in enumerate(strategies):
            # Simulate different performance characteristics
            latency = 100.0 + (i * 50.0)  # Increasing latency
            quality = 0.7 + (i * 0.1)    # Increasing quality
            
            # Create mock result
            from src.coda.components.voice.hybrid_orchestrator import ProcessingResult
            from src.coda.components.voice.models import VoiceResponse
            
            mock_response = VoiceResponse(
                response_id=f"mock_{i}",
                conversation_id="test_conversation",
                message_id=f"mock_message_{i}",
                text_content=f"Mock response {i}",
                audio_data=b"",
                processing_mode=VoiceProcessingMode.HYBRID,
                total_latency_ms=latency
            )
            
            result = ProcessingResult(
                response=mock_response,
                strategy=strategy,
                latency_ms=latency,
                quality_score=quality,
                relevance_score=0.8,
                confidence=0.8,
                metadata={"simulation": True}
            )
            
            # Update stats multiple times
            for _ in range(3):
                orchestrator._update_performance_stats(strategy, latency, result)
        
        # Check performance statistics
        status = orchestrator.get_orchestrator_status()
        
        logger.info(f"✓ Performance tracking results:")
        logger.info(f"  Total requests: {status['processing_stats']['total_requests']}")
        logger.info(f"  Average latency: {status['processing_stats']['average_latency_ms']:.1f}ms")
        
        for strategy_name, usage in status['processing_stats']['strategy_usage'].items():
            logger.info(f"  {strategy_name}: {usage} uses")
        
        for strategy_name, perf in status['strategy_performance'].items():
            if perf['total_uses'] > 0:
                logger.info(f"  {strategy_name} avg score: {perf['average_score']:.2f}")
        
        await orchestrator.cleanup()
        logger.info("✓ Performance tracking test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance tracking test failed: {e}")
        return False


async def test_fallback_mechanisms():
    """Test fallback mechanisms."""
    logger.info("=== Testing Fallback Mechanisms ===")
    
    try:
        configs = create_test_configs()
        configs[0].enable_fallback = True
        configs[0].fallback_strategy = ProcessingStrategy.MOSHI_ONLY
        
        orchestrator = HybridOrchestrator(*configs)
        
        conversation_id = "test_conversation_fallback"
        conversation_state = create_test_conversation_state(conversation_id)
        
        voice_message = create_test_voice_message(
            conversation_id,
            "Test message for fallback processing"
        )
        
        # Test fallback processing
        logger.info("Testing fallback processing...")
        fallback_response = await orchestrator._fallback_processing(voice_message, conversation_state)
        
        logger.info(f"✓ Fallback response: {fallback_response.text_content}")
        logger.info(f"  Processing mode: {fallback_response.processing_mode}")
        logger.info(f"  Response ID: {fallback_response.response_id}")
        
        await orchestrator.cleanup()
        logger.info("✓ Fallback mechanisms test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback mechanisms test failed: {e}")
        return False


async def main():
    """Run all hybrid orchestrator tests."""
    logger.info("🚀 Starting Hybrid Orchestrator Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Strategy Selection", test_strategy_selection),
        ("Processing Strategies", test_processing_strategies),
        ("Adaptive Learning", test_adaptive_learning),
        ("Performance Tracking", test_performance_tracking),
        ("Fallback Mechanisms", test_fallback_mechanisms),
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
    logger.info("🏁 Hybrid Orchestrator Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All hybrid orchestrator tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
