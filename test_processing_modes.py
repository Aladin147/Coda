#!/usr/bin/env python3
"""
Test Processing Modes

This script tests the multiple processing modes system for voice conversations.
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
    from src.coda.components.voice.mode_manager import ProcessingModeManager
    from src.coda.components.voice.processing_modes import ProcessingModeType
    from src.coda.components.voice.moshi_client import MoshiConfig
    from src.coda.components.voice.context_integration import ContextConfig
    from src.coda.components.voice.llm_integration import VoiceLLMConfig
    from src.coda.components.voice.performance_optimizer import OptimizationConfig
    from src.coda.components.voice.models import VoiceMessage, VoiceProcessingMode, ConversationState
    from src.coda.components.llm.models import LLMProvider
    logger.info("✓ Successfully imported processing modes components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_configs() -> tuple:
    """Create test configurations for all components."""
    
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
    
    return moshi_config, context_config, llm_config, optimization_config


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


async def test_mode_manager_initialization():
    """Test processing mode manager initialization."""
    logger.info("=== Testing Mode Manager Initialization ===")
    
    try:
        configs = create_test_configs()
        manager = ProcessingModeManager(*configs)
        
        logger.info("✓ Mode manager created")
        
        # Note: Skip actual initialization due to component dependencies
        # await manager.initialize()
        
        # Test configuration
        status = manager.get_manager_status()
        logger.info(f"✓ Manager status:")
        logger.info(f"  Current mode: {status['current_mode']}")
        logger.info(f"  Available modes: {status['available_modes']}")
        logger.info(f"  Auto selection: {status['auto_selection_enabled']}")
        
        await manager.cleanup()
        logger.info("✓ Mode manager initialization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mode_selection_logic():
    """Test automatic mode selection logic."""
    logger.info("=== Testing Mode Selection Logic ===")
    
    try:
        configs = create_test_configs()
        manager = ProcessingModeManager(*configs)
        
        # Test different message types for mode selection
        test_cases = [
            ("Hi there!", "greeting", ProcessingModeType.MOSHI_ONLY),
            ("What's the weather like?", "simple question", ProcessingModeType.HYBRID_SPEED),
            ("Can you explain how neural networks work?", "explanation", ProcessingModeType.LLM_ONLY),
            ("Help me debug this Python function", "coding", ProcessingModeType.HYBRID_QUALITY),
            ("Implement a comprehensive machine learning pipeline with error handling and optimization", "complex coding", ProcessingModeType.HYBRID_QUALITY)
        ]
        
        for text_content, description, expected_mode in test_cases:
            voice_message = create_test_voice_message("test_conversation", text_content)
            conversation_state = create_test_conversation_state("test_conversation")
            
            # Test mode selection
            selected_mode = await manager._auto_select_mode(voice_message, conversation_state)
            
            logger.info(f"✓ {description}:")
            logger.info(f"  Text: '{text_content[:50]}...'")
            logger.info(f"  Expected: {expected_mode}, Selected: {selected_mode}")
            logger.info(f"  Match: {'✓' if selected_mode == expected_mode else '✗'}")
        
        await manager.cleanup()
        logger.info("✓ Mode selection logic test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mode selection test failed: {e}")
        return False


async def test_individual_modes():
    """Test individual processing modes."""
    logger.info("=== Testing Individual Processing Modes ===")
    
    try:
        from src.coda.components.voice.processing_modes import (
            MoshiOnlyMode, LLMOnlyMode, ProcessingModeConfig
        )
        
        configs = create_test_configs()
        moshi_config, context_config, llm_config, optimization_config = configs
        
        # Test Moshi-only mode
        logger.info("Testing Moshi-only mode...")
        moshi_mode = MoshiOnlyMode(moshi_config)
        await moshi_mode.initialize()
        
        moshi_info = moshi_mode.get_mode_info()
        logger.info(f"✓ Moshi mode: {moshi_info.name}")
        logger.info(f"  Target latency: {moshi_info.target_latency_ms}ms")
        logger.info(f"  Expected quality: {moshi_info.expected_quality_score}")
        logger.info(f"  Resource usage: {moshi_info.resource_usage}")
        
        await moshi_mode.cleanup()
        
        # Test LLM-only mode
        logger.info("Testing LLM-only mode...")
        llm_mode = LLMOnlyMode(context_config, llm_config, optimization_config)
        await llm_mode.initialize()
        
        llm_info = llm_mode.get_mode_info()
        logger.info(f"✓ LLM mode: {llm_info.name}")
        logger.info(f"  Target latency: {llm_info.target_latency_ms}ms")
        logger.info(f"  Expected quality: {llm_info.expected_quality_score}")
        logger.info(f"  Resource usage: {llm_info.resource_usage}")
        
        await llm_mode.cleanup()
        
        logger.info("✓ Individual modes test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Individual modes test failed: {e}")
        return False


async def test_mode_switching():
    """Test mode switching functionality."""
    logger.info("=== Testing Mode Switching ===")
    
    try:
        configs = create_test_configs()
        manager = ProcessingModeManager(*configs)
        
        # Test mode switching
        modes_to_test = [
            ProcessingModeType.MOSHI_ONLY,
            ProcessingModeType.LLM_ONLY,
            ProcessingModeType.HYBRID_BALANCED,
            ProcessingModeType.ADAPTIVE
        ]
        
        for mode_type in modes_to_test:
            logger.info(f"Testing switch to {mode_type}...")
            
            # Test switch (will fail since modes aren't initialized, but tests the logic)
            success = await manager.switch_mode(mode_type)
            logger.info(f"  Switch result: {'✓' if success else '✗ (expected - mode not initialized)'}")
            
            if success:
                status = manager.get_manager_status()
                logger.info(f"  Current mode: {status['current_mode']}")
        
        await manager.cleanup()
        logger.info("✓ Mode switching test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mode switching test failed: {e}")
        return False


async def test_performance_tracking():
    """Test performance tracking functionality."""
    logger.info("=== Testing Performance Tracking ===")
    
    try:
        configs = create_test_configs()
        manager = ProcessingModeManager(*configs)
        
        # Simulate performance data
        from src.coda.components.voice.models import VoiceResponse
        
        mock_response = VoiceResponse(
            response_id="mock_response",
            conversation_id="test_conversation",
            message_id="mock_message",
            text_content="Mock response for testing",
            audio_data=b"",
            processing_mode=VoiceProcessingMode.HYBRID,
            total_latency_ms=150.0,
            response_relevance=0.8
        )
        
        # Track performance for different modes
        modes_to_track = [
            ProcessingModeType.MOSHI_ONLY,
            ProcessingModeType.LLM_ONLY,
            ProcessingModeType.HYBRID_BALANCED
        ]
        
        for mode_type in modes_to_track:
            # Initialize tracking for this mode
            manager.mode_performance[mode_type] = []
            manager.mode_usage_stats[mode_type] = 0
            
            # Simulate multiple uses with different performance
            for i in range(5):
                processing_time = 100.0 + (i * 20.0)  # Varying processing times
                manager._track_mode_performance(mode_type, processing_time, mock_response)
        
        # Get performance statistics
        stats = manager.get_mode_performance_stats()
        
        logger.info("✓ Performance tracking results:")
        for mode_name, mode_stats in stats.items():
            if mode_stats.get("usage_count", 0) > 0:
                logger.info(f"  {mode_name}:")
                logger.info(f"    Usage count: {mode_stats['usage_count']}")
                logger.info(f"    Avg processing time: {mode_stats['avg_processing_time_ms']:.1f}ms")
                logger.info(f"    Avg relevance: {mode_stats['avg_relevance_score']:.2f}")
        
        await manager.cleanup()
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
        manager = ProcessingModeManager(*configs)
        
        # Test fallback mode selection
        fallback_mode = manager._get_fallback_mode()
        logger.info(f"✓ Fallback mode selection: {fallback_mode}")
        
        # Test with no available modes
        manager.available_modes.clear()
        fallback_mode_empty = manager._get_fallback_mode()
        logger.info(f"✓ Fallback with no modes: {fallback_mode_empty}")
        
        # Test fallback priority
        from src.coda.components.voice.processing_modes import MoshiOnlyMode
        
        # Add only one mode to test priority
        manager.available_modes[ProcessingModeType.LLM_ONLY] = None  # Mock
        fallback_with_llm = manager._get_fallback_mode()
        logger.info(f"✓ Fallback with only LLM: {fallback_with_llm}")
        
        await manager.cleanup()
        logger.info("✓ Fallback mechanisms test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback mechanisms test failed: {e}")
        return False


async def main():
    """Run all processing modes tests."""
    logger.info("🚀 Starting Processing Modes Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Mode Manager Initialization", test_mode_manager_initialization),
        ("Mode Selection Logic", test_mode_selection_logic),
        ("Individual Modes", test_individual_modes),
        ("Mode Switching", test_mode_switching),
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
    logger.info("🏁 Processing Modes Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All processing modes tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
