#!/usr/bin/env python3
"""
Test Fallback Manager

This script tests the fallback management system for voice processing components.
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
    from src.coda.components.voice.fallback_manager import (
        FallbackManager, FallbackConfig, FallbackRule, FailureType, FallbackStrategy,
        FallbackPriority, CircuitBreaker, FailureEvent
    )
    from src.coda.components.voice.mode_manager import ProcessingModeType
    from src.coda.components.voice.models import VoiceMessage, VoiceResponse, VoiceProcessingMode
    logger.info("✓ Successfully imported fallback manager components")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    exit(1)


def create_test_config() -> FallbackConfig:
    """Create test configuration for fallback management."""
    
    # Define fallback rules
    rules = [
        FallbackRule(
            rule_id="timeout_retry",
            failure_types=[FailureType.TIMEOUT],
            components=["moshi", "llm"],
            strategy=FallbackStrategy.RETRY,
            priority=FallbackPriority.HIGH,
            max_retries=2,
            retry_delay_seconds=0.5
        ),
        FallbackRule(
            rule_id="component_alternative",
            failure_types=[FailureType.COMPONENT_FAILURE],
            components=["moshi"],
            strategy=FallbackStrategy.ALTERNATIVE_COMPONENT,
            priority=FallbackPriority.MEDIUM,
            alternative_component="llm"
        ),
        FallbackRule(
            rule_id="resource_degraded",
            failure_types=[FailureType.RESOURCE_EXHAUSTION],
            components=["llm"],
            strategy=FallbackStrategy.DEGRADED_MODE,
            priority=FallbackPriority.MEDIUM,
            degraded_mode=ProcessingModeType.MOSHI_ONLY
        ),
        FallbackRule(
            rule_id="emergency_fallback",
            failure_types=[FailureType.UNKNOWN_ERROR],
            components=[],  # Apply to all components
            strategy=FallbackStrategy.EMERGENCY_RESPONSE,
            priority=FallbackPriority.LOWEST
        )
    ]
    
    return FallbackConfig(
        enable_fallbacks=True,
        max_total_retries=3,
        failure_threshold=2,
        recovery_timeout_seconds=5.0,
        emergency_response_enabled=True,
        emergency_response_text="I'm experiencing technical difficulties. Please try again later.",
        enable_circuit_breaker=True,
        circuit_breaker_timeout_seconds=10.0,
        fallback_rules=rules
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


class MockFailingOperation:
    """Mock operation that can be configured to fail."""
    
    def __init__(self, failure_type: FailureType = None, fail_count: int = 0):
        self.failure_type = failure_type
        self.fail_count = fail_count
        self.call_count = 0
    
    async def __call__(self, voice_message: VoiceMessage) -> VoiceResponse:
        self.call_count += 1
        
        # Fail for the specified number of calls
        if self.call_count <= self.fail_count and self.failure_type:
            if self.failure_type == FailureType.TIMEOUT:
                raise asyncio.TimeoutError("Mock timeout error")
            elif self.failure_type == FailureType.COMPONENT_FAILURE:
                raise RuntimeError("Mock component failure")
            elif self.failure_type == FailureType.RESOURCE_EXHAUSTION:
                raise MemoryError("Mock resource exhaustion")
            elif self.failure_type == FailureType.NETWORK_ERROR:
                raise ConnectionError("Mock network error")
            else:
                raise Exception("Mock unknown error")
        
        # Success case
        return VoiceResponse(
            response_id=f"mock_{voice_message.message_id}",
            conversation_id=voice_message.conversation_id,
            message_id=voice_message.message_id,
            text_content=f"Mock response to: {voice_message.text_content[:50]}...",
            audio_data=b"",
            processing_mode=VoiceProcessingMode.HYBRID,
            total_latency_ms=100.0,
            response_relevance=0.8
        )


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    logger.info("=== Testing Circuit Breaker ===")
    
    try:
        circuit_breaker = CircuitBreaker("test_component", failure_threshold=2, timeout_seconds=1.0)
        
        # Initially should allow execution
        assert circuit_breaker.can_execute() == True
        logger.info("✓ Circuit breaker initially closed")
        
        # Record failures
        circuit_breaker.record_failure()
        assert circuit_breaker.can_execute() == True
        logger.info("✓ Circuit breaker still closed after 1 failure")
        
        circuit_breaker.record_failure()
        assert circuit_breaker.can_execute() == False
        logger.info("✓ Circuit breaker opened after 2 failures")
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        assert circuit_breaker.can_execute() == True
        logger.info("✓ Circuit breaker reset after timeout")
        
        # Test success reset
        circuit_breaker.record_failure()
        circuit_breaker.record_success()
        assert circuit_breaker.can_execute() == True
        logger.info("✓ Circuit breaker reset after success")
        
        # Check status
        status = circuit_breaker.get_status()
        logger.info(f"✓ Circuit breaker status: {status['is_open']}")
        
        logger.info("✓ Circuit breaker test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Circuit breaker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_manager_initialization():
    """Test fallback manager initialization."""
    logger.info("=== Testing Fallback Manager Initialization ===")
    
    try:
        config = create_test_config()
        manager = FallbackManager(config)
        
        logger.info("✓ Fallback manager created")
        logger.info(f"  Fallback rules: {len(manager.fallback_rules)}")
        logger.info(f"  Emergency response enabled: {config.emergency_response_enabled}")
        logger.info(f"  Circuit breaker enabled: {config.enable_circuit_breaker}")
        
        # Check initial statistics
        stats = manager.get_failure_statistics()
        logger.info(f"✓ Initial statistics:")
        logger.info(f"  Total failures: {stats['total_failures']}")
        logger.info(f"  Emergency responses: {stats['emergency_responses']}")
        logger.info(f"  Circuit breaker activations: {stats['circuit_breaker_activations']}")
        
        await manager.cleanup()
        logger.info("✓ Fallback manager initialization test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return False


async def test_retry_fallback():
    """Test retry fallback strategy."""
    logger.info("=== Testing Retry Fallback ===")
    
    try:
        config = create_test_config()
        manager = FallbackManager(config)
        
        voice_message = create_test_voice_message("test_conversation", "Test retry fallback")
        
        # Create operation that fails twice then succeeds
        failing_operation = MockFailingOperation(FailureType.TIMEOUT, fail_count=2)
        
        # Execute with fallback
        response = await manager.execute_with_fallback(
            failing_operation,
            "moshi",
            voice_message
        )
        
        logger.info(f"✓ Retry fallback successful:")
        logger.info(f"  Operation called: {failing_operation.call_count} times")
        logger.info(f"  Response: {response.text_content[:50]}...")
        
        # Check statistics
        stats = manager.get_failure_statistics()
        logger.info(f"✓ Failure statistics:")
        logger.info(f"  Total failures: {stats['total_failures']}")
        logger.info(f"  Recent failures: {stats['recent_failures']}")
        
        await manager.cleanup()
        logger.info("✓ Retry fallback test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Retry fallback test failed: {e}")
        return False


async def test_alternative_component_fallback():
    """Test alternative component fallback strategy."""
    logger.info("=== Testing Alternative Component Fallback ===")
    
    try:
        config = create_test_config()
        manager = FallbackManager(config)
        
        voice_message = create_test_voice_message("test_conversation", "Test alternative component")
        
        # Create operation that always fails
        failing_operation = MockFailingOperation(FailureType.COMPONENT_FAILURE, fail_count=10)
        
        # Execute with fallback
        response = await manager.execute_with_fallback(
            failing_operation,
            "moshi",
            voice_message
        )
        
        logger.info(f"✓ Alternative component fallback:")
        logger.info(f"  Response: {response.text_content[:50]}...")
        logger.info(f"  Processing mode: {response.processing_mode}")
        logger.info(f"  Response relevance: {response.response_relevance}")
        
        await manager.cleanup()
        logger.info("✓ Alternative component fallback test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Alternative component fallback test failed: {e}")
        return False


async def test_degraded_mode_fallback():
    """Test degraded mode fallback strategy."""
    logger.info("=== Testing Degraded Mode Fallback ===")
    
    try:
        config = create_test_config()
        manager = FallbackManager(config)
        
        voice_message = create_test_voice_message("test_conversation", "Test degraded mode")
        
        # Create operation that fails with resource exhaustion
        failing_operation = MockFailingOperation(FailureType.RESOURCE_EXHAUSTION, fail_count=10)
        
        # Execute with fallback
        response = await manager.execute_with_fallback(
            failing_operation,
            "llm",
            voice_message
        )
        
        logger.info(f"✓ Degraded mode fallback:")
        logger.info(f"  Response: {response.text_content[:50]}...")
        logger.info(f"  Processing mode: {response.processing_mode}")
        logger.info(f"  Latency: {response.total_latency_ms}ms")
        
        await manager.cleanup()
        logger.info("✓ Degraded mode fallback test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Degraded mode fallback test failed: {e}")
        return False


async def test_emergency_response():
    """Test emergency response fallback."""
    logger.info("=== Testing Emergency Response ===")
    
    try:
        config = create_test_config()
        manager = FallbackManager(config)
        
        voice_message = create_test_voice_message("test_conversation", "Test emergency response")
        
        # Create operation that always fails with unknown error
        failing_operation = MockFailingOperation(FailureType.UNKNOWN_ERROR, fail_count=10)
        
        # Execute with fallback
        response = await manager.execute_with_fallback(
            failing_operation,
            "unknown_component",
            voice_message
        )
        
        logger.info(f"✓ Emergency response:")
        logger.info(f"  Response: {response.text_content}")
        logger.info(f"  Response ID: {response.response_id}")
        logger.info(f"  Latency: {response.total_latency_ms}ms")
        
        # Check statistics
        stats = manager.get_failure_statistics()
        logger.info(f"✓ Emergency response statistics:")
        logger.info(f"  Emergency responses: {stats['emergency_responses']}")
        
        await manager.cleanup()
        logger.info("✓ Emergency response test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Emergency response test failed: {e}")
        return False


async def test_circuit_breaker_integration():
    """Test circuit breaker integration with fallback manager."""
    logger.info("=== Testing Circuit Breaker Integration ===")
    
    try:
        config = create_test_config()
        config.failure_threshold = 2  # Low threshold for testing
        manager = FallbackManager(config)
        
        voice_message = create_test_voice_message("test_conversation", "Test circuit breaker")
        
        # Create operation that always fails
        failing_operation = MockFailingOperation(FailureType.COMPONENT_FAILURE, fail_count=10)
        
        # Execute multiple times to trigger circuit breaker
        responses = []
        for i in range(4):
            response = await manager.execute_with_fallback(
                failing_operation,
                "test_component",
                voice_message
            )
            responses.append(response)
            logger.info(f"  Execution {i+1}: {response.response_id}")
        
        # Check circuit breaker status
        stats = manager.get_failure_statistics()
        logger.info(f"✓ Circuit breaker integration:")
        logger.info(f"  Circuit breaker activations: {stats['circuit_breaker_activations']}")
        logger.info(f"  Total failures: {stats['total_failures']}")
        
        # Check circuit breaker status
        cb_status = stats['circuit_breaker_status'].get('test_component', {})
        logger.info(f"  Circuit breaker open: {cb_status.get('is_open', False)}")
        
        await manager.cleanup()
        logger.info("✓ Circuit breaker integration test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Circuit breaker integration test failed: {e}")
        return False


async def test_fallback_rule_management():
    """Test fallback rule management."""
    logger.info("=== Testing Fallback Rule Management ===")
    
    try:
        config = create_test_config()
        manager = FallbackManager(config)
        
        initial_rule_count = len(manager.fallback_rules)
        logger.info(f"✓ Initial rule count: {initial_rule_count}")
        
        # Add new rule
        new_rule = FallbackRule(
            rule_id="test_rule",
            failure_types=[FailureType.NETWORK_ERROR],
            components=["test_component"],
            strategy=FallbackStrategy.RETRY,
            priority=FallbackPriority.HIGH
        )
        
        manager.add_fallback_rule(new_rule)
        assert len(manager.fallback_rules) == initial_rule_count + 1
        logger.info("✓ Successfully added new fallback rule")
        
        # Remove rule
        success = manager.remove_fallback_rule("test_rule")
        assert success == True
        assert len(manager.fallback_rules) == initial_rule_count
        logger.info("✓ Successfully removed fallback rule")
        
        # Try to remove non-existent rule
        success = manager.remove_fallback_rule("non_existent")
        assert success == False
        logger.info("✓ Correctly handled non-existent rule removal")
        
        await manager.cleanup()
        logger.info("✓ Fallback rule management test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback rule management test failed: {e}")
        return False


async def test_failure_statistics():
    """Test failure statistics and monitoring."""
    logger.info("=== Testing Failure Statistics ===")
    
    try:
        config = create_test_config()
        manager = FallbackManager(config)
        
        voice_message = create_test_voice_message("test_conversation", "Test statistics")
        
        # Generate various types of failures
        failure_types = [
            FailureType.TIMEOUT,
            FailureType.COMPONENT_FAILURE,
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.NETWORK_ERROR
        ]
        
        for failure_type in failure_types:
            failing_operation = MockFailingOperation(failure_type, fail_count=1)
            
            try:
                await manager.execute_with_fallback(
                    failing_operation,
                    f"component_{failure_type.value}",
                    voice_message
                )
            except:
                pass  # Expected to fail
        
        # Get comprehensive statistics
        stats = manager.get_failure_statistics()
        
        logger.info(f"✓ Failure statistics:")
        logger.info(f"  Total failures: {stats['total_failures']}")
        logger.info(f"  Recent failures: {stats['recent_failures']}")
        logger.info(f"  Failure by type: {stats['failure_by_type']}")
        logger.info(f"  Failure by component: {stats['failure_by_component']}")
        logger.info(f"  Circuit breakers: {len(stats['circuit_breaker_status'])}")
        
        await manager.cleanup()
        logger.info("✓ Failure statistics test completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failure statistics test failed: {e}")
        return False


async def main():
    """Run all fallback manager tests."""
    logger.info("🚀 Starting Fallback Manager Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Circuit Breaker", test_circuit_breaker),
        ("Manager Initialization", test_fallback_manager_initialization),
        ("Retry Fallback", test_retry_fallback),
        ("Alternative Component", test_alternative_component_fallback),
        ("Degraded Mode", test_degraded_mode_fallback),
        ("Emergency Response", test_emergency_response),
        ("Circuit Breaker Integration", test_circuit_breaker_integration),
        ("Rule Management", test_fallback_rule_management),
        ("Failure Statistics", test_failure_statistics),
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
    logger.info("🏁 Fallback Manager Test Results")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All fallback manager tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
