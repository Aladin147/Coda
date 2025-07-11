#!/usr/bin/env python3
"""
Quick validation test for Phase 1.5 fixes.

This script validates that the critical issues found in stress testing have been fixed:
1. Session history query bug (unhashable type: 'list')
2. Event coordinator initialization bug (await on non-coroutine)
3. Exception handling bug (duplicate exception classes)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fixes_validation")

# Import components
try:
    from coda.core.session_manager import SessionManager
    from coda.core.event_coordinator import EventCoordinator
    from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
    from coda.components.llm.manager import LLMManager
    from coda.core.events import get_event_bus
    
except ImportError as e:
    logger.error(f"Failed to import Coda components: {e}")
    exit(1)


async def test_session_history_fix():
    """Test that session history query is fixed."""
    logger.info("🧪 Testing Session History Fix...")
    
    try:
        session_manager = SessionManager()
        await session_manager.initialize()
        
        # Create session and add messages
        session_id = await session_manager.create_session()
        await session_manager.add_message_to_session(session_id, "user", "Test message 1")
        await session_manager.add_message_to_session(session_id, "assistant", "Test response 1")
        
        # This should not raise "unhashable type: 'list'" error
        history = session_manager.get_session_history(session_id)
        
        logger.info(f"✅ Session history retrieved successfully: {len(history)} messages")
        return True
        
    except Exception as e:
        logger.error(f"❌ Session history fix failed: {e}")
        return False


async def test_event_coordinator_fix():
    """Test that event coordinator initialization is fixed."""
    logger.info("🧪 Testing Event Coordinator Fix...")
    
    try:
        # Initialize event bus first
        event_bus = get_event_bus()
        await event_bus.start()
        
        # This should not raise "object NoneType can't be used in 'await' expression"
        event_coordinator = EventCoordinator()
        await event_coordinator.initialize()
        
        # Test event emission
        await event_coordinator.emit_gui_event("test_event", {"test": "data"})
        
        logger.info("✅ Event coordinator initialized and working successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Event coordinator fix failed: {e}")
        return False


async def test_llm_exception_handling_fix():
    """Test that LLM exception handling is fixed."""
    logger.info("🧪 Testing LLM Exception Handling Fix...")
    
    try:
        # Test with valid configuration first
        config = LLMConfig(
            providers={
                "ollama": ProviderConfig(
                    provider=LLMProvider.OLLAMA,
                    model="qwen3:30b-a3b",
                    host="http://localhost:11434",
                    temperature=0.7,
                    max_tokens=50,
                    system_message="/no_think Respond briefly."
                )
            },
            default_provider="ollama"
        )
        
        llm_manager = LLMManager(config)
        
        # This should not raise "catching classes that do not inherit from BaseException"
        response = await llm_manager.generate_response(
            prompt="/no_think Test message",
            provider="ollama"
        )
        
        logger.info(f"✅ LLM exception handling working: {response.content[:30]}...")
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM exception handling fix failed: {e}")
        return False


async def test_concurrent_operations():
    """Test concurrent operations work without issues."""
    logger.info("🧪 Testing Concurrent Operations...")
    
    try:
        session_manager = SessionManager()
        await session_manager.initialize()
        
        # Create multiple sessions concurrently
        session_tasks = []
        for i in range(10):
            task = session_manager.create_session(metadata={'test_id': i})
            session_tasks.append(task)
        
        session_ids = await asyncio.gather(*session_tasks)
        
        # Add messages concurrently
        message_tasks = []
        for i, session_id in enumerate(session_ids):
            for j in range(5):
                task = session_manager.add_message_to_session(
                    session_id, "user", f"Message {j} in session {i}"
                )
                message_tasks.append(task)
        
        await asyncio.gather(*message_tasks)
        
        # Test history retrieval (this was the original bug)
        for session_id in session_ids:
            history = session_manager.get_session_history(session_id)
            if len(history) != 5:
                raise Exception(f"Expected 5 messages, got {len(history)}")
        
        logger.info(f"✅ Concurrent operations working: {len(session_ids)} sessions, {len(message_tasks)} messages")
        return True
        
    except Exception as e:
        logger.error(f"❌ Concurrent operations fix failed: {e}")
        return False


async def test_integrated_pipeline():
    """Test the integrated pipeline works end-to-end."""
    logger.info("🧪 Testing Integrated Pipeline...")
    
    try:
        # Initialize all components
        session_manager = SessionManager()
        await session_manager.initialize()
        
        event_bus = get_event_bus()
        await event_bus.start()
        
        event_coordinator = EventCoordinator()
        await event_coordinator.initialize()
        
        config = LLMConfig(
            providers={
                "ollama": ProviderConfig(
                    provider=LLMProvider.OLLAMA,
                    model="qwen3:30b-a3b",
                    host="http://localhost:11434",
                    temperature=0.7,
                    max_tokens=50,
                    system_message="/no_think Respond briefly."
                )
            },
            default_provider="ollama"
        )
        
        llm_manager = LLMManager(config)
        
        # Test end-to-end pipeline
        session_id = await session_manager.create_session()
        
        # Process message through pipeline
        user_message = "What's 2+2?"
        response = await llm_manager.generate_response(
            prompt=f"/no_think {user_message}",
            provider="ollama"
        )
        
        # Store in session
        await session_manager.add_message_to_session(session_id, "user", user_message)
        await session_manager.add_message_to_session(session_id, "assistant", response.content)
        
        # Emit event
        await event_coordinator.emit_gui_event("pipeline_test", {
            "session_id": session_id,
            "message": user_message,
            "response": response.content
        })
        
        # Verify
        history = session_manager.get_session_history(session_id)
        stats = event_coordinator.get_event_stats()
        
        logger.info(f"✅ Integrated pipeline working: {len(history)} messages, {stats['total_events']} events")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated pipeline fix failed: {e}")
        return False


async def main():
    """Main validation function."""
    logger.info("🚀 VALIDATING PHASE 1.5 FIXES")
    logger.info("=" * 50)
    
    tests = [
        ("Session History Fix", test_session_history_fix),
        ("Event Coordinator Fix", test_event_coordinator_fix),
        ("LLM Exception Handling Fix", test_llm_exception_handling_fix),
        ("Concurrent Operations", test_concurrent_operations),
        ("Integrated Pipeline", test_integrated_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*30}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 VALIDATION RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status} {test_name}")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 ALL FIXES VALIDATED - SYSTEM IS ROBUST!")
        return 0
    else:
        logger.error(f"\n❌ {total_tests - passed_tests} TESTS FAILED - FIXES NEED ATTENTION")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
