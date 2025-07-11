#!/usr/bin/env python3
"""
Edge Case and Error Scenario Testing Suite for Coda Phase 1.5.

Tests edge cases, error conditions, malformed inputs, network failures,
component crashes, and recovery scenarios to ensure robust error handling.
"""

import asyncio
import logging
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, patch
import random
import string

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coda.core.assistant import CodaAssistant
from coda.core.config import CodaConfig
from coda.core.session_manager import SessionManager
from coda.core.event_coordinator import EventCoordinator
from coda.components.llm.manager import LLMManager
from coda.components.memory.manager import MemoryManager
from coda.interfaces.websocket.server import CodaWebSocketServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("edge_case_test")


@dataclass
class EdgeCaseTestResult:
    """Result of an edge case test."""
    test_name: str
    success: bool
    error_handled: bool
    recovery_successful: bool
    response_time: float
    error_message: Optional[str] = None
    expected_behavior: bool = True


class EdgeCaseTestSuite:
    """
    Comprehensive edge case and error scenario testing suite.
    
    Tests system robustness under various failure conditions,
    malformed inputs, and unexpected scenarios.
    """
    
    def __init__(self):
        self.config = CodaConfig()
        self.results: Dict[str, List[EdgeCaseTestResult]] = {}
        self.assistant: Optional[CodaAssistant] = None
        self.session_manager: Optional[SessionManager] = None
        self.event_coordinator: Optional[EventCoordinator] = None
        self.llm_manager: Optional[LLMManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.websocket_server: Optional[CodaWebSocketServer] = None
    
    async def run_comprehensive_edge_case_tests(self) -> Dict[str, Any]:
        """Run all edge case tests."""
        logger.info("🚀 Starting Comprehensive Edge Case Test Suite")
        logger.info("=" * 80)
        
        # Initialize system
        await self._initialize_system()
        
        # Test Category 1: Input Validation Edge Cases
        logger.info("🔍 Category 1: Input Validation Edge Cases")
        input_results = await self.test_input_validation_edge_cases()
        self.results["input_validation"] = input_results
        
        # Test Category 2: Session Management Edge Cases
        logger.info("🔍 Category 2: Session Management Edge Cases")
        session_results = await self.test_session_management_edge_cases()
        self.results["session_management"] = session_results
        
        # Test Category 3: Memory System Edge Cases
        logger.info("🔍 Category 3: Memory System Edge Cases")
        memory_results = await self.test_memory_system_edge_cases()
        self.results["memory_system"] = memory_results
        
        # Test Category 4: LLM Provider Edge Cases
        logger.info("🔍 Category 4: LLM Provider Edge Cases")
        llm_results = await self.test_llm_provider_edge_cases()
        self.results["llm_provider"] = llm_results
        
        # Test Category 5: WebSocket Edge Cases
        logger.info("🔍 Category 5: WebSocket Edge Cases")
        websocket_results = await self.test_websocket_edge_cases()
        self.results["websocket"] = websocket_results
        
        # Test Category 6: Event System Edge Cases
        logger.info("🔍 Category 6: Event System Edge Cases")
        event_results = await self.test_event_system_edge_cases()
        self.results["event_system"] = event_results
        
        # Test Category 7: Component Failure Scenarios
        logger.info("🔍 Category 7: Component Failure Scenarios")
        failure_results = await self.test_component_failure_scenarios()
        self.results["component_failures"] = failure_results
        
        # Test Category 8: Recovery and Resilience
        logger.info("🔍 Category 8: Recovery and Resilience")
        recovery_results = await self.test_recovery_and_resilience()
        self.results["recovery_resilience"] = recovery_results
        
        # Cleanup
        await self._cleanup_system()
        
        # Generate summary
        summary = self._generate_edge_case_summary()
        self.results["summary"] = summary
        
        logger.info("=" * 80)
        logger.info("🎉 Comprehensive Edge Case Test Suite Complete!")
        
        return self.results
    
    async def test_input_validation_edge_cases(self) -> List[EdgeCaseTestResult]:
        """Test input validation edge cases."""
        logger.info("  Testing input validation edge cases...")
        
        results = []
        
        # Test cases for malformed inputs
        test_cases = [
            ("empty_message", ""),
            ("null_message", None),
            ("very_long_message", "x" * 10000),
            ("unicode_message", "🚀🎉💻🔥⚡🌟🎯🛠️🎭🧠"),
            ("json_injection", '{"malicious": "payload", "type": "injection"}'),
            ("sql_injection", "'; DROP TABLE sessions; --"),
            ("script_injection", "<script>alert('xss')</script>"),
            ("binary_data", b'\x00\x01\x02\x03\x04\x05'.decode('latin-1')),
            ("control_characters", "\x00\x01\x02\x03\x04\x05\x06\x07"),
            ("newline_flood", "\n" * 1000),
            ("tab_flood", "\t" * 1000),
            ("space_flood", " " * 1000),
        ]
        
        session_id = await self.session_manager.create_session()
        
        for test_name, test_input in test_cases:
            result = await self._test_message_processing(
                test_name, test_input, session_id
            )
            results.append(result)
            logger.info(f"    {'✅' if result.success else '❌'} {test_name}: "
                       f"{'Handled' if result.error_handled else 'Failed'}")
        
        await self.session_manager.end_session(session_id)
        
        return results
    
    async def test_session_management_edge_cases(self) -> List[EdgeCaseTestResult]:
        """Test session management edge cases."""
        logger.info("  Testing session management edge cases...")
        
        results = []
        
        # Test invalid session operations
        test_cases = [
            ("invalid_session_id", "invalid-session-id"),
            ("empty_session_id", ""),
            ("null_session_id", None),
            ("very_long_session_id", "x" * 1000),
            ("malformed_uuid", "not-a-uuid"),
            ("deleted_session", None),  # Will be set to a deleted session
        ]
        
        # Create and delete a session for the deleted_session test
        deleted_session = await self.session_manager.create_session()
        await self.session_manager.end_session(deleted_session)
        test_cases[5] = ("deleted_session", deleted_session)
        
        for test_name, session_id in test_cases:
            result = await self._test_session_operation(test_name, session_id)
            results.append(result)
            logger.info(f"    {'✅' if result.success else '❌'} {test_name}: "
                       f"{'Handled' if result.error_handled else 'Failed'}")
        
        # Test session limit edge cases
        session_limit_result = await self._test_session_limits()
        results.append(session_limit_result)
        logger.info(f"    {'✅' if session_limit_result.success else '❌'} session_limits: "
                   f"{'Handled' if session_limit_result.error_handled else 'Failed'}")
        
        return results
    
    async def test_memory_system_edge_cases(self) -> List[EdgeCaseTestResult]:
        """Test memory system edge cases."""
        logger.info("  Testing memory system edge cases...")
        
        results = []
        
        # Test memory operations with edge case data
        test_cases = [
            ("empty_memory_content", ""),
            ("null_memory_content", None),
            ("very_large_memory", "x" * 100000),
            ("binary_memory_content", b'\x00\x01\x02\x03'.decode('latin-1')),
            ("unicode_memory_content", "🚀" * 1000),
            ("json_memory_content", '{"nested": {"deep": {"structure": "value"}}}'),
        ]
        
        for test_name, memory_content in test_cases:
            result = await self._test_memory_operation(test_name, memory_content)
            results.append(result)
            logger.info(f"    {'✅' if result.success else '❌'} {test_name}: "
                       f"{'Handled' if result.error_handled else 'Failed'}")
        
        # Test memory search edge cases
        search_cases = [
            ("empty_search_query", ""),
            ("null_search_query", None),
            ("very_long_search_query", "search " * 1000),
            ("special_char_search", "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
        ]
        
        for test_name, search_query in search_cases:
            result = await self._test_memory_search(test_name, search_query)
            results.append(result)
            logger.info(f"    {'✅' if result.success else '❌'} {test_name}: "
                       f"{'Handled' if result.error_handled else 'Failed'}")
        
        return results
    
    async def test_llm_provider_edge_cases(self) -> List[EdgeCaseTestResult]:
        """Test LLM provider edge cases."""
        logger.info("  Testing LLM provider edge cases...")
        
        results = []
        
        # Test LLM requests with edge case prompts
        test_cases = [
            ("empty_prompt", ""),
            ("null_prompt", None),
            ("very_long_prompt", "Please respond to this: " + "x" * 50000),
            ("unicode_prompt", "🚀🎉💻 Please respond in emojis only 🔥⚡🌟"),
            ("malformed_prompt", "\x00\x01\x02 Invalid characters \x03\x04"),
            ("recursive_prompt", "Please ask yourself this question: " * 100),
        ]
        
        for test_name, prompt in test_cases:
            result = await self._test_llm_request(test_name, prompt)
            results.append(result)
            logger.info(f"    {'✅' if result.success else '❌'} {test_name}: "
                       f"{'Handled' if result.error_handled else 'Failed'}")
        
        # Test invalid provider scenarios
        provider_cases = [
            ("invalid_provider", "invalid_provider"),
            ("empty_provider", ""),
            ("null_provider", None),
        ]
        
        for test_name, provider in provider_cases:
            result = await self._test_llm_provider(test_name, provider)
            results.append(result)
            logger.info(f"    {'✅' if result.success else '❌'} {test_name}: "
                       f"{'Handled' if result.error_handled else 'Failed'}")
        
        return results
    
    async def test_websocket_edge_cases(self) -> List[EdgeCaseTestResult]:
        """Test WebSocket edge cases."""
        logger.info("  Testing WebSocket edge cases...")
        
        results = []
        
        # Test WebSocket message edge cases
        test_cases = [
            ("empty_websocket_message", {}),
            ("null_websocket_message", None),
            ("malformed_json", "not json"),
            ("very_large_message", {"data": "x" * 100000}),
            ("invalid_event_type", {"type": "invalid_event", "data": {}}),
            ("missing_required_fields", {"type": "message"}),  # Missing data
            ("circular_reference", None),  # Will be created with circular ref
        ]
        
        # Create circular reference object
        circular_obj = {"self": None}
        circular_obj["self"] = circular_obj
        test_cases[6] = ("circular_reference", circular_obj)
        
        for test_name, message_data in test_cases:
            result = await self._test_websocket_message(test_name, message_data)
            results.append(result)
            logger.info(f"    {'✅' if result.success else '❌'} {test_name}: "
                       f"{'Handled' if result.error_handled else 'Failed'}")
        
        return results

    async def test_event_system_edge_cases(self) -> List[EdgeCaseTestResult]:
        """Test event system edge cases."""
        logger.info("  Testing event system edge cases...")

        results = []

        # Test event emission edge cases
        test_cases = [
            ("empty_event_type", "", {}),
            ("null_event_type", None, {}),
            ("invalid_event_type", "invalid_event", {}),
            ("empty_event_data", "test_event", {}),
            ("null_event_data", "test_event", None),
            ("very_large_event_data", "test_event", {"data": "x" * 50000}),
            ("circular_event_data", "test_event", None),  # Will be set with circular ref
        ]

        # Create circular reference for event data
        circular_data = {"self": None}
        circular_data["self"] = circular_data
        test_cases[6] = ("circular_event_data", "test_event", circular_data)

        for test_name, event_type, event_data in test_cases:
            result = await self._test_event_emission(test_name, event_type, event_data)
            results.append(result)
            logger.info(f"    {'✅' if result.success else '❌'} {test_name}: "
                       f"{'Handled' if result.error_handled else 'Failed'}")

        return results

    async def test_component_failure_scenarios(self) -> List[EdgeCaseTestResult]:
        """Test component failure scenarios."""
        logger.info("  Testing component failure scenarios...")

        results = []

        # Test LLM provider failure simulation
        llm_failure_result = await self._test_llm_failure_simulation()
        results.append(llm_failure_result)
        logger.info(f"    {'✅' if llm_failure_result.success else '❌'} llm_failure_simulation: "
                   f"{'Handled' if llm_failure_result.error_handled else 'Failed'}")

        # Test memory system failure simulation
        memory_failure_result = await self._test_memory_failure_simulation()
        results.append(memory_failure_result)
        logger.info(f"    {'✅' if memory_failure_result.success else '❌'} memory_failure_simulation: "
                   f"{'Handled' if memory_failure_result.error_handled else 'Failed'}")

        # Test WebSocket failure simulation
        websocket_failure_result = await self._test_websocket_failure_simulation()
        results.append(websocket_failure_result)
        logger.info(f"    {'✅' if websocket_failure_result.success else '❌'} websocket_failure_simulation: "
                   f"{'Handled' if websocket_failure_result.error_handled else 'Failed'}")

        return results

    async def test_recovery_and_resilience(self) -> List[EdgeCaseTestResult]:
        """Test recovery and resilience scenarios."""
        logger.info("  Testing recovery and resilience...")

        results = []

        # Test system recovery after errors
        recovery_result = await self._test_system_recovery()
        results.append(recovery_result)
        logger.info(f"    {'✅' if recovery_result.success else '❌'} system_recovery: "
                   f"{'Recovered' if recovery_result.recovery_successful else 'Failed'}")

        # Test graceful degradation
        degradation_result = await self._test_graceful_degradation()
        results.append(degradation_result)
        logger.info(f"    {'✅' if degradation_result.success else '❌'} graceful_degradation: "
                   f"{'Handled' if degradation_result.error_handled else 'Failed'}")

        return results

    # Helper methods for testing specific scenarios

    async def _test_message_processing(self, test_name: str, message: Any, session_id: str) -> EdgeCaseTestResult:
        """Test message processing with edge case input."""
        start_time = time.time()

        try:
            response = await self.assistant.process_text_message(
                message=message,
                session_id=session_id
            )

            response_time = time.time() - start_time

            # Check if response is valid
            success = response is not None and hasattr(response, 'content')

            return EdgeCaseTestResult(
                test_name=test_name,
                success=success,
                error_handled=True,
                recovery_successful=success,
                response_time=response_time
            )

        except Exception as e:
            response_time = time.time() - start_time

            # Error was caught and handled
            return EdgeCaseTestResult(
                test_name=test_name,
                success=False,
                error_handled=True,
                recovery_successful=False,
                response_time=response_time,
                error_message=str(e)
            )
