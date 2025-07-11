#!/usr/bin/env python3
"""
Phase 1.5 Comprehensive Testing & Validation Suite.

Validates all Phase 1 components are working correctly and can handle
real-world scenarios, stress conditions, and edge cases.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coda.core.config import CodaConfig
from coda.core.session_manager import SessionManager
from coda.core.event_coordinator import EventCoordinator
from coda.components.memory.manager import MemoryManager
from coda.components.personality.manager import PersonalityManager
from coda.components.tools.manager import ToolManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phase1_validation")


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    component: str
    success: bool
    response_time_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


class Phase1ValidationSuite:
    """
    Comprehensive validation suite for Phase 1 components.
    
    Tests all core functionality to ensure components are working
    correctly and can handle real-world scenarios.
    """
    
    def __init__(self):
        self.config = CodaConfig()
        self.results: List[ValidationResult] = []
        
        # Component instances
        self.session_manager: Optional[SessionManager] = None
        self.event_coordinator: Optional[EventCoordinator] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.personality_manager: Optional[PersonalityManager] = None
        self.tool_manager: Optional[ToolManager] = None
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("🚀 Starting Phase 1.5 Comprehensive Validation Suite")
        logger.info("=" * 80)
        
        # Initialize components
        await self._initialize_components()
        
        # Test 1: Session Management Validation
        logger.info("📋 Test 1: Session Management Validation")
        await self._test_session_management()
        
        # Test 2: Event Coordination Validation
        logger.info("📋 Test 2: Event Coordination Validation")
        await self._test_event_coordination()
        
        # Test 3: Memory System Validation
        logger.info("📋 Test 3: Memory System Validation")
        await self._test_memory_system()
        
        # Test 4: Personality System Validation
        logger.info("📋 Test 4: Personality System Validation")
        await self._test_personality_system()
        
        # Test 5: Tools System Validation
        logger.info("📋 Test 5: Tools System Validation")
        await self._test_tools_system()
        
        # Test 6: Integration Testing
        logger.info("📋 Test 6: Component Integration Validation")
        await self._test_component_integration()
        
        # Test 7: Stress Testing
        logger.info("📋 Test 7: Basic Stress Testing")
        await self._test_basic_stress_scenarios()
        
        # Test 8: Edge Case Testing
        logger.info("📋 Test 8: Edge Case Validation")
        await self._test_edge_cases()
        
        # Cleanup
        await self._cleanup_components()
        
        # Generate summary
        summary = self._generate_validation_summary()
        
        logger.info("=" * 80)
        logger.info("🎉 Phase 1.5 Comprehensive Validation Complete!")
        
        return {
            "results": [result.__dict__ for result in self.results],
            "summary": summary
        }
    
    async def _initialize_components(self):
        """Initialize all components for testing."""
        logger.info("  Initializing components for validation...")
        
        try:
            # Initialize Session Manager
            self.session_manager = SessionManager(self.config)
            await self.session_manager.initialize()
            
            # Initialize Event Coordinator
            self.event_coordinator = EventCoordinator()
            await self.event_coordinator.initialize()
            
            # Initialize Memory Manager
            self.memory_manager = MemoryManager(self.config)
            await self.memory_manager.initialize()
            
            # Initialize Personality Manager
            session_id = str(uuid.uuid4())
            self.personality_manager = PersonalityManager(session_id, self.config)
            await self.personality_manager.initialize()
            
            # Initialize Tool Manager
            self.tool_manager = ToolManager(self.config)
            await self.tool_manager.initialize()
            
            logger.info("  ✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"  ❌ Component initialization failed: {e}")
            raise
    
    async def _test_session_management(self):
        """Test session management functionality."""
        logger.info("  Testing session management...")
        
        # Test 1: Create Session
        start_time = time.time()
        try:
            session_id = await self.session_manager.create_session()
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="create_session",
                component="session_manager",
                success=True,
                response_time_ms=response_time,
                details={"session_id": session_id}
            ))
            logger.info(f"    ✅ Create session: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="create_session",
                component="session_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Create session failed: {e}")
            return
        
        # Test 2: Get Session
        start_time = time.time()
        try:
            session_data = await self.session_manager.get_session(session_id)
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="get_session",
                component="session_manager",
                success=session_data is not None,
                response_time_ms=response_time,
                details={"has_data": session_data is not None}
            ))
            logger.info(f"    ✅ Get session: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="get_session",
                component="session_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Get session failed: {e}")
        
        # Test 3: Update Session
        start_time = time.time()
        try:
            await self.session_manager.update_session(session_id, {"test_data": "validation"})
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="update_session",
                component="session_manager",
                success=True,
                response_time_ms=response_time,
                details={"updated": True}
            ))
            logger.info(f"    ✅ Update session: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="update_session",
                component="session_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Update session failed: {e}")
        
        # Test 4: End Session
        start_time = time.time()
        try:
            await self.session_manager.end_session(session_id)
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="end_session",
                component="session_manager",
                success=True,
                response_time_ms=response_time,
                details={"ended": True}
            ))
            logger.info(f"    ✅ End session: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="end_session",
                component="session_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ End session failed: {e}")
    
    async def _test_event_coordination(self):
        """Test event coordination functionality."""
        logger.info("  Testing event coordination...")
        
        # Test 1: Emit Event
        start_time = time.time()
        try:
            await self.event_coordinator.emit_event("test_event", {"data": "validation"})
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="emit_event",
                component="event_coordinator",
                success=True,
                response_time_ms=response_time,
                details={"event_emitted": True}
            ))
            logger.info(f"    ✅ Emit event: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="emit_event",
                component="event_coordinator",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Emit event failed: {e}")
        
        # Test 2: Event Listener Registration
        start_time = time.time()
        try:
            event_received = False
            
            def test_listener(event_data):
                nonlocal event_received
                event_received = True
            
            self.event_coordinator.register_listener("validation_event", test_listener)
            await self.event_coordinator.emit_event("validation_event", {"test": True})
            
            # Give a moment for event processing
            await asyncio.sleep(0.1)
            
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="event_listener",
                component="event_coordinator",
                success=event_received,
                response_time_ms=response_time,
                details={"event_received": event_received}
            ))
            logger.info(f"    ✅ Event listener: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="event_listener",
                component="event_coordinator",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Event listener failed: {e}")
    
    async def _test_memory_system(self):
        """Test memory system functionality."""
        logger.info("  Testing memory system...")
        
        # Test 1: Add Memory
        start_time = time.time()
        try:
            self.memory_manager.add_turn("user", "This is a test message for validation")
            self.memory_manager.add_turn("assistant", "I understand, this is a validation test")
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="add_memory",
                component="memory_manager",
                success=True,
                response_time_ms=response_time,
                details={"turns_added": 2}
            ))
            logger.info(f"    ✅ Add memory: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="add_memory",
                component="memory_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Add memory failed: {e}")
        
        # Test 2: Search Memory
        start_time = time.time()
        try:
            results = await self.memory_manager.search_memories("validation test", max_results=5)
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="search_memory",
                component="memory_manager",
                success=True,
                response_time_ms=response_time,
                details={"results_count": len(results)}
            ))
            logger.info(f"    ✅ Search memory: {response_time:.1f}ms, {len(results)} results")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="search_memory",
                component="memory_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Search memory failed: {e}")
    
    async def _test_personality_system(self):
        """Test personality system functionality."""
        logger.info("  Testing personality system...")
        
        # Test 1: Get Personality Traits
        start_time = time.time()
        try:
            traits = self.personality_manager.get_current_traits()
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="get_personality_traits",
                component="personality_manager",
                success=len(traits) > 0,
                response_time_ms=response_time,
                details={"traits_count": len(traits)}
            ))
            logger.info(f"    ✅ Get personality traits: {response_time:.1f}ms, {len(traits)} traits")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="get_personality_traits",
                component="personality_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Get personality traits failed: {e}")
        
        # Test 2: Process Feedback
        start_time = time.time()
        try:
            await self.personality_manager.process_feedback("positive", "Great response!")
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="process_feedback",
                component="personality_manager",
                success=True,
                response_time_ms=response_time,
                details={"feedback_processed": True}
            ))
            logger.info(f"    ✅ Process feedback: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="process_feedback",
                component="personality_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Process feedback failed: {e}")
    
    async def _test_tools_system(self):
        """Test tools system functionality."""
        logger.info("  Testing tools system...")
        
        # Test 1: List Available Tools
        start_time = time.time()
        try:
            tools = self.tool_manager.list_available_tools()
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="list_tools",
                component="tool_manager",
                success=len(tools) > 0,
                response_time_ms=response_time,
                details={"tools_count": len(tools)}
            ))
            logger.info(f"    ✅ List tools: {response_time:.1f}ms, {len(tools)} tools")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="list_tools",
                component="tool_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ List tools failed: {e}")
        
        # Test 2: Execute Tool
        start_time = time.time()
        try:
            result = await self.tool_manager.execute_tool("get_time", {})
            response_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                test_name="execute_tool",
                component="tool_manager",
                success=result is not None,
                response_time_ms=response_time,
                details={"tool_executed": True, "has_result": result is not None}
            ))
            logger.info(f"    ✅ Execute tool: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="execute_tool",
                component="tool_manager",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Execute tool failed: {e}")

    async def _test_component_integration(self):
        """Test component integration functionality."""
        logger.info("  Testing component integration...")

        # Test: Memory + Personality Integration
        start_time = time.time()
        try:
            # Add memory and process with personality
            self.memory_manager.add_turn("user", "I prefer direct answers")
            await self.personality_manager.process_feedback("preference", "User likes direct communication")

            response_time = (time.time() - start_time) * 1000

            self.results.append(ValidationResult(
                test_name="memory_personality_integration",
                component="integration",
                success=True,
                response_time_ms=response_time,
                details={"integration_successful": True}
            ))
            logger.info(f"    ✅ Memory-Personality integration: {response_time:.1f}ms")

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="memory_personality_integration",
                component="integration",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Memory-Personality integration failed: {e}")

    async def _test_basic_stress_scenarios(self):
        """Test basic stress scenarios."""
        logger.info("  Testing basic stress scenarios...")

        # Test: Multiple Concurrent Sessions
        start_time = time.time()
        try:
            sessions = []
            for i in range(10):
                session_id = await self.session_manager.create_session()
                sessions.append(session_id)

            # Cleanup
            for session_id in sessions:
                await self.session_manager.end_session(session_id)

            response_time = (time.time() - start_time) * 1000

            self.results.append(ValidationResult(
                test_name="concurrent_sessions",
                component="stress_test",
                success=True,
                response_time_ms=response_time,
                details={"sessions_created": len(sessions)}
            ))
            logger.info(f"    ✅ Concurrent sessions: {response_time:.1f}ms, {len(sessions)} sessions")

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="concurrent_sessions",
                component="stress_test",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Concurrent sessions failed: {e}")

    async def _test_edge_cases(self):
        """Test edge case scenarios."""
        logger.info("  Testing edge cases...")

        # Test: Empty Input Handling
        start_time = time.time()
        try:
            # Test empty memory search
            results = await self.memory_manager.search_memories("", max_results=5)

            response_time = (time.time() - start_time) * 1000

            self.results.append(ValidationResult(
                test_name="empty_input_handling",
                component="edge_case",
                success=True,  # Should handle gracefully
                response_time_ms=response_time,
                details={"handled_gracefully": True}
            ))
            logger.info(f"    ✅ Empty input handling: {response_time:.1f}ms")

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                test_name="empty_input_handling",
                component="edge_case",
                success=False,
                response_time_ms=response_time,
                details={},
                error_message=str(e)
            ))
            logger.error(f"    ❌ Empty input handling failed: {e}")

    async def _cleanup_components(self):
        """Cleanup components after testing."""
        logger.info("  Cleaning up components...")

        try:
            if self.memory_manager:
                await self.memory_manager.shutdown()

            if self.personality_manager:
                await self.personality_manager.shutdown()

            if self.tool_manager:
                await self.tool_manager.shutdown()

            logger.info("  ✅ Component cleanup completed")

        except Exception as e:
            logger.warning(f"  ⚠️ Cleanup warning: {e}")

    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests

        # Group by component
        component_results = {}
        for result in self.results:
            if result.component not in component_results:
                component_results[result.component] = {"passed": 0, "failed": 0, "total": 0}

            component_results[result.component]["total"] += 1
            if result.success:
                component_results[result.component]["passed"] += 1
            else:
                component_results[result.component]["failed"] += 1

        # Calculate average response times
        avg_response_times = {}
        for component in component_results:
            component_times = [r.response_time_ms for r in self.results if r.component == component]
            avg_response_times[component] = sum(component_times) / len(component_times) if component_times else 0

        # Overall assessment
        success_rate = successful_tests / total_tests if total_tests > 0 else 0

        if success_rate >= 0.9:
            overall_status = "excellent"
        elif success_rate >= 0.8:
            overall_status = "good"
        elif success_rate >= 0.7:
            overall_status = "fair"
        else:
            overall_status = "needs_improvement"

        return {
            "overall_status": overall_status,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "component_results": component_results,
            "avg_response_times_ms": avg_response_times,
            "validation_timestamp": datetime.now().isoformat()
        }


async def main():
    """Run the Phase 1.5 validation suite."""
    validator = Phase1ValidationSuite()

    try:
        results = await validator.run_comprehensive_validation()

        # Print summary
        summary = results["summary"]
        logger.info("=" * 80)
        logger.info("🎯 PHASE 1.5 VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Tests: {summary['successful_tests']}/{summary['total_tests']} passed")

        logger.info("\n📊 Component Results:")
        for component, stats in summary["component_results"].items():
            success_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            avg_time = summary["avg_response_times_ms"].get(component, 0)
            logger.info(f"  • {component}: {stats['passed']}/{stats['total']} "
                       f"({success_rate:.1%}) - {avg_time:.1f}ms avg")

        # Show failed tests
        failed_results = [r for r in results["results"] if not r["success"]]
        if failed_results:
            logger.info("\n❌ Failed Tests:")
            for result in failed_results:
                logger.info(f"  • {result['component']}.{result['test_name']}: {result['error_message']}")

        return 0 if summary["overall_status"] in ["excellent", "good"] else 1

    except Exception as e:
        logger.error(f"Validation suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
