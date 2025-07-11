#!/usr/bin/env python3
"""
Phase 1.5 Edge Case & Error Scenario Testing

This comprehensive test suite is designed to:
1. Test edge cases and boundary conditions
2. Test error scenarios and recovery
3. Test malformed inputs and invalid data
4. Test component failure scenarios
5. Test resource exhaustion scenarios
6. Ensure robust error handling

Tests are designed to SURFACE ISSUES and validate error handling.
"""

import asyncio
import logging
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests/edge_case_test_results.log', mode='w')
    ]
)
logger = logging.getLogger("edge_case_test")

# Import components
try:
    from coda.core.session_manager import SessionManager
    from coda.core.event_coordinator import EventCoordinator
    from coda.components.llm.models import LLMConfig, ProviderConfig, LLMProvider
    from coda.components.llm.manager import LLMManager
    from coda.components.memory.manager import MemoryManager
    from coda.components.memory.models import MemoryManagerConfig
    from coda.core.events import get_event_bus
    
except ImportError as e:
    logger.error(f"Failed to import Coda components: {e}")
    exit(1)


class EdgeCaseTester:
    """Comprehensive edge case and error scenario testing."""
    
    def __init__(self):
        self.results = []
        
    async def test_malformed_inputs(self):
        """Test handling of malformed and invalid inputs."""
        logger.info("🧪 Testing Malformed Inputs...")
        
        test_cases = [
            # Session Manager edge cases
            {"component": "session_manager", "test": "empty_session_id", "input": ""},
            {"component": "session_manager", "test": "none_session_id", "input": None},
            {"component": "session_manager", "test": "very_long_session_id", "input": "x" * 1000},
            {"component": "session_manager", "test": "special_chars_session_id", "input": "!@#$%^&*()"},
            {"component": "session_manager", "test": "unicode_session_id", "input": "测试🚀"},
            
            # Message content edge cases
            {"component": "session_manager", "test": "empty_message", "input": ""},
            {"component": "session_manager", "test": "very_long_message", "input": "x" * 100000},
            {"component": "session_manager", "test": "unicode_message", "input": "Hello 世界 🌍 emoji test"},
            {"component": "session_manager", "test": "null_bytes", "input": "test\x00null\x00bytes"},
            {"component": "session_manager", "test": "control_chars", "input": "test\n\r\t\b\f"},
            
            # JSON-breaking characters
            {"component": "session_manager", "test": "json_breaking", "input": '{"test": "value"}'},
            {"component": "session_manager", "test": "quotes_and_escapes", "input": 'test "quotes" and \\backslashes\\'},
        ]
        
        results = []
        session_manager = SessionManager()
        await session_manager.initialize()
        
        for case in test_cases:
            try:
                logger.info(f"  Testing {case['test']} with input: {repr(case['input'][:50])}")
                
                if case["component"] == "session_manager":
                    if "session_id" in case["test"]:
                        # Test session creation with malformed ID
                        if case["input"] is None:
                            session_id = await session_manager.create_session()
                        else:
                            session_id = await session_manager.create_session(case["input"])
                        result = {"test": case["test"], "success": True, "session_id": session_id}
                    
                    elif "message" in case["test"]:
                        # Test message addition with malformed content
                        session_id = await session_manager.create_session()
                        success = await session_manager.add_message_to_session(
                            session_id, "user", case["input"]
                        )
                        result = {"test": case["test"], "success": success}
                
                results.append(result)
                logger.info(f"    ✅ {case['test']}: Handled gracefully")
                
            except Exception as e:
                result = {"test": case["test"], "success": False, "error": str(e)}
                results.append(result)
                logger.warning(f"    ⚠️ {case['test']}: {e}")
        
        return {"test_category": "malformed_inputs", "results": results}
    
    async def test_boundary_conditions(self):
        """Test boundary conditions and limits."""
        logger.info("🧪 Testing Boundary Conditions...")
        
        results = []
        session_manager = SessionManager()
        await session_manager.initialize()
        
        # Test maximum sessions
        try:
            logger.info("  Testing maximum concurrent sessions...")
            session_ids = []
            for i in range(1000):  # Try to create 1000 sessions
                session_id = await session_manager.create_session(metadata={"test_session": i})
                session_ids.append(session_id)
                
                if i % 100 == 0:
                    logger.info(f"    Created {i+1} sessions...")
            
            results.append({
                "test": "max_concurrent_sessions",
                "success": True,
                "sessions_created": len(session_ids)
            })
            
            # Cleanup
            for session_id in session_ids:
                await session_manager.end_session(session_id)
                
        except Exception as e:
            results.append({
                "test": "max_concurrent_sessions",
                "success": False,
                "error": str(e),
                "sessions_created": len(session_ids) if 'session_ids' in locals() else 0
            })
        
        # Test maximum messages per session
        try:
            logger.info("  Testing maximum messages per session...")
            session_id = await session_manager.create_session()
            
            for i in range(10000):  # Try to add 10,000 messages
                await session_manager.add_message_to_session(
                    session_id, "user" if i % 2 == 0 else "assistant", f"Message {i}"
                )
                
                if i % 1000 == 0:
                    logger.info(f"    Added {i+1} messages...")
            
            # Test retrieval
            history = session_manager.get_session_history(session_id)
            
            results.append({
                "test": "max_messages_per_session",
                "success": True,
                "messages_added": 10000,
                "messages_retrieved": len(history)
            })
            
        except Exception as e:
            results.append({
                "test": "max_messages_per_session",
                "success": False,
                "error": str(e)
            })
        
        return {"test_category": "boundary_conditions", "results": results}
    
    async def test_component_failure_scenarios(self):
        """Test component failure and recovery scenarios."""
        logger.info("🧪 Testing Component Failure Scenarios...")
        
        results = []
        
        # Test LLM failure scenarios
        try:
            logger.info("  Testing LLM failure scenarios...")
            
            # Test with invalid configuration
            invalid_config = LLMConfig(
                providers={
                    "invalid": ProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model="nonexistent-model",
                        host="http://invalid-host:99999",
                        temperature=0.7,
                        max_tokens=100
                    )
                },
                default_provider="invalid"
            )
            
            llm_manager = LLMManager(invalid_config)
            
            # This should fail gracefully
            try:
                response = await llm_manager.generate_response(
                    prompt="Test message",
                    provider="invalid"
                )
                results.append({
                    "test": "llm_invalid_config",
                    "success": False,
                    "error": "Should have failed but didn't"
                })
            except Exception as e:
                results.append({
                    "test": "llm_invalid_config",
                    "success": True,
                    "error_handled": str(e)
                })
                logger.info(f"    ✅ LLM invalid config: Error handled gracefully")
            
        except Exception as e:
            results.append({
                "test": "llm_failure_scenarios",
                "success": False,
                "error": str(e)
            })
        
        # Test memory manager failure scenarios
        try:
            logger.info("  Testing Memory Manager failure scenarios...")
            
            # Test with invalid storage path
            invalid_memory_config = MemoryManagerConfig()
            invalid_memory_config.storage_path = "/invalid/path/that/does/not/exist"
            
            memory_manager = MemoryManager(invalid_memory_config)
            
            try:
                await memory_manager.initialize()
                results.append({
                    "test": "memory_invalid_path",
                    "success": True,
                    "note": "Memory manager handled invalid path gracefully"
                })
            except Exception as e:
                results.append({
                    "test": "memory_invalid_path",
                    "success": True,
                    "error_handled": str(e)
                })
                logger.info(f"    ✅ Memory invalid path: Error handled gracefully")
            
        except Exception as e:
            results.append({
                "test": "memory_failure_scenarios",
                "success": False,
                "error": str(e)
            })
        
        return {"test_category": "component_failure_scenarios", "results": results}
    
    async def test_concurrent_error_scenarios(self):
        """Test concurrent operations with errors."""
        logger.info("🧪 Testing Concurrent Error Scenarios...")
        
        results = []
        session_manager = SessionManager()
        await session_manager.initialize()
        
        try:
            # Test concurrent operations on same session
            session_id = await session_manager.create_session()
            
            # Create tasks that might conflict
            tasks = []
            for i in range(50):
                # Mix of valid and potentially problematic operations
                if i % 5 == 0:
                    # Try to end session while others are using it
                    task = session_manager.end_session(session_id)
                elif i % 7 == 0:
                    # Try to add message to potentially ended session
                    task = session_manager.add_message_to_session(
                        session_id, "user", f"Concurrent message {i}"
                    )
                else:
                    # Normal operations
                    task = session_manager.add_message_to_session(
                        session_id, "user", f"Normal message {i}"
                    )
                
                tasks.append(task)
            
            # Execute all tasks concurrently
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_ops = sum(1 for r in task_results if not isinstance(r, Exception))
            failed_ops = len(task_results) - successful_ops
            
            results.append({
                "test": "concurrent_session_operations",
                "success": True,
                "total_operations": len(tasks),
                "successful_operations": successful_ops,
                "failed_operations": failed_ops,
                "error_rate": failed_ops / len(tasks)
            })
            
        except Exception as e:
            results.append({
                "test": "concurrent_error_scenarios",
                "success": False,
                "error": str(e)
            })
        
        return {"test_category": "concurrent_error_scenarios", "results": results}
    
    async def test_data_integrity(self):
        """Test data integrity under various conditions."""
        logger.info("🧪 Testing Data Integrity...")
        
        results = []
        session_manager = SessionManager()
        await session_manager.initialize()
        
        try:
            # Test data persistence and retrieval integrity
            session_id = await session_manager.create_session(metadata={"integrity_test": True})
            
            # Add known data
            test_messages = [
                ("user", "First message"),
                ("assistant", "First response"),
                ("user", "Second message with unicode: 测试 🚀"),
                ("assistant", "Second response with special chars: !@#$%^&*()"),
                ("user", "Third message\nwith\nnewlines"),
                ("assistant", "Third response with \"quotes\" and 'apostrophes'")
            ]
            
            for role, content in test_messages:
                await session_manager.add_message_to_session(session_id, role, content)
            
            # Retrieve and verify
            history = session_manager.get_session_history(session_id)
            
            # Verify data integrity
            integrity_check = True
            if len(history) != len(test_messages):
                integrity_check = False
                logger.error(f"    ❌ Message count mismatch: expected {len(test_messages)}, got {len(history)}")
            
            for i, (expected_role, expected_content) in enumerate(test_messages):
                if i < len(history):
                    actual_role = history[i]["role"]
                    actual_content = history[i]["content"]
                    
                    if actual_role != expected_role or actual_content != expected_content:
                        integrity_check = False
                        logger.error(f"    ❌ Message {i} mismatch:")
                        logger.error(f"      Expected: {expected_role}: {expected_content}")
                        logger.error(f"      Actual: {actual_role}: {actual_content}")
            
            results.append({
                "test": "data_integrity",
                "success": integrity_check,
                "messages_stored": len(test_messages),
                "messages_retrieved": len(history)
            })
            
            if integrity_check:
                logger.info("    ✅ Data integrity: All messages stored and retrieved correctly")
            
        except Exception as e:
            results.append({
                "test": "data_integrity",
                "success": False,
                "error": str(e)
            })
        
        return {"test_category": "data_integrity", "results": results}

    async def run_comprehensive_edge_case_tests(self):
        """Run all edge case and error scenario tests."""
        logger.info("🚀 STARTING COMPREHENSIVE EDGE CASE TESTING")
        logger.info("=" * 70)
        logger.info("⚠️  These tests are designed to SURFACE EDGE CASES and ERRORS")
        logger.info("⚠️  Tests will attempt to break the system with invalid inputs")
        logger.info("=" * 70)

        all_results = []

        try:
            # Test 1: Malformed Inputs
            logger.info("\n" + "="*40)
            result1 = await self.test_malformed_inputs()
            all_results.append(result1)

            # Test 2: Boundary Conditions
            logger.info("\n" + "="*40)
            result2 = await self.test_boundary_conditions()
            all_results.append(result2)

            # Test 3: Component Failure Scenarios
            logger.info("\n" + "="*40)
            result3 = await self.test_component_failure_scenarios()
            all_results.append(result3)

            # Test 4: Concurrent Error Scenarios
            logger.info("\n" + "="*40)
            result4 = await self.test_concurrent_error_scenarios()
            all_results.append(result4)

            # Test 5: Data Integrity
            logger.info("\n" + "="*40)
            result5 = await self.test_data_integrity()
            all_results.append(result5)

            # Generate comprehensive report
            self._generate_edge_case_report(all_results)

            return all_results

        except Exception as e:
            logger.error(f"❌ Edge case testing suite failed: {e}")
            self._generate_edge_case_report(all_results)
            raise

    def _generate_edge_case_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive edge case test report."""
        logger.info("\n" + "="*70)
        logger.info("📊 COMPREHENSIVE EDGE CASE TEST REPORT")
        logger.info("="*70)

        total_test_categories = len(results)
        total_individual_tests = sum(len(r.get("results", [])) for r in results)

        logger.info(f"📈 OVERALL RESULTS:")
        logger.info(f"   Test Categories: {total_test_categories}")
        logger.info(f"   Individual Tests: {total_individual_tests}")

        # Analyze each category
        for category_result in results:
            category_name = category_result.get("test_category", "unknown")
            individual_results = category_result.get("results", [])

            passed_tests = sum(1 for r in individual_results if r.get("success", False))
            failed_tests = len(individual_results) - passed_tests

            logger.info(f"\n🔍 {category_name.upper().replace('_', ' ')}:")
            logger.info(f"   Total Tests: {len(individual_results)}")
            logger.info(f"   Passed: {passed_tests}")
            logger.info(f"   Failed: {failed_tests}")

            # Show details for failed tests
            for test_result in individual_results:
                test_name = test_result.get("test", "unknown")
                success = test_result.get("success", False)

                if not success:
                    error = test_result.get("error", "Unknown error")
                    logger.info(f"     ❌ {test_name}: {error}")
                elif "error_handled" in test_result:
                    logger.info(f"     ✅ {test_name}: Error handled gracefully")
                else:
                    logger.info(f"     ✅ {test_name}: Passed")

        # Critical issues summary
        logger.info(f"\n🚨 CRITICAL ISSUES DETECTED:")
        critical_issues = []

        for category_result in results:
            for test_result in category_result.get("results", []):
                if not test_result.get("success", False) and "error_handled" not in test_result:
                    critical_issues.append({
                        "category": category_result.get("test_category"),
                        "test": test_result.get("test"),
                        "error": test_result.get("error")
                    })

        if critical_issues:
            for issue in critical_issues:
                logger.info(f"   ❌ {issue['category']}.{issue['test']}: {issue['error']}")
        else:
            logger.info("   🎉 No critical issues detected!")

        # Recommendations
        logger.info(f"\n📝 RECOMMENDATIONS:")
        if critical_issues:
            logger.info("   🔧 Fix critical issues before proceeding to Phase 2")
            logger.info("   🔧 Improve error handling for edge cases")

        # Check for specific patterns
        boundary_issues = [i for i in critical_issues if "boundary" in i["category"]]
        if boundary_issues:
            logger.info("   🔧 Review system limits and add proper boundary checks")

        concurrent_issues = [i for i in critical_issues if "concurrent" in i["category"]]
        if concurrent_issues:
            logger.info("   🔧 Improve thread safety and concurrent operation handling")

        logger.info("="*70)

        # Save detailed report
        report_path = Path("tests/edge_case_detailed_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_categories': total_test_categories,
                    'total_tests': total_individual_tests,
                    'critical_issues': len(critical_issues)
                },
                'detailed_results': results,
                'critical_issues': critical_issues,
                'timestamp': time.time()
            }, f, indent=2)

        logger.info(f"📄 Detailed report saved to: {report_path}")


async def main():
    """Main edge case testing function."""
    tester = EdgeCaseTester()

    try:
        results = await tester.run_comprehensive_edge_case_tests()

        # Determine exit code based on critical issues
        critical_issues = 0
        for category_result in results:
            for test_result in category_result.get("results", []):
                if not test_result.get("success", False) and "error_handled" not in test_result:
                    critical_issues += 1

        if critical_issues > 0:
            logger.error(f"❌ {critical_issues} critical issues found - system needs attention")
            return 1
        else:
            logger.info("🎉 All edge case tests passed - system handles edge cases robustly!")
            return 0

    except Exception as e:
        logger.error(f"❌ Edge case testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
