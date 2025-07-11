#!/usr/bin/env python3
"""
Simple Phase 1.5 Validation Test.

Tests core functionality of Phase 1 components to ensure they work correctly.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_validation")


class SimpleValidationSuite:
    """Simple validation suite for Phase 1 components."""
    
    def __init__(self):
        self.results = []
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run validation tests."""
        logger.info("🚀 Starting Simple Phase 1.5 Validation")
        logger.info("=" * 60)
        
        # Test 1: Import Validation
        logger.info("📋 Test 1: Import Validation")
        await self._test_imports()
        
        # Test 2: Configuration Validation
        logger.info("📋 Test 2: Configuration Validation")
        await self._test_configuration()
        
        # Test 3: Session Manager Validation
        logger.info("📋 Test 3: Session Manager Validation")
        await self._test_session_manager()
        
        # Test 4: Event Coordinator Validation
        logger.info("📋 Test 4: Event Coordinator Validation")
        await self._test_event_coordinator()
        
        # Test 5: Memory System Validation
        logger.info("📋 Test 5: Memory System Validation")
        await self._test_memory_system()
        
        # Test 6: Tools System Validation
        logger.info("📋 Test 6: Tools System Validation")
        await self._test_tools_system()
        
        # Generate summary
        summary = self._generate_summary()
        
        logger.info("=" * 60)
        logger.info("🎉 Simple Validation Complete!")
        
        return {
            "results": self.results,
            "summary": summary
        }
    
    async def _test_imports(self):
        """Test that all core components can be imported."""
        logger.info("  Testing component imports...")
        
        imports_to_test = [
            ("coda.core.config", "CodaConfig"),
            ("coda.core.session_manager", "SessionManager"),
            ("coda.core.event_coordinator", "EventCoordinator"),
            ("coda.components.memory.manager", "MemoryManager"),
            ("coda.components.personality.manager", "PersonalityManager"),
            ("coda.components.tools.manager", "ToolManager"),
            ("coda.components.llm.manager", "LLMManager"),
            ("coda.core.assistant", "CodaAssistant"),
        ]
        
        for module_name, class_name in imports_to_test:
            start_time = time.time()
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                response_time = (time.time() - start_time) * 1000
                
                self.results.append({
                    "test": f"import_{class_name}",
                    "component": "imports",
                    "success": True,
                    "response_time_ms": response_time,
                    "details": {"module": module_name, "class": class_name}
                })
                logger.info(f"    ✅ {class_name}: {response_time:.1f}ms")
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.results.append({
                    "test": f"import_{class_name}",
                    "component": "imports",
                    "success": False,
                    "response_time_ms": response_time,
                    "error": str(e)
                })
                logger.error(f"    ❌ {class_name}: {e}")
    
    async def _test_configuration(self):
        """Test configuration system."""
        logger.info("  Testing configuration system...")
        
        start_time = time.time()
        try:
            from coda.core.config import CodaConfig
            config = CodaConfig()
            
            # Test basic config access
            has_voice = hasattr(config, 'voice')
            has_llm = hasattr(config, 'llm')
            has_memory = hasattr(config, 'memory')
            
            response_time = (time.time() - start_time) * 1000
            
            self.results.append({
                "test": "config_creation",
                "component": "configuration",
                "success": True,
                "response_time_ms": response_time,
                "details": {
                    "has_voice": has_voice,
                    "has_llm": has_llm,
                    "has_memory": has_memory
                }
            })
            logger.info(f"    ✅ Configuration: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append({
                "test": "config_creation",
                "component": "configuration",
                "success": False,
                "response_time_ms": response_time,
                "error": str(e)
            })
            logger.error(f"    ❌ Configuration: {e}")
    
    async def _test_session_manager(self):
        """Test session manager functionality."""
        logger.info("  Testing session manager...")
        
        start_time = time.time()
        try:
            from coda.core.session_manager import SessionManager

            # SessionManager doesn't need config, just storage path
            session_manager = SessionManager()

            # Test session creation
            session_id = await session_manager.create_session()

            # Test session retrieval (not async)
            session_data = session_manager.get_session(session_id)

            # Test session cleanup
            await session_manager.end_session(session_id)
            
            response_time = (time.time() - start_time) * 1000
            
            self.results.append({
                "test": "session_operations",
                "component": "session_manager",
                "success": True,
                "response_time_ms": response_time,
                "details": {
                    "session_created": session_id is not None,
                    "session_retrieved": session_data is not None
                }
            })
            logger.info(f"    ✅ Session Manager: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append({
                "test": "session_operations",
                "component": "session_manager",
                "success": False,
                "response_time_ms": response_time,
                "error": str(e)
            })
            logger.error(f"    ❌ Session Manager: {e}")
    
    async def _test_event_coordinator(self):
        """Test event coordinator functionality."""
        logger.info("  Testing event coordinator...")
        
        start_time = time.time()
        try:
            from coda.core.event_coordinator import EventCoordinator
            
            coordinator = EventCoordinator()
            await coordinator.initialize()
            
            # Test event emission (use correct method name)
            await coordinator.emit_gui_event("test_event", {"data": "validation"})
            
            response_time = (time.time() - start_time) * 1000
            
            self.results.append({
                "test": "event_operations",
                "component": "event_coordinator",
                "success": True,
                "response_time_ms": response_time,
                "details": {"event_emitted": True}
            })
            logger.info(f"    ✅ Event Coordinator: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append({
                "test": "event_operations",
                "component": "event_coordinator",
                "success": False,
                "response_time_ms": response_time,
                "error": str(e)
            })
            logger.error(f"    ❌ Event Coordinator: {e}")
    
    async def _test_memory_system(self):
        """Test memory system functionality."""
        logger.info("  Testing memory system...")
        
        start_time = time.time()
        try:
            from coda.components.memory.manager import MemoryManager
            from coda.components.memory.models import MemoryManagerConfig

            # Use proper MemoryManagerConfig
            config = MemoryManagerConfig()
            memory_manager = MemoryManager(config)
            await memory_manager.initialize()

            # Test memory operations
            memory_manager.add_turn("user", "Test message for validation")
            memory_manager.add_turn("assistant", "Test response for validation")

            # Test memory search (correct parameter name)
            results = await memory_manager.search_memories("validation", limit=5)

            # Cleanup
            await memory_manager.shutdown()
            
            response_time = (time.time() - start_time) * 1000
            
            self.results.append({
                "test": "memory_operations",
                "component": "memory_system",
                "success": True,
                "response_time_ms": response_time,
                "details": {
                    "turns_added": 2,
                    "search_results": len(results)
                }
            })
            logger.info(f"    ✅ Memory System: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append({
                "test": "memory_operations",
                "component": "memory_system",
                "success": False,
                "response_time_ms": response_time,
                "error": str(e)
            })
            logger.error(f"    ❌ Memory System: {e}")
    
    async def _test_tools_system(self):
        """Test tools system functionality."""
        logger.info("  Testing tools system...")
        
        start_time = time.time()
        try:
            from coda.components.tools.manager import ToolManager
            from coda.core.config import CodaConfig
            
            config = CodaConfig()
            tool_manager = ToolManager(config)
            await tool_manager.initialize()
            
            # Test tool listing (use correct method name)
            tools = tool_manager.get_available_tools()
            
            # Test tool execution (if tools are available)
            if tools:
                # Try to execute a simple tool like get_time using process_function_call
                try:
                    result = await tool_manager.process_function_call({
                        "name": "get_time",
                        "arguments": "{}"
                    })
                    tool_executed = result is not None and result.success
                except Exception:
                    tool_executed = False
            else:
                tool_executed = False
            
            # Cleanup
            await tool_manager.shutdown()
            
            response_time = (time.time() - start_time) * 1000
            
            self.results.append({
                "test": "tools_operations",
                "component": "tools_system",
                "success": True,
                "response_time_ms": response_time,
                "details": {
                    "tools_count": len(tools),
                    "tool_executed": tool_executed
                }
            })
            logger.info(f"    ✅ Tools System: {response_time:.1f}ms, {len(tools)} tools")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results.append({
                "test": "tools_operations",
                "component": "tools_system",
                "success": False,
                "response_time_ms": response_time,
                "error": str(e)
            })
            logger.error(f"    ❌ Tools System: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - successful_tests
        
        # Group by component
        component_results = {}
        for result in self.results:
            component = result["component"]
            if component not in component_results:
                component_results[component] = {"passed": 0, "failed": 0, "total": 0}
            
            component_results[component]["total"] += 1
            if result["success"]:
                component_results[component]["passed"] += 1
            else:
                component_results[component]["failed"] += 1
        
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
            "validation_timestamp": datetime.now().isoformat()
        }


async def main():
    """Run the simple validation suite."""
    validator = SimpleValidationSuite()
    
    try:
        results = await validator.run_validation()
        
        # Print summary
        summary = results["summary"]
        logger.info("=" * 60)
        logger.info("🎯 PHASE 1.5 SIMPLE VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Tests: {summary['successful_tests']}/{summary['total_tests']} passed")
        
        logger.info("\n📊 Component Results:")
        for component, stats in summary["component_results"].items():
            success_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  • {component}: {stats['passed']}/{stats['total']} ({success_rate:.1%})")
        
        # Show failed tests
        failed_results = [r for r in results["results"] if not r["success"]]
        if failed_results:
            logger.info("\n❌ Failed Tests:")
            for result in failed_results:
                logger.info(f"  • {result['component']}.{result['test']}: {result.get('error', 'Unknown error')}")
        
        return 0 if summary["overall_status"] in ["excellent", "good"] else 1
        
    except Exception as e:
        logger.error(f"Validation suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
