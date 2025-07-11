#!/usr/bin/env python3
"""
Comprehensive Stress Testing Suite for Coda Phase 1.5.

Tests system limits, concurrent operations, memory pressure, and performance
under extreme conditions to ensure robustness and identify breaking points.
"""

import asyncio
import logging
import time
import gc
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import json
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
logger = logging.getLogger("stress_test")


@dataclass
class StressTestMetrics:
    """Metrics collection for stress testing."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    response_times: List[float] = field(default_factory=list)
    memory_usage_samples: List[float] = field(default_factory=list)
    cpu_usage_samples: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    
    def record_operation(self, success: bool, response_time: float, error_type: str = None):
        """Record an operation result."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.response_times.append(response_time)
    
    def record_system_metrics(self):
        """Record current system metrics."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        self.memory_usage_samples.append(memory_mb)
        self.cpu_usage_samples.append(cpu_percent)
        
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        duration = (self.end_time or datetime.now()) - self.start_time
        
        return {
            "duration_seconds": duration.total_seconds(),
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.successful_operations / max(1, self.total_operations),
            "operations_per_second": self.total_operations / max(1, duration.total_seconds()),
            "avg_response_time_ms": sum(self.response_times) / max(1, len(self.response_times)) * 1000,
            "min_response_time_ms": min(self.response_times) * 1000 if self.response_times else 0,
            "max_response_time_ms": max(self.response_times) * 1000 if self.response_times else 0,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_cpu_percent": self.peak_cpu_percent,
            "avg_memory_mb": sum(self.memory_usage_samples) / max(1, len(self.memory_usage_samples)),
            "avg_cpu_percent": sum(self.cpu_usage_samples) / max(1, len(self.cpu_usage_samples)),
            "error_counts": self.error_counts,
            "memory_samples": len(self.memory_usage_samples),
            "cpu_samples": len(self.cpu_usage_samples)
        }


class ComprehensiveStressTestSuite:
    """
    Comprehensive stress testing suite for Coda system.
    
    Tests system limits, concurrent operations, memory pressure,
    and performance under extreme conditions.
    """
    
    def __init__(self):
        self.config = CodaConfig()
        self.results: Dict[str, Any] = {}
        self.assistant: Optional[CodaAssistant] = None
        self.session_manager: Optional[SessionManager] = None
        self.event_coordinator: Optional[EventCoordinator] = None
        self.llm_manager: Optional[LLMManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.websocket_server: Optional[CodaWebSocketServer] = None
        
        # Test configuration
        self.max_concurrent_sessions = 50
        self.max_messages_per_session = 100
        self.stress_duration_seconds = 300  # 5 minutes
        self.memory_pressure_mb = 1000  # 1GB memory pressure
        
    async def run_comprehensive_stress_tests(self) -> Dict[str, Any]:
        """Run all stress tests."""
        logger.info("🚀 Starting Comprehensive Stress Test Suite")
        logger.info("=" * 80)
        
        # Initialize system
        await self._initialize_system()
        
        # Test 1: Concurrent Session Stress
        logger.info("🔥 Test 1: Concurrent Session Stress")
        session_results = await self.test_concurrent_session_stress()
        self.results["concurrent_sessions"] = session_results
        
        # Test 2: High Message Volume Stress
        logger.info("🔥 Test 2: High Message Volume Stress")
        message_results = await self.test_high_message_volume_stress()
        self.results["high_message_volume"] = message_results
        
        # Test 3: Memory Pressure Stress
        logger.info("🔥 Test 3: Memory Pressure Stress")
        memory_results = await self.test_memory_pressure_stress()
        self.results["memory_pressure"] = memory_results
        
        # Test 4: WebSocket Connection Stress
        logger.info("🔥 Test 4: WebSocket Connection Stress")
        websocket_results = await self.test_websocket_connection_stress()
        self.results["websocket_connections"] = websocket_results
        
        # Test 5: Event System Stress
        logger.info("🔥 Test 5: Event System Stress")
        event_results = await self.test_event_system_stress()
        self.results["event_system"] = event_results
        
        # Test 6: LLM Provider Stress
        logger.info("🔥 Test 6: LLM Provider Stress")
        llm_results = await self.test_llm_provider_stress()
        self.results["llm_provider"] = llm_results
        
        # Test 7: Memory System Stress
        logger.info("🔥 Test 7: Memory System Stress")
        memory_sys_results = await self.test_memory_system_stress()
        self.results["memory_system"] = memory_sys_results
        
        # Test 8: System Limits Discovery
        logger.info("🔥 Test 8: System Limits Discovery")
        limits_results = await self.test_system_limits_discovery()
        self.results["system_limits"] = limits_results
        
        # Cleanup
        await self._cleanup_system()
        
        # Generate summary
        summary = self._generate_stress_test_summary()
        self.results["summary"] = summary
        
        logger.info("=" * 80)
        logger.info("🎉 Comprehensive Stress Test Suite Complete!")
        
        return self.results
    
    async def test_concurrent_session_stress(self) -> Dict[str, Any]:
        """Test system under concurrent session load."""
        logger.info("  Testing concurrent session handling...")
        
        metrics = StressTestMetrics()
        
        # Monitor system metrics
        monitoring_task = asyncio.create_task(self._monitor_system_metrics(metrics, 60))
        
        async def create_session_and_chat():
            """Create a session and perform multiple chat operations."""
            try:
                start_time = time.time()
                
                # Create session
                session_id = await self.session_manager.create_session()
                
                # Perform multiple chat operations
                for i in range(10):
                    message = f"Test message {i} in session {session_id[:8]}"
                    
                    # Process message through pipeline
                    response = await self.assistant.process_text_message(
                        message=message,
                        session_id=session_id
                    )
                    
                    # Small delay to simulate realistic usage
                    await asyncio.sleep(0.1)
                
                # Cleanup session
                await self.session_manager.end_session(session_id)
                
                response_time = time.time() - start_time
                metrics.record_operation(True, response_time)
                
            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_operation(False, response_time, type(e).__name__)
                logger.warning(f"Session operation failed: {e}")
        
        # Run concurrent sessions
        logger.info(f"    Creating {self.max_concurrent_sessions} concurrent sessions...")
        
        tasks = []
        for i in range(self.max_concurrent_sessions):
            task = asyncio.create_task(create_session_and_chat())
            tasks.append(task)
        
        # Wait for all sessions to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        metrics.end_time = datetime.now()
        summary = metrics.get_summary()
        
        logger.info(f"    ✅ Concurrent sessions: {summary['success_rate']:.1%} success rate")
        logger.info(f"    ✅ Peak memory: {summary['peak_memory_mb']:.1f}MB")
        logger.info(f"    ✅ Avg response time: {summary['avg_response_time_ms']:.1f}ms")
        
        return summary
    
    async def test_high_message_volume_stress(self) -> Dict[str, Any]:
        """Test system under high message volume."""
        logger.info("  Testing high message volume handling...")
        
        metrics = StressTestMetrics()
        
        # Create a single session for message flooding
        session_id = await self.session_manager.create_session()
        
        # Monitor system metrics
        monitoring_task = asyncio.create_task(self._monitor_system_metrics(metrics, 120))
        
        async def send_message_burst():
            """Send a burst of messages."""
            try:
                start_time = time.time()
                
                message = f"Stress test message {random.randint(1000, 9999)}"
                
                response = await self.assistant.process_text_message(
                    message=message,
                    session_id=session_id
                )
                
                response_time = time.time() - start_time
                metrics.record_operation(True, response_time)
                
            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_operation(False, response_time, type(e).__name__)
        
        # Send high volume of messages
        logger.info(f"    Sending {self.max_messages_per_session} messages rapidly...")
        
        tasks = []
        for i in range(self.max_messages_per_session):
            task = asyncio.create_task(send_message_burst())
            tasks.append(task)
            
            # Small delay to prevent overwhelming the system
            if i % 10 == 0:
                await asyncio.sleep(0.1)
        
        # Wait for all messages to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cleanup
        await self.session_manager.end_session(session_id)
        
        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        metrics.end_time = datetime.now()
        summary = metrics.get_summary()
        
        logger.info(f"    ✅ Message volume: {summary['operations_per_second']:.1f} msg/sec")
        logger.info(f"    ✅ Success rate: {summary['success_rate']:.1%}")
        logger.info(f"    ✅ Avg response time: {summary['avg_response_time_ms']:.1f}ms")
        
        return summary
    
    async def test_memory_pressure_stress(self) -> Dict[str, Any]:
        """Test system under memory pressure."""
        logger.info("  Testing memory pressure handling...")
        
        metrics = StressTestMetrics()
        
        # Create memory pressure by allocating large objects
        memory_hogs = []
        
        try:
            # Allocate memory in chunks
            chunk_size = 50 * 1024 * 1024  # 50MB chunks
            target_memory = self.memory_pressure_mb * 1024 * 1024
            
            logger.info(f"    Creating {self.memory_pressure_mb}MB memory pressure...")
            
            while len(memory_hogs) * chunk_size < target_memory:
                # Create large string to consume memory
                chunk = 'x' * chunk_size
                memory_hogs.append(chunk)
                
                # Record memory usage
                metrics.record_system_metrics()
                
                # Test system functionality under pressure
                if len(memory_hogs) % 5 == 0:
                    try:
                        start_time = time.time()
                        
                        session_id = await self.session_manager.create_session()
                        response = await self.assistant.process_text_message(
                            message="Memory pressure test message",
                            session_id=session_id
                        )
                        await self.session_manager.end_session(session_id)
                        
                        response_time = time.time() - start_time
                        metrics.record_operation(True, response_time)
                        
                    except Exception as e:
                        response_time = time.time() - start_time
                        metrics.record_operation(False, response_time, type(e).__name__)
            
            # Test system under maximum memory pressure
            logger.info("    Testing system functionality under memory pressure...")
            
            for i in range(10):
                try:
                    start_time = time.time()
                    
                    session_id = await self.session_manager.create_session()
                    response = await self.assistant.process_text_message(
                        message=f"High memory pressure test {i}",
                        session_id=session_id
                    )
                    await self.session_manager.end_session(session_id)
                    
                    response_time = time.time() - start_time
                    metrics.record_operation(True, response_time)
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    metrics.record_operation(False, response_time, type(e).__name__)
                
                await asyncio.sleep(0.5)
        
        finally:
            # Clean up memory
            logger.info("    Cleaning up memory pressure...")
            memory_hogs.clear()
            gc.collect()
        
        metrics.end_time = datetime.now()
        summary = metrics.get_summary()
        
        logger.info(f"    ✅ Memory pressure: {summary['peak_memory_mb']:.1f}MB peak")
        logger.info(f"    ✅ System stability: {summary['success_rate']:.1%}")
        logger.info(f"    ✅ Performance impact: {summary['avg_response_time_ms']:.1f}ms avg")
        
        return summary

    async def test_websocket_connection_stress(self) -> Dict[str, Any]:
        """Test WebSocket server under connection stress."""
        logger.info("  Testing WebSocket connection stress...")

        metrics = StressTestMetrics()

        # Start WebSocket server if not already running
        if not self.websocket_server:
            self.websocket_server = CodaWebSocketServer(self.config)
            await self.websocket_server.start()

        # Simulate multiple WebSocket connections
        connections = []

        try:
            logger.info("    Creating multiple WebSocket connections...")

            # Create connections (simulated)
            for i in range(20):
                try:
                    start_time = time.time()

                    # Simulate connection creation and message sending
                    connection_id = f"stress_test_conn_{i}"

                    # Simulate sending events through WebSocket
                    await self.websocket_server.broadcast_event(
                        "stress_test",
                        {"connection_id": connection_id, "message": f"Test message {i}"}
                    )

                    response_time = time.time() - start_time
                    metrics.record_operation(True, response_time)

                    connections.append(connection_id)

                except Exception as e:
                    response_time = time.time() - start_time
                    metrics.record_operation(False, response_time, type(e).__name__)

                # Record system metrics
                metrics.record_system_metrics()

                await asyncio.sleep(0.1)

            # Test broadcast performance
            logger.info("    Testing broadcast performance...")

            for i in range(50):
                try:
                    start_time = time.time()

                    await self.websocket_server.broadcast_event(
                        "broadcast_stress_test",
                        {"broadcast_id": i, "timestamp": time.time()}
                    )

                    response_time = time.time() - start_time
                    metrics.record_operation(True, response_time)

                except Exception as e:
                    response_time = time.time() - start_time
                    metrics.record_operation(False, response_time, type(e).__name__)

        finally:
            # Cleanup connections
            connections.clear()

        metrics.end_time = datetime.now()
        summary = metrics.get_summary()

        logger.info(f"    ✅ WebSocket stress: {summary['success_rate']:.1%} success rate")
        logger.info(f"    ✅ Broadcast performance: {summary['operations_per_second']:.1f} ops/sec")

        return summary

    async def test_event_system_stress(self) -> Dict[str, Any]:
        """Test event coordination system under stress."""
        logger.info("  Testing event system stress...")

        metrics = StressTestMetrics()

        # Test high-frequency event generation
        event_types = [
            "conversation_turn", "memory_store", "llm_response",
            "system_status", "component_health", "performance_metric"
        ]

        async def generate_events():
            """Generate high-frequency events."""
            for i in range(100):
                try:
                    start_time = time.time()

                    event_type = random.choice(event_types)
                    event_data = {
                        "event_id": i,
                        "timestamp": time.time(),
                        "data": f"stress_test_data_{i}",
                        "session_id": f"stress_session_{i % 10}"
                    }

                    # Emit event through coordinator
                    await self.event_coordinator.emit_event(event_type, event_data)

                    response_time = time.time() - start_time
                    metrics.record_operation(True, response_time)

                except Exception as e:
                    response_time = time.time() - start_time
                    metrics.record_operation(False, response_time, type(e).__name__)

                # Record system metrics periodically
                if i % 10 == 0:
                    metrics.record_system_metrics()

                await asyncio.sleep(0.01)  # High frequency

        # Run concurrent event generation
        logger.info("    Generating high-frequency events...")

        tasks = []
        for i in range(5):  # 5 concurrent event generators
            task = asyncio.create_task(generate_events())
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end_time = datetime.now()
        summary = metrics.get_summary()

        logger.info(f"    ✅ Event system: {summary['operations_per_second']:.1f} events/sec")
        logger.info(f"    ✅ Success rate: {summary['success_rate']:.1%}")

        return summary

    async def test_llm_provider_stress(self) -> Dict[str, Any]:
        """Test LLM provider under stress."""
        logger.info("  Testing LLM provider stress...")

        metrics = StressTestMetrics()

        # Test concurrent LLM requests
        async def make_llm_request():
            """Make a single LLM request."""
            try:
                start_time = time.time()

                prompt = f"Stress test prompt {random.randint(1000, 9999)}: Please respond briefly."

                response = await self.llm_manager.generate_response(
                    prompt=prompt,
                    provider="ollama"
                )

                response_time = time.time() - start_time
                metrics.record_operation(True, response_time)

            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_operation(False, response_time, type(e).__name__)

        # Run concurrent LLM requests
        logger.info("    Making concurrent LLM requests...")

        tasks = []
        for i in range(20):  # 20 concurrent requests
            task = asyncio.create_task(make_llm_request())
            tasks.append(task)

            # Record system metrics
            if i % 5 == 0:
                metrics.record_system_metrics()

            await asyncio.sleep(0.2)  # Stagger requests

        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end_time = datetime.now()
        summary = metrics.get_summary()

        logger.info(f"    ✅ LLM stress: {summary['success_rate']:.1%} success rate")
        logger.info(f"    ✅ Avg response time: {summary['avg_response_time_ms']:.1f}ms")

        return summary

    async def test_memory_system_stress(self) -> Dict[str, Any]:
        """Test memory system under stress."""
        logger.info("  Testing memory system stress...")

        metrics = StressTestMetrics()

        # Test high-volume memory operations
        async def memory_operations():
            """Perform memory operations."""
            try:
                start_time = time.time()

                # Store memory
                memory_content = f"Stress test memory {random.randint(1000, 9999)}"
                self.memory_manager.add_turn("user", memory_content)
                self.memory_manager.add_turn("assistant", f"Response to {memory_content}")

                # Search memory
                search_results = await self.memory_manager.search_memories(
                    query="stress test",
                    max_results=5
                )

                response_time = time.time() - start_time
                metrics.record_operation(True, response_time)

            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_operation(False, response_time, type(e).__name__)

        # Run concurrent memory operations
        logger.info("    Performing high-volume memory operations...")

        tasks = []
        for i in range(100):  # 100 memory operations
            task = asyncio.create_task(memory_operations())
            tasks.append(task)

            # Record system metrics
            if i % 20 == 0:
                metrics.record_system_metrics()

            await asyncio.sleep(0.05)

        await asyncio.gather(*tasks, return_exceptions=True)

        metrics.end_time = datetime.now()
        summary = metrics.get_summary()

        logger.info(f"    ✅ Memory system: {summary['operations_per_second']:.1f} ops/sec")
        logger.info(f"    ✅ Success rate: {summary['success_rate']:.1%}")

        return summary

    async def test_system_limits_discovery(self) -> Dict[str, Any]:
        """Discover system limits and breaking points."""
        logger.info("  Discovering system limits...")

        limits = {
            "max_concurrent_sessions": 0,
            "max_messages_per_second": 0,
            "memory_limit_mb": 0,
            "breaking_points": []
        }

        # Test maximum concurrent sessions
        logger.info("    Testing maximum concurrent sessions...")

        session_count = 0
        try:
            sessions = []
            while session_count < 200:  # Test up to 200 sessions
                session_id = await self.session_manager.create_session()
                sessions.append(session_id)
                session_count += 1

                # Test if system is still responsive
                if session_count % 10 == 0:
                    try:
                        test_response = await self.assistant.process_text_message(
                            message="System limit test",
                            session_id=session_id
                        )
                        limits["max_concurrent_sessions"] = session_count
                    except Exception as e:
                        limits["breaking_points"].append(f"Session limit at {session_count}: {e}")
                        break

            # Cleanup sessions
            for session_id in sessions:
                try:
                    await self.session_manager.end_session(session_id)
                except:
                    pass

        except Exception as e:
            limits["breaking_points"].append(f"Session creation failed at {session_count}: {e}")

        logger.info(f"    ✅ Max concurrent sessions: {limits['max_concurrent_sessions']}")

        return limits

    async def _initialize_system(self):
        """Initialize all system components."""
        logger.info("  Initializing system components...")

        # Initialize core components
        self.assistant = CodaAssistant(self.config)
        await self.assistant.initialize()

        # Get component references
        self.session_manager = self.assistant.session_manager
        self.event_coordinator = self.assistant.event_coordinator
        self.llm_manager = self.assistant.llm_manager
        self.memory_manager = self.assistant.memory_manager

        # Initialize WebSocket server
        self.websocket_server = CodaWebSocketServer(self.config)
        await self.websocket_server.start()

        logger.info("  ✅ System components initialized")

    async def _cleanup_system(self):
        """Cleanup system components."""
        logger.info("  Cleaning up system components...")

        try:
            if self.websocket_server:
                await self.websocket_server.stop()

            if self.assistant:
                await self.assistant.shutdown()

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

        # Force garbage collection
        gc.collect()

        logger.info("  ✅ System cleanup complete")

    async def _monitor_system_metrics(self, metrics: StressTestMetrics, duration: int):
        """Monitor system metrics for specified duration."""
        end_time = time.time() + duration

        while time.time() < end_time:
            metrics.record_system_metrics()
            await asyncio.sleep(1)

    def _generate_stress_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive stress test summary."""
        summary = {
            "overall_status": "success",
            "total_tests": 8,
            "passed_tests": 0,
            "failed_tests": 0,
            "performance_metrics": {},
            "system_limits": {},
            "recommendations": []
        }

        # Analyze each test result
        test_results = [
            ("concurrent_sessions", "Concurrent Session Stress"),
            ("high_message_volume", "High Message Volume Stress"),
            ("memory_pressure", "Memory Pressure Stress"),
            ("websocket_connections", "WebSocket Connection Stress"),
            ("event_system", "Event System Stress"),
            ("llm_provider", "LLM Provider Stress"),
            ("memory_system", "Memory System Stress"),
            ("system_limits", "System Limits Discovery")
        ]

        for test_key, test_name in test_results:
            if test_key in self.results:
                result = self.results[test_key]

                # Check if test passed (success rate > 80%)
                if isinstance(result, dict) and "success_rate" in result:
                    if result["success_rate"] > 0.8:
                        summary["passed_tests"] += 1
                    else:
                        summary["failed_tests"] += 1
                        summary["recommendations"].append(f"Improve {test_name} performance")
                else:
                    summary["passed_tests"] += 1  # Assume passed if no success rate

        # Extract performance metrics
        if "concurrent_sessions" in self.results:
            summary["performance_metrics"]["max_concurrent_sessions"] = self.results["concurrent_sessions"].get("total_operations", 0)

        if "high_message_volume" in self.results:
            summary["performance_metrics"]["max_messages_per_second"] = self.results["high_message_volume"].get("operations_per_second", 0)

        if "memory_pressure" in self.results:
            summary["performance_metrics"]["peak_memory_mb"] = self.results["memory_pressure"].get("peak_memory_mb", 0)

        # Extract system limits
        if "system_limits" in self.results:
            summary["system_limits"] = self.results["system_limits"]

        # Overall assessment
        if summary["passed_tests"] >= 7:
            summary["overall_status"] = "excellent"
        elif summary["passed_tests"] >= 5:
            summary["overall_status"] = "good"
        elif summary["passed_tests"] >= 3:
            summary["overall_status"] = "fair"
        else:
            summary["overall_status"] = "needs_improvement"

        return summary


async def main():
    """Run the comprehensive stress test suite."""
    stress_tester = ComprehensiveStressTestSuite()

    try:
        results = await stress_tester.run_comprehensive_stress_tests()

        # Print summary
        summary = results["summary"]
        logger.info("=" * 80)
        logger.info("🔥 COMPREHENSIVE STRESS TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")

        if summary["performance_metrics"]:
            logger.info("\n⚡ Performance Metrics:")
            for metric, value in summary["performance_metrics"].items():
                logger.info(f"  • {metric}: {value}")

        if summary["system_limits"]:
            logger.info("\n🚧 System Limits:")
            limits = summary["system_limits"]
            if "max_concurrent_sessions" in limits:
                logger.info(f"  • Max concurrent sessions: {limits['max_concurrent_sessions']}")
            if "breaking_points" in limits and limits["breaking_points"]:
                logger.info("  • Breaking points:")
                for bp in limits["breaking_points"]:
                    logger.info(f"    - {bp}")

        if summary["recommendations"]:
            logger.info("\n💡 Recommendations:")
            for rec in summary["recommendations"]:
                logger.info(f"  • {rec}")

        return 0 if summary["overall_status"] in ["excellent", "good"] else 1

    except Exception as e:
        logger.error(f"Stress test suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
