#!/usr/bin/env python3
"""
Phase 1.5 Stress Testing Suite

This comprehensive stress testing suite is designed to:
1. Push the system to its limits
2. Identify breaking points and bottlenecks
3. Test concurrent operations
4. Validate memory and resource management
5. Ensure no fallbacks are used under stress

Tests are designed to SURFACE ISSUES, not just pass.
"""

import asyncio
import logging
import sys
import time
import psutil
import gc
import threading
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for detailed analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests/stress_test_results.log', mode='w')
    ]
)
logger = logging.getLogger("stress_test")

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


class SystemMonitor:
    """Monitor system resources during stress tests."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.metrics = []
        
        def monitor():
            while self.monitoring:
                try:
                    cpu_percent = self.process.cpu_percent()
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    
                    self.metrics.append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_mb,
                        'threads': self.process.num_threads()
                    })
                    time.sleep(0.1)  # Sample every 100ms
                except Exception as e:
                    logger.warning(f"Monitoring error: {e}")
                    
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return metrics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        return self.metrics
    
    def get_peak_usage(self):
        """Get peak resource usage."""
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_mb'] for m in self.metrics]
        
        return {
            'peak_cpu': max(cpu_values),
            'peak_memory_mb': max(memory_values),
            'avg_cpu': statistics.mean(cpu_values),
            'avg_memory_mb': statistics.mean(memory_values),
            'samples': len(self.metrics)
        }


class StressTester:
    """Comprehensive stress testing suite."""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.results = {}
        
    async def test_concurrent_sessions(self, num_sessions: int = 50):
        """Test concurrent session creation and management."""
        logger.info(f"🔥 STRESS TEST: Concurrent Sessions ({num_sessions} sessions)")
        
        self.monitor.start_monitoring()
        start_time = time.time()
        
        try:
            session_manager = SessionManager()
            await session_manager.initialize()
            
            # Create sessions concurrently
            tasks = []
            for i in range(num_sessions):
                task = session_manager.create_session(metadata={'test_id': i})
                tasks.append(task)
            
            session_ids = await asyncio.gather(*tasks)
            
            # Test concurrent operations on sessions
            operation_tasks = []
            for i, session_id in enumerate(session_ids):
                # Add messages to each session
                for j in range(10):  # 10 messages per session
                    task = session_manager.add_message_to_session(
                        session_id, 
                        "user" if j % 2 == 0 else "assistant",
                        f"Test message {j} in session {i}"
                    )
                    operation_tasks.append(task)
            
            await asyncio.gather(*operation_tasks)
            
            # Test concurrent session queries (fix: get_session_history is not async)
            histories = []
            for session_id in session_ids:
                history = session_manager.get_session_history(session_id)
                histories.append(history)
            
            # Cleanup
            cleanup_tasks = []
            for session_id in session_ids:
                cleanup_tasks.append(session_manager.end_session(session_id))
            
            await asyncio.gather(*cleanup_tasks)
            
            end_time = time.time()
            metrics = self.monitor.stop_monitoring()
            peak_usage = self.monitor.get_peak_usage()
            
            result = {
                'test': 'concurrent_sessions',
                'sessions_created': len(session_ids),
                'messages_added': len(operation_tasks),
                'total_time': end_time - start_time,
                'avg_time_per_session': (end_time - start_time) / num_sessions,
                'peak_memory_mb': peak_usage.get('peak_memory_mb', 0),
                'peak_cpu': peak_usage.get('peak_cpu', 0),
                'success': True
            }
            
            logger.info(f"✅ Concurrent sessions test completed: {result}")
            return result
            
        except Exception as e:
            self.monitor.stop_monitoring()
            logger.error(f"❌ Concurrent sessions test failed: {e}")
            return {'test': 'concurrent_sessions', 'success': False, 'error': str(e)}
    
    async def test_high_volume_messages(self, messages_per_second: int = 10, duration_seconds: int = 30):
        """Test high volume message processing."""
        logger.info(f"🔥 STRESS TEST: High Volume Messages ({messages_per_second}/sec for {duration_seconds}s)")
        
        self.monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # Setup components
            session_manager = SessionManager()
            await session_manager.initialize()
            
            llm_config = LLMConfig(
                providers={
                    "ollama": ProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model="qwen3:30b-a3b",
                        host="http://localhost:11434",
                        temperature=0.7,
                        max_tokens=50,  # Short responses for speed
                        system_message="/no_think Respond with exactly 5 words."
                    )
                },
                default_provider="ollama"
            )
            
            llm_manager = LLMManager(llm_config)
            session_id = await session_manager.create_session()
            
            # Generate high volume of messages
            message_count = 0
            response_times = []
            errors = []
            
            end_time = start_time + duration_seconds
            
            while time.time() < end_time:
                batch_start = time.time()
                
                # Send batch of messages
                batch_tasks = []
                for i in range(messages_per_second):
                    if time.time() >= end_time:
                        break
                    
                    message = f"Quick test message {message_count + i}"
                    task = self._process_message_with_timing(
                        llm_manager, session_manager, session_id, message
                    )
                    batch_tasks.append(task)
                
                # Process batch
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Collect results
                for result in batch_results:
                    if isinstance(result, Exception):
                        errors.append(str(result))
                    else:
                        response_times.append(result)
                        message_count += 1
                
                # Rate limiting
                batch_duration = time.time() - batch_start
                if batch_duration < 1.0:
                    await asyncio.sleep(1.0 - batch_duration)
            
            end_time = time.time()
            metrics = self.monitor.stop_monitoring()
            peak_usage = self.monitor.get_peak_usage()
            
            result = {
                'test': 'high_volume_messages',
                'messages_processed': message_count,
                'errors': len(errors),
                'error_rate': len(errors) / max(message_count + len(errors), 1),
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'total_time': end_time - start_time,
                'messages_per_second': message_count / (end_time - start_time),
                'peak_memory_mb': peak_usage.get('peak_memory_mb', 0),
                'peak_cpu': peak_usage.get('peak_cpu', 0),
                'success': len(errors) < message_count * 0.1  # Less than 10% error rate
            }
            
            logger.info(f"✅ High volume messages test completed: {result}")
            return result
            
        except Exception as e:
            self.monitor.stop_monitoring()
            logger.error(f"❌ High volume messages test failed: {e}")
            return {'test': 'high_volume_messages', 'success': False, 'error': str(e)}
    
    async def _process_message_with_timing(self, llm_manager, session_manager, session_id, message):
        """Process a single message and return timing."""
        start = time.time()
        try:
            response = await llm_manager.generate_response(
                prompt=f"/no_think {message}",
                provider="ollama"
            )
            
            await session_manager.add_message_to_session(session_id, "user", message)
            await session_manager.add_message_to_session(session_id, "assistant", response.content)
            
            return time.time() - start
        except Exception as e:
            raise Exception(f"Message processing failed: {e}")
    
    async def test_memory_pressure(self, large_sessions: int = 20, messages_per_session: int = 100):
        """Test system under memory pressure."""
        logger.info(f"🔥 STRESS TEST: Memory Pressure ({large_sessions} sessions, {messages_per_session} messages each)")
        
        self.monitor.start_monitoring()
        start_time = time.time()
        
        try:
            session_manager = SessionManager()
            await session_manager.initialize()
            
            memory_manager = MemoryManager(MemoryManagerConfig())
            await memory_manager.initialize()
            
            # Create large sessions with many messages
            session_ids = []
            for i in range(large_sessions):
                session_id = await session_manager.create_session(metadata={'large_session': i})
                session_ids.append(session_id)
                
                # Add many messages to create memory pressure
                for j in range(messages_per_session):
                    long_message = f"This is a long test message {j} in session {i}. " * 20  # ~1KB message
                    await session_manager.add_message_to_session(session_id, "user", long_message)
                    
                    # Also add to memory manager
                    memory_manager.add_turn("user", long_message)
                    memory_manager.add_turn("assistant", f"Response to message {j} in session {i}")
            
            # Force garbage collection
            gc.collect()
            
            # Test retrieval under memory pressure
            retrieval_times = []
            for session_id in session_ids:
                start_retrieval = time.time()
                history = session_manager.get_session_history(session_id)
                retrieval_times.append(time.time() - start_retrieval)
                
                # Verify data integrity
                if len(history) != messages_per_session:
                    raise Exception(f"Data integrity issue: expected {messages_per_session}, got {len(history)}")
            
            end_time = time.time()
            metrics = self.monitor.stop_monitoring()
            peak_usage = self.monitor.get_peak_usage()
            
            result = {
                'test': 'memory_pressure',
                'sessions_created': len(session_ids),
                'total_messages': large_sessions * messages_per_session,
                'avg_retrieval_time': statistics.mean(retrieval_times),
                'max_retrieval_time': max(retrieval_times),
                'total_time': end_time - start_time,
                'peak_memory_mb': peak_usage.get('peak_memory_mb', 0),
                'peak_cpu': peak_usage.get('peak_cpu', 0),
                'success': True
            }
            
            logger.info(f"✅ Memory pressure test completed: {result}")
            return result
            
        except Exception as e:
            self.monitor.stop_monitoring()
            logger.error(f"❌ Memory pressure test failed: {e}")
            return {'test': 'memory_pressure', 'success': False, 'error': str(e)}

    async def test_event_system_overload(self, events_per_second: int = 100, duration_seconds: int = 20):
        """Test event system under heavy load."""
        logger.info(f"🔥 STRESS TEST: Event System Overload ({events_per_second}/sec for {duration_seconds}s)")

        self.monitor.start_monitoring()
        start_time = time.time()

        try:
            # Initialize event system
            event_bus = get_event_bus()
            await event_bus.start()

            event_coordinator = EventCoordinator()
            await event_coordinator.initialize()

            # Track event processing
            events_sent = 0
            events_processed = 0
            processing_times = []

            # Event handler to track processing
            async def test_handler(event):
                nonlocal events_processed
                events_processed += 1

            # Subscribe to test events
            event_coordinator.subscribe_to_events("stress_test_event", test_handler)

            end_time = start_time + duration_seconds

            while time.time() < end_time:
                batch_start = time.time()

                # Send batch of events
                for i in range(events_per_second):
                    if time.time() >= end_time:
                        break

                    event_start = time.time()
                    await event_coordinator.emit_gui_event("stress_test_event", {
                        "event_id": events_sent,
                        "timestamp": time.time(),
                        "data": f"Stress test event {events_sent}"
                    })
                    processing_times.append(time.time() - event_start)
                    events_sent += 1

                # Rate limiting
                batch_duration = time.time() - batch_start
                if batch_duration < 1.0:
                    await asyncio.sleep(1.0 - batch_duration)

            # Wait for event processing to complete
            await asyncio.sleep(1.0)

            end_time = time.time()
            metrics = self.monitor.stop_monitoring()
            peak_usage = self.monitor.get_peak_usage()

            result = {
                'test': 'event_system_overload',
                'events_sent': events_sent,
                'events_processed': events_processed,
                'processing_rate': events_processed / events_sent if events_sent > 0 else 0,
                'avg_event_time': statistics.mean(processing_times) if processing_times else 0,
                'max_event_time': max(processing_times) if processing_times else 0,
                'total_time': end_time - start_time,
                'events_per_second': events_sent / (end_time - start_time),
                'peak_memory_mb': peak_usage.get('peak_memory_mb', 0),
                'peak_cpu': peak_usage.get('peak_cpu', 0),
                'success': events_processed >= events_sent * 0.95  # 95% processing rate
            }

            logger.info(f"✅ Event system overload test completed: {result}")
            return result

        except Exception as e:
            self.monitor.stop_monitoring()
            logger.error(f"❌ Event system overload test failed: {e}")
            return {'test': 'event_system_overload', 'success': False, 'error': str(e)}

    async def test_concurrent_component_operations(self, concurrent_operations: int = 20):
        """Test concurrent operations across multiple components."""
        logger.info(f"🔥 STRESS TEST: Concurrent Component Operations ({concurrent_operations} concurrent ops)")

        self.monitor.start_monitoring()
        start_time = time.time()

        try:
            # Initialize all components
            session_manager = SessionManager()
            await session_manager.initialize()

            memory_manager = MemoryManager(MemoryManagerConfig())
            await memory_manager.initialize()

            llm_config = LLMConfig(
                providers={
                    "ollama": ProviderConfig(
                        provider=LLMProvider.OLLAMA,
                        model="qwen3:30b-a3b",
                        host="http://localhost:11434",
                        temperature=0.7,
                        max_tokens=30,
                        system_message="/no_think Respond briefly."
                    )
                },
                default_provider="ollama"
            )
            llm_manager = LLMManager(llm_config)

            # Create concurrent operations mixing all components
            tasks = []

            for i in range(concurrent_operations):
                # Mix different types of operations
                if i % 4 == 0:
                    # Session operations
                    task = self._concurrent_session_ops(session_manager, i)
                elif i % 4 == 1:
                    # Memory operations
                    task = self._concurrent_memory_ops(memory_manager, i)
                elif i % 4 == 2:
                    # LLM operations
                    task = self._concurrent_llm_ops(llm_manager, i)
                else:
                    # Mixed operations
                    task = self._concurrent_mixed_ops(session_manager, memory_manager, llm_manager, i)

                tasks.append(task)

            # Execute all operations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze results
            successful_ops = sum(1 for r in results if not isinstance(r, Exception))
            failed_ops = len(results) - successful_ops
            errors = [str(r) for r in results if isinstance(r, Exception)]

            end_time = time.time()
            metrics = self.monitor.stop_monitoring()
            peak_usage = self.monitor.get_peak_usage()

            result = {
                'test': 'concurrent_component_operations',
                'total_operations': len(tasks),
                'successful_operations': successful_ops,
                'failed_operations': failed_ops,
                'success_rate': successful_ops / len(tasks),
                'errors': errors[:5],  # First 5 errors for analysis
                'total_time': end_time - start_time,
                'ops_per_second': len(tasks) / (end_time - start_time),
                'peak_memory_mb': peak_usage.get('peak_memory_mb', 0),
                'peak_cpu': peak_usage.get('peak_cpu', 0),
                'success': failed_ops < len(tasks) * 0.1  # Less than 10% failure rate
            }

            logger.info(f"✅ Concurrent component operations test completed: {result}")
            return result

        except Exception as e:
            self.monitor.stop_monitoring()
            logger.error(f"❌ Concurrent component operations test failed: {e}")
            return {'test': 'concurrent_component_operations', 'success': False, 'error': str(e)}

    async def _concurrent_session_ops(self, session_manager, op_id):
        """Concurrent session operations."""
        session_id = await session_manager.create_session(metadata={'op_id': op_id})

        for i in range(5):
            await session_manager.add_message_to_session(
                session_id, "user", f"Concurrent message {i} from op {op_id}"
            )

        history = session_manager.get_session_history(session_id)
        await session_manager.end_session(session_id)

        return len(history)

    async def _concurrent_memory_ops(self, memory_manager, op_id):
        """Concurrent memory operations."""
        for i in range(5):
            memory_manager.add_turn("user", f"Concurrent memory test {i} from op {op_id}")
            memory_manager.add_turn("assistant", f"Response {i} from op {op_id}")

        return True

    async def _concurrent_llm_ops(self, llm_manager, op_id):
        """Concurrent LLM operations."""
        response = await llm_manager.generate_response(
            prompt=f"/no_think Concurrent test {op_id}",
            provider="ollama"
        )
        return len(response.content)

    async def _concurrent_mixed_ops(self, session_manager, memory_manager, llm_manager, op_id):
        """Mixed concurrent operations."""
        # Create session
        session_id = await session_manager.create_session(metadata={'mixed_op': op_id})

        # Add to memory
        memory_manager.add_turn("user", f"Mixed op {op_id}")

        # Get LLM response
        response = await llm_manager.generate_response(
            prompt=f"/no_think Mixed operation {op_id}",
            provider="ollama"
        )

        # Store in session
        await session_manager.add_message_to_session(session_id, "user", f"Mixed op {op_id}")
        await session_manager.add_message_to_session(session_id, "assistant", response.content)

        return True

    async def run_comprehensive_stress_tests(self):
        """Run all stress tests and generate comprehensive report."""
        logger.info("🚀 STARTING COMPREHENSIVE STRESS TESTING SUITE")
        logger.info("=" * 80)
        logger.info("⚠️  WARNING: These tests are designed to SURFACE ISSUES")
        logger.info("⚠️  Tests will push system to limits and may cause failures")
        logger.info("=" * 80)

        all_results = []

        try:
            # Test 1: Concurrent Sessions
            logger.info("\n" + "="*50)
            result1 = await self.test_concurrent_sessions(50)
            all_results.append(result1)

            # Brief recovery time
            await asyncio.sleep(2)

            # Test 2: High Volume Messages
            logger.info("\n" + "="*50)
            result2 = await self.test_high_volume_messages(15, 30)  # 15 msg/sec for 30s
            all_results.append(result2)

            # Brief recovery time
            await asyncio.sleep(2)

            # Test 3: Memory Pressure
            logger.info("\n" + "="*50)
            result3 = await self.test_memory_pressure(25, 150)  # 25 sessions, 150 msgs each
            all_results.append(result3)

            # Brief recovery time
            await asyncio.sleep(2)

            # Test 4: Event System Overload
            logger.info("\n" + "="*50)
            result4 = await self.test_event_system_overload(150, 20)  # 150 events/sec for 20s
            all_results.append(result4)

            # Brief recovery time
            await asyncio.sleep(2)

            # Test 5: Concurrent Component Operations
            logger.info("\n" + "="*50)
            result5 = await self.test_concurrent_component_operations(30)
            all_results.append(result5)

            # Generate comprehensive report
            self._generate_stress_test_report(all_results)

            return all_results

        except Exception as e:
            logger.error(f"❌ Stress testing suite failed: {e}")
            self._generate_stress_test_report(all_results)
            raise

    def _generate_stress_test_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive stress test report."""
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE STRESS TEST REPORT")
        logger.info("="*80)

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get('success', False))
        failed_tests = total_tests - passed_tests

        logger.info(f"📈 OVERALL RESULTS:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        logger.info(f"\n🔍 DETAILED RESULTS:")

        for result in results:
            test_name = result.get('test', 'unknown')
            success = result.get('success', False)
            status = "✅ PASS" if success else "❌ FAIL"

            logger.info(f"\n   {status} {test_name.upper()}")

            if test_name == 'concurrent_sessions':
                logger.info(f"      Sessions Created: {result.get('sessions_created', 0)}")
                logger.info(f"      Messages Added: {result.get('messages_added', 0)}")
                logger.info(f"      Total Time: {result.get('total_time', 0):.2f}s")
                logger.info(f"      Peak Memory: {result.get('peak_memory_mb', 0):.1f}MB")

            elif test_name == 'high_volume_messages':
                logger.info(f"      Messages Processed: {result.get('messages_processed', 0)}")
                logger.info(f"      Error Rate: {result.get('error_rate', 0)*100:.1f}%")
                logger.info(f"      Avg Response Time: {result.get('avg_response_time', 0)*1000:.1f}ms")
                logger.info(f"      Messages/Second: {result.get('messages_per_second', 0):.1f}")

            elif test_name == 'memory_pressure':
                logger.info(f"      Total Messages: {result.get('total_messages', 0)}")
                logger.info(f"      Peak Memory: {result.get('peak_memory_mb', 0):.1f}MB")
                logger.info(f"      Avg Retrieval Time: {result.get('avg_retrieval_time', 0)*1000:.1f}ms")

            elif test_name == 'event_system_overload':
                logger.info(f"      Events Sent: {result.get('events_sent', 0)}")
                logger.info(f"      Processing Rate: {result.get('processing_rate', 0)*100:.1f}%")
                logger.info(f"      Events/Second: {result.get('events_per_second', 0):.1f}")

            elif test_name == 'concurrent_component_operations':
                logger.info(f"      Total Operations: {result.get('total_operations', 0)}")
                logger.info(f"      Success Rate: {result.get('success_rate', 0)*100:.1f}%")
                logger.info(f"      Ops/Second: {result.get('ops_per_second', 0):.1f}")

            if not success and 'error' in result:
                logger.info(f"      Error: {result['error']}")

        # Performance summary
        logger.info(f"\n⚡ PERFORMANCE SUMMARY:")
        peak_memories = [r.get('peak_memory_mb', 0) for r in results if 'peak_memory_mb' in r]
        peak_cpus = [r.get('peak_cpu', 0) for r in results if 'peak_cpu' in r]

        if peak_memories:
            logger.info(f"   Peak Memory Usage: {max(peak_memories):.1f}MB")
            logger.info(f"   Avg Memory Usage: {statistics.mean(peak_memories):.1f}MB")

        if peak_cpus:
            logger.info(f"   Peak CPU Usage: {max(peak_cpus):.1f}%")
            logger.info(f"   Avg CPU Usage: {statistics.mean(peak_cpus):.1f}%")

        # Issues and recommendations
        logger.info(f"\n🚨 ISSUES DETECTED:")
        issues_found = False

        for result in results:
            if not result.get('success', False):
                issues_found = True
                logger.info(f"   ❌ {result.get('test', 'unknown')}: {result.get('error', 'Unknown error')}")

        # Performance warnings
        for result in results:
            if result.get('peak_memory_mb', 0) > 1000:  # > 1GB
                issues_found = True
                logger.info(f"   ⚠️  High memory usage in {result.get('test')}: {result.get('peak_memory_mb'):.1f}MB")

            if result.get('error_rate', 0) > 0.05:  # > 5% error rate
                issues_found = True
                logger.info(f"   ⚠️  High error rate in {result.get('test')}: {result.get('error_rate')*100:.1f}%")

        if not issues_found:
            logger.info("   🎉 No critical issues detected!")

        logger.info(f"\n📝 RECOMMENDATIONS:")
        if failed_tests > 0:
            logger.info("   🔧 Fix failing tests before proceeding to Phase 2")
        if any(r.get('peak_memory_mb', 0) > 500 for r in results):
            logger.info("   🔧 Consider memory optimization for large-scale deployments")
        if any(r.get('error_rate', 0) > 0.02 for r in results):
            logger.info("   🔧 Improve error handling and retry mechanisms")

        logger.info("="*80)

        # Save detailed report to file
        report_path = Path("tests/stress_test_detailed_report.json")
        import json
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': (passed_tests/total_tests)*100
                },
                'detailed_results': results,
                'timestamp': time.time()
            }, f, indent=2)

        logger.info(f"📄 Detailed report saved to: {report_path}")


async def main():
    """Main stress testing function."""
    tester = StressTester()

    try:
        results = await tester.run_comprehensive_stress_tests()

        # Determine exit code based on results
        failed_tests = sum(1 for r in results if not r.get('success', False))
        if failed_tests > 0:
            logger.error(f"❌ {failed_tests} stress tests failed - system needs attention")
            return 1
        else:
            logger.info("🎉 All stress tests passed - system is robust!")
            return 0

    except Exception as e:
        logger.error(f"❌ Stress testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
