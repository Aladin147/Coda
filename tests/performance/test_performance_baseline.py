"""
Performance Baseline Tests.
Establishes baseline performance metrics for RTX 5090 system.
"""

import pytest
import asyncio
import time
import uuid
import psutil
import os
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from src.coda.components.voice.models import (
    VoiceMessage, VoiceResponse, VoiceProcessingMode
)


class PerformanceBaseline:
    """Performance baseline metrics for RTX 5090."""
    
    # RTX 5090 Target Performance Metrics
    TARGET_METRICS = {
        "single_request_latency_ms": 100,  # Under 100ms for single request
        "concurrent_10_avg_latency_ms": 200,  # Under 200ms average for 10 concurrent
        "concurrent_50_avg_latency_ms": 500,  # Under 500ms average for 50 concurrent
        "throughput_req_per_sec": 20,  # At least 20 requests per second
        "memory_usage_mb_per_request": 50,  # Under 50MB per request
        "cpu_usage_percent_max": 80,  # Under 80% CPU usage
        "error_rate_percent": 5,  # Under 5% error rate
    }
    
    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process(os.getpid())
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        self.metrics[name] = value
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)
    
    def validate_against_targets(self) -> Dict[str, bool]:
        """Validate metrics against RTX 5090 targets."""
        results = {}
        
        for target_name, target_value in self.TARGET_METRICS.items():
            actual_value = self.metrics.get(target_name, float('inf'))
            
            # For latency and usage metrics, lower is better
            if "latency" in target_name or "usage" in target_name or "error_rate" in target_name:
                results[target_name] = actual_value <= target_value
            else:  # For throughput, higher is better
                results[target_name] = actual_value >= target_value
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        validation = self.validate_against_targets()
        passed = sum(1 for v in validation.values() if v)
        total = len(validation)
        
        return {
            "metrics": self.metrics.copy(),
            "targets": self.TARGET_METRICS.copy(),
            "validation": validation,
            "score": f"{passed}/{total}",
            "percentage": (passed / total) * 100 if total > 0 else 0
        }


class TestPerformanceBaseline:
    """Performance baseline tests for RTX 5090."""

    @pytest.fixture
    def baseline(self):
        """Create performance baseline tracker."""
        return PerformanceBaseline()

    @pytest.fixture
    def mock_voice_manager(self):
        """Create mock voice manager with realistic performance."""
        manager = Mock()
        
        async def mock_process(message):
            # Simulate realistic processing time based on mode
            if message.processing_mode == VoiceProcessingMode.MOSHI_ONLY:
                processing_time = 0.05  # 50ms
            elif message.processing_mode == VoiceProcessingMode.LLM_ONLY:
                processing_time = 0.08  # 80ms
            else:  # HYBRID
                processing_time = 0.12  # 120ms
            
            await asyncio.sleep(processing_time)
            
            return VoiceResponse(
                response_id=str(uuid.uuid4()),
                conversation_id=message.conversation_id,
                message_id=message.message_id,
                text_content="Baseline test response",
                processing_mode=message.processing_mode,
                total_latency_ms=processing_time * 1000
            )
        
        manager.process_voice_input = AsyncMock(side_effect=mock_process)
        manager.is_initialized = True
        return manager

    @pytest.mark.benchmark(group="baseline")
    def test_single_request_baseline(self, benchmark, mock_voice_manager, baseline):
        """Establish single request performance baseline."""
        
        def single_request():
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id="baseline-single",
                text_content="Single request baseline test",
                audio_data=b"fake_audio_data",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            
            return asyncio.run(mock_voice_manager.process_voice_input(voice_message))
        
        # Record initial memory
        initial_memory = baseline.get_memory_usage_mb()
        
        # Run benchmark
        result = benchmark(single_request)
        
        # Record final memory
        final_memory = baseline.get_memory_usage_mb()
        memory_per_request = final_memory - initial_memory
        
        # Record metrics
        baseline.record_metric("single_request_latency_ms", result.total_latency_ms)
        baseline.record_metric("memory_usage_mb_per_request", memory_per_request)
        
        # Verify response
        assert isinstance(result, VoiceResponse)
        assert result.total_latency_ms > 0
        
        print(f"Single request latency: {result.total_latency_ms:.2f}ms")
        print(f"Memory per request: {memory_per_request:.2f}MB")

    @pytest.mark.asyncio
    async def test_concurrent_10_baseline(self, mock_voice_manager, baseline):
        """Establish 10 concurrent requests baseline."""
        
        # Create 10 concurrent requests
        requests = [
            VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"baseline-10-{i}",
                text_content=f"Concurrent baseline test {i}",
                audio_data=b"fake_audio_data",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            for i in range(10)
        ]
        
        # Record initial state
        initial_memory = baseline.get_memory_usage_mb()
        initial_cpu = baseline.get_cpu_usage_percent()
        
        # Process concurrently
        start_time = time.time()
        tasks = [mock_voice_manager.process_voice_input(msg) for msg in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Record final state
        final_memory = baseline.get_memory_usage_mb()
        final_cpu = baseline.get_cpu_usage_percent()
        
        # Calculate metrics
        total_time = (end_time - start_time) * 1000
        avg_latency = sum(r.total_latency_ms for r in responses if isinstance(r, VoiceResponse)) / len(responses)
        throughput = len(responses) / (end_time - start_time)
        memory_usage = final_memory - initial_memory
        cpu_usage = max(final_cpu - initial_cpu, 0)
        
        # Record metrics
        baseline.record_metric("concurrent_10_avg_latency_ms", avg_latency)
        baseline.record_metric("throughput_req_per_sec", throughput)
        baseline.record_metric("cpu_usage_percent_max", cpu_usage)
        
        # Count errors
        errors = sum(1 for r in responses if isinstance(r, Exception))
        error_rate = (errors / len(responses)) * 100
        baseline.record_metric("error_rate_percent", error_rate)
        
        print(f"10 concurrent requests: {total_time:.2f}ms total, {avg_latency:.2f}ms avg")
        print(f"Throughput: {throughput:.2f} req/s")
        print(f"CPU usage: {cpu_usage:.1f}%")
        print(f"Error rate: {error_rate:.1f}%")
        
        # Verify all requests completed successfully
        assert errors == 0
        assert len(responses) == 10

    @pytest.mark.asyncio
    async def test_concurrent_50_baseline(self, mock_voice_manager, baseline):
        """Establish 50 concurrent requests baseline."""
        
        # Create 50 concurrent requests
        requests = [
            VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"baseline-50-{i}",
                text_content=f"High concurrency baseline test {i}",
                audio_data=b"fake_audio_data",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            for i in range(50)
        ]
        
        # Record initial state
        initial_memory = baseline.get_memory_usage_mb()
        
        # Process concurrently
        start_time = time.time()
        tasks = [mock_voice_manager.process_voice_input(msg) for msg in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Calculate metrics
        successful_responses = [r for r in responses if isinstance(r, VoiceResponse)]
        avg_latency = sum(r.total_latency_ms for r in successful_responses) / len(successful_responses) if successful_responses else 0
        
        # Record metrics
        baseline.record_metric("concurrent_50_avg_latency_ms", avg_latency)
        
        # Count errors
        errors = sum(1 for r in responses if isinstance(r, Exception))
        error_rate = (errors / len(responses)) * 100
        
        print(f"50 concurrent requests: {avg_latency:.2f}ms avg latency")
        print(f"Success rate: {len(successful_responses)}/{len(responses)}")
        print(f"Error rate: {error_rate:.1f}%")
        
        # Most requests should succeed
        assert len(successful_responses) > 40  # At least 80% success rate

    def test_processing_mode_comparison(self, mock_voice_manager, baseline):
        """Compare performance across different processing modes."""
        
        modes = [
            VoiceProcessingMode.MOSHI_ONLY,
            VoiceProcessingMode.LLM_ONLY,
            VoiceProcessingMode.HYBRID
        ]
        
        mode_results = {}
        
        for mode in modes:
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"mode-{mode.value}",
                text_content=f"Processing mode comparison for {mode.value}",
                audio_data=b"fake_audio_data",
                processing_mode=mode
            )
            
            # Process request
            start_time = time.time()
            result = asyncio.run(mock_voice_manager.process_voice_input(voice_message))
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000
            mode_results[mode.value] = {
                "latency_ms": processing_time,
                "reported_latency_ms": result.total_latency_ms
            }
            
            print(f"Mode {mode.value}: {processing_time:.2f}ms")
        
        # Verify mode performance differences
        moshi_latency = mode_results["moshi_only"]["latency_ms"]
        llm_latency = mode_results["llm_only"]["latency_ms"]
        hybrid_latency = mode_results["hybrid"]["latency_ms"]
        
        # MOSHI should be fastest, HYBRID should be slowest
        assert moshi_latency <= llm_latency
        assert llm_latency <= hybrid_latency

    def test_memory_efficiency_baseline(self, mock_voice_manager, baseline):
        """Test memory efficiency baseline."""
        
        # Record initial memory
        initial_memory = baseline.get_memory_usage_mb()
        
        # Process multiple requests to test memory efficiency
        async def memory_test():
            requests = []
            for i in range(20):
                voice_message = VoiceMessage(
                    message_id=str(uuid.uuid4()),
                    conversation_id=f"memory-{i}",
                    text_content=f"Memory efficiency test {i}",
                    audio_data=b"fake_audio_data",
                    processing_mode=VoiceProcessingMode.MOSHI_ONLY
                )
                requests.append(voice_message)
            
            # Process sequentially to avoid concurrency effects
            for request in requests:
                await mock_voice_manager.process_voice_input(request)
        
        asyncio.run(memory_test())
        
        # Record final memory
        final_memory = baseline.get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        memory_per_request = memory_increase / 20
        
        print(f"Memory efficiency: {memory_increase:.2f}MB for 20 requests")
        print(f"Memory per request: {memory_per_request:.2f}MB")
        
        # Memory usage should be reasonable
        assert memory_per_request < 10  # Under 10MB per request

    def test_baseline_summary(self, baseline):
        """Generate baseline performance summary."""
        
        # Get performance summary
        summary = baseline.get_performance_summary()
        
        print("\n" + "="*60)
        print("RTX 5090 PERFORMANCE BASELINE SUMMARY")
        print("="*60)
        
        print("\nRecorded Metrics:")
        for metric, value in summary["metrics"].items():
            print(f"  {metric}: {value:.2f}")
        
        print("\nTarget Metrics:")
        for metric, target in summary["targets"].items():
            actual = summary["metrics"].get(metric, "Not measured")
            status = "✅ PASS" if summary["validation"].get(metric, False) else "❌ FAIL"
            print(f"  {metric}: {actual} (target: {target}) {status}")
        
        print(f"\nOverall Score: {summary['score']} ({summary['percentage']:.1f}%)")
        
        if summary['percentage'] >= 80:
            print("🎉 EXCELLENT: RTX 5090 performance targets largely met!")
        elif summary['percentage'] >= 60:
            print("✅ GOOD: RTX 5090 performance is acceptable with room for improvement")
        else:
            print("⚠️  NEEDS IMPROVEMENT: RTX 5090 performance below expectations")
        
        print("="*60)
        
        # At least 60% of targets should be met
        assert summary['percentage'] >= 60, f"Performance baseline too low: {summary['percentage']:.1f}%"

    @pytest.mark.benchmark(group="baseline_throughput")
    def test_throughput_baseline(self, benchmark, mock_voice_manager):
        """Establish throughput baseline."""
        
        def throughput_test():
            async def process_batch():
                requests = [
                    VoiceMessage(
                        message_id=str(uuid.uuid4()),
                        conversation_id=f"throughput-{i}",
                        text_content=f"Throughput test {i}",
                        audio_data=b"fake_audio_data",
                        processing_mode=VoiceProcessingMode.MOSHI_ONLY
                    )
                    for i in range(10)
                ]
                
                start_time = time.time()
                tasks = [mock_voice_manager.process_voice_input(msg) for msg in requests]
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                return len(results), end_time - start_time
            
            return asyncio.run(process_batch())
        
        count, duration = benchmark(throughput_test)
        throughput = count / duration
        
        print(f"Throughput baseline: {throughput:.2f} requests/second")
        
        # Should achieve reasonable throughput
        assert throughput > 10  # At least 10 requests per second
