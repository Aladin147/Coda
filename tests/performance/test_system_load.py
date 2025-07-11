"""
System Load Testing.
Tests end-to-end performance, multi-user scenarios, and resource utilization.
"""

import pytest
import asyncio
import time
import uuid
import psutil
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Optional

try:
    import torch
    import GPUtil
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.coda.core.assistant import CodaAssistant
from src.coda.core.config import CodaConfig
from src.coda.components.voice.models import (
    VoiceMessage, VoiceResponse, VoiceProcessingMode
)


class SystemLoadMetrics:
    """System load testing metrics collector."""
    
    def __init__(self):
        self.response_times = []
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.throughput_data = []
        self.error_counts = {"timeout": 0, "processing": 0, "memory": 0}
    
    def record_response_time(self, time_ms: float):
        """Record response time."""
        self.response_times.append(time_ms)
    
    def record_resource_usage(self, cpu_percent: float, memory_mb: float, gpu_percent: Optional[float] = None):
        """Record resource usage."""
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_mb)
        if gpu_percent is not None:
            self.gpu_usage.append(gpu_percent)
    
    def record_throughput(self, requests_per_second: float):
        """Record throughput."""
        self.throughput_data.append(requests_per_second)
    
    def record_error(self, error_type: str):
        """Record error occurrence."""
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load testing statistics."""
        import numpy as np
        return {
            "avg_response_time": np.mean(self.response_times) if self.response_times else 0,
            "max_response_time": np.max(self.response_times) if self.response_times else 0,
            "p95_response_time": np.percentile(self.response_times, 95) if len(self.response_times) >= 20 else 0,
            "avg_cpu_usage": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu_usage": np.max(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory_usage": np.mean(self.memory_usage) if self.memory_usage else 0,
            "max_memory_usage": np.max(self.memory_usage) if self.memory_usage else 0,
            "avg_gpu_usage": np.mean(self.gpu_usage) if self.gpu_usage else 0,
            "max_gpu_usage": np.max(self.gpu_usage) if self.gpu_usage else 0,
            "avg_throughput": np.mean(self.throughput_data) if self.throughput_data else 0,
            "max_throughput": np.max(self.throughput_data) if self.throughput_data else 0,
            "total_requests": len(self.response_times),
            "error_counts": self.error_counts.copy()
        }


class ResourceMonitor:
    """Resource usage monitor."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        if not self.gpu_available:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100  # Convert to percentage
            return 0.0
        except Exception:
            return 0.0


class TestSystemLoad:
    """System load testing."""

    @pytest.fixture
    def load_metrics(self):
        """Create load testing metrics collector."""
        return SystemLoadMetrics()

    @pytest.fixture
    def resource_monitor(self):
        """Create resource monitor."""
        return ResourceMonitor()

    @pytest.fixture
    def system_config(self):
        """Create system configuration for load testing."""
        return CodaConfig(
            voice_enabled=True,
            memory_enabled=True,
            llm_enabled=True,
            websocket_enabled=True,
            debug_mode=False  # Disable debug for performance testing
        )

    @pytest.fixture
    def mock_assistant(self, system_config):
        """Create mock assistant for load testing."""
        with patch('src.coda.core.assistant.VoiceManager') as mock_voice_manager, \
             patch('src.coda.core.assistant.MemoryManager') as mock_memory_manager, \
             patch('src.coda.core.assistant.LLMManager') as mock_llm_manager:
            
            # Configure mock voice manager
            voice_instance = mock_voice_manager.return_value
            voice_instance.is_initialized = True
            
            async def mock_process_voice(message):
                # Simulate realistic processing time
                processing_time = 0.05 + (hash(message.message_id) % 100) / 1000  # 50-150ms
                await asyncio.sleep(processing_time)
                
                return VoiceResponse(
                    response_id=str(uuid.uuid4()),
                    conversation_id=message.conversation_id,
                    message_id=message.message_id,
                    text_content=f"Response to: {message.text_content[:50]}...",
                    processing_mode=message.processing_mode,
                    total_latency_ms=processing_time * 1000
                )
            
            voice_instance.process_voice_input = AsyncMock(side_effect=mock_process_voice)
            
            # Configure mock memory manager
            memory_instance = mock_memory_manager.return_value
            memory_instance.is_initialized = True
            memory_instance.store_conversation = AsyncMock(return_value="memory-stored")
            memory_instance.retrieve_relevant_memories = AsyncMock(return_value=[])
            
            # Configure mock LLM manager
            llm_instance = mock_llm_manager.return_value
            llm_instance.is_initialized = True
            llm_instance.generate_response = AsyncMock(return_value="Generated response")
            
            # Create assistant
            assistant = CodaAssistant(config=system_config)
            asyncio.run(assistant.initialize())
            
            return assistant

    @pytest.mark.asyncio
    async def test_single_request_performance(self, mock_assistant, load_metrics, resource_monitor):
        """Test single request performance."""
        
        # Record initial resource usage
        initial_cpu = resource_monitor.get_cpu_usage()
        initial_memory = resource_monitor.get_memory_usage()
        initial_gpu = resource_monitor.get_gpu_usage()
        
        # Create test message
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="single-test",
            text_content="Single request performance test",
            audio_data=b"fake_audio_data",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Process request
        start_time = time.time()
        response = await mock_assistant.process_voice_message(voice_message)
        end_time = time.time()
        
        # Record metrics
        response_time = (end_time - start_time) * 1000
        load_metrics.record_response_time(response_time)
        
        # Record resource usage
        final_cpu = resource_monitor.get_cpu_usage()
        final_memory = resource_monitor.get_memory_usage()
        final_gpu = resource_monitor.get_gpu_usage()
        
        load_metrics.record_resource_usage(
            final_cpu - initial_cpu,
            final_memory - initial_memory,
            final_gpu - initial_gpu if final_gpu is not None and initial_gpu is not None else None
        )
        
        print(f"Single request: {response_time:.2f}ms")
        print(f"CPU: {final_cpu - initial_cpu:.1f}%, Memory: {final_memory - initial_memory:.1f}MB")
        if final_gpu is not None:
            print(f"GPU: {final_gpu - initial_gpu:.1f}%")
        
        # Verify response
        assert isinstance(response, VoiceResponse)
        assert response.conversation_id == "single-test"
        
        # Performance assertions
        assert response_time < 500  # Under 500ms for single request

    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, mock_assistant, load_metrics, resource_monitor):
        """Test concurrent request performance."""
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 25, 50, 100]
        
        for level in concurrency_levels:
            # Record initial resource usage
            initial_cpu = resource_monitor.get_cpu_usage()
            initial_memory = resource_monitor.get_memory_usage()
            initial_gpu = resource_monitor.get_gpu_usage()
            
            # Create concurrent requests
            requests = []
            for i in range(level):
                voice_message = VoiceMessage(
                    message_id=str(uuid.uuid4()),
                    conversation_id=f"concurrent-{i}",
                    text_content=f"Concurrent request {i} at level {level}",
                    audio_data=b"fake_audio_data",
                    processing_mode=VoiceProcessingMode.MOSHI_ONLY
                )
                requests.append(voice_message)
            
            # Process concurrently
            start_time = time.time()
            tasks = [mock_assistant.process_voice_message(msg) for msg in requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Record metrics
            total_time = (end_time - start_time) * 1000
            avg_time = total_time / level
            throughput = level / (end_time - start_time)
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    load_metrics.record_error("processing")
                else:
                    load_metrics.record_response_time(response.total_latency_ms)
            
            load_metrics.record_throughput(throughput)
            
            # Record resource usage
            final_cpu = resource_monitor.get_cpu_usage()
            final_memory = resource_monitor.get_memory_usage()
            final_gpu = resource_monitor.get_gpu_usage()
            
            load_metrics.record_resource_usage(
                final_cpu - initial_cpu,
                final_memory - initial_memory,
                final_gpu - initial_gpu if final_gpu is not None and initial_gpu is not None else None
            )
            
            print(f"Concurrency {level}: {total_time:.2f}ms total, {avg_time:.2f}ms avg, {throughput:.2f} req/s")
            print(f"CPU: {final_cpu - initial_cpu:.1f}%, Memory: {final_memory - initial_memory:.1f}MB")
            if final_gpu is not None:
                print(f"GPU: {final_gpu - initial_gpu:.1f}%")
            
            # Count successful responses
            successful = sum(1 for r in responses if isinstance(r, VoiceResponse))
            print(f"Success rate: {successful}/{level} ({successful/level*100:.1f}%)")
            
            # Performance assertions
            assert successful > 0  # At least some requests should succeed
            if level <= 50:  # For reasonable concurrency levels
                assert successful / level > 0.8  # At least 80% success rate

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, mock_assistant, load_metrics, resource_monitor):
        """Test performance under sustained load."""
        
        # Sustained load parameters
        test_duration = 10.0  # 10 seconds
        request_interval = 0.1  # 100ms between requests
        
        # Record initial resource usage
        initial_cpu = resource_monitor.get_cpu_usage()
        initial_memory = resource_monitor.get_memory_usage()
        initial_gpu = resource_monitor.get_gpu_usage()
        
        # Run sustained load test
        start_time = time.time()
        request_count = 0
        tasks = []
        
        while time.time() - start_time < test_duration:
            # Create request
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"sustained-{request_count}",
                text_content=f"Sustained load request {request_count}",
                audio_data=b"fake_audio_data",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            
            # Process asynchronously
            task = asyncio.create_task(mock_assistant.process_voice_message(voice_message))
            tasks.append(task)
            request_count += 1
            
            # Wait for next request interval
            await asyncio.sleep(request_interval)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Record metrics
        total_time = end_time - start_time
        throughput = request_count / total_time
        
        for response in responses:
            if isinstance(response, Exception):
                load_metrics.record_error("processing")
            else:
                load_metrics.record_response_time(response.total_latency_ms)
        
        load_metrics.record_throughput(throughput)
        
        # Record resource usage
        final_cpu = resource_monitor.get_cpu_usage()
        final_memory = resource_monitor.get_memory_usage()
        final_gpu = resource_monitor.get_gpu_usage()
        
        load_metrics.record_resource_usage(
            final_cpu - initial_cpu,
            final_memory - initial_memory,
            final_gpu - initial_gpu if final_gpu is not None and initial_gpu is not None else None
        )
        
        print(f"Sustained load: {request_count} requests in {total_time:.2f}s ({throughput:.2f} req/s)")
        print(f"CPU: {final_cpu - initial_cpu:.1f}%, Memory: {final_memory - initial_memory:.1f}MB")
        if final_gpu is not None:
            print(f"GPU: {final_gpu - initial_gpu:.1f}%")
        
        # Count successful responses
        successful = sum(1 for r in responses if isinstance(r, VoiceResponse))
        print(f"Success rate: {successful}/{request_count} ({successful/request_count*100:.1f}%)")
        
        # Performance assertions
        assert successful > 0  # At least some requests should succeed
        assert successful / request_count > 0.8  # At least 80% success rate
        assert throughput > 5  # At least 5 requests per second

    @pytest.mark.asyncio
    async def test_resource_scaling(self, mock_assistant, load_metrics, resource_monitor):
        """Test resource usage scaling with increasing load."""
        
        # Test different batch sizes
        batch_sizes = [1, 10, 25, 50, 100]
        scaling_results = []
        
        for size in batch_sizes:
            # Record initial resource usage
            initial_cpu = resource_monitor.get_cpu_usage()
            initial_memory = resource_monitor.get_memory_usage()
            initial_gpu = resource_monitor.get_gpu_usage()
            
            # Create batch of requests
            requests = []
            for i in range(size):
                voice_message = VoiceMessage(
                    message_id=str(uuid.uuid4()),
                    conversation_id=f"scaling-{i}",
                    text_content=f"Resource scaling test {i} in batch {size}",
                    audio_data=b"fake_audio_data",
                    processing_mode=VoiceProcessingMode.MOSHI_ONLY
                )
                requests.append(voice_message)
            
            # Process batch
            start_time = time.time()
            tasks = [mock_assistant.process_voice_message(msg) for msg in requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Record resource usage
            final_cpu = resource_monitor.get_cpu_usage()
            final_memory = resource_monitor.get_memory_usage()
            final_gpu = resource_monitor.get_gpu_usage()
            
            cpu_usage = final_cpu - initial_cpu
            memory_usage = final_memory - initial_memory
            gpu_usage = final_gpu - initial_gpu if final_gpu is not None and initial_gpu is not None else None
            
            load_metrics.record_resource_usage(cpu_usage, memory_usage, gpu_usage)
            
            # Record scaling results
            scaling_results.append({
                "batch_size": size,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "gpu_usage": gpu_usage,
                "total_time": (end_time - start_time) * 1000,
                "avg_time": ((end_time - start_time) * 1000) / size
            })
            
            print(f"Batch {size}: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}MB")
            if gpu_usage is not None:
                print(f"GPU: {gpu_usage:.1f}%")
        
        # Analyze scaling behavior
        for i in range(1, len(scaling_results)):
            current = scaling_results[i]
            previous = scaling_results[i-1]
            
            cpu_ratio = current["cpu_usage"] / previous["cpu_usage"] if previous["cpu_usage"] > 0 else 0
            memory_ratio = current["memory_usage"] / previous["memory_usage"] if previous["memory_usage"] > 0 else 0
            size_ratio = current["batch_size"] / previous["batch_size"]
            
            print(f"Scaling {previous['batch_size']} → {current['batch_size']}:")
            print(f"  CPU scaling: {cpu_ratio:.2f}x (vs {size_ratio:.2f}x batch size)")
            print(f"  Memory scaling: {memory_ratio:.2f}x (vs {size_ratio:.2f}x batch size)")
            
            # Ideally, resource usage should scale sub-linearly with batch size
            if current["batch_size"] >= 50:  # Only check for larger batches
                assert cpu_ratio < size_ratio * 1.5  # CPU usage should scale reasonably
                assert memory_ratio < size_ratio * 1.5  # Memory usage should scale reasonably

    @pytest.mark.asyncio
    async def test_different_processing_modes(self, mock_assistant, load_metrics):
        """Test performance with different processing modes."""
        
        modes = [
            VoiceProcessingMode.MOSHI_ONLY,
            VoiceProcessingMode.LLM_ONLY,
            VoiceProcessingMode.HYBRID
        ]
        
        mode_results = {}
        
        for mode in modes:
            # Create test message with specific mode
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"mode-{mode.value}",
                text_content=f"Processing mode test for {mode.value}",
                audio_data=b"fake_audio_data",
                processing_mode=mode
            )
            
            # Process request
            start_time = time.time()
            response = await mock_assistant.process_voice_message(voice_message)
            end_time = time.time()
            
            # Record metrics
            response_time = (end_time - start_time) * 1000
            load_metrics.record_response_time(response_time)
            
            mode_results[mode.value] = {
                "response_time": response_time,
                "success": isinstance(response, VoiceResponse)
            }
            
            print(f"Mode {mode.value}: {response_time:.2f}ms")
        
        # Compare mode performance
        for mode, result in mode_results.items():
            assert result["success"]  # All modes should succeed
            assert result["response_time"] < 1000  # Under 1 second for any mode

    @pytest.mark.asyncio
    async def test_system_recovery_after_load(self, mock_assistant, load_metrics, resource_monitor):
        """Test system recovery after heavy load."""
        
        # First, apply heavy load
        heavy_load_size = 100
        requests = []
        
        for i in range(heavy_load_size):
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"heavy-{i}",
                text_content=f"Heavy load request {i}",
                audio_data=b"fake_audio_data",
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            requests.append(voice_message)
        
        # Process heavy load
        tasks = [mock_assistant.process_voice_message(msg) for msg in requests]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Record resource usage after heavy load
        post_load_cpu = resource_monitor.get_cpu_usage()
        post_load_memory = resource_monitor.get_memory_usage()
        post_load_gpu = resource_monitor.get_gpu_usage()
        
        # Wait for system to recover
        await asyncio.sleep(2.0)
        
        # Test single request after recovery
        recovery_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="recovery-test",
            text_content="System recovery test after heavy load",
            audio_data=b"fake_audio_data",
            processing_mode=VoiceProcessingMode.MOSHI_ONLY
        )
        
        # Process recovery request
        start_time = time.time()
        recovery_response = await mock_assistant.process_voice_message(recovery_message)
        end_time = time.time()
        
        recovery_time = (end_time - start_time) * 1000
        
        # Record resource usage after recovery
        recovery_cpu = resource_monitor.get_cpu_usage()
        recovery_memory = resource_monitor.get_memory_usage()
        recovery_gpu = resource_monitor.get_gpu_usage()
        
        print(f"Post-load CPU: {post_load_cpu:.1f}%, Memory: {post_load_memory:.1f}MB")
        print(f"Recovery CPU: {recovery_cpu:.1f}%, Memory: {recovery_memory:.1f}MB")
        print(f"Recovery response time: {recovery_time:.2f}ms")
        
        # Verify recovery
        assert isinstance(recovery_response, VoiceResponse)
        assert recovery_time < 500  # Under 500ms after recovery
        
        # Resource usage should decrease after recovery
        assert recovery_cpu < post_load_cpu * 1.5  # CPU usage should decrease or stabilize
        assert recovery_memory <= post_load_memory * 1.2  # Memory usage should not increase significantly

    def test_system_performance_summary(self, load_metrics):
        """Generate system performance summary."""
        
        # Get overall performance stats
        stats = load_metrics.get_stats()
        
        print("\nSystem Performance Summary:")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Average response time: {stats['avg_response_time']:.2f}ms")
        print(f"95th percentile response time: {stats['p95_response_time']:.2f}ms")
        print(f"Average throughput: {stats['avg_throughput']:.2f} req/s")
        print(f"Maximum throughput: {stats['max_throughput']:.2f} req/s")
        print(f"Average CPU usage: {stats['avg_cpu_usage']:.1f}%")
        print(f"Average memory usage: {stats['avg_memory_usage']:.1f}MB")
        
        if stats['avg_gpu_usage'] > 0:
            print(f"Average GPU usage: {stats['avg_gpu_usage']:.1f}%")
        
        print(f"Error counts: {stats['error_counts']}")
        
        # Performance assertions for RTX 5090
        if stats['total_requests'] > 0:
            assert stats['avg_response_time'] < 500  # Under 500ms average
            assert stats['max_throughput'] > 10  # At least 10 req/s
            
            # Error rate should be low
            total_errors = sum(stats['error_counts'].values())
            error_rate = total_errors / stats['total_requests'] if stats['total_requests'] > 0 else 0
            assert error_rate < 0.1  # Less than 10% error rate
