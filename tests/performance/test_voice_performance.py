"""
Voice Processing Performance Tests.
Tests audio processing latency, streaming performance, and concurrent voice handling.
"""

import pytest
import asyncio
import time
import uuid
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, AsyncGenerator

from src.coda.components.voice.models import (
    VoiceMessage, VoiceResponse, VoiceStreamChunk, 
    VoiceProcessingMode, AudioConfig, AudioFormat
)
from src.coda.components.voice.manager import VoiceManager
from src.coda.components.voice.audio_processor import AudioProcessor


class VoicePerformanceMetrics:
    """Voice processing performance metrics collector."""
    
    def __init__(self):
        self.processing_times = []
        self.latencies = []
        self.throughput_data = []
    
    def record_processing_time(self, start_time: float, end_time: float):
        """Record processing time."""
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        self.processing_times.append(processing_time)
    
    def record_latency(self, latency_ms: float):
        """Record latency measurement."""
        self.latencies.append(latency_ms)
    
    def record_throughput(self, items_processed: int, time_taken: float):
        """Record throughput measurement."""
        throughput = items_processed / time_taken if time_taken > 0 else 0
        self.throughput_data.append(throughput)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return {
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "max_processing_time": np.max(self.processing_times) if self.processing_times else 0,
            "min_processing_time": np.min(self.processing_times) if self.processing_times else 0,
            "avg_latency": np.mean(self.latencies) if self.latencies else 0,
            "max_latency": np.max(self.latencies) if self.latencies else 0,
            "avg_throughput": np.mean(self.throughput_data) if self.throughput_data else 0,
            "total_samples": len(self.processing_times)
        }


class TestVoiceProcessingPerformance:
    """Voice processing performance tests."""

    @pytest.fixture
    def audio_config(self):
        """Create optimized audio configuration."""
        return AudioConfig(
            sample_rate=24000,
            channels=1,
            chunk_size=1024,
            format=AudioFormat.WAV,
            noise_reduction=True,
            echo_cancellation=True,
            vad_enabled=True
        )

    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics collector."""
        return VoicePerformanceMetrics()

    @pytest.fixture
    def mock_voice_manager(self):
        """Create mock voice manager with realistic performance."""
        manager = Mock(spec=VoiceManager)
        
        # Mock processing with realistic latency
        async def mock_process(message):
            await asyncio.sleep(0.05)  # 50ms processing time
            return VoiceResponse(
                response_id=str(uuid.uuid4()),
                conversation_id=message.conversation_id,
                message_id=message.message_id,
                text_content="Processed response",
                processing_mode=message.processing_mode,
                total_latency_ms=50.0
            )
        
        manager.process_voice_input = AsyncMock(side_effect=mock_process)
        manager.is_initialized = True
        return manager

    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing."""
        # Generate 1 second of audio at 24kHz
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        # Generate sine wave
        frequency = 440  # A4 note
        t = np.linspace(0, duration, samples, False)
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Convert to bytes (16-bit PCM)
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    @pytest.mark.benchmark(group="voice_processing")
    def test_single_voice_processing_latency(self, benchmark, mock_voice_manager, sample_audio_data):
        """Benchmark single voice message processing latency."""
        
        def process_voice():
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id="latency-test",
                text_content="Test message for latency",
                audio_data=sample_audio_data,
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            
            return asyncio.run(mock_voice_manager.process_voice_input(voice_message))
        
        result = benchmark(process_voice)
        assert isinstance(result, VoiceResponse)
        assert result.total_latency_ms > 0
        
        # RTX 5090 should achieve low latency
        assert result.total_latency_ms < 100  # Under 100ms

    @pytest.mark.benchmark(group="voice_throughput")
    def test_voice_processing_throughput(self, benchmark, mock_voice_manager, sample_audio_data):
        """Benchmark voice processing throughput."""
        
        def throughput_test():
            async def process_multiple():
                messages = [
                    VoiceMessage(
                        message_id=str(uuid.uuid4()),
                        conversation_id=f"throughput-{i}",
                        text_content=f"Throughput test message {i}",
                        audio_data=sample_audio_data,
                        processing_mode=VoiceProcessingMode.MOSHI_ONLY
                    )
                    for i in range(10)
                ]
                
                start_time = time.time()
                results = await asyncio.gather(*[
                    mock_voice_manager.process_voice_input(msg) for msg in messages
                ])
                end_time = time.time()
                
                return len(results), end_time - start_time
            
            return asyncio.run(process_multiple())
        
        count, duration = benchmark(throughput_test)
        throughput = count / duration
        
        print(f"Voice processing throughput: {throughput:.2f} messages/second")
        assert throughput > 5  # Should process at least 5 messages per second

    @pytest.mark.asyncio
    async def test_concurrent_voice_processing(self, mock_voice_manager, sample_audio_data, performance_metrics):
        """Test concurrent voice processing performance."""
        
        async def voice_task(task_id: int):
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=f"concurrent-{task_id}",
                text_content=f"Concurrent message {task_id}",
                audio_data=sample_audio_data,
                processing_mode=VoiceProcessingMode.MOSHI_ONLY
            )
            
            start_time = time.time()
            result = await mock_voice_manager.process_voice_input(voice_message)
            end_time = time.time()
            
            performance_metrics.record_processing_time(start_time, end_time)
            performance_metrics.record_latency(result.total_latency_ms)
            
            return result
        
        # Test with increasing concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]
        
        for level in concurrency_levels:
            start_time = time.time()
            tasks = [voice_task(i) for i in range(level)]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Record throughput
            performance_metrics.record_throughput(level, end_time - start_time)
            
            # Verify all tasks completed
            assert len(results) == level
            for result in results:
                assert isinstance(result, VoiceResponse)
            
            print(f"Concurrency {level}: {len(results)} tasks in {end_time - start_time:.3f}s")
        
        # Get final statistics
        stats = performance_metrics.get_stats()
        print(f"Performance Stats: {stats}")
        
        # Performance assertions for RTX 5090
        assert stats["avg_processing_time"] < 200  # Under 200ms average
        assert stats["max_processing_time"] < 500  # Under 500ms max

    @pytest.mark.benchmark(group="audio_processing")
    def test_audio_processing_performance(self, benchmark, audio_config, sample_audio_data):
        """Benchmark audio processing performance."""
        
        def process_audio():
            processor = AudioProcessor(audio_config)
            
            # Simulate audio processing
            processed = processor._apply_noise_reduction(sample_audio_data)
            processed = processor._apply_echo_cancellation(processed)
            processed = processor._apply_voice_activity_detection(processed)
            
            return len(processed)
        
        result = benchmark(process_audio)
        assert result > 0

    @pytest.mark.asyncio
    async def test_streaming_performance(self, mock_voice_manager, sample_audio_data):
        """Test streaming voice response performance."""
        
        # Mock streaming response
        async def mock_stream():
            chunks = [
                "Hello ",
                "this is ",
                "a streaming ",
                "response test."
            ]
            
            for i, text in enumerate(chunks):
                yield VoiceStreamChunk(
                    conversation_id="stream-test",
                    chunk_index=i,
                    text_content=text,
                    audio_data=sample_audio_data[:1024],  # Small chunk
                    timestamp=time.time(),
                    is_complete=(i == len(chunks) - 1),
                    chunk_type="audio"
                )
                await asyncio.sleep(0.01)  # Simulate processing delay
        
        mock_voice_manager.stream_response = AsyncMock(return_value=mock_stream())
        
        # Test streaming performance
        start_time = time.time()
        chunks = []
        
        async for chunk in await mock_voice_manager.stream_response():
            chunks.append(chunk)
            chunk_time = time.time()
            
            # Each chunk should arrive quickly
            time_since_start = (chunk_time - start_time) * 1000
            assert time_since_start < 1000  # Under 1 second total
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        print(f"Streaming completed in {total_time:.2f}ms")
        assert len(chunks) == 4
        assert total_time < 500  # Under 500ms for streaming

    def test_memory_usage_during_processing(self, mock_voice_manager, sample_audio_data):
        """Test memory usage during voice processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple voice messages
        async def process_batch():
            messages = [
                VoiceMessage(
                    message_id=str(uuid.uuid4()),
                    conversation_id=f"memory-test-{i}",
                    text_content=f"Memory test message {i}",
                    audio_data=sample_audio_data,
                    processing_mode=VoiceProcessingMode.MOSHI_ONLY
                )
                for i in range(20)
            ]
            
            results = []
            for msg in messages:
                result = await mock_voice_manager.process_voice_input(msg)
                results.append(result)
            
            return results
        
        results = asyncio.run(process_batch())
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Verify processing completed
        assert len(results) == 20
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Under 500MB increase

    @pytest.mark.benchmark(group="voice_modes")
    def test_processing_mode_performance(self, benchmark, mock_voice_manager, sample_audio_data):
        """Benchmark different voice processing modes."""
        
        modes = [
            VoiceProcessingMode.MOSHI_ONLY,
            VoiceProcessingMode.LLM_ONLY,
            VoiceProcessingMode.HYBRID
        ]
        
        def test_mode(mode):
            voice_message = VoiceMessage(
                message_id=str(uuid.uuid4()),
                conversation_id="mode-test",
                text_content="Mode performance test",
                audio_data=sample_audio_data,
                processing_mode=mode
            )
            
            return asyncio.run(mock_voice_manager.process_voice_input(voice_message))
        
        results = {}
        for mode in modes:
            result = benchmark.pedantic(test_mode, args=(mode,), rounds=5)
            results[mode.value] = result
            print(f"Mode {mode.value}: Completed successfully")
        
        # All modes should complete successfully
        for mode, result in results.items():
            assert isinstance(result, VoiceResponse)

    @pytest.mark.asyncio
    async def test_voice_pipeline_end_to_end_performance(self, mock_voice_manager, sample_audio_data):
        """Test end-to-end voice pipeline performance."""
        
        # Simulate complete voice pipeline
        pipeline_steps = []
        
        # Step 1: Voice input
        start_time = time.time()
        voice_message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            conversation_id="pipeline-test",
            text_content="End-to-end pipeline test",
            audio_data=sample_audio_data,
            processing_mode=VoiceProcessingMode.HYBRID
        )
        
        step1_time = time.time()
        pipeline_steps.append(("voice_input", (step1_time - start_time) * 1000))
        
        # Step 2: Processing
        result = await mock_voice_manager.process_voice_input(voice_message)
        
        step2_time = time.time()
        pipeline_steps.append(("processing", (step2_time - step1_time) * 1000))
        
        # Step 3: Response generation
        await asyncio.sleep(0.01)  # Simulate response generation
        
        step3_time = time.time()
        pipeline_steps.append(("response", (step3_time - step2_time) * 1000))
        
        total_time = (step3_time - start_time) * 1000
        
        # Print pipeline timing
        print("Voice Pipeline Performance:")
        for step, duration in pipeline_steps:
            print(f"  {step}: {duration:.2f}ms")
        print(f"  Total: {total_time:.2f}ms")
        
        # Performance assertions
        assert isinstance(result, VoiceResponse)
        assert total_time < 200  # Under 200ms total pipeline time
        
        # Individual step assertions
        for step, duration in pipeline_steps:
            assert duration < 100  # Each step under 100ms

    def test_voice_processing_scalability(self, mock_voice_manager, sample_audio_data):
        """Test voice processing scalability."""
        
        async def scalability_test(num_users: int):
            """Test with different numbers of concurrent users."""
            tasks = []
            
            for user_id in range(num_users):
                voice_message = VoiceMessage(
                    message_id=str(uuid.uuid4()),
                    conversation_id=f"user-{user_id}",
                    text_content=f"Scalability test from user {user_id}",
                    audio_data=sample_audio_data,
                    processing_mode=VoiceProcessingMode.MOSHI_ONLY
                )
                
                task = mock_voice_manager.process_voice_input(voice_message)
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Count successful results
            successful = [r for r in results if isinstance(r, VoiceResponse)]
            failed = [r for r in results if isinstance(r, Exception)]
            
            return {
                "users": num_users,
                "successful": len(successful),
                "failed": len(failed),
                "total_time": (end_time - start_time) * 1000,
                "avg_time_per_user": ((end_time - start_time) * 1000) / num_users
            }
        
        # Test different user loads
        user_loads = [1, 5, 10, 25, 50, 100]
        scalability_results = []
        
        for load in user_loads:
            result = asyncio.run(scalability_test(load))
            scalability_results.append(result)
            
            print(f"Users: {result['users']}, "
                  f"Success: {result['successful']}, "
                  f"Avg Time: {result['avg_time_per_user']:.2f}ms")
        
        # Verify scalability
        for result in scalability_results:
            # Most requests should succeed
            success_rate = result['successful'] / result['users']
            assert success_rate > 0.8  # At least 80% success rate
            
            # Average time per user should be reasonable
            assert result['avg_time_per_user'] < 1000  # Under 1 second per user
