"""
Test suite for Phase 1: Core Infrastructure of the voice processing system.

This test suite verifies all components implemented in Phase 1:
- Environment setup and dependencies
- Base audio processor and VAD
- Configuration system
- Audio pipeline
- VRAM management
"""

import pytest
import asyncio
import torch
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.coda.components.voice.models import (
    VoiceConfig,
    AudioConfig,
    MoshiConfig,
    ExternalLLMConfig,
    AudioFormat,
    VoiceProcessingMode
)
from src.coda.components.voice.config import (
    ConfigurationManager,
    ConfigurationTemplate,
    load_voice_config
)
from src.coda.components.voice.audio_processor import (
    AudioProcessor,
    VoiceActivityDetector
)
from src.coda.components.voice.pipeline import (
    AudioPipeline,
    PipelineManager,
    AudioBuffer,
    PipelineState
)
from src.coda.components.voice.vram_manager import (
    DynamicVRAMManager,
    VRAMMonitor,
    MemoryPressure
)
from src.coda.components.voice.utils import (
    PerformanceMonitor,
    VRAMManager,
    AudioUtils,
    LatencyTracker
)


class TestEnvironmentSetup:
    """Test environment setup and dependencies."""
    
    def test_pytorch_available(self):
        """Test that PyTorch is available."""
        assert torch.__version__ is not None
        assert len(torch.__version__) > 0
    
    def test_cuda_detection(self):
        """Test CUDA detection (should work even with warnings)."""
        # CUDA should be detected even if there are compatibility warnings
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            assert torch.cuda.device_count() > 0
            device_name = torch.cuda.get_device_name(0)
            assert len(device_name) > 0
    
    def test_audio_dependencies(self):
        """Test audio processing dependencies."""
        # Test numpy
        test_array = np.array([1, 2, 3, 4])
        assert len(test_array) == 4
        
        # Test basic audio operations
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)
        assert len(audio_data) == samples


class TestAudioProcessor:
    """Test audio processor and VAD functionality."""
    
    @pytest.fixture
    async def audio_config(self):
        """Create test audio configuration."""
        return AudioConfig(
            sample_rate=16000,
            channels=1,
            vad_enabled=True,
            vad_threshold=0.5,
            noise_reduction=True,
            echo_cancellation=True,
            auto_gain_control=True
        )
    
    @pytest.fixture
    async def audio_processor(self, audio_config):
        """Create and initialize audio processor."""
        processor = AudioProcessor()
        await processor.initialize(audio_config)
        return processor
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        # Generate sine wave
        t = np.linspace(0, duration, samples)
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    async def test_audio_processor_initialization(self, audio_config):
        """Test audio processor initialization."""
        processor = AudioProcessor()
        await processor.initialize(audio_config)
        
        assert processor.config == audio_config
        assert processor.sample_rate == audio_config.sample_rate
        assert processor.channels == audio_config.channels
    
    async def test_process_input_audio(self, audio_processor, sample_audio):
        """Test input audio processing."""
        processed = await audio_processor.process_input_audio(sample_audio)
        
        assert isinstance(processed, bytes)
        assert len(processed) > 0
        # Should be same length or similar (processing might change length slightly)
        assert abs(len(processed) - len(sample_audio)) <= 100
    
    async def test_process_output_audio(self, audio_processor, sample_audio):
        """Test output audio processing."""
        processed = await audio_processor.process_output_audio(sample_audio)
        
        assert isinstance(processed, bytes)
        assert len(processed) > 0
    
    async def test_voice_activity_detection(self, audio_processor, sample_audio):
        """Test voice activity detection."""
        # Test with actual audio (should detect activity)
        has_activity = await audio_processor.detect_voice_activity(sample_audio)
        assert isinstance(has_activity, bool)
        
        # Test with silence
        silence = b'\x00' * len(sample_audio)
        has_silence = await audio_processor.detect_voice_activity(silence)
        assert isinstance(has_silence, bool)
    
    async def test_extract_features(self, audio_processor, sample_audio):
        """Test audio feature extraction."""
        features = await audio_processor.extract_features(sample_audio)
        
        assert isinstance(features, dict)
        assert 'duration_ms' in features
        assert 'sample_rate' in features
        assert 'amplitude_mean' in features
        assert features['duration_ms'] > 0
        assert features['sample_rate'] == 16000
    
    def test_supported_formats(self, audio_processor):
        """Test supported audio formats."""
        formats = audio_processor.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert 'wav' in formats


class TestVoiceActivityDetector:
    """Test VAD functionality."""
    
    @pytest.fixture
    async def vad(self):
        """Create and initialize VAD."""
        vad = VoiceActivityDetector(threshold=0.5, sample_rate=16000)
        
        # Mock the model loading to avoid downloading
        with patch('torch.hub.load') as mock_load:
            mock_model = Mock()
            mock_model.return_value = torch.tensor([0.8])  # High speech probability
            mock_load.return_value = (mock_model, None)
            
            await vad.initialize({'vad_threshold': 0.5, 'sample_rate': 16000})
        
        return vad
    
    async def test_vad_initialization(self):
        """Test VAD initialization."""
        vad = VoiceActivityDetector()
        
        with patch('torch.hub.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = (mock_model, None)
            
            await vad.initialize({'vad_threshold': 0.5, 'sample_rate': 16000})
            
            assert vad.model is not None
            assert vad.threshold == 0.5
            assert vad.sample_rate == 16000
    
    async def test_detect_activity(self, vad):
        """Test activity detection."""
        # Create sample audio
        audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()
        
        is_speech = await vad.detect_activity(audio_data)
        assert isinstance(is_speech, bool)
        assert vad.get_confidence_score() >= 0.0
    
    def test_set_sensitivity(self, vad):
        """Test sensitivity adjustment."""
        vad.set_sensitivity(0.3)
        assert vad.threshold == 0.3
        
        vad.set_sensitivity(1.5)  # Should be clamped to 1.0
        assert vad.threshold == 1.0
        
        vad.set_sensitivity(-0.1)  # Should be clamped to 0.0
        assert vad.threshold == 0.0


class TestConfigurationSystem:
    """Test configuration management."""
    
    def test_configuration_templates(self):
        """Test configuration templates."""
        dev_config = ConfigurationTemplate.development()
        prod_config = ConfigurationTemplate.production()
        light_config = ConfigurationTemplate.lightweight()
        test_config = ConfigurationTemplate.testing()
        
        assert isinstance(dev_config, VoiceConfig)
        assert isinstance(prod_config, VoiceConfig)
        assert isinstance(light_config, VoiceConfig)
        assert isinstance(test_config, VoiceConfig)
        
        # Verify different modes
        assert dev_config.mode == VoiceProcessingMode.MOSHI_ONLY
        assert prod_config.mode == VoiceProcessingMode.HYBRID
        assert light_config.mode == VoiceProcessingMode.TRADITIONAL
        assert test_config.mode == VoiceProcessingMode.MOSHI_ONLY
    
    def test_configuration_manager(self, tmp_path):
        """Test configuration manager."""
        config_manager = ConfigurationManager(config_dir=str(tmp_path))
        
        # Test loading template config
        config = config_manager.get_template_config("development")
        assert isinstance(config, VoiceConfig)
        
        # Test saving and loading config
        config_manager.save_config(config, "test_config")
        loaded_config = config_manager.load_config("test_config")
        
        assert loaded_config.mode == config.mode
        assert loaded_config.audio.sample_rate == config.audio.sample_rate
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigurationManager()
        
        # Test valid config
        valid_config = ConfigurationTemplate.development()
        validation = config_manager.validate_config(valid_config)
        
        assert isinstance(validation, dict)
        assert 'valid' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        
        # Test invalid config (over-allocation)
        invalid_config = ConfigurationTemplate.development()
        invalid_config.total_vram = "8GB"
        invalid_config.moshi.vram_allocation = "6GB"
        invalid_config.external_llm.vram_allocation = "6GB"
        invalid_config.reserved_system = "2GB"  # Total: 14GB > 8GB
        
        validation = config_manager.validate_config(invalid_config)
        assert not validation['valid']
        assert len(validation['errors']) > 0


class TestAudioPipeline:
    """Test audio pipeline functionality."""
    
    @pytest.fixture
    async def audio_config(self):
        """Create test audio configuration."""
        return AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            vad_enabled=False,  # Disable for testing
            noise_reduction=False,
            echo_cancellation=False,
            auto_gain_control=False
        )
    
    @pytest.fixture
    async def audio_pipeline(self, audio_config):
        """Create audio pipeline."""
        pipeline = AudioPipeline(audio_config)
        return pipeline
    
    async def test_pipeline_initialization(self, audio_pipeline):
        """Test pipeline initialization."""
        assert audio_pipeline.state == PipelineState.IDLE
        
        await audio_pipeline.initialize()
        assert audio_pipeline.input_stream is not None
        assert audio_pipeline.output_stream is not None
    
    async def test_pipeline_start_stop(self, audio_pipeline):
        """Test pipeline start and stop."""
        await audio_pipeline.start()
        assert audio_pipeline.state == PipelineState.RUNNING
        
        await audio_pipeline.stop()
        assert audio_pipeline.state == PipelineState.IDLE
    
    async def test_audio_processing(self, audio_pipeline):
        """Test audio processing through pipeline."""
        await audio_pipeline.start()
        
        # Create sample audio
        sample_audio = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()
        
        # Process input
        chunk = await audio_pipeline.process_input(sample_audio)
        assert chunk.data is not None
        assert chunk.sequence_number >= 0
        
        # Queue output
        success = await audio_pipeline.queue_output(sample_audio)
        assert success
        
        await audio_pipeline.stop()
    
    def test_audio_buffer(self):
        """Test audio buffer functionality."""
        buffer = AudioBuffer(max_size=5)
        
        # Create test chunk
        chunk = Mock()
        chunk.chunk_id = "test_chunk"
        
        # Test put and get
        assert buffer.put(chunk, block=False)
        assert buffer.size() == 1
        
        retrieved = buffer.get(block=False)
        assert retrieved == chunk
        assert buffer.size() == 0
        
        # Test buffer full
        for i in range(6):  # More than max_size
            test_chunk = Mock()
            test_chunk.chunk_id = f"chunk_{i}"
            buffer.put(test_chunk, block=False)
        
        stats = buffer.get_stats()
        assert stats['dropped_chunks'] > 0


class TestVRAMManagement:
    """Test VRAM management functionality."""
    
    @pytest.fixture
    def voice_config(self):
        """Create test voice configuration."""
        return ConfigurationTemplate.testing()
    
    @pytest.fixture
    def vram_monitor(self):
        """Create VRAM monitor."""
        return VRAMMonitor(device_id=0)
    
    @pytest.fixture
    async def vram_manager(self, voice_config):
        """Create VRAM manager."""
        manager = DynamicVRAMManager(voice_config)
        await manager.initialize()
        return manager
    
    def test_vram_monitor_initialization(self, vram_monitor):
        """Test VRAM monitor initialization."""
        assert vram_monitor.device_id == 0
        assert vram_monitor.total_vram_mb >= 0
    
    def test_vram_allocation_info(self, vram_monitor):
        """Test VRAM allocation info."""
        allocation = vram_monitor.get_current_allocation()
        
        assert hasattr(allocation, 'total_mb')
        assert hasattr(allocation, 'allocated_mb')
        assert hasattr(allocation, 'free_mb')
        assert allocation.total_mb >= 0
        assert allocation.allocated_mb >= 0
    
    async def test_component_registration(self, vram_manager):
        """Test component registration."""
        # Register component
        success = vram_manager.register_component(
            component_id="test_component",
            max_mb=1024.0,
            priority=5,
            can_resize=True
        )
        assert success
        
        # Try to register same component again
        success = vram_manager.register_component(
            component_id="test_component",
            max_mb=512.0,
            priority=3
        )
        assert not success  # Should fail
    
    async def test_memory_allocation(self, vram_manager):
        """Test memory allocation and deallocation."""
        # Register component
        vram_manager.register_component("test_comp", max_mb=1024.0, priority=5)
        
        # Allocate memory
        success = vram_manager.allocate("test_comp", 512.0)
        assert success
        
        # Check allocation
        summary = vram_manager.get_allocation_summary()
        assert summary['components']['test_comp']['allocated_mb'] == 512.0
        
        # Deallocate memory
        success = vram_manager.deallocate("test_comp", 256.0)
        assert success
        
        # Check remaining allocation
        summary = vram_manager.get_allocation_summary()
        assert summary['components']['test_comp']['allocated_mb'] == 256.0
    
    async def test_memory_pressure_detection(self, vram_manager):
        """Test memory pressure detection."""
        pressure = vram_manager.get_memory_pressure()
        assert isinstance(pressure, MemoryPressure)
    
    async def test_memory_optimization(self, vram_manager):
        """Test memory optimization."""
        results = vram_manager.optimize_memory()
        
        assert isinstance(results, dict)
        assert 'freed_mb' in results
        assert 'actions_taken' in results
        assert 'pressure_before' in results


class TestUtilities:
    """Test utility functions and classes."""
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        monitor = PerformanceMonitor(window_size=10)
        
        # Record some chunks
        for i in range(5):
            monitor.record_chunk_processed()
        
        metrics = monitor.get_current_metrics()
        assert metrics.throughput_chunks_per_sec >= 0
        assert metrics.cpu_usage >= 0
    
    def test_latency_tracker(self):
        """Test latency tracking."""
        tracker = LatencyTracker("test_operation")
        
        # Simulate operation
        tracker.start()
        import time
        time.sleep(0.01)  # 10ms
        latency = tracker.stop()
        
        assert latency > 0
        assert latency < 100  # Should be less than 100ms
        
        stats = tracker.get_stats()
        assert stats['count'] == 1
        assert stats['avg'] > 0
    
    def test_audio_utils(self):
        """Test audio utility functions."""
        # Test format validation
        assert AudioUtils.validate_audio_format("wav")
        assert AudioUtils.validate_audio_format("mp3")
        assert not AudioUtils.validate_audio_format("invalid")
        
        # Test duration calculation
        duration = AudioUtils.calculate_audio_duration(
            audio_data=b'\x00' * 32000,  # 32000 bytes
            sample_rate=16000,
            channels=1,
            bit_depth=16
        )
        assert abs(duration - 1.0) < 0.01  # Should be ~1 second
        
        # Test size calculation
        size = AudioUtils.calculate_audio_size(
            duration_seconds=1.0,
            sample_rate=16000,
            channels=1,
            bit_depth=16
        )
        assert size == 32000  # 16000 samples * 1 channel * 2 bytes


@pytest.mark.asyncio
async def test_phase1_integration():
    """Integration test for Phase 1 components."""
    # Create configuration
    config = ConfigurationTemplate.testing()
    
    # Test configuration loading
    config_manager = ConfigurationManager()
    validation = config_manager.validate_config(config)
    assert validation['valid']
    
    # Test audio pipeline
    pipeline = AudioPipeline(config.audio)
    await pipeline.initialize()
    await pipeline.start()
    
    # Test audio processing
    sample_audio = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()
    chunk = await pipeline.process_input(sample_audio)
    assert chunk is not None
    
    # Test VRAM management
    vram_manager = DynamicVRAMManager(config)
    await vram_manager.initialize()
    
    success = vram_manager.register_component("pipeline", max_mb=512.0, priority=5)
    assert success
    
    success = vram_manager.allocate("pipeline", 256.0)
    assert success
    
    # Cleanup
    await pipeline.stop()
    await vram_manager.cleanup()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
