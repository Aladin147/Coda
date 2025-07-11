"""
Comprehensive tests for AudioProcessor to increase coverage from 14% to 80%+.
Targets specific uncovered lines in audio_processor.py.
"""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
import torch
import torchaudio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from src.coda.components.voice.audio_processor import AudioProcessor
from src.coda.components.voice.models import AudioConfig, AudioFormat, VoiceConfig
from src.coda.components.voice.exceptions import AudioProcessingError, ValidationError


class TestAudioProcessorComprehensive:
    """Comprehensive tests for AudioProcessor covering all major functionality."""

    @pytest_asyncio.fixture
    async def audio_config(self):
        """Create test audio configuration."""
        return AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            format=AudioFormat.WAV,
            noise_reduction=True,
            echo_cancellation=True,
            vad_enabled=True
        )

    @pytest_asyncio.fixture
    async def voice_config(self, audio_config):
        """Create test voice configuration."""
        return VoiceConfig(
            audio=audio_config,
            provider="moshi",
            enabled=True
        )

    @pytest_asyncio.fixture
    async def audio_processor(self, audio_config):
        """Create AudioProcessor instance."""
        processor = AudioProcessor(audio_config)
        await processor.initialize()
        return processor

    @pytest_asyncio.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing."""
        # Generate 1 second of sine wave at 440Hz
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        return audio

    def test_audio_processor_initialization(self, audio_config):
        """Test AudioProcessor initialization with different configurations."""
        # Test basic initialization
        processor = AudioProcessor(audio_config)
        assert processor.config == audio_config
        assert not processor.is_initialized
        
        # Test initialization with noise reduction disabled
        config_no_noise = AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            format=AudioFormat.WAV,
            noise_reduction=False,
            echo_cancellation=False,
            vad_enabled=False
        )
        processor_no_noise = AudioProcessor(config_no_noise)
        assert processor_no_noise.config.noise_reduction is False

    @pytest.mark.asyncio
    async def test_audio_processor_initialize(self, audio_config):
        """Test AudioProcessor initialization process."""
        processor = AudioProcessor(audio_config)
        
        # Test successful initialization
        await processor.initialize()
        assert processor.is_initialized
        
        # Test double initialization (should not raise error)
        await processor.initialize()
        assert processor.is_initialized

    @pytest.mark.asyncio
    async def test_audio_processor_cleanup(self, audio_processor):
        """Test AudioProcessor cleanup process."""
        assert audio_processor.is_initialized
        
        await audio_processor.cleanup()
        assert not audio_processor.is_initialized
        
        # Test double cleanup (should not raise error)
        await audio_processor.cleanup()
        assert not audio_processor.is_initialized

    @pytest.mark.asyncio
    async def test_process_input_audio_basic(self, audio_processor, sample_audio_data):
        """Test basic input audio processing."""
        result = await audio_processor.process_input_audio(sample_audio_data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # Single channel
        assert result.dtype == torch.float32

    @pytest.mark.asyncio
    async def test_process_input_audio_with_enhancements(self, audio_config, sample_audio_data):
        """Test input audio processing with all enhancements enabled."""
        # Enable all enhancements
        audio_config.noise_reduction = True
        audio_config.echo_cancellation = True
        audio_config.vad_enabled = True
        
        processor = AudioProcessor(audio_config)
        await processor.initialize()
        
        with patch.object(processor, '_apply_noise_reduction', return_value=sample_audio_data) as mock_noise:
            with patch.object(processor, '_apply_echo_cancellation', return_value=sample_audio_data) as mock_echo:
                with patch.object(processor, '_detect_voice_activity', return_value=True) as mock_vad:
                    result = await processor.process_input_audio(sample_audio_data)
                    
                    mock_noise.assert_called_once()
                    mock_echo.assert_called_once()
                    mock_vad.assert_called_once()
                    assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_process_output_audio_basic(self, audio_processor, sample_audio_data):
        """Test basic output audio processing."""
        result = await audio_processor.process_output_audio(sample_audio_data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # Single channel
        assert result.dtype == torch.float32

    @pytest.mark.asyncio
    async def test_process_output_audio_with_enhancements(self, audio_config, sample_audio_data):
        """Test output audio processing with enhancements."""
        audio_config.noise_reduction = True
        
        processor = AudioProcessor(audio_config)
        await processor.initialize()
        
        with patch.object(processor, '_apply_output_enhancement', return_value=sample_audio_data) as mock_enhance:
            result = await processor.process_output_audio(sample_audio_data)
            
            mock_enhance.assert_called_once()
            assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_extract_features(self, audio_processor, sample_audio_data):
        """Test audio feature extraction."""
        features = await audio_processor.extract_features(sample_audio_data)
        
        assert isinstance(features, dict)
        assert 'mfcc' in features
        assert 'spectral_centroid' in features
        assert 'zero_crossing_rate' in features
        assert 'energy' in features

    @pytest.mark.asyncio
    async def test_detect_voice_activity(self, audio_processor, sample_audio_data):
        """Test voice activity detection."""
        # Test with voice-like signal
        is_voice = await audio_processor.detect_voice_activity(sample_audio_data)
        assert isinstance(is_voice, bool)
        
        # Test with silence
        silence = torch.zeros_like(sample_audio_data)
        is_silence = await audio_processor.detect_voice_activity(silence)
        assert isinstance(is_silence, bool)

    @pytest.mark.asyncio
    async def test_convert_audio_format(self, audio_processor, sample_audio_data):
        """Test audio format conversion."""
        # Test WAV to MP3 conversion
        converted = await audio_processor.convert_audio_format(
            sample_audio_data, AudioFormat.WAV, AudioFormat.MP3
        )
        assert isinstance(converted, bytes)
        
        # Test same format (should return original)
        same_format = await audio_processor.convert_audio_format(
            sample_audio_data, AudioFormat.WAV, AudioFormat.WAV
        )
        assert torch.equal(same_format, sample_audio_data)

    @pytest.mark.asyncio
    async def test_resample_audio(self, audio_processor, sample_audio_data):
        """Test audio resampling."""
        # Test upsampling
        upsampled = await audio_processor.resample_audio(sample_audio_data, 16000, 32000)
        assert upsampled.shape[-1] == sample_audio_data.shape[-1] * 2
        
        # Test downsampling
        downsampled = await audio_processor.resample_audio(sample_audio_data, 16000, 8000)
        assert downsampled.shape[-1] == sample_audio_data.shape[-1] // 2

    @pytest.mark.asyncio
    async def test_normalize_audio(self, audio_processor, sample_audio_data):
        """Test audio normalization."""
        # Create audio with different amplitude
        loud_audio = sample_audio_data * 2.0
        
        normalized = await audio_processor.normalize_audio(loud_audio)
        
        # Check that normalized audio is within expected range
        assert torch.max(torch.abs(normalized)) <= 1.0
        assert torch.max(torch.abs(normalized)) > 0.5  # Should be reasonably loud

    @pytest.mark.asyncio
    async def test_apply_noise_reduction(self, audio_processor, sample_audio_data):
        """Test noise reduction functionality."""
        # Add some noise
        noise = torch.randn_like(sample_audio_data) * 0.1
        noisy_audio = sample_audio_data + noise
        
        denoised = await audio_processor._apply_noise_reduction(noisy_audio)
        
        assert isinstance(denoised, torch.Tensor)
        assert denoised.shape == noisy_audio.shape

    @pytest.mark.asyncio
    async def test_apply_echo_cancellation(self, audio_processor, sample_audio_data):
        """Test echo cancellation functionality."""
        # Create echo by adding delayed version
        delay_samples = 800  # 50ms at 16kHz
        echo = torch.cat([torch.zeros(1, delay_samples), sample_audio_data * 0.3], dim=1)
        echo_audio = sample_audio_data + echo[:, :sample_audio_data.shape[1]]
        
        processed = await audio_processor._apply_echo_cancellation(echo_audio)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == echo_audio.shape

    @pytest.mark.asyncio
    async def test_error_handling_invalid_input(self, audio_processor):
        """Test error handling with invalid inputs."""
        # Test with None input
        with pytest.raises(AudioProcessingError):
            await audio_processor.process_input_audio(None)
        
        # Test with wrong tensor shape
        invalid_audio = torch.randn(3, 3, 1000)  # 3D tensor
        with pytest.raises(AudioProcessingError):
            await audio_processor.process_input_audio(invalid_audio)
        
        # Test with empty tensor
        empty_audio = torch.empty(1, 0)
        with pytest.raises(AudioProcessingError):
            await audio_processor.process_input_audio(empty_audio)

    @pytest.mark.asyncio
    async def test_error_handling_uninitialized(self, audio_config, sample_audio_data):
        """Test error handling when processor is not initialized."""
        processor = AudioProcessor(audio_config)
        
        with pytest.raises(AudioProcessingError, match="not initialized"):
            await processor.process_input_audio(sample_audio_data)

    @pytest.mark.asyncio
    async def test_different_audio_formats(self, audio_processor):
        """Test processing with different audio formats."""
        formats_to_test = [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FLAC]
        
        for audio_format in formats_to_test:
            # Create sample data for each format
            sample_data = torch.randn(1, 16000)  # 1 second at 16kHz
            
            # Test processing (should handle format internally)
            result = await audio_processor.process_input_audio(sample_data)
            assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_batch_processing(self, audio_processor):
        """Test batch audio processing."""
        # Create batch of audio samples
        batch_size = 4
        batch_audio = torch.randn(batch_size, 16000)
        
        results = []
        for i in range(batch_size):
            result = await audio_processor.process_input_audio(batch_audio[i:i+1])
            results.append(result)
        
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, torch.Tensor)

    def test_get_supported_formats(self, audio_processor):
        """Test getting supported audio formats."""
        formats = audio_processor.get_supported_formats()
        
        assert isinstance(formats, list)
        assert AudioFormat.WAV in formats
        assert len(formats) > 0

    @pytest.mark.asyncio
    async def test_performance_metrics(self, audio_processor, sample_audio_data):
        """Test performance metrics collection."""
        # Process audio and check if metrics are collected
        await audio_processor.process_input_audio(sample_audio_data)
        
        metrics = audio_processor.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'processing_time_ms' in metrics
        assert 'samples_processed' in metrics

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, audio_processor):
        """Test concurrent audio processing."""
        # Create multiple audio samples
        samples = [torch.randn(1, 16000) for _ in range(3)]
        
        # Process concurrently
        tasks = [audio_processor.process_input_audio(sample) for sample in samples]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, torch.Tensor)
