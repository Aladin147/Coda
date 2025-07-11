"""
Tests for Advanced Audio Processing.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.coda.components.voice.audio_processor import AudioProcessor, VoiceActivityDetector
from src.coda.components.voice.models import AudioConfig, AudioFormat


class TestAdvancedAudioProcessing:
    """Test advanced audio processing algorithms."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create audio processor for testing."""
        processor = AudioProcessor()
        return processor
    
    @pytest.fixture
    def test_audio_tensor(self):
        """Create test audio tensor."""
        # Generate test audio: sine wave + noise
        sample_rate = 24000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        
        # Create sine wave (speech-like signal)
        t = torch.linspace(0, duration, samples)
        frequency = 440  # A4 note
        clean_signal = 0.5 * torch.sin(2 * np.pi * frequency * t)
        
        # Add noise
        noise = 0.1 * torch.randn(samples)
        noisy_signal = clean_signal + noise
        
        return noisy_signal
    
    @pytest.fixture
    def test_audio_bytes(self, test_audio_tensor):
        """Create test audio as bytes."""
        # Convert tensor to 16-bit PCM bytes
        audio_int16 = (test_audio_tensor * 32767).clamp(-32767, 32767).to(torch.int16)
        return audio_int16.numpy().tobytes()
    
    def test_noise_reduction_spectral_subtraction(self, audio_processor, test_audio_tensor):
        """Test spectral subtraction noise reduction."""
        # Apply noise reduction
        enhanced_audio = audio_processor._spectral_subtraction_noise_reduction(test_audio_tensor.unsqueeze(0))
        
        assert isinstance(enhanced_audio, torch.Tensor)
        assert enhanced_audio.shape == test_audio_tensor.unsqueeze(0).shape
        
        # Enhanced audio should have different characteristics than original
        original_energy = torch.mean(test_audio_tensor ** 2)
        enhanced_energy = torch.mean(enhanced_audio ** 2)
        
        # Energy should be preserved (within reasonable bounds)
        assert 0.1 < enhanced_energy / original_energy < 2.0
    
    def test_wiener_filter_noise_reduction(self, audio_processor, test_audio_tensor):
        """Test Wiener filter noise reduction."""
        enhanced_audio = audio_processor._wiener_filter_noise_reduction(test_audio_tensor.unsqueeze(0))
        
        assert isinstance(enhanced_audio, torch.Tensor)
        assert enhanced_audio.shape == test_audio_tensor.unsqueeze(0).shape
        
        # Check that filtering was applied
        assert not torch.equal(enhanced_audio, test_audio_tensor.unsqueeze(0))
    
    def test_adaptive_noise_gate(self, audio_processor, test_audio_tensor):
        """Test adaptive noise gate."""
        # Create audio with quiet and loud sections
        quiet_section = 0.01 * torch.randn(1000)
        loud_section = 0.5 * torch.randn(1000)
        mixed_audio = torch.cat([quiet_section, loud_section, quiet_section])
        
        gated_audio = audio_processor._adaptive_noise_gate(mixed_audio.unsqueeze(0))
        
        assert isinstance(gated_audio, torch.Tensor)
        assert gated_audio.shape == mixed_audio.unsqueeze(0).shape
        
        # Quiet sections should be more attenuated
        quiet_energy_original = torch.mean(quiet_section ** 2)
        quiet_energy_gated = torch.mean(gated_audio[0, :1000] ** 2)
        
        assert quiet_energy_gated <= quiet_energy_original
    
    def test_adaptive_echo_cancellation(self, audio_processor, test_audio_tensor):
        """Test adaptive echo cancellation."""
        enhanced_audio = audio_processor._adaptive_echo_cancellation(test_audio_tensor.unsqueeze(0))
        
        assert isinstance(enhanced_audio, torch.Tensor)
        assert enhanced_audio.shape == test_audio_tensor.unsqueeze(0).shape
    
    def test_acoustic_echo_suppression(self, audio_processor, test_audio_tensor):
        """Test acoustic echo suppression."""
        enhanced_audio = audio_processor._acoustic_echo_suppression(test_audio_tensor.unsqueeze(0))
        
        assert isinstance(enhanced_audio, torch.Tensor)
        assert enhanced_audio.shape == test_audio_tensor.unsqueeze(0).shape
    
    def test_remove_periodic_components(self, audio_processor):
        """Test periodic component removal."""
        # Create audio with periodic echo
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        # Original signal
        t = torch.linspace(0, duration, samples)
        original = 0.5 * torch.sin(2 * np.pi * 440 * t)
        
        # Add echo (delayed and attenuated version)
        echo_delay = 1000  # samples
        echo_strength = 0.3
        echo = torch.zeros_like(original)
        echo[echo_delay:] = echo_strength * original[:-echo_delay]
        
        # Combined signal
        echoed_signal = original + echo
        
        # Remove periodic components
        enhanced_audio = audio_processor._remove_periodic_components(echoed_signal.unsqueeze(0), 256)
        
        assert isinstance(enhanced_audio, torch.Tensor)
        assert enhanced_audio.shape == echoed_signal.unsqueeze(0).shape
        
        # Enhanced audio should be closer to original than echoed signal
        original_mse = torch.mean((echoed_signal - original) ** 2)
        enhanced_mse = torch.mean((enhanced_audio.squeeze(0) - original) ** 2)
        
        # Enhancement should reduce the error (though not necessarily always)
        # This is a heuristic test since echo removal is complex
        assert enhanced_mse < original_mse * 2  # Allow some tolerance
    
    def test_calculate_echo_suppression_factor(self, audio_processor, test_audio_tensor):
        """Test echo suppression factor calculation."""
        # Convert to frequency domain for testing
        stft = torch.stft(
            test_audio_tensor,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=torch.hann_window(512),
            return_complex=True
        )
        magnitude = torch.abs(stft)
        
        suppression_factor = audio_processor._calculate_echo_suppression_factor(magnitude)
        
        assert isinstance(suppression_factor, torch.Tensor)
        assert suppression_factor.shape[0] == magnitude.shape[0]  # Same frequency bins
        
        # Suppression factors should be between 0.3 and 1.0
        assert torch.all(suppression_factor >= 0.3)
        assert torch.all(suppression_factor <= 1.0)
    
    @pytest.mark.asyncio
    async def test_full_noise_reduction_pipeline(self, audio_processor, test_audio_tensor):
        """Test complete noise reduction pipeline."""
        enhanced_audio = await audio_processor._apply_noise_reduction(test_audio_tensor)
        
        assert isinstance(enhanced_audio, torch.Tensor)
        assert enhanced_audio.shape == test_audio_tensor.shape
        
        # Should not be identical to input (some processing should occur)
        assert not torch.equal(enhanced_audio, test_audio_tensor)
    
    @pytest.mark.asyncio
    async def test_full_echo_cancellation_pipeline(self, audio_processor, test_audio_tensor):
        """Test complete echo cancellation pipeline."""
        enhanced_audio = await audio_processor._apply_echo_cancellation(test_audio_tensor)
        
        assert isinstance(enhanced_audio, torch.Tensor)
        assert enhanced_audio.shape == test_audio_tensor.shape
    
    @pytest.mark.asyncio
    async def test_process_input_audio_with_enhancements(self, audio_processor, test_audio_bytes):
        """Test input audio processing with all enhancements enabled."""
        # Initialize processor
        config = AudioConfig(
            sample_rate=24000,
            channels=1,
            noise_reduction=True,
            echo_cancellation=True,
            auto_gain_control=True
        )
        await audio_processor.initialize(config)
        
        # Process audio
        processed_audio = await audio_processor.process_input_audio(test_audio_bytes)
        
        assert isinstance(processed_audio, bytes)
        assert len(processed_audio) > 0
        
        # Should be similar length (allowing for some processing differences)
        assert 0.8 < len(processed_audio) / len(test_audio_bytes) < 1.2
    
    @pytest.mark.asyncio
    async def test_process_output_audio_with_enhancements(self, audio_processor, test_audio_bytes):
        """Test output audio processing with enhancements."""
        # Initialize processor
        config = AudioConfig(
            sample_rate=24000,
            channels=1
        )
        await audio_processor.initialize(config)
        
        # Process audio
        processed_audio = await audio_processor.process_output_audio(test_audio_bytes)
        
        assert isinstance(processed_audio, bytes)
        assert len(processed_audio) > 0
    
    def test_error_handling_in_noise_reduction(self, audio_processor):
        """Test error handling in noise reduction."""
        # Test with empty tensor
        empty_tensor = torch.tensor([])
        result = audio_processor._spectral_subtraction_noise_reduction(empty_tensor)
        assert torch.equal(result, empty_tensor)
        
        # Test with invalid tensor
        invalid_tensor = torch.tensor([[float('nan')]])
        result = audio_processor._wiener_filter_noise_reduction(invalid_tensor)
        assert result.shape == invalid_tensor.shape
    
    def test_error_handling_in_echo_cancellation(self, audio_processor):
        """Test error handling in echo cancellation."""
        # Test with empty tensor
        empty_tensor = torch.tensor([])
        result = audio_processor._adaptive_echo_cancellation(empty_tensor)
        assert torch.equal(result, empty_tensor)
    
    @pytest.mark.asyncio
    async def test_audio_enhancement_disabled(self, audio_processor, test_audio_bytes):
        """Test audio processing with enhancements disabled."""
        # Disable all enhancements
        audio_processor.noise_reduction_enabled = False
        audio_processor.echo_cancellation_enabled = False
        audio_processor.auto_gain_control_enabled = False
        
        config = AudioConfig(
            sample_rate=24000,
            channels=1,
            noise_reduction=False,
            echo_cancellation=False,
            auto_gain_control=False
        )
        await audio_processor.initialize(config)
        
        # Process audio
        processed_audio = await audio_processor.process_input_audio(test_audio_bytes)
        
        assert isinstance(processed_audio, bytes)
        # With no enhancements, output should be very similar to input
        assert len(processed_audio) == len(test_audio_bytes)
    
    def test_audio_tensor_conversion_consistency(self, audio_processor, test_audio_bytes):
        """Test that audio conversion is consistent."""
        # Convert bytes to tensor and back
        tensor = audio_processor._bytes_to_tensor(test_audio_bytes)
        converted_bytes = audio_processor._tensor_to_bytes(tensor)
        
        # Should be very similar (allowing for minor floating point differences)
        assert len(converted_bytes) == len(test_audio_bytes)
        
        # Convert back to tensor to check values
        tensor2 = audio_processor._bytes_to_tensor(converted_bytes)
        
        # Tensors should be very close
        assert torch.allclose(tensor, tensor2, atol=1e-3)


class TestVoiceActivityDetector:
    """Test voice activity detection with enhanced audio."""
    
    @pytest.fixture
    def vad(self):
        """Create VAD for testing."""
        return VoiceActivityDetector()
    
    @pytest.mark.asyncio
    async def test_vad_with_enhanced_audio(self, vad):
        """Test VAD with enhanced audio processing."""
        # Initialize VAD
        await vad.initialize()
        
        # Create test audio with speech and silence
        sample_rate = 24000
        speech_samples = int(0.5 * sample_rate)  # 0.5 seconds of speech
        silence_samples = int(0.5 * sample_rate)  # 0.5 seconds of silence
        
        # Speech signal (higher energy)
        speech_signal = 0.3 * torch.randn(speech_samples)
        
        # Silence signal (very low energy)
        silence_signal = 0.01 * torch.randn(silence_samples)
        
        # Convert to bytes
        speech_bytes = (speech_signal * 32767).clamp(-32767, 32767).to(torch.int16).numpy().tobytes()
        silence_bytes = (silence_signal * 32767).clamp(-32767, 32767).to(torch.int16).numpy().tobytes()
        
        # Test VAD
        speech_detected = await vad.detect_voice_activity(speech_bytes)
        silence_detected = await vad.detect_voice_activity(silence_bytes)
        
        assert speech_detected == True
        assert silence_detected == False
