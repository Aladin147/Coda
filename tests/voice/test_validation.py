"""
Tests for validation utilities.
"""

import pytest
import io
import wave
from pathlib import Path
from unittest.mock import patch, mock_open

from src.coda.components.voice.validation import (
    validate_audio_data, validate_voice_config, validate_conversation_id,
    validate_timeout, validate_file_path, validate_audio_config, validate_moshi_config
)
from src.coda.components.voice.models import VoiceConfig, AudioConfig, MoshiConfig
from src.coda.components.voice.exceptions import ValidationError, ConfigurationError, ErrorCodes


class TestAudioValidation:
    """Test audio data validation."""
    
    def test_validate_audio_data_empty(self):
        """Test validation of empty audio data."""
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(b"")
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_FORMAT_INVALID
        assert "empty" in str(exc_info.value)
    
    def test_validate_audio_data_too_small(self):
        """Test validation of too small audio data."""
        small_data = b"tiny"  # Less than 44 bytes
        
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(small_data)
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_FORMAT_INVALID
        assert "too small" in str(exc_info.value)
    
    def test_validate_audio_data_too_large(self):
        """Test validation of too large audio data."""
        large_data = b"x" * (60 * 1024 * 1024)  # 60MB
        
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(large_data, max_size_mb=50)
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_SIZE_EXCEEDED
        assert "too large" in str(exc_info.value)
        assert exc_info.value.context["size_bytes"] == len(large_data)
    
    def test_validate_audio_data_valid_wav(self):
        """Test validation of valid WAV audio data."""
        # Create a minimal valid WAV file
        wav_data = io.BytesIO()
        with wave.open(wav_data, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(b'\x00\x00' * 1000)  # 1000 frames of silence
        
        audio_bytes = wav_data.getvalue()
        
        metadata = validate_audio_data(audio_bytes)
        
        assert metadata["format"] == "wav"
        assert metadata["channels"] == 1
        assert metadata["sample_rate"] == 16000
        assert metadata["sample_width"] == 2
        assert metadata["frame_count"] == 1000
        assert metadata["duration_seconds"] == 1000 / 16000
    
    def test_validate_audio_data_unsupported_sample_rate(self):
        """Test validation with unsupported sample rate."""
        wav_data = io.BytesIO()
        with wave.open(wav_data, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(11025)  # Unsupported rate
            wav_file.writeframes(b'\x00\x00' * 100)
        
        audio_bytes = wav_data.getvalue()
        
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(audio_bytes)
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_FORMAT_INVALID
        assert "Unsupported sample rate" in str(exc_info.value)
    
    def test_validate_audio_data_unsupported_channels(self):
        """Test validation with unsupported channel count."""
        wav_data = io.BytesIO()
        with wave.open(wav_data, 'wb') as wav_file:
            wav_file.setnchannels(5)  # 5.1 surround - unsupported
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(b'\x00\x00' * 100)
        
        audio_bytes = wav_data.getvalue()
        
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(audio_bytes)
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_FORMAT_INVALID
        assert "Unsupported channel count" in str(exc_info.value)
    
    def test_validate_audio_data_too_long(self):
        """Test validation of audio that's too long."""
        wav_data = io.BytesIO()
        with wave.open(wav_data, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # 10 minutes of audio (600 seconds)
            wav_file.writeframes(b'\x00\x00' * (16000 * 600))
        
        audio_bytes = wav_data.getvalue()
        
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(audio_bytes, max_duration_seconds=300)  # 5 minute limit
        
        assert exc_info.value.error_code == ErrorCodes.AUDIO_SIZE_EXCEEDED
        assert "too long" in str(exc_info.value)
    
    def test_validate_audio_data_raw_format(self):
        """Test validation of raw audio data."""
        # Raw audio data that can't be parsed as WAV
        raw_data = b'\x00\x01' * 1000  # 2000 bytes of raw audio
        
        metadata = validate_audio_data(raw_data)
        
        assert metadata["format"] == "raw"
        assert metadata["size_bytes"] == 2000
        assert "estimated_duration_seconds" in metadata


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_voice_config_none(self):
        """Test validation of None voice config."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_voice_config(None)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_MISSING
    
    def test_validate_voice_config_valid(self):
        """Test validation of valid voice config."""
        config = VoiceConfig(
            audio=AudioConfig(
                sample_rate=16000,
                channels=1,
                format="wav"
            ),
            moshi=MoshiConfig(
                device="cpu",
                vram_allocation="2GB"
            )
        )
        
        # Should not raise any exception
        validate_voice_config(config)
    
    def test_validate_audio_config_invalid_sample_rate(self):
        """Test validation of invalid sample rate."""
        config = AudioConfig(
            sample_rate=12000,  # Unsupported
            channels=1,
            format="wav"
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_audio_config(config)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "Invalid sample rate" in str(exc_info.value)
    
    def test_validate_audio_config_invalid_channels(self):
        """Test validation of invalid channel count."""
        config = AudioConfig(
            sample_rate=16000,
            channels=5,  # Unsupported
            format="wav"
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_audio_config(config)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "Invalid channel count" in str(exc_info.value)
    
    def test_validate_moshi_config_invalid_device(self):
        """Test validation of invalid device."""
        config = MoshiConfig(
            device="quantum",  # Invalid device
            vram_allocation="2GB"
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_moshi_config(config)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "Invalid device" in str(exc_info.value)
    
    def test_validate_moshi_config_invalid_vram_format(self):
        """Test validation of invalid VRAM allocation format."""
        config = MoshiConfig(
            device="cpu",
            vram_allocation="2TB"  # Invalid unit
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_moshi_config(config)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "Invalid VRAM allocation format" in str(exc_info.value)
    
    def test_validate_moshi_config_invalid_vram_value(self):
        """Test validation of invalid VRAM allocation value."""
        config = MoshiConfig(
            device="cpu",
            vram_allocation="-2GB"  # Negative value
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_moshi_config(config)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "Invalid VRAM allocation value" in str(exc_info.value)


class TestConversationValidation:
    """Test conversation ID validation."""
    
    def test_validate_conversation_id_empty(self):
        """Test validation of empty conversation ID."""
        with pytest.raises(ValidationError) as exc_info:
            validate_conversation_id("")
        
        assert exc_info.value.error_code == ErrorCodes.CONVERSATION_NOT_FOUND
    
    def test_validate_conversation_id_none(self):
        """Test validation of None conversation ID."""
        with pytest.raises(ValidationError) as exc_info:
            validate_conversation_id(None)
        
        assert exc_info.value.error_code == ErrorCodes.CONVERSATION_NOT_FOUND
    
    def test_validate_conversation_id_wrong_type(self):
        """Test validation of wrong type conversation ID."""
        with pytest.raises(ValidationError) as exc_info:
            validate_conversation_id(123)
        
        assert exc_info.value.error_code == ErrorCodes.CONVERSATION_INVALID_STATE
        assert "must be string" in str(exc_info.value)
    
    def test_validate_conversation_id_too_long(self):
        """Test validation of too long conversation ID."""
        long_id = "x" * 150  # Longer than 100 chars
        
        with pytest.raises(ValidationError) as exc_info:
            validate_conversation_id(long_id)
        
        assert exc_info.value.error_code == ErrorCodes.CONVERSATION_INVALID_STATE
        assert "too long" in str(exc_info.value)
    
    def test_validate_conversation_id_valid(self):
        """Test validation of valid conversation ID."""
        valid_id = "conversation-123-abc"
        
        # Should not raise any exception
        validate_conversation_id(valid_id)


class TestTimeoutValidation:
    """Test timeout validation."""
    
    def test_validate_timeout_none(self):
        """Test validation of None timeout."""
        result = validate_timeout(None, max_timeout=60.0)
        assert result == 60.0  # Should return max_timeout
    
    def test_validate_timeout_valid(self):
        """Test validation of valid timeout."""
        result = validate_timeout(10.5)
        assert result == 10.5
    
    def test_validate_timeout_wrong_type(self):
        """Test validation of wrong type timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout("10")
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "must be numeric" in str(exc_info.value)
    
    def test_validate_timeout_too_small(self):
        """Test validation of too small timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout(0.05, min_timeout=0.1)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "too small" in str(exc_info.value)
    
    def test_validate_timeout_too_large(self):
        """Test validation of too large timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout(500.0, max_timeout=300.0)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
        assert "too large" in str(exc_info.value)


class TestFilePathValidation:
    """Test file path validation."""
    
    def test_validate_file_path_empty(self):
        """Test validation of empty file path."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path("")
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_MISSING
    
    def test_validate_file_path_none(self):
        """Test validation of None file path."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(None)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_MISSING
    
    def test_validate_file_path_not_exists(self):
        """Test validation of non-existent file path."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path("/non/existent/file.txt", must_exist=True)
        
        assert exc_info.value.error_code == ErrorCodes.CONFIG_MISSING
        assert "does not exist" in str(exc_info.value)
    
    def test_validate_file_path_not_file(self):
        """Test validation of path that's not a file."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False):
            
            with pytest.raises(ValidationError) as exc_info:
                validate_file_path("/some/directory", must_exist=True)
            
            assert exc_info.value.error_code == ErrorCodes.CONFIG_INVALID
            assert "not a file" in str(exc_info.value)
    
    def test_validate_file_path_valid(self):
        """Test validation of valid file path."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            
            result = validate_file_path("/valid/file.txt", must_exist=True)
            assert isinstance(result, Path)
            assert str(result) == "/valid/file.txt"
    
    def test_validate_file_path_no_existence_check(self):
        """Test validation without existence check."""
        result = validate_file_path("/any/path.txt", must_exist=False)
        assert isinstance(result, Path)
        assert str(result) == "/any/path.txt"


if __name__ == "__main__":
    pytest.main([__file__])
