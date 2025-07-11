"""
Input validation utilities for the voice processing system.

This module provides validation functions for audio data, configurations,
and other inputs to ensure system stability and security.
"""

import io
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .exceptions import (
    AudioProcessingError,
    ConfigurationError,
    ErrorCodes,
    ValidationError,
    create_error,
)
from .models import AudioConfig, MoshiConfig, VoiceConfig

# Audio validation constants
MAX_AUDIO_SIZE_MB = 50  # Maximum audio file size in MB
MAX_AUDIO_DURATION_SECONDS = 300  # Maximum audio duration in seconds
SUPPORTED_SAMPLE_RATES = [8000, 16000, 22050, 24000, 44100, 48000]
SUPPORTED_CHANNELS = [1, 2]  # Mono and stereo
MIN_AUDIO_SIZE_BYTES = 44  # Minimum size for a valid audio header


def validate_audio_data(
    audio_data: bytes,
    max_size_mb: Optional[float] = None,
    max_duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate audio data and return metadata.

    Args:
        audio_data: Raw audio data bytes
        max_size_mb: Maximum allowed size in MB
        max_duration_seconds: Maximum allowed duration in seconds

    Returns:
        Dict containing audio metadata

    Raises:
        ValidationError: If audio data is invalid
        AudioProcessingError: If audio cannot be processed
    """
    if not audio_data:
        raise create_error(ValidationError, "Audio data is empty", ErrorCodes.AUDIO_FORMAT_INVALID)

    if len(audio_data) < MIN_AUDIO_SIZE_BYTES:
        raise create_error(
            ValidationError,
            f"Audio data too small: {len(audio_data)} bytes (minimum: {MIN_AUDIO_SIZE_BYTES})",
            ErrorCodes.AUDIO_FORMAT_INVALID,
            size_bytes=len(audio_data),
        )

    # Check size limit
    max_size_bytes = (max_size_mb or MAX_AUDIO_SIZE_MB) * 1024 * 1024
    if len(audio_data) > max_size_bytes:
        raise create_error(
            ValidationError,
            f"Audio data too large: {len(audio_data) / 1024 / 1024:.1f}MB "
            f"(maximum: {max_size_mb or MAX_AUDIO_SIZE_MB}MB)",
            ErrorCodes.AUDIO_SIZE_EXCEEDED,
            size_bytes=len(audio_data),
            max_size_bytes=max_size_bytes,
        )

    # Try to parse as WAV to get metadata
    metadata = {}
    try:
        with io.BytesIO(audio_data) as audio_io:
            with wave.open(audio_io, "rb") as wav_file:
                metadata = {
                    "format": "wav",
                    "channels": wav_file.getnchannels(),
                    "sample_rate": wav_file.getframerate(),
                    "sample_width": wav_file.getsampwidth(),
                    "frame_count": wav_file.getnframes(),
                    "duration_seconds": wav_file.getnframes() / wav_file.getframerate(),
                    "size_bytes": len(audio_data),
                }

                # Validate sample rate
                if metadata["sample_rate"] not in SUPPORTED_SAMPLE_RATES:
                    raise create_error(
                        ValidationError,
                        f"Unsupported sample rate: {metadata['sample_rate']}Hz "
                        f"(supported: {SUPPORTED_SAMPLE_RATES})",
                        ErrorCodes.AUDIO_FORMAT_INVALID,
                        sample_rate=metadata["sample_rate"],
                    )

                # Validate channels
                if metadata["channels"] not in SUPPORTED_CHANNELS:
                    raise create_error(
                        ValidationError,
                        f"Unsupported channel count: {metadata['channels']} "
                        f"(supported: {SUPPORTED_CHANNELS})",
                        ErrorCodes.AUDIO_FORMAT_INVALID,
                        channels=metadata["channels"],
                    )

                # Check duration limit
                max_duration = max_duration_seconds or MAX_AUDIO_DURATION_SECONDS
                if metadata["duration_seconds"] > max_duration:
                    raise create_error(
                        ValidationError,
                        f"Audio too long: {metadata['duration_seconds']:.1f}s "
                        f"(maximum: {max_duration}s)",
                        ErrorCodes.AUDIO_SIZE_EXCEEDED,
                        duration_seconds=metadata["duration_seconds"],
                        max_duration_seconds=max_duration,
                    )

    except wave.Error as e:
        # Try to handle as raw audio data
        metadata = {
            "format": "raw",
            "size_bytes": len(audio_data),
            "estimated_duration_seconds": len(audio_data) / (16000 * 2),  # Assume 16kHz, 16-bit
        }

        # For raw audio, we can't validate format details
        # but we can still check basic size constraints

    except Exception as e:
        raise create_error(
            AudioProcessingError,
            f"Failed to parse audio data: {str(e)}",
            ErrorCodes.AUDIO_CORRUPTION,
            original_error=str(e),
        )

    return metadata


def validate_voice_config(config: VoiceConfig) -> None:
    """
    Validate voice configuration.

    Args:
        config: Voice configuration to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not config:
        raise create_error(
            ConfigurationError, "Voice configuration is None", ErrorCodes.CONFIG_MISSING
        )

    # Validate audio config
    if config.audio:
        validate_audio_config(config.audio)

    # Validate Moshi config
    if config.moshi:
        validate_moshi_config(config.moshi)

    # Validate processing mode
    if hasattr(config, "processing_mode") and config.processing_mode:
        valid_modes = ["moshi_only", "llm_enhanced", "hybrid", "adaptive"]
        if config.processing_mode not in valid_modes:
            raise create_error(
                ConfigurationError,
                f"Invalid processing mode: {config.processing_mode} " f"(valid: {valid_modes})",
                ErrorCodes.CONFIG_INVALID,
                processing_mode=config.processing_mode,
            )


def validate_audio_config(config: AudioConfig) -> None:
    """
    Validate audio configuration.

    Args:
        config: Audio configuration to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not config:
        raise create_error(
            ConfigurationError, "Audio configuration is None", ErrorCodes.CONFIG_MISSING
        )

    # Validate sample rate
    if config.sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise create_error(
            ConfigurationError,
            f"Invalid sample rate: {config.sample_rate}Hz "
            f"(supported: {SUPPORTED_SAMPLE_RATES})",
            ErrorCodes.CONFIG_INVALID,
            sample_rate=config.sample_rate,
        )

    # Validate channels
    if config.channels not in SUPPORTED_CHANNELS:
        raise create_error(
            ConfigurationError,
            f"Invalid channel count: {config.channels} " f"(supported: {SUPPORTED_CHANNELS})",
            ErrorCodes.CONFIG_INVALID,
            channels=config.channels,
        )


def validate_moshi_config(config: MoshiConfig) -> None:
    """
    Validate Moshi configuration.

    Args:
        config: Moshi configuration to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not config:
        raise create_error(
            ConfigurationError, "Moshi configuration is None", ErrorCodes.CONFIG_MISSING
        )

    # Validate device
    valid_devices = ["cpu", "cuda", "auto"]
    if config.device not in valid_devices:
        raise create_error(
            ConfigurationError,
            f"Invalid device: {config.device} (valid: {valid_devices})",
            ErrorCodes.CONFIG_INVALID,
            device=config.device,
        )

    # Validate VRAM allocation format
    if config.vram_allocation:
        if not config.vram_allocation.endswith(("MB", "GB")):
            raise create_error(
                ConfigurationError,
                f"Invalid VRAM allocation format: {config.vram_allocation} "
                "(must end with 'MB' or 'GB')",
                ErrorCodes.CONFIG_INVALID,
                vram_allocation=config.vram_allocation,
            )

        # Extract numeric value
        try:
            value = float(config.vram_allocation[:-2])
            if value <= 0:
                raise ValueError("Must be positive")
        except ValueError:
            raise create_error(
                ConfigurationError,
                f"Invalid VRAM allocation value: {config.vram_allocation}",
                ErrorCodes.CONFIG_INVALID,
                vram_allocation=config.vram_allocation,
            )


def validate_conversation_id(conversation_id: str) -> None:
    """
    Validate conversation ID format.

    Args:
        conversation_id: Conversation ID to validate

    Raises:
        ValidationError: If conversation ID is invalid
    """
    if not conversation_id:
        raise create_error(
            ValidationError, "Conversation ID is empty", ErrorCodes.CONVERSATION_NOT_FOUND
        )

    if not isinstance(conversation_id, str):
        raise create_error(
            ValidationError,
            f"Conversation ID must be string, got {type(conversation_id)}",
            ErrorCodes.CONVERSATION_INVALID_STATE,
            conversation_id_type=type(conversation_id).__name__,
        )

    if len(conversation_id) > 100:
        raise create_error(
            ValidationError,
            f"Conversation ID too long: {len(conversation_id)} chars (max: 100)",
            ErrorCodes.CONVERSATION_INVALID_STATE,
            conversation_id_length=len(conversation_id),
        )


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate file path.

    Args:
        file_path: File path to validate
        must_exist: Whether file must exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid
    """
    if not file_path:
        raise create_error(ValidationError, "File path is empty", ErrorCodes.CONFIG_MISSING)

    path = Path(file_path)

    if must_exist and not path.exists():
        raise create_error(
            ValidationError,
            f"File does not exist: {path}",
            ErrorCodes.CONFIG_MISSING,
            file_path=str(path),
        )

    if must_exist and not path.is_file():
        raise create_error(
            ValidationError,
            f"Path is not a file: {path}",
            ErrorCodes.CONFIG_INVALID,
            file_path=str(path),
        )

    return path


def validate_timeout(
    timeout: Optional[float], min_timeout: float = 0.1, max_timeout: float = 300.0
) -> float:
    """
    Validate timeout value.

    Args:
        timeout: Timeout value in seconds
        min_timeout: Minimum allowed timeout
        max_timeout: Maximum allowed timeout

    Returns:
        Validated timeout value

    Raises:
        ValidationError: If timeout is invalid
    """
    if timeout is None:
        return max_timeout

    if not isinstance(timeout, (int, float)):
        raise create_error(
            ValidationError,
            f"Timeout must be numeric, got {type(timeout)}",
            ErrorCodes.CONFIG_INVALID,
            timeout_type=type(timeout).__name__,
        )

    if timeout < min_timeout:
        raise create_error(
            ValidationError,
            f"Timeout too small: {timeout}s (minimum: {min_timeout}s)",
            ErrorCodes.CONFIG_INVALID,
            timeout=timeout,
            min_timeout=min_timeout,
        )

    if timeout > max_timeout:
        raise create_error(
            ValidationError,
            f"Timeout too large: {timeout}s (maximum: {max_timeout}s)",
            ErrorCodes.CONFIG_INVALID,
            timeout=timeout,
            max_timeout=max_timeout,
        )

    return float(timeout)
